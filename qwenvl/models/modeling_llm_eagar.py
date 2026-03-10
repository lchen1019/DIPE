from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers import Qwen2RMSNorm
from transformers import Qwen2Config
from .modeling_utils import Qwen25_SigLIPPreTrainedModel, rotate_half

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def decoupled_eager_attention_forward(
    module: nn.Module,
    static_query: torch.Tensor,
    dynamic_query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    inter_attention_mask: torch.Tensor,
    intra_attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    chunk_size: int = 128,
    output_attentions: bool = False,
    **kwargs,
):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    bsz, num_heads, q_len, head_dim = static_query.shape

    if inter_attention_mask.dim() == 3:
        inter_attention_mask = inter_attention_mask.unsqueeze(1)
    if intra_attention_mask.dim() == 3:
        intra_attention_mask = intra_attention_mask.unsqueeze(1)

    chunk_size = (((4096 * 4096) / q_len) // 128) * 128
    chunk_size = min(chunk_size, q_len)
    chunk_size = int(chunk_size)

    output_chunks = []
    attn_weights_chunks = []

    for i in range(0, q_len, chunk_size):
        end = min(i + chunk_size, q_len)

        q_static_chunk = static_query[:, :, i:end, :]
        q_dynamic_chunk = dynamic_query[:, :, i:end, :]

        mask_inter_chunk = inter_attention_mask[:, :, i:end, :]
        mask_intra_chunk = intra_attention_mask[:, :, i:end, :]

        attn_static_chunk = torch.matmul(q_static_chunk, key.transpose(-1, -2).contiguous()) * scaling
        attn_dynamic_chunk = torch.matmul(q_dynamic_chunk, key.transpose(-1, -2).contiguous()) * scaling

        attn_weights = torch.where(mask_intra_chunk, attn_dynamic_chunk, attn_static_chunk)

        mask_keep = mask_intra_chunk | mask_inter_chunk
        
        min_val = torch.tensor(-1e7, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(mask_keep, attn_weights, min_val)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

        chunk_output = torch.matmul(attn_weights, value)
        output_chunks.append(chunk_output)
        
        # attention weights
        if output_attentions:
            with torch.no_grad():
                avg_attn = attn_weights.mean(dim=1, keepdim=True).detach().to(torch.float16).cpu()
                attn_weights_chunks.append(avg_attn)
            del attn_static_chunk, attn_dynamic_chunk
        else:
            del attn_static_chunk, attn_dynamic_chunk, attn_weights

    attn_output = torch.cat(output_chunks, dim=2)
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    # attention weights are averaged over heads and returned in float16 to save memory, and only if output_attentions is True
    full_attn_weights = torch.cat(attn_weights_chunks, dim=2) if output_attentions else None

    return attn_output, full_attn_weights


class Qwen25_SigLIPRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen25_SigLIP has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def apply_multimodal_rotary_pos_emb(x, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class Qwen25_SigLIPAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.rotary_emb = Qwen25_SigLIPRotaryEmbedding(config=config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC

        static_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        kv_valid_mask: Optional[torch.Tensor] = None,
        q_valid_mask: Optional[torch.Tensor] = None,
        # MODIFIED: Changed from _block_mask to standard boolean mask
        inter_mask: Optional[torch.Tensor] = None,
        intra_mask: Optional[torch.Tensor] = None,

        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # TODO
        # We performed a key-value update first, then a rope (this can be optimized)
        cos, sin = position_embeddings
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states_dynamic = apply_multimodal_rotary_pos_emb(
            key_states, cos, sin, self.rope_scaling["mrope_section"]
        )
        query_states_dynamic = apply_multimodal_rotary_pos_emb(
            query_states, cos[:, :, -query_states.shape[2]:], sin[:, :, -query_states.shape[2]:], self.rope_scaling["mrope_section"]
        )

        static_cos, static_sin = static_position_embeddings
        query_states_static = apply_multimodal_rotary_pos_emb(
            query_states, static_cos[:, :, -query_states.shape[2]:], static_sin[:, :, -query_states.shape[2]:], self.rope_scaling["mrope_section"]
        )
    
        k_dyn_bshd = key_states_dynamic.transpose(1, 2)
        k_dyn_bhsd = k_dyn_bshd[kv_valid_mask].unsqueeze(0).transpose(1, 2)

        v_bshd = value_states.transpose(1, 2)
        v_bhsd = v_bshd[kv_valid_mask].unsqueeze(0).transpose(1, 2)

        q_dyn_bshd = query_states_dynamic.transpose(1, 2)
        q_static_bshd = query_states_static.transpose(1, 2)
        q_dyn_bhsd = q_dyn_bshd[q_valid_mask].unsqueeze(0).transpose(1, 2)
        q_static_bhsd = q_static_bshd[q_valid_mask].unsqueeze(0).transpose(1, 2)

        attn_output, attn_weights = decoupled_eager_attention_forward(
            self,
            static_query=q_static_bhsd,
            dynamic_query=q_dyn_bhsd,
            key=k_dyn_bhsd,
            value=v_bhsd,
            inter_attention_mask=inter_mask[None, None, :, :],
            intra_attention_mask=intra_mask[None, None, :, :],
            scaling=self.scaling,
            dropout_p=self.attention_dropout if self.training else 0.0,
            output_attentions=output_attentions,
        )

        if not self.training:
            _attn_output = torch.zeros((bsz, q_len, self.num_heads, self.head_dim), dtype=attn_output.dtype, device=attn_output.device)
            _attn_output[q_valid_mask] = attn_output.squeeze(0)
            attn_output = _attn_output

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Qwen25_SigLIPDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen25_SigLIPAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC

        # args for DIPE
        static_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        kv_valid_mask: Optional[torch.Tensor] = None,
        q_valid_mask: Optional[torch.Tensor] = None,
        inter_mask: Optional[torch.Tensor] = None,
        intra_mask: Optional[torch.Tensor] = None,

        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,

            static_position_embeddings=static_position_embeddings,
            kv_valid_mask=kv_valid_mask,
            q_valid_mask=q_valid_mask,
            inter_mask=inter_mask,
            intra_mask=intra_mask,

            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class Qwen25_SigLIPTextModel(Qwen25_SigLIPPreTrainedModel):
    config: Qwen2Config

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen25_SigLIPDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen25_SigLIPRotaryEmbedding(config=config)
        self.static_rotary_emb = Qwen25_SigLIPRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        
        # args for DIPE
        static_position_ids: Optional[torch.LongTensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None
        
        
        # =====================================================================
        # Create standard boolean masks for eagar attention
        # =====================================================================
        B, L = visual_mask.shape
        device = visual_mask.device
        
        position_map = torch.arange(L, device=device).unsqueeze(0).expand(B, L).reshape(-1)
        batch_map = torch.arange(B, device=device).unsqueeze(1).expand(B, L).reshape(-1)

        is_visual_flat = visual_mask.reshape(-1)
        is_text_flat = text_mask.reshape(-1)
        valid_mask_flat = valid_mask.reshape(-1)
        
        k_batch_ids = batch_map[valid_mask_flat].int()
        k_pos_ids   = position_map[valid_mask_flat].int()
        k_is_visual = is_visual_flat[valid_mask_flat]
        k_is_text   = is_text_flat[valid_mask_flat]

        # prefilling or training
        if inputs_embeds.shape[1] == valid_mask.shape[1]:
            q_valid_mask = valid_mask

            q_batch_ids = k_batch_ids
            q_pos_ids   = k_pos_ids
            q_is_visual = k_is_visual
            q_is_text   = k_is_text
        # decoding
        else:
            seqlens = valid_mask.sum(dim=1)
            cu_seqlens = seqlens.cumsum(dim=0).to(dtype=torch.long)
            q_indices = cu_seqlens - 1
            q_valid_mask = torch.ones((B, 1), device=device, dtype=torch.bool).expand(B, inputs_embeds.shape[1])
            
            q_batch_ids = k_batch_ids[q_indices]
            q_pos_ids   = k_pos_ids[q_indices]
            q_is_visual = k_is_visual[q_indices]
            q_is_text   = k_is_text[q_indices]

        # import pdb; pdb.set_trace()
        if self.training:
            flat_position_ids = position_ids[0, 0].reshape(-1)
            is_new_sample = flat_position_ids[1:] < flat_position_ids[:-1]
            sample_boundaries = torch.cat([torch.tensor([False], device=device), is_new_sample])
            packed_batch_ids = sample_boundaries.cumsum(dim=0).int()
            # During training with packing, batch_ids are derived from position_ids
            q_batch_ids = packed_batch_ids[q_valid_mask.view(-1)]
            k_batch_ids = packed_batch_ids[valid_mask.view(-1)]

        q_pos_ids_exp = q_pos_ids.unsqueeze(1)
        k_pos_ids_exp = k_pos_ids.unsqueeze(0)
        q_batch_ids_exp = q_batch_ids.unsqueeze(1)
        k_batch_ids_exp = k_batch_ids.unsqueeze(0)
        q_is_visual_exp = q_is_visual.unsqueeze(1)
        k_is_visual_exp = k_is_visual.unsqueeze(0)
        q_is_text_exp = q_is_text.unsqueeze(1)
        k_is_text_exp = k_is_text.unsqueeze(0)

        same_batch_mask = (q_batch_ids_exp == k_batch_ids_exp)
        causal_mask = (q_pos_ids_exp >= k_pos_ids_exp)
        
        # Intra-modal mask: (text->text) or (visual->visual)
        intra_modal_logic = (q_is_visual_exp & k_is_visual_exp) | (q_is_text_exp & k_is_text_exp)
        intra_mask = same_batch_mask & causal_mask & intra_modal_logic

        # Inter-modal mask: (text->visual) or (visual->text)
        inter_modal_logic = (q_is_visual_exp & k_is_text_exp) | (q_is_text_exp & k_is_visual_exp)
        inter_mask = same_batch_mask & causal_mask & inter_modal_logic
        # =====================================================================

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        static_position_embeddings = self.static_rotary_emb(hidden_states, static_position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None, # sequence packing is handled by masks
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,

                # args for DIPE
                static_position_embeddings=static_position_embeddings,
                kv_valid_mask=valid_mask,
                q_valid_mask=q_valid_mask,
                # MODIFIED: Pass standard boolean masks
                intra_mask=intra_mask,
                inter_mask=inter_mask,

                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )