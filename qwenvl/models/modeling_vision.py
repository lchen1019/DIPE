# Copyright 2025 Lin Chen. All rights reserved.
# This code is based on the implementation of Siglip, Qwen2.5-VL
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils import TransformersKwargs, auto_docstring
from transformers import Qwen2RMSNorm
from transformers.modeling_outputs import BaseModelOutput
from .configuration_qwen25_siglip import Siglip2VisionConfig
from transformers.utils.generic import check_model_inputs
from .modeling_utils import Qwen25_SigLIPPreTrainedModel

logger = logging.get_logger(__name__)
try:
    from torchvision.transforms.functional import gaussian_blur
    _has_torchvision = True
except ImportError:
    _has_torchvision = False


class Merger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        # self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).view(-1, self.hidden_size)
        x = self.mlp(x)
        return x

class Siglip2VisionEmbeddings(nn.Module):
    """
    from qwen3-vl
    linear -> 3d conv to support native resolution
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.num_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.patch_embedding = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.patch_embedding(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1),
            self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1),
            self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1),
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            # attn_output, _, attn_lse = attention_interface(
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scale,
                dropout=0.0 if not self.training else self.dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scale,
                    dropout=0.0 if not self.training else self.dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)
        
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Union[Siglip2VisionConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
 
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Siglip2EncoderLayer`].

    Args:
        config: Siglip2VisionConfig
    """

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                attention_mask,
                **kwargs,
            )
        return BaseModelOutput(last_hidden_state=hidden_states)


class Siglip2VisionTransformerPretrainedModel(Qwen25_SigLIPPreTrainedModel):
    config: Siglip2VisionConfig
    _input_embed_layer = "patch_embedding"
    _can_record_outputs = {
        "hidden_states": Siglip2EncoderLayer,
        "attentions": Siglip2Attention,
    }
    _no_split_modules = [
        "Siglip2VisionEmbeddings",
        "Siglip2EncoderLayer",
    ]

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)
        self.encoder = Siglip2Encoder(config)
        self.adaptor = Merger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.num_grid_per_side = int(config.num_patches ** 0.5)
        self.post_init()

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.position_embedding.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.position_embedding.weight.dtype, device=self.position_embedding.weight.device
        )
        pos_embeds = self.position_embedding(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        grid_thw (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(pixel_values)
        resized_positional_embeddings = self.fast_pos_embed_interpolate(grid_thw)

        # Add positional embeddings to patch embeddings
        hidden_states = hidden_states + resized_positional_embeddings

        # if attention_mask is not None and self.config._attn_implementation != "flash_attention_2":
        #     # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
        #     encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        # else:
        #     encoder_attention_mask = attention_mask

        # from qwenvl for packing sequence
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = encoder_outputs.last_hidden_state

        # vision-language adaptor
        hidden_state = self.adaptor(hidden_state)
        
        return hidden_state



__all__ = ["Siglip2VisionTransformerPretrainedModel"]
