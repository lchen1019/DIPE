from transformers import Trainer
from transformers.utils import logging

from qwenvl.models import Qwen25_SigLIPModel, Siglip2VisionTransformerPretrainedModel
    
logger = logging.get_logger(__name__)

def return_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    **kwargs
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    import qwenvl

    qwenvl.models.modeling_qwen25_siglip.create_causal_mask = (
        return_mask
    )
    qwenvl.models.modeling_qwen25_siglip.create_sliding_window_causal_mask = (
        return_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including embeddings, encoder layers, and adaptor.
    Outputs the indices of trainable/non-trainable layers and the adaptor status.
    """
    # 1. Check Embeddings status
    is_embeddings_trainable = any(
        param.requires_grad for param in self.embeddings.parameters()
    )

    trainable_layers = []
    non_trainable_layers = []

    # 2. Check Encoder Layers status (mapped from self.blocks to self.encoder.layers)
    # We use self.encoder.layers based on the Siglip2Encoder definition
    for layer_idx, layer in enumerate(self.encoder.layers):
        # Using 'any' ensures that if LoRA or specific norms are trainable, it counts as trainable
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # 3. Check Adaptor/Merger status (mapped from self.adaptor to self.adaptor)
    is_adaptor_trainable = any(param.requires_grad for param in self.adaptor.parameters())

    # Print results
    print(f"Vision Module - Embeddings Trainable: {is_embeddings_trainable}")
    
    print("Vision Module - Encoder Layers:")
    print(
        f"Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )
    
    print(f"Vision Module - Adaptor (Merger) Trainable: {is_adaptor_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.language_model.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.language_model.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "adaptor" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )

        if self.args.bf16:
            current_eps = optimizer_kwargs.get("eps", 1e-8)
            if current_eps < 1e-7:
                print(f"WARNING: BF16 training detected. Boosting AdamW epsilon from {current_eps} to 1e-7 to prevent NaNs.")
                optimizer_kwargs["eps"] = 1e-7
        
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    # import pdb; pdb.set_trace()
    return self.optimizer


Trainer.create_optimizer = create_optimizer
Qwen25_SigLIPModel.print_trainable_parameters = print_trainable_parameters
Siglip2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
