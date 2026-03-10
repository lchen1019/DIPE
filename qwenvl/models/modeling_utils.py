# Copyright 2025 Lin Chen. All rights reserved.
# This code is based on the implementation of Qwen2.5-VL
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

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring
from .configuration_qwen25_siglip import Qwen25_SigLIPConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@auto_docstring
class Qwen25_SigLIPPreTrainedModel(PreTrainedModel):
    config: Qwen25_SigLIPConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen25_SigLIPDecoderLayer", "Siglip2EncoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
