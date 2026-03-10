# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import AutoProcessor, AutoTokenizer, Trainer
from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.models import Qwen25_SigLIPForConditionalGeneration
from qwenvl.models import Qwen2Config, Siglip2VisionConfig, Qwen25_SigLIPConfig
from safetensors.torch import load_file

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.adaptor.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.adaptor.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def load_weights_from_folder(model, folder_path, load):
    """
    Loads weights from a folder containing safetensors or bin files.
    :param model: The model to load weights into.
    :param folder_path: The path to the folder containing the weights.
    load to cpu to avoid deepspeed
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.safetensors') or f.endswith('.bin')]
    files.sort()
    # tqdm
    model_keys = set(model.state_dict().keys())
    new_state_dict = {}
    for i in tqdm(range(len(files))):
        file = files[i]
        file_path = os.path.join(folder_path, file)
        
        if file.endswith('.safetensors'):
            state_dict = load_file(file_path, device="cpu")
        else:
            state_dict = torch.load(file_path, map_location="cpu")

        for key in state_dict.keys():
            if load == 'visual':
                new_key = key.replace("vision_model.", "")
                if 'embeddings.position_embedding.weight' in new_key:
                    new_key = 'position_embedding.weight'
            elif load == 'language_model':
                new_key = key.replace("model.", "")
            else:
                raise ValueError(f'Unknown load type: {load}')
            
            if new_key in model_keys:
                new_state_dict[new_key] = state_dict[key]

    if load == 'visual':
        # reshape embeddings.patch_embedding.weight to conv3d
        src_weight = new_state_dict['embeddings.patch_embedding.weight']
        # conv2d_weight = src_weight.view(1152, 3, 16, 16) # patch 16x16, channel 3
        conv2d_weight = src_weight.view(1152, 16, 16, 3) 
        conv2d_weight = conv2d_weight.permute(0, 3, 1, 2).contiguous()
        conv3d_weight = conv2d_weight.unsqueeze(2)
        conv3d_weight = conv3d_weight.repeat(1, 1, 2, 1, 1)
        conv3d_weight = conv3d_weight / 2.0
        new_state_dict['embeddings.patch_embedding.weight'] = conv3d_weight
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f'Loaded weights from folder: {folder_path}: {msg}')
    return model


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    data_args.model_type = "qwen2.5vl"

    processor = None
    tokenizer = None
    model = None


    # TODO
    # this can be optimized
    if model_args.model_name_or_path and os.path.exists(model_args.model_name_or_path):
        print(f'the initlized model is {model_args.model_name_or_path}...')

        model = Qwen25_SigLIPForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )

        try:
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        except:
            print(f'{model_args.model_name_or_path} has no processor, load from {model_args.processor_path}')
            processor = None

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # truncation=True,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        print("Loading model from scratch...")
        # config
        text_config = Qwen2Config.from_pretrained(model_args.llm_path)
        vision_config = Siglip2VisionConfig.from_pretrained(model_args.visual_encoder_path)

        # flash attn
        text_config._attn_implementation = "flash_attention_2"
        vision_config._attn_implementation = "flash_attention_2"

        del text_config._name_or_path
        del vision_config._name_or_path
        config = Qwen25_SigLIPConfig(text_config=text_config.to_dict(), vision_config=vision_config.to_dict())
        config._attn_implementation = "flash_attention_2"

        model = Qwen25_SigLIPForConditionalGeneration(config)

        load_weights_from_folder(model.language_model, model_args.llm_path, load="language_model")
        load_weights_from_folder(model.visual, model_args.visual_encoder_path, load="visual")

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.llm_path,
            cache_dir=training_args.cache_dir,
            # truncation=True,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    

    if processor is None:
        processor = AutoProcessor.from_pretrained(model_args.processor_path)
    
    # check    
    assert processor is not None, "processor is None"
    assert tokenizer is not None, "tokenizer is None"
    assert model is not None, "model is None"
    
    # packing
    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    # gradient checkpointing
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # LoRA
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.model.print_trainable_parameters()
            model.visual.print_trainable_parameters()
    
    # data
    data_module = make_supervised_data_module(processor, data_args=data_args)

    processor.save_pretrained(training_args.output_dir)

    # trainer 
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
