#!/bin/bash

# set env
export WANDB_PROJECT="qwen-vl-finetune"
export WANDB_API_KEY="xxxxxxxxxxxxxxxxx"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=scripts/deepspeed/zero2.json

# Model configuration
mllm="/path/to/qwen25_3b_pretrain"
llm="/path/to/Qwen/Qwen2.5-3B"
visual="/path/to/google/siglip2-so400m-patch16-naflex"
processor="/path/to/Qwen/processor_config" # qwen3-vl for siglip2

# Training hyperparameters
lr=2e-5
batch_size=8
grad_accum_steps=2

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=llava_next_sft

# Output configuration
run_name="qwen25_3b_sft"
output_dir=/apdcephfs_gy4/share_303999727/capychen/ckpt/${run_name}

# Training arguments
# max_pixels和min_pixels控制visual token的个数，4-1280个patches
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${mllm} \
    --llm_path ${llm} \
    --visual_encoder_path ${visual} \
    --processor_path ${processor} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels $((32*32*1280)) \
    --min_pixels $((32*32*4)) \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 32 \
    --run_name ${run_name} \
    --report_to wandb"

# export CUDA_LAUNCH_BLOCKING=1 
# export TORCH_USE_CUDA_DSA=1
# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}