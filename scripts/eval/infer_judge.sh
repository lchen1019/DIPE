#!/bin/bash
LOCAL_IP=$LOCAL_IP
LOG_DIR="eval_output/logs/transformers_instances"
mkdir -p ${LOG_DIR}

declare -A INSTANCE_CONFIG
INSTANCE_CONFIG[0]="0,1:8200"
INSTANCE_CONFIG[1]="2,3:8201"
INSTANCE_CONFIG[2]="4,5:8202"
INSTANCE_CONFIG[3]="6,7:8203"

echo "============================================"
echo "Starting Transformers FastAPI instances..."
echo "============================================"

for i in {0..3}; do
    CONFIG=${INSTANCE_CONFIG[$i]}
    GPU_ID=$(echo $CONFIG | cut -d':' -f1)
    PORT=$(echo $CONFIG | cut -d':' -f2)
    
    echo "Starting instance $i on GPU $GPU_ID, port $PORT..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    vllm serve /path/to/judge/model \
        --served-model-name qwen_25 \
        --api-key sk-abc123 \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 1 \
        --trust-remote-code \
        --dtype auto \
        --gpu_memory_utilization 0.70 \
        --port $PORT \
        --host $LOCAL_IP &
    
    echo "Instance $i started."
done
