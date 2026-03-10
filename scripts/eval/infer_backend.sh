#!/bin/bash
CONFIG_PATH="scripts/eval/distribution_config.env"
source $CONFIG_PATH
MODEL_PATH=$EVAL_MODEL_PATH

LOCAL_IP=$LOCAL_IP
LOG_DIR="eval_output/logs/transformers_instances"
mkdir -p ${LOG_DIR}

declare -A INSTANCE_CONFIG
INSTANCE_CONFIG[0]="0:8100"
INSTANCE_CONFIG[1]="1:8101"
INSTANCE_CONFIG[2]="2:8102"
INSTANCE_CONFIG[3]="3:8103"
INSTANCE_CONFIG[4]="4:8104"
INSTANCE_CONFIG[5]="5:8105"
INSTANCE_CONFIG[6]="6:8106"
INSTANCE_CONFIG[7]="7:8107"

echo "============================================"
echo "Starting Transformers FastAPI instances..."
echo "============================================"

for i in {0..7}; do
    CONFIG=${INSTANCE_CONFIG[$i]}
    GPU_ID=$(echo $CONFIG | cut -d':' -f1)
    PORT=$(echo $CONFIG | cut -d':' -f2)
    
    echo "Starting instance $i on GPU $GPU_ID, port $PORT..."

    CUDA_VISIBLE_DEVICES=$GPU_ID python qwenvl/eval/transformers_backend.py \
        --port ${PORT} \
        --host ${LOCAL_IP} \
        --model-path ${MODEL_PATH} \
        > ${LOG_DIR}/instance_${i}.log 2>&1 &
    
    echo "Instance $i started."
done
