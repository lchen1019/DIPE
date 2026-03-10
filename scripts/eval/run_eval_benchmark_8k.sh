# model config
export VLMEVALKIT_USE_MODELSCOPE=1
export OPENAI_API_KEY=sk-abc123
export OPENAI_API_BASE="http://xxx.xxx.xxx.xxx:7777/v1/chat/completions" # judge model
export LOCAL_LLM=qwen_25
export MACHINE_IP=${LOCAL_IP}
export MACHINE_PORT=7777
export LMDEPLOY_API_KEY=sk-abc123

# data config
export MODELSCOPE_CACHE="/apdcephfs/share_302735770/allenqmpeng/MODELSCOPE_CACHE"
export HF_HOME="/apdcephfs/share_302735770/allenqmpeng/HF_home_cache"
export LMUData="/apdcephfs/share_302735770/allenqmpeng/LMUData"

# eval config
MODEL=api_qwen2_5_siglip_3b_instruct
OUTPUT_DIR=eval_output

CONFIG_PATH="scripts/eval/distribution_config.env"
source $CONFIG_PATH
SAVE_NAME=${SAVE_NAME}
MODE=${MODE}

echo "=============================="
echo "Starting eval benchmark..."
echo "SAVE_NAME: ${SAVE_NAME}"
echo "MODE: ${MODE}"
echo "=============================="

data_list=(
    # Perception
    HRBench4K
    HRBench8K
    VStarBench
    POPE
    CountBenchQA
    BLINK

    # Document Understanding
    InfoVQA_VAL
    ChartQA_TEST
    DocVQA_VAL
    TextVQA_VAL
    AI2D_TEST
    OCRBench

    # General VQA
    RealWorldQA
    MMStar
    MMBench_DEV_EN_V11
    MMBench_DEV_CN_V11
    MMVP
    MathVista_MINI
    MathVision_MINI
)


for DATA in ${data_list[@]}
do
    echo "evaluate ${DATA}";
    echo "save name: ${SAVE_NAME}";
    mkdir -p $OUTPUT_DIR/${SAVE_NAME}_8k
    python /path/to/eval_benchmarks_8k/run_vlmeval.py \
    --data ${DATA} \
    --model $MODEL \
    --api-nproc 64 \
    --judge qwen_25 \
    --work-dir $OUTPUT_DIR/${SAVE_NAME}_8k | tee -a $OUTPUT_DIR/${SAVE_NAME}_8k/${DATA}.log
done
