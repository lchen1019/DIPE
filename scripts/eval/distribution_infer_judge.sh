#!/bin/bash
# ==============================================================================
#        Multi-node inference backend automated deployment script
# ==============================================================================

CONFIG_PATH="scripts/eval/distribution_config.env"
REMOTE_USER="root"
CONDA_ACTIVATE="/jizhicfs/capychen/miniconda3/bin/activate"
CONDA_ENV="modified_eval"
INFER_SCRIPT="scripts/eval/infer_judge.sh"
LB_SCRIPT="scripts/eval/balance_deploy_api.sh"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ 错误: 配置文件 $CONFIG_PATH 不存在"
    exit 1
fi

source $CONFIG_PATH
PROJECT_PATH=${PROJECT_PATH}
WORKER_HOSTS=${WORKER_HOSTS}

echo "🔍 Project Path: $PROJECT_PATH"
echo "🔍 Worker hosts: $WORKER_HOSTS"

echo "🧹 cleaning up old inference processes on each node...."
pkill -f "$INFER_SCRIPT" || true
for HOST in $WORKER_HOSTS; do
    ssh -n ${REMOTE_USER}@${HOST} "pkill -f '$INFER_SCRIPT' || true" &
done
wait
echo "🧹 Cleaning complete."


# Installation Environment
echo "Environmental installation"
for HOST in $WORKER_HOSTS; do
    echo "on ${HOST} Environmental installation..."
    # your installation command
done
echo "wait for all installation tasks to complete...."
wait
echo "all node environments are installed!"


for HOST in $WORKER_HOSTS; do
    echo "🚀 starting ${HOST} ..."
    START_CMD="source ~/.bashrc; \
        source $CONDA_ACTIVATE; \
        conda activate $CONDA_ENV; \
        export LOCAL_IP=${HOST}; \
        cd $PROJECT_PATH; \
        nohup bash $INFER_SCRIPT > backend_deploy.log 2>&1 & sleep 2"
    echo ${START_CMD}
    ssh -n ${REMOTE_USER}@${HOST} "${START_CMD}"
done


echo "🚀 starting local backend service..."
START_CMD="source $CONDA_ACTIVATE; \
    conda activate $CONDA_ENV; \
    cd $PROJECT_PATH; \
    nohup bash $INFER_SCRIPT > backend_deploy.log 2>&1 & sleep 2"
eval "${START_CMD}"
wait
echo "✅ All backend nodes have been started and commands have been sent."


echo "⚖️ Starting the load balancing service locally (API Gateway)..."
(
    source $CONDA_ACTIVATE
    conda activate $CONDA_ENV
    cd $PROJECT_PATH
    bash $LB_SCRIPT judge
)
