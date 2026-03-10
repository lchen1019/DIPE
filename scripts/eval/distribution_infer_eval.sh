#!/bin/bash

# ==============================================================================
#                 Multi-Node Inference Backend Automated Deployment Script
# ==============================================================================
CONFIG_PATH="scripts/eval/distribution_config.env"
REMOTE_USER="root"
CONDA_ACTIVATE="/jizhicfs/capychen/miniconda3/bin/activate"
CONDA_ENV="modified_eval"
INFER_SCRIPT="scripts/eval/infer_backend.sh"
LB_SCRIPT="scripts/eval/balance_deploy_api.sh"


# --- 2. Parse Worker Hosts ---
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Error: Configuration file $CONFIG_PATH does not exist"
    exit 1
fi

source $CONFIG_PATH
PROJECT_PATH=${PROJECT_PATH}
WORKER_HOSTS=${WORKER_HOSTS}

echo "🔍 Project path: $PROJECT_PATH"
echo "🔍 Identified Worker nodes: $WORKER_HOSTS"


# --- 3. Environment Preparation and Cleanup ---
echo "🧹 Cleaning up old inference processes on all nodes..."
pkill -f "$INFER_SCRIPT" || true
for HOST in $WORKER_HOSTS; do
    ssh -n ${REMOTE_USER}@${HOST} "pkill -f '$INFER_SCRIPT' || true" &
done
wait
echo "🧹 Cleanup complete."


# Install Environment
echo "Environment installation"
for HOST in $WORKER_HOSTS; do
    echo "Starting environment installation on node ${HOST}..."
    ssh ${REMOTE_USER}@${HOST} your commard here
done
echo "Waiting for all installation tasks to complete..."
wait
echo "Environment installation completed on all nodes!"


# --- 4. Deploy Backend Services ---
echo "🚀 Deploying multi-node backend services..."
# A. Start on all remote Workers
for HOST in $WORKER_HOSTS; do
    echo "🚀 Starting backend service remotely on node ${HOST}..."

    START_CMD="source ~/.bashrc; \
        source $CONDA_ACTIVATE; \
        conda activate $CONDA_ENV; \
        export LOCAL_IP=${HOST}; \
        cd $PROJECT_PATH; \
        nohup bash $INFER_SCRIPT > backend_deploy.log 2>&1 & sleep 2"

    echo ${START_CMD}
    # Use ssh to execute the startup command
    ssh -n ${REMOTE_USER}@${HOST} "${START_CMD}"
done

# B. Deploy a backend service locally
echo "🚀 Starting local backend service..."
START_CMD="source $CONDA_ACTIVATE; \
    conda activate $CONDA_ENV; \
    cd $PROJECT_PATH; \
    nohup bash $INFER_SCRIPT > backend_deploy.log 2>&1 & sleep 2"
eval "${START_CMD}"

wait
echo "✅ Startup commands sent to all backend nodes."

# --- 5. Start Local Load Balancing Service ---
echo "⚖️ Starting local load balancing service (API Gateway)..."
(
    source $CONDA_ACTIVATE
    conda activate $CONDA_ENV
    cd $PROJECT_PATH
    bash $LB_SCRIPT eval
)