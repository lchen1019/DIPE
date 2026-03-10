#!/bin/bash

if [ "$1" == "eval" ]; then
    FASTAPI_SCRIPT="qwenvl/eval/least_con_balancer.py"
else
    FASTAPI_SCRIPT="qwenvl/eval/least_con_balancer_vllm.py"
fi

echo $FASTAPI_SCRIPT

FASTAPI_PORT=7777
LOCAL_IP=$LOCAL_IP

echo "============================================"
echo "Starting FastAPI Load Balancer on port 7777..."
echo "============================================"

python ${FASTAPI_SCRIPT} \
    --port ${FASTAPI_PORT} \
    --host ${LOCAL_IP}
