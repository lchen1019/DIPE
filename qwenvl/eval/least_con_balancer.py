import os
import asyncio
import time
import random
import httpx
import yaml

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# configure backend servers
API_KEY = "sk-abc123"
HEALTH_CHECK_INTERVAL = 30
REQUEST_TIMEOUT = 300

class Backend:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.healthy = True
        self.active_requests = 0  # 当前正在处理的请求数
        self.lock = asyncio.Lock() # 确保计数器并发安全

LOCAL_IP = os.getenv("LOCAL_IP", "localhost")
# BACKENDS_DATA = [
#     {"url": f"http://{LOCAL_IP}:8100", "name": "instance-0", "healthy": True},
#     {"url": f"http://{LOCAL_IP}:8101", "name": "instance-1", "healthy": True},
#     {"url": f"http://{LOCAL_IP}:8102", "name": "instance-2", "healthy": True},
#     {"url": f"http://{LOCAL_IP}:8103", "name": "instance-3", "healthy": True},
#     {"url": f"http://{LOCAL_IP}:8104", "name": "instance-4", "healthy": True},
#     {"url": f"http://{LOCAL_IP}:8105", "name": "instance-5", "healthy": True},
#     {"url": f"http://{LOCAL_IP}:8106", "name": "instance-6", "healthy": True},
#     {"url": f"http://{LOCAL_IP}:8107", "name": "instance-7", "healthy": True},
# ]

# WORKER_CONFIG_PATH = "scripts/eval/distribution_config.yaml"
# worker_hosts = []
# with open(WORKER_CONFIG_PATH, 'r') as f:
#     config = yaml.safe_load(f)
#     worker_hosts = config.get('worker_hosts', [])
# worker_hosts.append(LOCAL_IP)
hosts_str = os.environ.get('WORKER_HOSTS', "")
worker_hosts = hosts_str.split()
worker_hosts.append(LOCAL_IP)


BACKENDS_DATA = [
    {"url": f"http://{ip}:{8100+i}", "name": f"instance-{i}", "healthy": True} 
    for i in range(8) for ip in worker_hosts
]

backends_list = [Backend(b["name"], b["url"]) for b in BACKENDS_DATA]


class ChatMessage(BaseModel):
    role: str
    content: Any # 修改为Any以兼容多模态或字符串


class ChatCompletionRequest(BaseModel):
    model: str = "api_qwen2_5_siglip_3b_instruct"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False


class LoadBalancer:
    def __init__(self, backends: List[Backend]):
        self.backends = backends
    
    async def get_least_loaded_backend(self) -> Optional[Backend]:
        """
        策略：最小连接数 (Least Connections)
        选择当前 active_requests 最小且健康的后端
        """
        # 筛选健康节点
        healthy_backends = [b for b in self.backends if b.healthy]
        
        if not healthy_backends:
            return None
        
        # 核心逻辑：按 active_requests 排序，取最小的
        # 如果有多个负载相同的，随机选择一个，避免羊群效应
        min_load = min(b.active_requests for b in healthy_backends)
        candidates = [b for b in healthy_backends if b.active_requests == min_load]
        
        return random.choice(candidates)

    def mark_unhealthy(self, backend: Backend):
        backend.healthy = False
    
    def mark_healthy(self, backend: Backend):
        backend.healthy = True

load_balancer = LoadBalancer(backends_list)


@asynccontextmanager
async def track_requests(backend: Backend):
    async with backend.lock:
        backend.active_requests += 1
    try:
        yield
    finally:
        async with backend.lock:
            backend.active_requests -= 1


async def health_check_task():
    async with httpx.AsyncClient() as client:
        while True:
            for backend in backends_list:
                try:
                    response = await client.get(f"{backend.url}/health", timeout=5.0)
                    if response.status_code == 200:
                        load_balancer.mark_healthy(backend)
                    else:
                        load_balancer.mark_unhealthy(backend)
                except Exception:
                    load_balancer.mark_unhealthy(backend)
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    health_check = asyncio.create_task(health_check_task())
    yield
    health_check.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """健康检查端点"""
    healthy_count = sum(1 for b in BACKENDS_DATA if b["healthy"])
    return {
        "status": "healthy" if healthy_count > 0 else "unhealthy",
        "healthy_backends": healthy_count,
        "total_backends": len(BACKENDS_DATA)
    }


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "api_qwen2_5_siglip_3b_instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization"
            }
        ]
    }


async def proxy_request(backend: Backend, path: str, request_data: Dict, stream: bool = False):
    url = f"{backend.url}{path}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    
    async with httpx.AsyncClient() as client:
        if stream:
            async with client.stream("POST", url, json=request_data, headers=headers, timeout=REQUEST_TIMEOUT) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
        else:
            response = await client.post(url, json=request_data, headers=headers, timeout=REQUEST_TIMEOUT)
            yield response

@app.get("/backends/status")
async def backends_status():
    """查看当前负载分布"""
    return {
        "backends": [
            {
                "name": b.name,
                "url": b.url,
                "healthy": b.healthy,
                "active_requests": b.active_requests # 关键指标
            }
            for b in backends_list
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 使用最小负载策略获取后端
    backend = await load_balancer.get_least_loaded_backend()
    
    if not backend:
        raise HTTPException(status_code=503, detail="No healthy backend available")
    
    request_data = request.model_dump(exclude_none=True)
    
    # 使用 track_requests 上下文管理器包裹整个请求过程
    # 注意：对于流式响应，必须确保生成器结束时才释放计数
    
    try:
        if request.stream:
            async def generate_with_tracking():
                # 进入上下文，计数+1
                async with track_requests(backend):
                    try:
                        async for chunk in proxy_request(backend, "/v1/chat/completions", request_data, stream=True):
                            yield chunk
                    except Exception:
                        load_balancer.mark_unhealthy(backend)
                        raise
                # 离开上下文，计数-1 (生成器结束时触发)

            return StreamingResponse(generate_with_tracking(), media_type="text/event-stream")
        else:
            # 非流式请求
            async with track_requests(backend):
                async for response in proxy_request(backend, "/v1/chat/completions", request_data, stream=False):
                    if response.status_code != 200:
                        load_balancer.mark_unhealthy(backend)
                        raise HTTPException(status_code=response.status_code, detail=response.text)
                    return JSONResponse(content=response.json())

    except httpx.TimeoutException:
        load_balancer.mark_unhealthy(backend)
        raise HTTPException(status_code=504, detail="Backend timeout")

    except Exception as e:
        # 如果在建立连接前就报错了，这里需要确保不会导致计数器泄漏
        # 但由于 track_requests 包裹了核心逻辑，通常是安全的
        raise HTTPException(status_code=502, detail=f"Proxy error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)
