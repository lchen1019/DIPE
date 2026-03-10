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
        self.active_requests = 0
        self.lock = asyncio.Lock()

LOCAL_IP = os.getenv("LOCAL_IP", "localhost")
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
    content: Any


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
        healthy_backends = [b for b in self.backends if b.healthy]
        
        if not healthy_backends:
            return None
        
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
    healthy_count = sum(1 for b in BACKENDS_DATA if b["healthy"])
    return {
        "status": "healthy" if healthy_count > 0 else "unhealthy",
        "healthy_backends": healthy_count,
        "total_backends": len(BACKENDS_DATA)
    }


@app.get("/v1/models")
async def list_models():
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
    return {
        "backends": [
            {
                "name": b.name,
                "url": b.url,
                "healthy": b.healthy,
                "active_requests": b.active_requests
            }
            for b in backends_list
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    backend = await load_balancer.get_least_loaded_backend()
    
    if not backend:
        raise HTTPException(status_code=503, detail="No healthy backend available")
    
    request_data = request.model_dump(exclude_none=True)
    
    try:
        if request.stream:
            async def generate_with_tracking():
                async with track_requests(backend):
                    try:
                        async for chunk in proxy_request(backend, "/v1/chat/completions", request_data, stream=True):
                            yield chunk
                    except Exception:
                        load_balancer.mark_unhealthy(backend)
                        raise

            return StreamingResponse(generate_with_tracking(), media_type="text/event-stream")
        else:
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
        raise HTTPException(status_code=502, detail=f"Proxy error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)
