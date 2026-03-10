import os
import time
import torch
import uvicorn
import uuid
import traceback
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from typing import List, Optional, Any
from contextlib import asynccontextmanager
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from qwenvl.models import Qwen25_SigLIPForConditionalGeneration
from fastapi import FastAPI, HTTPException
from qwen_vl_utils import process_vision_info


# ================= 配置区域 =================
MODEL_PATH = None
model = None
tokenizer = None
processor = None


class ChatMessage(BaseModel):
    role: str
    content: Any # 修改为Any以兼容多模态或字符串  # Qwen-VL 可能包含图片列表，这里做宽泛定义


class ChatCompletionRequest(BaseModel):
    model: str = "qwen-transformers"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.1
    presence_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


# ================= 生命周期管理 =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, processor
    print(f"Loading: {MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.padding_side = "left" # Batch 推理必须设置左填充
        processor = AutoProcessor.from_pretrained(MODEL_PATH, tokenizer=tokenizer, trust_remote_code=True)
        
        # 加载模型
        # model = AutoModelForImageTextToText.from_pretrained(
        #     MODEL_PATH,
        #     dtype="auto",
        #     attn_implementation="flash_attention_2",
        #     device_map="auto"
        # ).eval()

        model = Qwen25_SigLIPForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
            device_map="auto",
        ).eval()
    
    except Exception as e:
        print(f"Loading failed: {e}")
        raise e
    
    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan, title="Qwen Transformers API")


# ================= 辅助函数 =================
def create_openai_chunk(content: str, model_name: str, finish_reason: str = None):
    """构造 OpenAI 格式的流式响应块"""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason
            }
        ]
    }


# ================= API 接口 =================

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "qwen-transformers"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model, processor
    
    # OpenAI format (image_url) 转化为 Qwen format content list
    qwen_messages = []
    for msg in request.messages:
        role = msg.role
        content = msg.content
        
        # 准备该条消息的内容列表
        new_content = []
        
        # 情况1: content 是纯字符串
        if isinstance(content, str):
            new_content.append({"type": "text", "text": content})
            
        # 情况2: content 是列表 (多模态输入)
        elif isinstance(content, list):
            for item in content:
                # 兼容 Pydantic 对象或字典访问
                item_type = getattr(item, 'type', item.get('type') if isinstance(item, dict) else None)
                
                if item_type == "text":
                    text_val = getattr(item, 'text', item.get('text') if isinstance(item, dict) else "")
                    new_content.append({"type": "text", "text": text_val})
                    
                elif item_type == "image_url":
                    # 获取 image_url 对象
                    image_url_obj = getattr(item, 'image_url', item.get('image_url') if isinstance(item, dict) else {})
                    # 获取 url 字符串
                    url = getattr(image_url_obj, 'url', image_url_obj.get('url') if isinstance(image_url_obj, dict) else "")
                    
                    # Qwen2.5-VL 格式: {"type": "image", "image": url}
                    # 注意：如果是本地路径，确保 url 是绝对路径或相对路径字符串
                    new_content.append({"type": "image", "image": url})
        
        qwen_messages.append({
            "role": role,
            "content": new_content
        })

    try:
        # 1. 获取纯文本 Prompt (不进行 Tokenize)
        text = processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 2. 提取视觉信息 (Image/Video)
        # process_vision_info 支持处理单条 conversation (list of dicts)
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        
        # 3. 使用 processor 处理文本和视觉输入
        # 注意：text 需要放入列表中以匹配 batch 维度 (Batch Size = 1)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # ==================== 修改结束 ====================

        inputs = inputs.to(model.device)

        # 构建 generation_kwargs
        generation_kwargs = {}
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        # 修正了原代码这里的逻辑错误，通常 top_p 对应 top_p
        if request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            generation_kwargs["top_k"] = request.top_k
        if request.max_tokens is not None:
            generation_kwargs["max_new_tokens"] = request.max_tokens
        if request.repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = request.repetition_penalty
        
        print(generation_kwargs)
        print('aaa', request.repetition_penalty)
        
        # 如果没有设置 do_sample，temperature 设置可能会报错或无效，通常建议：
        if generation_kwargs.get("temperature", 0) == 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True

        # 生成
        generated_ids = model.generate(**inputs, **generation_kwargs)

        # 裁剪掉 Input 部分的 Token
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 解码
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # 非流式返回
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}]
        }
    except Exception as e:
        print(f"生成错误: {e}")
        error_msg = traceback.format_exc()
        print(f"生成发生严重错误:\n{error_msg}") 
        
        # 调试信息
        print("=== Debug Info ===")
        if 'inputs' in locals():
            for k, v in inputs.items():
                if hasattr(v, 'shape'):
                    print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        print("==================")

        raise HTTPException(status_code=500, detail=f"生成错误: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--host", type=str)
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    
    uvicorn.run(app, host=args.host, port=args.port)
