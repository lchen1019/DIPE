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


MODEL_PATH = None
model = None
tokenizer = None
processor = None


class ChatMessage(BaseModel):
    role: str
    content: Any

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, processor
    print(f"Loading: {MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.padding_side = "left"
        processor = AutoProcessor.from_pretrained(MODEL_PATH, tokenizer=tokenizer, trust_remote_code=True)
        
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


def create_openai_chunk(content: str, model_name: str, finish_reason: str = None):
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



@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "qwen-transformers"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model, processor
    
    qwen_messages = []
    for msg in request.messages:
        role = msg.role
        content = msg.content
        
        new_content = []
        
        if isinstance(content, str):
            new_content.append({"type": "text", "text": content})
            
        elif isinstance(content, list):
            for item in content:
                item_type = getattr(item, 'type', item.get('type') if isinstance(item, dict) else None)
                
                if item_type == "text":
                    text_val = getattr(item, 'text', item.get('text') if isinstance(item, dict) else "")
                    new_content.append({"type": "text", "text": text_val})
                    
                elif item_type == "image_url":
                    image_url_obj = getattr(item, 'image_url', item.get('image_url') if isinstance(item, dict) else {})
                    url = getattr(image_url_obj, 'url', image_url_obj.get('url') if isinstance(image_url_obj, dict) else "")
                    new_content.append({"type": "image", "image": url})
        
        qwen_messages.append({
            "role": role,
            "content": new_content
        })

    try:
        text = processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(model.device)

        generation_kwargs = {}
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            generation_kwargs["top_k"] = request.top_k
        if request.max_tokens is not None:
            generation_kwargs["max_new_tokens"] = request.max_tokens
        if request.repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = request.repetition_penalty
        
        if generation_kwargs.get("temperature", 0) == 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True

        generated_ids = model.generate(**inputs, **generation_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}]
        }
    except Exception as e:
        print(f"error: {e}")
        error_msg = traceback.format_exc()
        print(f"error:\n{error_msg}") 
        
        print("=== Debug Info ===")
        if 'inputs' in locals():
            for k, v in inputs.items():
                if hasattr(v, 'shape'):
                    print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        print("==================")

        raise HTTPException(status_code=500, detail=f"error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--host", type=str)
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    
    uvicorn.run(app, host=args.host, port=args.port)
