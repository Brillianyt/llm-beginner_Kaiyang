#!/usr/bin/env python3
"""
Simple inference server using HuggingFace Transformers + Accelerate.
Uses device_map="auto" to automatically offload weights to CPU when GPU memory is limited.
"""

import argparse, os, sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import uvicorn

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="Qwen2.5-Coder-7B-Instruct Server")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: list[str] | None = None


class GenerateResponse(BaseModel):
    text: str
    finish_reason: str | None = None


_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        model_path = os.environ.get(
            "MODEL_PATH",
            "/mnt/workspace/llm-beginner_Kaiyang/task-6-coding-agent/models/Qwen2.5-Coder-7B-Instruct"
        )
        print(f"Loading model from {model_path} ...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        # device_map="auto" automatically distributes layers across GPU/CPU
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        _model.eval()
        print("Model loaded successfully.")
        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, "
                  f"{torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
    return _model, _tokenizer


@app.post("/v1/completions", response_model=GenerateResponse)
async def completions(req: GenerateRequest):
    model, tokenizer = get_model()
    inputs = tokenizer(req.prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=None,
            return_dict_in_generate=True,
            output_scores=False,
        )

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Check if stopped due to EOS
    finish_reason = None
    if outputs[0][-1] == tokenizer.eos_token_id:
        finish_reason = "stop"

    return GenerateResponse(text=generated_text, finish_reason=finish_reason)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    """Minimal OpenAI-compatible chat endpoint."""
    body = await req.json()
    model, tokenizer = get_model()

    # Extract messages
    messages = body.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": body.get("prompt", "")}]

    # Build prompt from messages (simple concat for code model)
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"

    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.9)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    finish_reason = None
    if outputs[0][-1] == tokenizer.eos_token_id:
        finish_reason = "stop"

    return JSONResponse({
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 0,
        "model": "Qwen2.5-Coder-7B-Instruct",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated_text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": outputs[0].shape[0] - inputs["input_ids"].shape[1],
            "total_tokens": outputs[0].shape[0],
        }
    })


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "Qwen2.5-Coder-7B-Instruct",
            "object": "model",
            "created": 0,
            "owned_by": "local",
        }]
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model-path", type=str,
        default="/mnt/workspace/llm-beginner_Kaiyang/task-6-coding-agent/models/Qwen2.5-Coder-7B-Instruct")
    args = parser.parse_args()
    os.environ["MODEL_PATH"] = args.model_path

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
