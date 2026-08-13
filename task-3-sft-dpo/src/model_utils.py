"""模型加载与 LoRA 注入的统一入口。

关键点：

1. 通过 ``AutoModelForCausalLM`` + ``AutoTokenizer`` 加载 Qwen2.5；
2. 训练时把 base 参数 ``requires_grad=False``，再注入 LoRA；
3. 提供 ``reference model`` 工厂：返回冻结的 SFT-LoRA 同架构但不参与反向；
4. 设备自适应：优先 cuda、否则 mps、否则 cpu。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = Path("models/Qwen2.5-0.5B")


def detect_device(prefer_cuda: bool = True) -> torch.device:
    """按 cuda → mps → cpu 顺序挑选设备。"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(model_path: str | Path = DEFAULT_MODEL_PATH):
    """加载 tokenizer，处理 pad_token 缺失的常见情况。"""
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tok.pad_token is None:
        # Qwen2.5 默认 pad = eos。
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """加载 base 模型的 ``eval`` 副本，不注入 LoRA。"""
    if dtype is None:
        dtype = torch.float32 if device is None or device.type == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def load_sft_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    lora_ckpt: str | Path | None = None,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    r: int = 8,
    alpha: int = 16,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """加载 base → 注入 LoRA → 若提供 ``lora_ckpt`` 则加载 SFT 权重。

    训练模式（默认 ``requires_grad`` 已由 ``inject_lora`` 设好）。
    """
    from .lora import inject_lora, load_lora_state_dict
    if dtype is None:
        dtype = torch.float32 if device is None or device.type == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    inject_lora(model, target_modules=target_modules, r=r, alpha=alpha)
    if lora_ckpt is not None:
        ckpt_file = Path(lora_ckpt)
        if ckpt_file.is_dir():
            # 兼容：「目录方式」保存：把 lora_A / lora_B 合成一个 state_dict 在该目录下
            from torch import load as torch_load
            state = torch_load(ckpt_file / "lora_state.pt", map_location="cpu")
        else:
            from torch import load as torch_load
            state = torch_load(ckpt_file, map_location="cpu")
        load_lora_state_dict(model, state)
    if device is not None:
        model = model.to(device)
    model.train()
    return model


def load_reference_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    sft_ckpt: str | Path | None = None,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    r: int = 8,
    alpha: int = 16,
    device: torch.device | None = None,
) -> nn.Module:
    """加载冻结 reference model（用于 DPO）。

    关键：reference 与 policy 同架构、同初始权重；训练时只跑 forward，不
    参与反向，因此 ``requires_grad=False``。
    """
    from .lora import inject_lora, load_lora_state_dict
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if (device is not None and device.type != "cpu") else torch.float32,
        trust_remote_code=True,
    )
    inject_lora(model, target_modules=target_modules, r=r, alpha=alpha)
    if sft_ckpt is not None:
        ckpt_file = Path(sft_ckpt)
        if ckpt_file.is_dir():
            from torch import load as torch_load
            state = torch_load(ckpt_file / "lora_state.pt", map_location="cpu")
        else:
            from torch import load as torch_load
            state = torch_load(ckpt_file, map_location="cpu")
        load_lora_state_dict(model, state)
    if device is not None:
        model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def freeze_all_but_lora(model: nn.Module) -> None:
    """确保除 ``lora_A`` / ``lora_B`` 之外的参数都被冻结。"""
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


def count_trainable(model: nn.Module) -> Tuple[int, int, float]:
    """返回 ``(trainable, total, ratio)``。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total if total > 0 else 0.0
    return trainable, total, ratio
