"""base / SFT / DPO 三个模型在相同指令上的输出对比。

调用方式：

```bash
python -m src.compare --model_path models/Qwen2.5-0.5B \
    --sft_ckpt ckpt/sft --dpo_ckpt ckpt/dpo \
    --output figures/compare.json
```

输出 JSON 结构：

```json
{
  "instructions": [...],
  "results": [
    {"instruction": "...",
     "base": "...", "sft": "...", "dpo": "..."},
    ...
  ]
}
```
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from .chat import format_messages
from .data_utils import SMOKE_SFT_SAMPLES
from .model_utils import (
    DEFAULT_MODEL_PATH,
    detect_device,
    load_base_model,
    load_reference_model,
    load_sft_model,
    load_tokenizer,
)


DEFAULT_INSTRUCTIONS = [
    "请用一句话解释 LoRA 的核心思想。",
    "介绍一下 DPO 与 RLHF 的区别。",
    "把下面这句话翻译成英文：「机器学习让计算机从数据中学习规律。」",
    "写一个 Python 函数，判断一个数是否为素数。",
    "为什么训练神经网络要用交叉熵而不是 MSE？",
]


def _extract_first_user(samples: list[dict]) -> List[str]:
    """从 SFT 样本里抽出 user 指令，作为对比 prompt。"""
    out: List[str] = []
    for s in samples:
        for m in s.get("messages", []):
            if m.get("role") == "user":
                out.append(m["content"])
                break
    return out


def generate_response(
    model,
    tokenizer,
    instruction: str,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """单条指令的贪心 / 采样生成。"""
    msgs = [{"role": "user", "content": instruction}]
    text = format_messages(msgs)
    # 末尾截断到 assistant 提示符之后。
    if not text.endswith("assistant\n"):
        text = text.rstrip("\n") + "\n"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-3),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][input_ids.size(1):]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text.strip()


def _maybe_load(model_path: Path, ckpt_path: Path | None, role: str, device, dtype=None):
    """按需加载模型，缺失则返回 ``None``。"""
    if ckpt_path is None:
        return None
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[{role}] 跳过：{ckpt_path} 不存在")
        return None
    if role == "base":
        return load_base_model(model_path, device=device, dtype=dtype)
    if role == "sft":
        return load_sft_model(model_path, lora_ckpt=ckpt_path, device=device)
    if role == "dpo":
        return load_reference_model(model_path, sft_ckpt=ckpt_path, device=device)
    raise ValueError(f"未知 role: {role}")


def compare(
    model_path: Path = DEFAULT_MODEL_PATH,
    sft_ckpt: Path | None = None,
    dpo_ckpt: Path | None = None,
    instructions: List[str] | None = None,
    output_path: Path | None = None,
    max_new_tokens: int = 128,
) -> dict:
    """主流程：加载三个模型、生成、并落盘 JSON。"""
    device = detect_device()
    print(f"[compare] device={device}, model={model_path}")
    if not model_path.exists():
        print(f"[compare] 跳过：模型 {model_path} 不存在")
        return {"instructions": instructions or DEFAULT_INSTRUCTIONS, "results": [],
                "skipped": f"model {model_path} missing"}

    tokenizer = load_tokenizer(model_path)

    if instructions is None:
        # 优先从 SFT 样本里抽 user 指令；没有则用默认集合。
        instructions = DEFAULT_INSTRUCTIONS

    base_model = _maybe_load(model_path, None, "base", device)
    sft_model = _maybe_load(model_path, sft_ckpt, "sft", device)
    dpo_model = _maybe_load(model_path, dpo_ckpt, "dpo", device)

    results = []
    for inst in instructions:
        rec = {"instruction": inst}
        if base_model is not None:
            rec["base"] = generate_response(base_model, tokenizer, inst, device, max_new_tokens)
        if sft_model is not None:
            rec["sft"] = generate_response(sft_model, tokenizer, inst, device, max_new_tokens)
        if dpo_model is not None:
            rec["dpo"] = generate_response(dpo_model, tokenizer, inst, device, max_new_tokens)
        results.append(rec)
        # 打印一行方便实时观察。
        print(f"\n[Q] {inst}")
        for k in ("base", "sft", "dpo"):
            if k in rec:
                print(f"  [{k}] {rec[k][:200]}")

    output = {"instructions": instructions, "results": results}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[compare] 写入 {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="base/SFT/DPO 对比生成")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sft_ckpt", type=Path, default=Path("ckpt/sft"))
    parser.add_argument("--dpo_ckpt", type=Path, default=Path("ckpt/dpo"))
    parser.add_argument("--output", type=Path, default=Path("figures/compare.json"))
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    compare(
        model_path=args.model_path,
        sft_ckpt=args.sft_ckpt,
        dpo_ckpt=args.dpo_ckpt,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
