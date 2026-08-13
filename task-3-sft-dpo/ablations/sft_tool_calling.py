"""S5：贯通任务五 —— 用 moss-003-sft-plugin 训一版带工具调用的 SFT 模型。

工具调用 SFT 与普通 SFT 几乎相同，但有两点需要注意：

1. **数据**：plugin 数据的 assistant content 中会出现 JSON 格式的工具调
   用（即 ``[TOOL_CALL]`` / ``<function_calls>`` / ``<invoke name=...>``）。我们的
   loss masking **不应该** mask 掉工具调用段；它就是 assistant 正常输出的一部分。
2. **prompt 模板**：建议在 system turn 加一句工具说明，让模型习惯
   输出 function-call 结构。

本脚本实现：

- 加载 ``data/moss-sft-plugin/moss-003-sft-with-tools-no-text2image.jsonl``；
- 把 system turn 替换为「你是一个工具调用助手……」；
- 与 ``train_sft.py`` 一样做 LoRA SFT；
- 评估输出：检查模型是否能生成合法的 function-call 格式。

输出：CKPT 到 ``ckpt/sft-tool/``，
评测 JSON 到 ``figures/s5_tool_calling.json``。

用法：
```bash
python ablations/sft_tool_calling.py --smoke
```
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.chat import build_labels, format_messages
from src.data_utils import load_moss_sft, load_sft_smoke
from src.lora import inject_lora, lora_state_dict
from src.model_utils import DEFAULT_MODEL_PATH, detect_device, load_tokenizer


TOOL_SYSTEM_PROMPT = (
    "你是一个具备工具调用能力的助手；当问题需要调用外部工具时，按以下格式输出：\n"
    "  <invoke name=\"{tool_name}\">{args}</invoke>\n"
    "其中 ``{args}`` 是合法的 JSON 对象。"
)

# 简单正则：从生成文本里抽取 ``<invoke name=...>`` 段。
INVOKE_PATTERN = re.compile(r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL)


def _inject_tool_sys_prompt(messages: List[dict]) -> List[dict]:
    """在 ``messages`` 头部注入工具说明。"""
    if not messages:
        return [{"role": "system", "content": TOOL_SYSTEM_PROMPT}]
    head, rest = messages[0], messages[1:]
    if head.get("role") == "system":
        new_head = {"role": "system", "content": head["content"] + "\n" + TOOL_SYSTEM_PROMPT}
    else:
        new_head = {"role": "system", "content": TOOL_SYSTEM_PROMPT}
        rest = messages
    return [new_head] + list(rest)


def _fix_msgs(samples: List[dict]) -> List[dict]:
    """对每条样本注入 system prompt。"""
    return [{"messages": _inject_tool_sys_prompt(s["messages"])} for s in samples]


def _format_for_sft(samples: List[dict], tokenizer, max_length: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """一次性把所有样本 tokenize + mask；返回 ``(input_ids, labels)``。

    简化版：每条样本独立，不做 padding（建议训练时再 collate）。
    """
    all_ids, all_labels = [], []
    for s in samples:
        msgs = s["messages"]
        text = format_messages(msgs)
        enc = tokenizer(text, truncation=True, max_length=max_length,
                        return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"][0]
        labels = build_labels(ids, msgs, tokenizer=tokenizer)
        all_ids.append(ids)
        all_labels.append(labels)
    return all_ids, all_labels


def collate(ids_list, labels_list, pad_id: int) -> dict:
    max_len = max(s.size(0) for s in ids_list)
    out_ids, out_lbl = [], []
    for i, l in zip(ids_list, labels_list):
        pad = max_len - i.size(0)
        if pad > 0:
            out_ids.append(torch.cat([i, torch.full((pad,), pad_id, dtype=torch.long)]))
            out_lbl.append(torch.cat([l, torch.full((pad,), -100, dtype=torch.long)]))
        else:
            out_ids.append(i)
            out_lbl.append(l)
    return {"input_ids": torch.stack(out_ids), "labels": torch.stack(out_lbl)}


def _check_tool_format(text: str) -> dict:
    """检查生成文本里工具调用格式是否合法。"""
    matches = INVOKE_PATTERN.findall(text)
    if not matches:
        return {"calls": 0, "valid_json": 0, "rate": 0.0}
    valid = 0
    for name, args in matches:
        try:
            args_dict = json.loads(args)
            if isinstance(args_dict, dict):
                valid += 1
        except json.JSONDecodeError:
            pass
    return {"calls": len(matches), "valid_json": valid, "rate": valid / len(matches)}


def main() -> None:
    parser = argparse.ArgumentParser(description="S5: 工具调用 SFT")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", type=Path, default=Path("data/moss-sft-plugin"))
    parser.add_argument("--output_dir", type=Path, default=Path("ckpt/sft-tool"))
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", type=Path, default=Path("figures/s5_tool_calling.json"))
    args = parser.parse_args()

    # 1. 数据
    if args.smoke:
        samples = load_sft_smoke()
        print(f"[S5] SMOKE 模式，使用 {len(samples)} 条内置样本")
    else:
        try:
            samples = load_moss_sft(args.data_dir, split="with-tools", max_samples=args.max_samples)
            print(f"[S5] 加载 {len(samples)} 条 plugin 数据")
        except FileNotFoundError as e:
            print(f"[S5] 数据缺失：{e}")
            samples = load_sft_smoke()

    samples = _fix_msgs(samples)

    # 2. 模型
    if not args.model_path.exists():
        print(f"[S5] 模型缺失：{args.model_path}；仅做格式检查管线验证。")
        sample = '<invoke name="sum">{"a": 1, "b": 2}</invoke> 错的格式'
        out = {"skipped": "model missing", "tool_format_check": _check_tool_format(sample)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    device = detect_device()
    tokenizer = load_tokenizer(args.model_path)
    from transformers import AutoModelForCausalLM
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=dtype, trust_remote_code=True,
    )
    inject_lora(model, target_modules=["q_proj", "v_proj"], r=8, alpha=16)
    model = model.to(device)

    # 3. 训练（与 train_sft.py 保持一致）
    ids_list, labels_list = _format_for_sft(samples, tokenizer, max_length=1024)
    loader = DataLoader(
        list(zip(ids_list, labels_list)),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate([x[0] for x in b], [x[1] for x in b], tokenizer.pad_token_id),
    )
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    n_steps = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            n_steps += 1
        print(f"[S5] epoch {epoch+1} done")

    # 4. 保存
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    state["__meta__"] = {"with_tools": True, "n_samples": len(samples)}
    torch.save(state, args.output_dir / "lora_state.pt")
    print(f"[S5] LoRA 权重保存到 {args.output_dir}")

    # 5. 评估：随便抽一条测试生成（demo）
    test_prompt = "请计算 12 + 34 的结果。"
    msgs = [{"role": "user", "content": test_prompt}]
    text = format_messages(msgs)
    if not text.endswith("assistant\n"):
        text = text.rstrip("\n") + "\n"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        gen = model.generate(input_ids, max_new_tokens=64, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    text_gen = tokenizer.decode(gen[0, input_ids.size(1):], skip_special_tokens=True)
    fmt = _check_tool_format(text_gen)
    out = {
        "test_prompt": test_prompt,
        "generated": text_gen,
        "tool_format_check": fmt,
        "n_samples": len(samples),
        "n_steps": n_steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S5] 写入 {args.output}")


if __name__ == "__main__":
    main()
