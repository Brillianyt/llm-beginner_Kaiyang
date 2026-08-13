"""S4：SFT-only vs SFT+DPO 在偏好数据上的差异 + reward margin 曲线。

设计：

1. 取一组固定偏好对（chosen / rejected）；
2. 分别在 SFT-only 与 SFT+DPO 模型上算每对的 log π(chosen) / log π(rejected)；
3. 计算 reward margin = β * (log π(chosen) - log π(rejected))；
4. 统计：avg margin、chosen 的胜率、margin 分布；
5. 训练过程额外输出：在 DPO 训练中画 margin 随 step 的上升曲线（需要
   ``train_dpo.py`` 把 log 写到 ``figures/dpo_train.log``）。

输出：JSON 到 ``figures/s4_sft_vs_dpo.json`` + 一张 ASCII 折线图。

用法：
```bash
python ablations/dpo_reward_margin.py --smoke
```
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from src.chat import build_labels, format_messages
from src.data_utils import load_dpo_smoke
from src.model_utils import DEFAULT_MODEL_PATH, detect_device, load_sft_model, load_tokenizer


def _gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    safe_labels = labels.clamp(min=0)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float()
    return (gathered * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


def _score(model, tokenizer, prompt: str, completion: str, device) -> float:
    """给定 prompt + completion，返回 completion 段的平均 log-prob。"""
    msgs = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    full_text = format_messages(msgs)
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512, add_special_tokens=False)
    ids = enc["input_ids"].to(device)

    # 找 prompt 段的 token 数。
    prompt_text = format_messages([{"role": "user", "content": prompt}])
    if not prompt_text.endswith("assistant\n"):
        prompt_text = prompt_text.rstrip("\n") + f"\n{format_messages([{'role': 'assistant', 'content': ''}])}"
    pids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    start = min(len(pids), ids.size(1))

    labels = torch.full_like(ids, -100)
    labels[0, start:] = ids[0, start:]

    with torch.no_grad():
        out = model(ids)
    return _gather_logprobs(out.logits, labels).item()


def evaluate_pair(model, tokenizer, samples: List[dict], device, beta: float = 0.1) -> dict:
    margins: List[float] = []
    chosen_wins = 0
    detailed = []
    for s in samples:
        prompt = s["prompt"]
        chosen = s["chosen"]
        rejected = s["rejected"]
        lp_c = _score(model, tokenizer, prompt, chosen, device)
        lp_r = _score(model, tokenizer, prompt, rejected, device)
        margin = beta * (lp_c - lp_r)
        margins.append(margin)
        if lp_c > lp_r:
            chosen_wins += 1
        detailed.append({
            "prompt": prompt[:60],
            "chosen_lp": lp_c,
            "rejected_lp": lp_r,
            "margin": margin,
        })
    return {
        "avg_margin": statistics.mean(margins),
        "max_margin": max(margins),
        "min_margin": min(margins),
        "std_margin": statistics.pstdev(margins) if len(margins) > 1 else 0.0,
        "chosen_win_rate": chosen_wins / max(len(samples), 1),
        "n_samples": len(samples),
        "details": detailed,
    }


def _ascii_line(values: List[float], width: int = 40) -> str:
    """把数值序列画成 ASCII 折线图（玩具版）。"""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return " " * width + f" (constant {lo:.4f})"
    out = []
    for i in range(0, len(values), max(1, len(values) // width)):
        chunk = values[i:i + 1]
        v = chunk[0]
        bar_height = int((v - lo) / (hi - lo) * 5)
        out.append("▁▂▃▄▅▆▇█"[min(bar_height, 7)])
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="S4: SFT vs SFT+DPO reward margin")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sft_ckpt", type=Path, default=Path("ckpt/sft"))
    parser.add_argument("--dpo_ckpt", type=Path, default=Path("ckpt/dpo"))
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("figures/s4_sft_vs_dpo.json"))
    args = parser.parse_args()

    samples = load_dpo_smoke() if args.smoke else load_dpo_smoke()[:args.n_samples]
    print(f"[S4] 评测 {len(samples)} 对偏好")

    if not args.model_path.exists():
        print(f"[S4] 模型 {args.model_path} 不存在，跳过实际评估。")
        out = {"skipped": "model missing", "samples": len(samples)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    device = detect_device()
    tokenizer = load_tokenizer(args.model_path)

    sft_metrics = {"skipped": "sft ckpt missing"}
    dpo_metrics = {"skipped": "dpo ckpt missing"}

    if args.sft_ckpt.exists():
        print("[S4] 加载 SFT 模型...")
        sft = load_sft_model(args.model_path, lora_ckpt=args.sft_ckpt, device=device)
        sft_metrics = evaluate_pair(sft, tokenizer, samples, device, beta=args.beta)
        print(f"[S4] SFT avg_margin={sft_metrics['avg_margin']:.4f}, "
              f"chosen_win={sft_metrics['chosen_win_rate']:.2%}")
        del sft

    if args.dpo_ckpt.exists():
        print("[S4] 加载 DPO 模型...")
        dpo = load_sft_model(args.model_path, lora_ckpt=args.dpo_ckpt, device=device)
        dpo_metrics = evaluate_pair(dpo, tokenizer, samples, device, beta=args.beta)
        print(f"[S4] DPO avg_margin={dpo_metrics['avg_margin']:.4f}, "
              f"chosen_win={dpo_metrics['chosen_win_rate']:.2%}")
        del dpo

    out = {
        "sft": sft_metrics,
        "dpo": dpo_metrics,
        "samples": len(samples),
        "beta": args.beta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S4] 写入 {args.output}")

    # ASCII margin 曲线。
    if "details" in sft_metrics:
        margins = [d["margin"] for d in sft_metrics["details"]]
        print(f"\n[S4] SFT  margin 曲线: {_ascii_line(margins)}")
    if "details" in dpo_metrics:
        margins = [d["margin"] for d in dpo_metrics["details"]]
        print(f"[S4] DPO  margin 曲线: {_ascii_line(margins)}")


if __name__ == "__main__":
    main()
