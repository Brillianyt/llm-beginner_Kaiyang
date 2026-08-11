"""S2: 重做 extrapolation eval(rope 和 abs 模型已训好,只跑 eval 部分)。

加载 ckpt/s2_rope.pt 和 ckpt/s2_abs.pt,eval @ block=64/128/256。
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.config import MiniGPTConfig
from src.model import MiniGPT
from src.tokenizer import BPETokenizer

CKPT_DIR = HERE / "ckpt"
DATA_DIR = HERE / "data"


def estimate_dev_ppl(model, dev_text, tok, block):
    model.eval()
    dev_ids = tok.encode(dev_text)[:4096]
    nll, n_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, max(1, len(dev_ids) - 1), block):
            window = dev_ids[i : i + block + 1]
            if len(window) < 2:
                break
            chunk = torch.tensor([window], dtype=torch.long, device=next(model.parameters()).device)
            logits = model(chunk)
            nll += F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1), reduction="sum",
            ).item()
            n_tok += chunk.size(1) - 1
    model.train()
    return math.exp(nll / n_tok) if n_tok > 0 else float("inf")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.from_pretrained(str(CKPT_DIR / "tokenizer.json"))
    dev_text = (DATA_DIR / "dev.txt").read_text(encoding="utf-8")

    test_blocks = [64, 128, 256]
    max_block = max(test_blocks) + 1

    rows = []
    for ckpt_name, label in [("s2_rope.pt", "rope"), ("s2_abs.pt", "absolute")]:
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"!! {ckpt_path} 不存在", flush=True)
            continue
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        cfg_dict = ckpt["config"]
        eval_cfg = MiniGPTConfig(
            vocab_size=cfg_dict["vocab_size"],
            d_model=cfg_dict["d_model"],
            n_heads=cfg_dict["n_heads"],
            n_layers=cfg_dict["n_layers"],
            d_ff=cfg_dict["d_ff"],
            block_size=max_block,
            dropout=cfg_dict.get("dropout", 0.1),
            weight_tying=cfg_dict.get("weight_tying", True),
            pos_encoding=cfg_dict.get("pos_encoding", "rope"),
        )
        print(f"\n[S2 eval] loading {label} model, eval cfg block_size={eval_cfg.block_size}", flush=True)
        model = MiniGPT(eval_cfg).to(device)

        # 容错加载:跳过 shape 不匹配的(SinusoidalPE pe 在训练时可能被扩展过)
        sd = ckpt["model_state_dict"]
        msd = model.state_dict()
        loaded_keys = 0
        for k in msd:
            if k in sd and sd[k].shape == msd[k].shape:
                msd[k].copy_(sd[k])
                loaded_keys += 1
        ret = model.load_state_dict(msd, strict=False)
        if ret.missing_keys:
            print(f"  missing in ckpt: {ret.missing_keys}", flush=True)
        print(f"  loaded {loaded_keys} matching keys", flush=True)
        model.eval()

        for test_block in test_blocks:
            ppl = estimate_dev_ppl(model, dev_text, tok, test_block)
            rows.append({"model": label, "block_size": test_block, "ppl": ppl})
            print(f"  [{label} @ block={test_block}] dev_ppl = {ppl:.2f}", flush=True)

    # 写报告
    out_path = HERE / "figures" / "s2_pe_extrapolation.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_model = {(r["model"], r["block_size"]): r["ppl"] for r in rows}
    lines = [
        "# S2: 绝对 PE vs RoPE 长序列外推对比",
        "",
        "## 实验设置",
        "",
        f"- 架构:256d × 4L × d_ff=1024,~6.3M 参数(rope 与 absolute 参数量相同,因为 SinusoidalPE 不需要 learnable)",
        f"- 训练 block_size=64(故意短,留 extrapolation 空间)",
        f"- 训练 iters=2000,数据=18MB SkyPile 子集",
        "",
        "## 结果",
        "",
        "| 模型 | eval @ block=64 (训练长度) | eval @ block=128 (2×外推) | eval @ block=256 (4×外推) |",
        "|---|---|---|---|",
    ]
    for model_name in ["rope", "absolute"]:
        ppl_64 = by_model.get((model_name, 64), float("inf"))
        ppl_128 = by_model.get((model_name, 128), float("inf"))
        ppl_256 = by_model.get((model_name, 256), float("inf"))
        lines.append(f"| {model_name} | {ppl_64:.2f} | {ppl_128:.2f} | {ppl_256:.2f} |")

    lines += [
        "",
        "## 退化比例(以 block=64 ppl 为基准)",
        "",
        "| 模型 | block=64 | block=128 退化 | block=256 退化 |",
        "|---|---|---|---|",
    ]
    for model_name in ["rope", "absolute"]:
        ppl_64 = by_model.get((model_name, 64), float("inf"))
        ppl_128 = by_model.get((model_name, 128), float("inf"))
        ppl_256 = by_model.get((model_name, 256), float("inf"))
        deg_128 = ((ppl_128 - ppl_64) / ppl_64 * 100) if ppl_64 > 0 else float("inf")
        deg_256 = ((ppl_256 - ppl_64) / ppl_64 * 100) if ppl_64 > 0 else float("inf")
        lines.append(f"| {model_name} | {ppl_64:.2f} | +{deg_128:.1f}% | +{deg_256:.1f}% |")

    lines += [
        "",
        "## 解读",
        "",
        "- RoPE 的旋转角 θ_pos,i = pos / base^(2i/d) 对任意 pos 都定义,可自然外推",
        "- Sinusoidal 绝对 PE 也可外推,但训练时没见过的位置模式泛化能力差",
        "- 两者的退化差距见上面表格",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[S2 eval] wrote {out_path}", flush=True)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()