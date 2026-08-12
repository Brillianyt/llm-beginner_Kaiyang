"""从 ckpt/tensorboard/ 读 event 文件,画出 loss / dev_ppl / LR 曲线。

不依赖 tensorboard 服务,直接用 EventAccumulator 读 event 数据 + matplotlib 画。
输出:figures/tensorboard_curve.png(多子图)+ 文本摘要打印控制台。
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).resolve().parent
TB_DIR = HERE / "ckpt" / "tensorboard"
OUT = HERE / "figures" / "tensorboard_curve.png"


def read_scalars(ea, tag):
    """读一个 tag 的所有 (step, value) 点,按 step 排序。"""
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals = [e.value for e in events]
    # 已经按 step 升序(SummaryWriter 写入顺序),但保险起见排一遍
    pairs = sorted(zip(steps, vals), key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def main():
    if not TB_DIR.exists() or not any(TB_DIR.iterdir()):
        print(f"[plot] no event files in {TB_DIR}", flush=True)
        sys.exit(1)

    # 合并所有 event 文件(可能跨多次 training run)
    ea = EventAccumulator(str(TB_DIR), size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags()["scalars"]
    print(f"[plot] available tags: {available}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 子图 1: train/loss
    ax = axes[0]
    if "train/loss" in available:
        steps, vals = read_scalars(ea, "train/loss")
        ax.plot(steps, vals, color="C0", alpha=0.4, label="train/loss (per-iter)")
    if "eval/train_loss" in available:
        steps, vals = read_scalars(ea, "eval/train_loss")
        ax.plot(steps, vals, "-o", color="C0", ms=4, label="eval train_loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 子图 2: eval/dev_ppl
    ax = axes[1]
    if "eval/dev_ppl" in available:
        steps, vals = read_scalars(ea, "eval/dev_ppl")
        ax.plot(steps, vals, "-o", color="C2", ms=4, label="dev_ppl")
        ax.axhline(50, ls=":", color="red", alpha=0.5, label="threshold 50")
        # 标 best(min ppl)
        best_step, best_ppl = min(zip(steps, vals), key=lambda x: x[1])
        ax.scatter([best_step], [best_ppl], s=80, c="red", zorder=5,
                   label=f"best = {best_ppl:.2f} @ step {best_step}")
        ax.set_yscale("log")
    if "eval/best_dev_ppl" in available:
        steps, vals = read_scalars(ea, "eval/best_dev_ppl")
        if steps:
            ax.axhline(vals[-1], ls="--", color="gray", alpha=0.4,
                       label=f"final best = {vals[-1]:.2f}")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity (log scale)")
    ax.set_title("Dev Perplexity (eval-method)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # 子图 3: train/lr(整条连贯)
    ax = axes[2]
    if "train/lr" in available:
        steps, vals = read_scalars(ea, "train/lr")
        ax.plot(steps, vals, color="C3", lw=0.8, alpha=0.8)
        # 标 LR 的局部最高点(每次 warmup 起点)——resume 时 LR 重新归 0 再 warmup
        # 用 LR 的"局部"模式不够明显,这里只画曲线
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate (log scale)")
    ax.set_yscale("log")
    ax.set_title("LR Schedule (warmup + cosine per run)")
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("MiniGPT on SkyPile subset — tensorboard curve", fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"[plot] saved {OUT}", flush=True)

    # 文本摘要
    if "eval/dev_ppl" in available:
        steps, vals = read_scalars(ea, "eval/dev_ppl")
        if steps:
            best_step, best_ppl = min(zip(steps, vals), key=lambda x: x[1])
            print(f"\n[summary]", flush=True)
            print(f"  total eval points: {len(steps)}", flush=True)
            print(f"  first dev_ppl: {vals[0]:.2f} @ step {steps[0]}", flush=True)
            print(f"  final dev_ppl: {vals[-1]:.2f} @ step {steps[-1]}", flush=True)
            print(f"  best dev_ppl:   {best_ppl:.2f} @ step {best_step}", flush=True)
            if "eval/best_dev_ppl" in available:
                _, fb = read_scalars(ea, "eval/best_dev_ppl")
                if fb:
                    print(f"  final best (summary): {fb[-1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
