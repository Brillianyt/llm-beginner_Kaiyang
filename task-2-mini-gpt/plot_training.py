"""解析训练日志,绘制 train_loss / dev_loss / dev_ppl 曲线。

训练日志格式(train.py 输出):
  iter    N | lr X | train_loss Y | elapsed Z
    [eval] train_loss=A dev_loss=B dev_ppl=C
    [ckpt] saved best.pt (dev_ppl=C)

输出:
  figures/training_curves.png  (三联子图:loss / ppl / lr)

用法:
  python plot_training.py [--log /tmp/train.log] [--out figures/training_curves.png]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无头环境
import matplotlib.pyplot as plt


ITER_TRAIN_RE = re.compile(
    r"^iter\s+(\d+)\s*\|.*train_loss\s+([\d.]+).*elapsed\s+([\d.]+)s"
)
# 旧格式:[eval] train_loss=X dev_loss=Y dev_ppl=Z
EVAL_RE_OLD = re.compile(
    r"\[eval\]\s+train_loss=([\d.]+)\s+dev_loss=([\d.]+)\s+dev_ppl=([\d.]+)"
)
# 新格式:[eval] train_loss=X dev_ppl(eval-method)=Z
EVAL_RE_NEW = re.compile(
    r"\[eval\]\s+train_loss=([\d.]+)\s+dev_ppl\(eval-method\)=([\d.]+)"
)
# ckpt 格式两版通用:[ckpt] saved best.pt (dev_ppl=Z)
CKPT_RE = re.compile(r"\[ckpt\]\s+saved best\.pt\s+\(dev_ppl=([\d.]+)\)")
LR_RE = re.compile(r"\|\s*lr\s+([\d.eE+-]+)")


def parse_log(path: Path):
    """返回 dicts:iter_train / evals / ckpts / lrs。

    evals 元素:(iter, train_loss, dev_loss_or_nan, ppl)
    新格式没有 dev_loss,用 NaN 占位。
    """
    iter_train: list[tuple[int, float, float]] = []
    evals: list[tuple[int, float, float, float]] = []
    ckpts: list[tuple[int, float]] = []
    lrs: list[tuple[int, float]] = []

    last_iter_seen = -1
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ITER_TRAIN_RE.search(line)
        if m:
            it, tl, el = int(m.group(1)), float(m.group(2)), float(m.group(3))
            iter_train.append((it, tl, el))
            last_iter_seen = it
            ml = LR_RE.search(line)
            if ml:
                lrs.append((it, float(ml.group(1))))
            continue

        m = EVAL_RE_OLD.search(line)
        if m:
            tl, dl, ppl = float(m.group(1)), float(m.group(2)), float(m.group(3))
            evals.append((last_iter_seen, tl, dl, ppl))
            continue

        m = EVAL_RE_NEW.search(line)
        if m:
            tl, ppl = float(m.group(1)), float(m.group(2))
            import math
            evals.append((last_iter_seen, tl, math.nan, ppl))
            continue

        m = CKPT_RE.search(line)
        if m:
            ppl = float(m.group(1))
            ckpts.append((last_iter_seen, ppl))

    return iter_train, evals, ckpts, lrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log", type=Path,
        default=Path("/tmp/train_skypile.log"),
        help="训练日志路径",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "figures" / "training_curves.png",
    )
    ap.add_argument("--title", type=str, default="MiniGPT on SkyPile subset (18MB)")
    args = ap.parse_args()

    iter_train, evals, ckpts, lrs = parse_log(args.log)
    if not iter_train:
        raise SystemExit(f"日志 {args.log} 里没匹配到训练行,先确认 train.py 跑过了")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 子图 1: loss 曲线(train 与 dev)
    ax = axes[0]
    it_train = [t[0] for t in iter_train]
    loss_train = [t[1] for t in iter_train]
    ax.plot(it_train, loss_train, alpha=0.4, label="train (per-iter)", color="C0")
    if evals:
        it_e = [e[0] for e in evals]
        ax.plot(it_e, [e[1] for e in evals], "-o", label="train (eval avg)", color="C0", ms=4)
        ax.plot(it_e, [e[2] for e in evals], "-o", label="dev (eval avg)", color="C1", ms=4)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图 2: dev perplexity
    ax = axes[1]
    if evals:
        it_e = [e[0] for e in evals]
        ppl = [e[3] for e in evals]
        ax.plot(it_e, ppl, "-o", color="C2", ms=4, label="dev ppl")
        # 标 best ckpt
        if ckpts:
            best_iter, best_ppl = min(ckpts, key=lambda x: x[1])
            ax.axhline(best_ppl, ls="--", color="gray", alpha=0.5,
                       label=f"best = {best_ppl:.2f} @ iter {best_iter}")
            ax.scatter([best_iter], [best_ppl], s=80, c="red", zorder=5, label="best ckpt")
        ax.axhline(50, ls=":", color="orange", alpha=0.5, label="threshold = 50")
    ax.set_xlabel("iteration")
    ax.set_ylabel("perplexity")
    ax.set_title("Dev perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图 3: 学习率
    ax = axes[2]
    if lrs:
        ax.plot([l[0] for l in lrs], [l[1] for l in lrs], color="C3")
    ax.set_xlabel("iteration")
    ax.set_ylabel("learning rate")
    ax.set_title("LR schedule (warmup + cosine)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[plot] saved {args.out}")
    print(f"[plot] train iters: {len(iter_train)}, evals: {len(evals)}, ckpts: {len(ckpts)}")
    if ckpts:
        best_iter, best_ppl = min(ckpts, key=lambda x: x[1])
        print(f"[plot] best ckpt: iter={best_iter}, dev_ppl={best_ppl:.2f}")
    if evals:
        last = evals[-1]
        print(f"[plot] final eval: iter={last[0]}, train_loss={last[1]:.4f}, "
              f"dev_loss={last[2]:.4f}, dev_ppl={last[3]:.2f}")


if __name__ == "__main__":
    main()