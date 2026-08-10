"""汇总所有 ablation 的最佳 dev_acc，生成报告与对比热图。

每个 ablation 用独立子进程跑模型加载/推理，避免 Python 模块缓存污染。
子进程代码写到临时 .py 文件，再 python <tmp> 调用。
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

ROOT = Path(__file__).resolve().parent

ZH_FONT = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=10)
ZH_FONT_TITLE = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=13)
ZH_FONT_BIG = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=15)


def collect_logs():
    base = ROOT
    results = {}
    a_dir = base / 'ablation_for_add_and_norm'
    for mode in ['baseline', 'no_residual', 'no_layernorm', 'no_residual_no_ln']:
        log_path = a_dir / 'logs' / f'{mode}.json'
        if log_path.exists():
            log = json.loads(log_path.read_text(encoding='utf-8'))
            results[f'addnorm_{mode}'] = {
                'best': log['best_dev_acc'],
                'epochs': log['epochs'],
                'ckpt': str((a_dir / 'ckpt' / f'best_{mode}.pt').resolve()),
                'src':  str((a_dir / 'src').resolve()),
            }
    h_dir = base / 'ablation_for_heads_layers'
    for tag in ['h2l2', 'h4l2', 'h4l4_default', 'h8l6']:
        log_path = h_dir / 'logs' / f'{tag}.json'
        if log_path.exists():
            log = json.loads(log_path.read_text(encoding='utf-8'))
            results[f'hl_{tag}'] = {
                'best': log['best_dev_acc'],
                'params': log.get('params'),
                'config': log.get('config'),
                'epochs': log['epochs'],
                'ckpt': str((h_dir / 'ckpt' / f'best_{tag}.pt').resolve()),
                'src':  str((h_dir / 'src').resolve()),
            }
    r_dir = base / 'ablation_for_rope'
    log_path = r_dir / 'logs' / 'rope.json'
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding='utf-8'))
        results['rope'] = {
            'best': log['best_dev_acc'],
            'epochs': log['epochs'],
            'ckpt': str((r_dir / 'ckpt' / 'best_rope.pt').resolve()),
            'src':  str((r_dir / 'src').resolve()),
        }
    return results


SUBPROC_TEMPLATE = '''
import sys, os
# 同时改 sys.path 与 cwd，确保 'src' 能被作为包找到
sys.path.insert(0, os.getcwd())
sys.path.insert(0, r"{src_path}")
# 让 'src' 在父目录下能被识别为 package
sys.path.insert(0, os.path.dirname(r"{src_path}"))
sys.path.insert(0, r"{src_parent}")

# 清缓存
for k in list(sys.modules.keys()):
    if k == "src" or k.startswith("src."):
        del sys.modules[k]

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from src.model import TransformerClassifier
from transformers import AutoTokenizer

ZH_FONT = fm.FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=10)
ZH_FONT_TITLE = fm.FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=13)

ckpt = torch.load(r"{ckpt}", map_location="cpu", weights_only=False)
cfg = ckpt["config"]
model = TransformerClassifier(
    vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
    n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
    d_ff=cfg["d_ff"], num_classes=cfg.get("num_classes", 2),
    max_len=cfg["max_len"], padding_idx=0,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=False)
text = r"""{sample}"""
ids = tokenizer.encode(text, max_length=200, padding="max_length",
                        truncation=True, return_tensors="pt")
valid_len = int((ids != 0).sum().item())
ids = ids[:, :valid_len]

with torch.no_grad():
    logits, attn_list = model(ids, return_attn_weights=True)

last = attn_list[-1][0].cpu().numpy()
entropies = []
for h in last:
    p = h + 1e-12
    entropies.append(-(p * np.log(p)).sum(-1).mean())
head = int(np.argmin(entropies))
attn = last[head]
tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())
pred = int(logits.argmax(-1).item())
prob = float(torch.softmax(logits, -1)[0, pred].item())

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(attn, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45, ha="right", fontproperties=ZH_FONT)
ax.set_yticklabels(tokens, fontproperties=ZH_FONT)
title = "{name} | head=%d pred=%d p=%.2f" % (head, pred, prob)
ax.set_title(title, fontproperties=ZH_FONT_TITLE)
plt.tight_layout()
plt.savefig(r"{save_path}", dpi=110, bbox_inches="tight")
plt.close()
'''


def plot_attn_subprocess(name, ckpt, src_path, sample, save_path):
    code = SUBPROC_TEMPLATE.format(
        src_path=src_path,
        src_parent=str(Path(src_path).parent.resolve()),
        ckpt=ckpt,
        sample=sample.replace('"""', '\\"\\"\\"'),
        name=name, save_path=save_path,
    )
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, 'HF_ENDPOINT': 'https://hf-mirror.com'},
        )
        if result.returncode != 0:
            print(f'  ! {name} subprocess failed:')
            err_lines = result.stderr.strip().split('\n')
            for line in err_lines[-5:]:
                print('   ', line)
        else:
            print(f'  -> {Path(save_path).name}')
    finally:
        os.unlink(tmp)


def make_comparison_plot(results, save_path):
    items = [(k, v['best']) for k, v in results.items()]
    items.sort(key=lambda x: -x[1])
    names = [k for k, _ in items]
    accs  = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))
    bars = ax.barh(range(len(names)), accs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontproperties=ZH_FONT)
    ax.set_xlabel('Best dev accuracy', fontproperties=ZH_FONT_TITLE)
    ax.set_title('Ablation comparison (ChnSentiCorp dev set)',
                 fontproperties=ZH_FONT_BIG)
    ax.invert_yaxis()
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.001, bar.get_y() + bar.get_height()/2,
                f'{acc:.4f}', va='center', fontproperties=ZH_FONT)
    ax.set_xlim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  -> {save_path.name}')


def make_curves_plot(results, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, info in results.items():
        epochs = info['epochs']
        xs = [e['epoch'] for e in epochs]
        ys = [e['dev_acc'] for e in epochs]
        ax.plot(xs, ys, marker='o', label=name)
    ax.set_xlabel('Epoch', fontproperties=ZH_FONT_TITLE)
    ax.set_ylabel('Dev accuracy', fontproperties=ZH_FONT_TITLE)
    ax.set_title('Training curves across ablations', fontproperties=ZH_FONT_BIG)
    ax.legend(prop=ZH_FONT, fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  -> {save_path.name}')


def main():
    print('Collecting ablation logs...')
    results = collect_logs()
    print(f'  {len(results)} ablation results found.')
    for k, v in results.items():
        print(f'    {k}: best={v["best"]:.4f}')

    summary = {k: v['best'] for k, v in results.items()}
    out = ROOT / 'ablation_summary.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    print(f'\nSummary -> {out.name}')

    figs = ROOT / 'figures' / 'ablation'
    figs.mkdir(parents=True, exist_ok=True)
    make_comparison_plot(results, figs / 'ablation_bar.png')
    make_curves_plot(results, figs / 'ablation_curves.png')

    sample = '这家酒店的服务非常好，房间干净整洁，前台态度也很热情，下次还会再来。'
    print('\nGenerating cross-ablation attention heatmaps...')
    for name, info in results.items():
        tag = name.replace('_', '-')
        plot_attn_subprocess(name, info['ckpt'], info['src'], sample,
                              str((figs / f'attn_{tag}.png').resolve()))


if __name__ == '__main__':
    main()