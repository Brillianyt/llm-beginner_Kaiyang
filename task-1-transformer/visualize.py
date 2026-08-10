"""M5: 生成注意力热图。

输出 ≥ 3 张：1 正面（预测对）、1 负面（预测对）、1 长句。
挑样本时跑一遍模型过滤：只挑预测==真实标签 的样本，保证可视化有意义。
热图选取最后一层中「最聚焦」的 head（注意力熵最小的 head = 语义头）。
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')   # 无头模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import TransformerClassifier, load_for_eval

OUT_DIR = ROOT / 'figures'
OUT_DIR.mkdir(exist_ok=True)

# 注册中文字体（显式注册 SimHei）
ZH_FONT = None
for path in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']:
    if Path(path).exists():
        ZH_FONT = fm.FontProperties(fname=path, size=10)
        ZH_FONT_TITLE = fm.FontProperties(fname=path, size=12)
        break

if ZH_FONT is None:
    ZH_FONT = fm.FontProperties()
    ZH_FONT_TITLE = fm.FontProperties()


def plot_attention_heatmap(tokens, attn, title, save_path):
    """画一张注意力热图（带中文字体）。"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontproperties=ZH_FONT)
    ax.set_yticklabels(tokens, fontproperties=ZH_FONT)
    ax.set_xlabel('Key', fontproperties=ZH_FONT_TITLE)
    ax.set_ylabel('Query', fontproperties=ZH_FONT_TITLE)
    ax.set_title(title, fontproperties=ZH_FONT_TITLE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ZH_FONT)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention weight', fontproperties=ZH_FONT)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {save_path.name}')


def pick_correct_sample(df, label, model, tokenizer, max_len=60):
    """从 df 中挑一个 label 类别、且模型预测正确、长度 ≤ max_len 的样本。"""
    candidates = df[df['label'] == label].copy()
    candidates['length'] = candidates['text'].str.len()
    candidates = candidates[candidates['length'] <= max_len].sort_values('length')
    for _, row in candidates.iterrows():
        text = str(row['text'])
        ids = tokenizer.encode(text, max_length=200, padding='max_length',
                                truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(ids)
            pred = int(logits.argmax(dim=-1).item())
        if pred == label:
            return text, pred
    # fallback：返回最短的
    return str(candidates.iloc[0]['text']), int(candidates.iloc[0]['label'])


def pick_long_sample(df, model, tokenizer):
    """挑一个相对长的样本（40-80 字），且模型预测正确。"""
    df = df.copy()
    df['length'] = df['text'].str.len()
    candidates = df[(df['length'] > 40) & (df['length'] < 90)].sort_values('length', ascending=False)
    for _, row in candidates.iterrows():
        text = str(row['text'])
        if len(text) > 100:
            text = text[:100]
        label = int(row['label'])
        ids = tokenizer.encode(text, max_length=120, padding='max_length',
                                truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(ids)
            pred = int(logits.argmax(dim=-1).item())
        if pred == label:
            return text, pred
    return str(candidates.iloc[0]['text'])[:100], int(candidates.iloc[0]['label'])


def entropy(attn):
    """注意力分布的熵，越小说明越聚焦。"""
    p = attn + 1e-12
    return -(p * np.log(p)).sum(-1).mean()


def main():
    print('Loading model...')
    model, tokenize_fn = load_for_eval(str(ROOT / 'ckpt' / 'best.pt'))
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', use_fast=False)

    import pandas as pd
    df = pd.read_parquet(ROOT / 'data' / 'validation.parquet')

    print('Picking samples...')
    pos_text, pos_pred = pick_correct_sample(df, label=1, model=model, tokenizer=tokenizer)
    print(f'  positive (true=1, pred={pos_pred}): {pos_text[:80]}')
    neg_text, neg_pred = pick_correct_sample(df, label=0, model=model, tokenizer=tokenizer)
    print(f'  negative (true=0, pred={neg_pred}): {neg_text[:80]}')
    long_text, long_pred = pick_long_sample(df, model, tokenizer)
    print(f'  long (true={long_pred}, pred={long_pred}): {long_text[:80]}...')

    samples = [('positive', pos_text, 1),
               ('negative', neg_text, 0),
               ('long',     long_text, long_pred)]

    for tag, text, true_label in samples:
        print(f'\n[{tag}]')
        ids = tokenize_fn(text).unsqueeze(0)
        valid_len = int((ids != 0).sum().item())
        ids_short = ids[:, :valid_len]

        with torch.no_grad():
            logits, attn_list = model(ids_short, return_attn_weights=True)

        # 选最后一层注意力熵最小的 head（语义最聚焦）
        last_layer_attn = attn_list[-1][0].cpu().numpy()    # (H, T, T)
        head_entropies = [entropy(h) for h in last_layer_attn]
        HEAD = int(np.argmin(head_entropies))
        print(f'  最后一层共 {len(head_entropies)} 个 head，熵={[f"{e:.2f}" for e in head_entropies]}')
        print(f'  选 head {HEAD} (entropy={head_entropies[HEAD]:.2f})')

        attn = last_layer_attn[HEAD]    # (T, T)
        tokens = tokenizer.convert_ids_to_tokens(ids_short[0].tolist())
        pred = int(logits.argmax(dim=-1).item())
        prob = float(torch.softmax(logits, dim=-1)[0, pred].item())
        title = f'[{tag}] layer4 head{HEAD}  true={true_label} pred={pred} p={prob:.2f}'
        out = OUT_DIR / f'attn_{tag}_layer4_head{HEAD}.png'
        plot_attention_heatmap(tokens, attn, title, out)

    print('\n[Done]')


if __name__ == '__main__':
    main()