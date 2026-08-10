"""S1: head 数 / 层数 消融实验。

固定 d_model=128，d_ff=512，跑 4 组配置：
  - (h=2, l=2)  小模型
  - (h=4, l=2)  多头 + 少层
  - (h=4, l=4)  默认（对照）
  - (h=8, l=6)  大模型

每组配置单独训，结果写到 logs/<config>.json 与 ckpt/ 下。
"""
import argparse
import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
CKPT_DIR = ROOT / 'ckpt'
LOG_DIR  = ROOT / 'logs'
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

CONFIGS = [
    {'n_heads': 2, 'n_layers': 2, 'tag': 'h2l2'},
    {'n_heads': 4, 'n_layers': 2, 'tag': 'h4l2'},
    {'n_heads': 4, 'n_layers': 4, 'tag': 'h4l4_default'},
    {'n_heads': 8, 'n_layers': 6, 'tag': 'h8l6'},
]


def cosine_warmup(opt, warmup, total):
    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        p = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def run_one(cfg, epochs=4, batch_size=32, lr=3e-4, warmup=500,
            d_model=128, d_ff=512, max_len=256, device='cuda'):
    tag = cfg['tag']
    print(f'\n========== Config: {tag} (heads={cfg["n_heads"]}, layers={cfg["n_layers"]}) ==========')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    import pandas as pd
    from transformers import AutoTokenizer
    from src.model import TransformerClassifier

    train_df = pd.read_parquet(DATA_DIR / 'train.parquet')
    dev_df   = pd.read_parquet(DATA_DIR / 'validation.parquet')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', use_fast=False)
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id or 0

    def collate(batch):
        texts  = [b['text']  for b in batch]
        labels = [b['label'] for b in batch]
        enc = tokenizer(texts, max_length=max_len, padding='max_length',
                         truncation=True, return_tensors='pt')
        return enc['input_ids'], torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(train_df.to_dict('records'), batch_size=batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    dev_loader   = DataLoader(dev_df.to_dict('records'), batch_size=batch_size * 2,
                              shuffle=False, collate_fn=collate, num_workers=0)

    model = TransformerClassifier(
        vocab_size=vocab_size, d_model=d_model,
        n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
        d_ff=d_ff, num_classes=2, max_len=max_len, padding_idx=pad_idx,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {total_params:,}')

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = cosine_warmup(optimizer, warmup, total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    log = {'config': cfg, 'params': total_params, 'epochs': []}
    ckpt_path = CKPT_DIR / f'best_{tag}.pt'

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start = time.time()
        for ids, labels in tqdm(train_loader, desc=f'[{tag}] E{epoch+1}', leave=False):
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, labels in dev_loader:
                ids, labels = ids.to(device), labels.to(device)
                logits = model(ids)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        dev_acc = correct / total
        elapsed = time.time() - start
        print(f'  loss={avg_loss:.4f} dev_acc={dev_acc:.4f} ({elapsed:.1f}s)')
        log['epochs'].append({'epoch': epoch+1, 'loss': avg_loss, 'dev_acc': dev_acc, 'elapsed': elapsed})

        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'd_model': d_model, 'n_heads': cfg['n_heads'],
                    'n_layers': cfg['n_layers'], 'd_ff': d_ff,
                    'vocab_size': vocab_size, 'max_len': max_len,
                    'num_classes': 2, 'tag': tag,
                },
                'dev_acc': dev_acc, 'epoch': epoch,
            }, ckpt_path)

    log['best_dev_acc'] = best_acc
    with open(LOG_DIR / f'{tag}.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f'  [{tag}] best dev_acc = {best_acc:.4f}')
    return best_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--only', default=None, help='只跑指定 tag，如 h2l2')
    args = p.parse_args()

    summary = {}
    for cfg in CONFIGS:
        if args.only and cfg['tag'] != args.only:
            continue
        best = run_one(cfg, epochs=args.epochs, batch_size=args.batch_size,
                       lr=args.lr, device=args.device)
        summary[cfg['tag']] = {'heads': cfg['n_heads'], 'layers': cfg['n_layers'], 'best_dev_acc': best}

    # 汇总
    with open(LOG_DIR / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('\n=== Summary ===')
    for tag, r in summary.items():
        print(f'  {tag:20s} h={r["heads"]} l={r["layers"]}  dev_acc={r["best_dev_acc"]:.4f}')


if __name__ == '__main__':
    main()