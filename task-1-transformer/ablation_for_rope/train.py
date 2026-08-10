"""S4: 训练 RoPE 版本 Transformer，与 baseline（sin PE）对比。

只跑一组配置（h=4, l=4, d_model=128）以与主实验直接对比。
"""
import argparse
import json
import math
import sys
import time
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


def cosine_warmup(opt, warmup, total):
    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        p = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=500)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--d_ff', type=int, default=512)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f'==== RoPE training ====')
    print(f'Device: {device}')

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
        enc = tokenizer(texts, max_length=args.max_len, padding='max_length',
                         truncation=True, return_tensors='pt')
        return enc['input_ids'], torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(train_df.to_dict('records'), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    dev_loader   = DataLoader(dev_df.to_dict('records'), batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=collate, num_workers=0)

    model = TransformerClassifier(
        vocab_size=vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, num_classes=2, max_len=args.max_len, padding_idx=pad_idx,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total_params:,}')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_warmup(optimizer, args.warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss()

    log = {'ablation': 'rope', 'epochs': []}
    best_acc = 0.0
    ckpt_path = CKPT_DIR / 'best_rope.pt'

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        start = time.time()
        for ids, labels in tqdm(train_loader, desc=f'[RoPE] Epoch {epoch+1}/{args.epochs}'):
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
                    'd_model': args.d_model, 'n_heads': args.n_heads,
                    'n_layers': args.n_layers, 'd_ff': args.d_ff,
                    'vocab_size': vocab_size, 'max_len': args.max_len,
                    'num_classes': 2, 'ablation': 'rope',
                },
                'dev_acc': dev_acc, 'epoch': epoch,
            }, ckpt_path)

    log['best_dev_acc'] = best_acc
    log['params'] = total_params
    with open(LOG_DIR / 'rope.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f'\n[RoPE] Best dev_acc = {best_acc:.4f}')


if __name__ == '__main__':
    main()