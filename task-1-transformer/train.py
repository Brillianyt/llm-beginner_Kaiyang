"""M3: 训练 Transformer 文本分类器。
用法: python train.py
"""
import os
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]          # 仓库根目录
DATA_DIR = Path(__file__).resolve().parent / 'data'  # task-1-transformer/data
CKPT_DIR = Path(__file__).resolve().parent / 'ckpt'
CKPT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# 训练超参
# ----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


# ----------------------------------------------------------------------
# Cosine LR Schedule with Warmup
# ----------------------------------------------------------------------
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------------------
# 主训练
# ----------------------------------------------------------------------
def main():
    args = get_args()
    device = torch.device(args.device)
    print(f'Device: {device}')

    # ---- 数据加载 ----
    print('Loading data...')
    import pandas as pd
    from transformers import AutoTokenizer

    train_df = pd.read_parquet(DATA_DIR / 'train.parquet')
    dev_df   = pd.read_parquet(DATA_DIR / 'validation.parquet')
    print(f'Train: {len(train_df)}, Dev: {len(dev_df)}')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', use_fast=False)
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id or 0

    def collate_fn(batch):
        texts  = [item['text']  for item in batch]
        labels = [item['label'] for item in batch]
        enc = tokenizer(
            texts,
            max_length=args.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return enc['input_ids'], torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(train_df.to_dict('records'), batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=0)
    dev_loader   = DataLoader(dev_df.to_dict('records'),   batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ---- 模型 ----
    print('Building model...')
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        num_classes=2,
        max_len=args.max_len,
        padding_idx=pad_idx,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    # ---- 优化器 + LR Schedule ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    loss_fn = nn.CrossEntropyLoss()

    # ---- 训练循环 ----
    best_dev_acc = 0.0
    ckpt_path = CKPT_DIR / 'best.pt'

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        start = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, (ids, labels) in enumerate(pbar):
            ids    = ids.to(device)
            labels = labels.to(device)

            logits = model(ids)
            loss   = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - start

        # ---- Dev 评估 ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, labels in dev_loader:
                ids    = ids.to(device)
                labels = labels.to(device)
                logits = model(ids)
                preds  = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        dev_acc = correct / total
        print(f'Epoch {epoch+1} | loss={avg_loss:.4f} | dev_acc={dev_acc:.4f} | best={best_dev_acc:.4f} | {elapsed:.1f}s')

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'd_model':   args.d_model,
                    'n_heads':   args.n_heads,
                    'n_layers':  args.n_layers,
                    'd_ff':      args.d_ff,
                    'vocab_size': vocab_size,
                    'max_len':   args.max_len,
                    'num_classes': 2,
                },
                'dev_acc': dev_acc,
                'epoch': epoch,
            }, ckpt_path)
            print(f'  [*] New best saved: {ckpt_path}')

    print(f'\nTraining done. Best dev_acc={best_dev_acc:.4f}')


if __name__ == '__main__':
    from src.model import TransformerClassifier
    main()
