"""M4: Toy 语言模型训练 + future-token independence 验证。

复用同一个手写 attention + TransformerBlock，加上 causal mask（上三角 -inf），
在唐诗数据上做 next-token prediction。

末尾额外验证：
  1. causal mask 自检（test_causal_mask 等价，已在 eval/run.py 覆盖）
  2. 实测：随机扰动未来 token embedding，验证过去位置输出不变
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.attention import scaled_dot_product_attention, MultiHeadAttention
from src.block import TransformerBlock, PositionalEncoding


class ToyLM(nn.Module):
    """小型 Transformer Decoder（用 causal mask 复用 encoder block）。"""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 max_len=128, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, max_len, dropout)

        # 因果 mask 缓存：上三角 True = 屏蔽
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask, persistent=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff=d_model*2, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ids):
        T = ids.size(1)
        x = self.embed(ids) * math.sqrt(self.d_model if hasattr(self, 'd_model') else 64)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x, mask=self.causal_mask[:T, :T])
        x = self.norm(x)
        return self.head(x)


def load_tang_text():
    """加载唐诗数据集。"""
    p = ROOT.parent / 'poetryFromTang.txt'
    text = p.read_text(encoding='utf-8', errors='ignore')
    # 清理：只保留中文字符
    chars = sorted(set(c for c in text if '一' <= c <= '鿿'))
    chars = ['<pad>'] + chars
    vocab = {c: i for i, c in enumerate(chars)}
    # 把原文切成字符序列
    sequence = [vocab[c] for c in text if c in vocab]
    return sequence, vocab


def make_batches(seq, block_size=64, batch_size=32):
    """切训练样本：next-token prediction。"""
    import random
    random.shuffle(seq)
    n = (len(seq) // (block_size + 1)) * (block_size + 1)
    seq = seq[:n]
    ids = torch.tensor(seq, dtype=torch.long).view(-1, block_size + 1)
    x = ids[:, :-1]
    y = ids[:, 1:]
    for i in range(0, x.size(0), batch_size):
        yield x[i:i+batch_size], y[i:i+batch_size]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    print('Loading 唐诗...')
    seq, vocab = load_tang_text()
    print(f'  vocab size: {len(vocab)}, total tokens: {len(seq)}')

    model = ToyLM(vocab_size=len(vocab), d_model=64, n_heads=4, n_layers=2,
                   max_len=128).to(device)
    print(f'  params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    # ----- 训练 -----
    BLOCK = 64
    EPOCHS = 3
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batch = 0
        for x, y in make_batches(seq, block_size=BLOCK, batch_size=32):
            x, y = x.to(device), y.to(device)
            logits = model(x)                  # (B, T, V)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        print(f'  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/n_batch:.4f}')

    # ----- 验证 1: 未来 token 改动不影响过去位置输出 -----
    print('\n[验证 1] causal mask 是否真的阻止了未来信息泄漏')
    model.eval()
    test_text = '床前明月光疑是地上霜'
    test_ids = torch.tensor([vocab[c] for c in test_text if c in vocab],
                             dtype=torch.long).unsqueeze(0).to(device)
    T = test_ids.size(1)

    # 干净 forward
    with torch.no_grad():
        logits_clean = model(test_ids)

    # 改动最后一个 token（未来）的 embedding（在 embed 层手动扰动）
    with torch.no_grad():
        # 提取 embedding 层注入扰动
        x = model.embed(test_ids) * math.sqrt(64)
        x = model.pe(x)
        # 在 x 上修改最后一个位置
        x_perturbed = x.clone()
        x_perturbed[:, -1, :] += torch.randn_like(x_perturbed[:, -1, :]) * 10
        for block in model.blocks:
            x_perturbed = block(x_perturbed, mask=model.causal_mask[:T, :T])
        x_perturbed = model.norm(x_perturbed)
        logits_perturbed = model.head(x_perturbed)

    diff = (logits_clean[:, :-1] - logits_perturbed[:, :-1]).abs().max().item()
    print(f'  past 位置输出最大差异: {diff:.6e}')
    verdict = 'PASS' if diff < 1e-5 else 'FAIL'
    print(f'  [{verdict}] causal mask 正确阻止了未来位置 V 改动影响过去')

    # ----- 验证 2: 生成长诗 -----
    print('\n[验证 2] 给定 prompt 续写')
    prompt = '床前明月光'
    prompt_ids = [vocab[c] for c in prompt if c in vocab]
    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(20):
            ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
            logits = model(ids)
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
    inv_vocab = {i: c for c, i in vocab.items()}
    text = ''.join(inv_vocab.get(i, '?') for i in generated)
    print(f'  prompt: {prompt}')
    print(f'  生成:   {text}')


if __name__ == '__main__':
    main()