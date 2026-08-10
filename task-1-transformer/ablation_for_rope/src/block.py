"""S4: TransformerBlock 使用 RoPE（无需 PositionalEncoding 模块）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class SublayerOutput(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048,
                 dropout: float = 0.1, use_rope: bool = True, max_len: int = 5000):
        super().__init__()
        # use_rope=True 时不再加 sin PE
        self.self_attn = MultiHeadAttention(h=n_heads, d_model=d_model, dropout=dropout,
                                            use_rope=use_rope, max_len=max_len)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = nn.ModuleList([
            SublayerOutput(d_model, dropout) for _ in range(2)
        ])

    def forward(self, x, mask=None):
        x = self.sublayers[0](x, lambda _x: self.self_attn(_x, _x, _x, mask))
        x = self.sublayers[1](x, self.feed_forward)
        return x