"""M2: TransformerBlock —— 一个 Transformer 编码器层。
参考 Harvard NLP Annotated Transformer 的 Pre-LN 变体（收敛更稳定）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """Sinusoidal 位置编码，与 token embedding 相加。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """FFN：两层 Linear + GELU 激活。

    FFN(x) = W2 * GELU(W1 * x)
    GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    """
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class SublayerOutput(nn.Module):
    """封装「残差 + LayerNorm」，使主线路更整洁。"""
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Pre-LN：先 LayerNorm 再 attention/FFN（比 Post-LN 更稳定）
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """一个 Transformer 编码器层。

    顺序：MultiHeadAttention -> Add&Norm -> FeedForward -> Add&Norm

    采用 Pre-LN（LayerNorm 在残差之前），收敛更稳定。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(h=n_heads, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.sublayers = nn.ModuleList([
            SublayerOutput(d_model, dropout) for _ in range(2)
        ])

    def forward(self, x, mask=None):
        """x: (B, T, D), mask: (T, T)"""
        # 1. Multi-Head Self-Attention + Residual
        x = self.sublayers[0](x, lambda _x: self.self_attn(_x, _x, _x, mask))
        # 2. FeedForward + Residual
        x = self.sublayers[1](x, self.feed_forward)
        return x


import math
