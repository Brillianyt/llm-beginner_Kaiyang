"""Ablation: 拆解 Add（残差）与 Norm（LayerNorm）。

对应 README 加分项 S2：「拆掉 residual 或 LayerNorm，记录训练是否还能收敛」。

支持 4 种 ablation 模式（通过 ABLATION_MODE 全局变量控制）:
  - 'baseline'          : Pre-LN + Residual（默认，等价于主实验）
  - 'no_residual'       : 去掉残差连接（保留 LN）
  - 'no_layernorm'      : 去掉 LayerNorm（保留残差）
  - 'no_residual_no_ln' : 残差和 LN 都去掉

训练脚本中通过命令行 --ablation 切换。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


# 全局 ablation 模式开关（被 train.py 通过 monkey-patch 设置）
ABLATION_MODE = 'baseline'


class PositionalEncoding(nn.Module):
    """Sinusoidal 位置编码。"""

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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class SublayerOutput(nn.Module):
    """封装「残差 + LayerNorm」，按 ablation 模式选择性地禁用某个部分。"""
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, sublayer):
        mode = ABLATION_MODE

        # 1) 计算 sublayer 输出
        if mode == 'no_layernorm' or mode == 'no_residual_no_ln':
            # 不做 LN
            inner = self.dropout(sublayer(x))
        else:
            # Pre-LN：先 LN 再 sublayer
            inner = self.dropout(sublayer(self.norm(x)))

        # 2) 是否加残差
        if mode == 'no_residual' or mode == 'no_residual_no_ln':
            return inner
        return x + inner


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(h=n_heads, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = nn.ModuleList([
            SublayerOutput(d_model, dropout) for _ in range(2)
        ])

    def forward(self, x, mask=None):
        x = self.sublayers[0](x, lambda _x: self.self_attn(_x, _x, _x, mask))
        x = self.sublayers[1](x, self.feed_forward)
        return x