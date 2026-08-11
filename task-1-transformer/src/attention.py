"""M1: 手写注意力层。
参考 Harvard NLP Annotated Transformer (http://nlp.seas.harvard.edu/annotated-transformer/)
不依赖 nn.MultiheadAttention 或其他高层封装。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def clones(module, n):
    """产生 n 个相同层的深拷贝（权重不共享）。"""
    return nn.ModuleList([module for _ in range(n)])


# ----------------------------------------------------------------------
# 1. Scaled Dot-Product Attention（核心注意力单元）
# ----------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    """缩放点积注意力。

    公式：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        Q: (B, H, T, D)  查询
        K: (B, H, T, D)  键
        V: (B, H, T, D)  值
        mask: (T, T) 或 (B, T, T)，1/True = 屏蔽位置（填 -1e9）

    Returns:
        (B, H, T, D) 加权输出
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask == 1 => 屏蔽（填 -1e9 让 softmax 后接近 0）
        scores = scores.masked_fill(mask == 1, float('-1e9'))

    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, V)
    return out


# ----------------------------------------------------------------------
# 2. Multi-Head Attention
# ----------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """多头注意力。

    思想：把 Q/K/V 分成 H 个 head 并行做注意力，再 concat 回来。
    这样每个 head 可以关注不同的表示子空间。

    输入/输出：(batch, seq_len, d_model)
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h          # 每个 head 的维度
        self.dropout_p = dropout

        # Q/K/V/O 四个投影，d_model -> d_model（无 bias 更稳定）
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        self.dropout = nn.Dropout(dropout)

        # 以下属性 forward 时暂存，节省重复计算
        self.attn_weights: torch.Tensor = None

    def forward(self, query, key, value, mask=None):
        """参数命名沿用标准习惯：query=Q, key=K, value=V。

        Args:
            query: (B, T, D)
            key:   (B, T, D)
            value: (B, T, D)
            mask:  (T, T) 上三角 causal mask，或 (B, T, T) 任意 mask
                   mask == 1 表示屏蔽

        Returns:
            (B, T, D) 注意力输出
        """
        if mask is not None:
            # 自动 broadcast 到 (B, H, T, T)
            # mask 形状：(T, T) / (B, T, T) / (B, H, T, T)
            if mask.dim() == 2:                  # (T, T) -> (1, 1, T, T)
                _mask = mask[None, None, :, :]
            elif mask.dim() == 3:                # (B, T, T) -> (B, 1, T, T)
                _mask = mask[:, None, :, :]
            else:                                # 已是 (B, H, T, T)
                _mask = mask
        else:
            _mask = None

        B = query.size(0)

        # -------- 投影 + 分 head --------
        # Q, K, V: (B, T, D) -> (B, T, H, d_k) -> (B, H, T, d_k)
        query, key, value = [
            lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # -------- 注意力 --------
        x, self.attn_weights = self._attention(query, key, value, mask=_mask)

        # -------- concat head -> 还原形状 --------
        # (B, H, T, d_k) -> (B, T, D)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # -------- 输出投影 --------
        return self.linears[-1](x)

    def _attention(self, Q, K, V, mask=None):
        """计算注意力并返回权重（供可视化）。"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-1e9'))
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout_p > 0:
            attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, V), attn_weights
