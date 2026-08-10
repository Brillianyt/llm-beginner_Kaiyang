"""S4 Ablation: 用 RoPE 替换 sin PE。

本文件在 attention.py 加上 RoPE 旋转函数，由 block.py 在 Q/K 投影后调用。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, n):
    return nn.ModuleList([module for _ in range(n)])


# ---------- RoPE 旋转 ----------
def precompute_rope_cache(head_dim: int, max_len: int = 5000, base: float = 10000.0,
                          device=None, dtype=torch.float32):
    """生成 RoPE 的 cos/sin 缓存。

    返回:
        cos: (max_len, head_dim)
        sin: (max_len, head_dim)
    """
    # θ_i = base^(-2i/d), i = 0, 1, ..., d/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                             dtype=dtype) / head_dim))
    t = torch.arange(max_len, device=device, dtype=dtype)   # (max_len,)
    # 外积：(max_len, d/2)
    freqs = torch.outer(t, inv_freq)
    # 复制成 (max_len, head_dim) —— half 前半 cos，后半 sin
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对最后一维应用 RoPE 旋转。

    Args:
        x: (B, H, T, D)
        cos, sin: (T, D)

    Returns:
        旋转后的 x，shape 不变
    """
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    cos1 = cos[..., :half]
    sin1 = sin[..., :half]

    rotated = torch.cat([
        x1 * cos1 - x2 * sin1,
        x1 * sin1 + x2 * cos1,
    ], dim=-1)
    return rotated


# ---------- 缩放点积（保持不变） ----------
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-1e9'))
    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, V)
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1,
                 use_rope: bool = False, max_len: int = 5000):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        self.use_rope = use_rope
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor = None

        if use_rope:
            cos, sin = precompute_rope_cache(self.d_k, max_len)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        B = query.size(0)

        Q = self.linears[0](query).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.linears[1](key).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.linears[2](value).view(B, -1, self.h, self.d_k).transpose(1, 2)

        # 应用 RoPE（在 attention 之前）
        if self.use_rope:
            T = Q.size(-2)
            cos = self.rope_cos[:T].to(Q.dtype)
            sin = self.rope_sin[:T].to(Q.dtype)
            Q = apply_rope(Q, cos, sin)
            K = apply_rope(K, cos, sin)

        x, self.attn_weights = self._attention(Q, K, V, mask=mask)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.linears[3](x)

    def _attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-1e9'))
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout.p > 0:
            attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, V), attn_weights