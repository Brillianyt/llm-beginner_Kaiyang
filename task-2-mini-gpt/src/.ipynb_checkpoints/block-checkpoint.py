"""M2/M3: Pre-LN TransformerBlock + SwiGLU MLP。

设计要点:
- Pre-LN:子层前归一化(参考 task-1/src/block.py 风格)
- SwiGLU:FFN(x) = W3(silu(W1(x)) * W2(x)),三个线性层,参数量与 GELU 双线性层相当
- kv_cache / position_offset / return_cache 透传给注意力层
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalMultiHeadAttention
from .rope import RotaryPositionalEmbedding


class SwiGLU(nn.Module):
    """SwiGLU FFN:LLaMA 风格的 gated MLP。

    FFN(x) = W3( silu(W1(x)) * W2(x) )
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 无 bias,LLaMA 风格
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    """一个 decoder block:Pre-LN CausalAttn → Add → Pre-LN SwiGLU → Add。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_heads, rope, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
        return_cache: bool = False,
    ):
        # Pre-LN: 子层前归一化
        attn_out = self.attn(
            self.attn_norm(x),
            kv_cache=kv_cache,
            position_offset=position_offset,
            return_cache=return_cache,
        )
        if return_cache:
            attn_out, new_cache = attn_out
        else:
            new_cache = None
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        if return_cache:
            return x, new_cache
        return x