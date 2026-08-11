"""M2/M3: 因果多头自注意力 + KV cache。

设计要点:
- 手写 scaled-dot-product attention(不调 F.scaled_dot_product_attention),保证教学价值
- KV cache 拼接在 K/V 的 seq 维(dim=-2)
- 因果 mask 维度:(T_query, T_key) = (current T, past+T);增量解码时 T=1,总 key 长度=past+1
- position_offset 必须由调用方显式传入(用于 RoPE);非 cache 路径默认 0,与全量 forward 数值一致
- dropout 在 inference(模型 eval 模式)下自动跳过
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryPositionalEmbedding


class CausalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RotaryPositionalEmbedding | None,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} 必须能整除 n_heads={n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope  # None 表示已用 absolute PE(在 embedding 上加过)

        # 无 bias 更稳定(GPT-2 风格)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
        return_cache: bool = False,
    ):
        """x: (B, T, d_model)

        kv_cache: None 或 (K_past, V_past),shape 各为 (B, H, T_past, head_dim)
        position_offset: 用于 RoPE,通常等于 K_past 的 seq 长度
        return_cache: True 时返回 (out, (K_new, V_new)),其中 K_new/V_new 已拼接 cache

        返回: (B, T, d_model);若 return_cache=True 则附带新的 (K_full, V_full)
        """
        B, T, _ = x.shape

        # 1. Q/K/V 投影 + 分头
        # (B, T, D) -> (B, T, H, d_h) -> (B, H, T, d_h)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE 只加在 Q 和 K_new(不加 V);如果使用 absolute PE,这里跳过
        if self.rope is not None:
            q = self.rope(q, position_offset=position_offset)
            k_new = self.rope(k_new, position_offset=position_offset)

        # 3. 拼接 KV cache(若有)
        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k_new], dim=-2)
            v = torch.cat([v_past, v_new], dim=-2)
        else:
            k, v = k_new, v_new

        # 4. 因果 mask:形状 (T_q, T_k)
        #    允许 query 位置 i(全局位置 i+position_offset)关注 key 位置 ≤ i+position_offset
        #    即 mask[q, k] = (k > q + position_offset)
        T_k = k.size(-2)
        # 用广播构造 mask (T, T_k)
        q_pos = torch.arange(T, device=x.device).unsqueeze(1) + position_offset   # (T, 1)
        k_pos = torch.arange(T_k, device=x.device).unsqueeze(0)                     # (1, T_k)
        mask = k_pos > q_pos   # True = 屏蔽

        # 5. Scaled dot-product attention(手写)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale   # (B, H, T, T_k)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        # dropout 在 eval 模式自动关闭(self.training 控制)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v                              # (B, H, T, head_dim)

        # 6. 合头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        if return_cache:
            # 返回的是当前步拼接后的 K/V(下次调用直接复用)
            return out, (k, v)
        return out