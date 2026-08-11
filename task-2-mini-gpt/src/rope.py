"""M2: 旋转位置编码(RoPE) + Sinusoidal 绝对位置编码(供 S2 对比)。

实现 RoFormer (Su et al. 2021) 的相邻两维配对方案:
  对 head_dim 维的向量,把它切成 (d/2) 个相邻对
  每一对 (x_2i, x_2i+1) 看作一个二维向量,用位置 pos 对应的旋转矩阵作用

  旋转角度 θ_pos,i = pos / base^(2i/d),i ∈ [0, d/2)

关键约定:
- forward 必须显式接收 position_offset,这是 KV cache 增量解码正确性的唯一保证
- cos/sin 缓存到 max_seq_len,position_offset 用于切片
"""
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Vaswani et al. 2017 风格的正弦绝对位置编码。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    与 RoPE 的关键差异:
    - 绝对 PE **直接加到 token embedding 上**(attention 输入端),而 RoPE 加在 Q/K(attention 内)
    - 绝对 PE 在训练长度 block_size 之外**没有任何位置信号**(对应位置没有 PE 项),
      实际使用时只能截断或外推(外推效果很差)
    - RoPE 在训练长度之外仍可用,因为旋转角 θ 可以任意扩展

    S2 实验目的:对比两者在 seq > block_size 时的表现。
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        # 注意:dropout 由调用方负责(self.dropout),这里不重复
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model);返回 x + PE[:, :T, :]

        若 T > max_len,自动扩展 PE 表(用于长序列外推)。
        """
        T = x.size(1)
        if T > self.max_len:
            self.extend(T)
        return x + self.pe[:, :T, :]

    def extend(self, new_max_len: int) -> None:
        """如果序列长度超过 max_len,扩展 PE 表(只追加,前面的不变)。"""
        if new_max_len <= self.max_len:
            return
        device = self.pe.device
        pe_new = torch.zeros(new_max_len, self.d_model, device=device)
        position = torch.arange(0, new_max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe_new[:, 0::2] = torch.sin(position * div_term)
        pe_new[:, 1::2] = torch.cos(position * div_term)
        pe_new = pe_new.unsqueeze(0)
        self.register_buffer("pe", pe_new)
        self.max_len = new_max_len


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim 必须为偶数,得到 {head_dim}"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # inv_freq[i] = 1 / base^(2i/head_dim),shape (head_dim/2,)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        """预计算 cos/sin 表,shape (1, 1, max_seq_len, head_dim)。

        必须跟 inv_freq 在同一 device(否则 outer 报错);model.to(device) 后
        inv_freq 跟着到新 device,但 torch.arange 默认 CPU,这里显式对齐。
        """
        self.max_seq_len = max_seq_len
        device = self.inv_freq.device
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (S,)
        freqs = torch.outer(t, self.inv_freq)                              # (S, head_dim/2)
        # 复制一份让维度配对相邻两维
        emb = torch.cat((freqs, freqs), dim=-1)                           # (S, head_dim)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """把 (x_even, x_odd) → (-x_odd, x_even)。

        等价于把相邻两维对调并对前一半取负。
        """
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # 交错回去:[-x_odd[0], x_even[0], -x_odd[1], x_even[1], ...]
        rotated = torch.stack((-x_odd, x_even), dim=-1)
        return rotated.flatten(-2)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """x: (B, H, T, head_dim);position_offset: KV cache 已累积长度。"""
        T = x.size(-2)
        if position_offset + T > self.max_seq_len:
            # 缓存不够,动态扩容
            new_max = max(self.max_seq_len * 2, position_offset + T)
            self._build_cache(new_max)
        cos = self.cos_cached[:, :, position_offset : position_offset + T, :]
        sin = self.sin_cached[:, :, position_offset : position_offset + T, :]
        return (x * cos) + (self._rotate_half(x) * sin)