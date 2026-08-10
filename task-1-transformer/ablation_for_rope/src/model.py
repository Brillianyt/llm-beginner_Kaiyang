"""S4: TransformerClassifier 使用 RoPE（无独立 PE 模块）。
"""
import math
import torch
import torch.nn as nn
from .block import TransformerBlock


class TransformerClassifier(nn.Module):
    """RoPE 版 Transformer 分类器。

    架构：Embedding + N * TransformerBlock（RoPE 在 MHA 内部）+ Pooling + ClassifierHead
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        num_classes: int = 2,
        max_len: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        # 不再需要 PositionalEncoding —— RoPE 在 MHA 内部

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout,
                             use_rope=True, max_len=max_len)
            for _ in range(n_layers)
        ])

        # 分类头
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        # 权重初始化（xavier_uniform）
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, token_ids, return_attn_weights=False):
        """前向。

        Args:
            token_ids: (B, T) LongTensor
            return_attn_weights: 若 True 返回 (logits, attn_weights_list)

        Returns:
            (B, num_classes) logits
            或 (logits, attn_weights_list) 当 return_attn_weights=True
        """
        # Padding mask: (B, T)，True = pad 位置
        pad_mask = (token_ids == self.padding_idx)

        # 1. Embedding（不加 PE —— RoPE 在 attention 内）
        x = self.embed(token_ids) * math.sqrt(self.d_model)

        # 2. N 层 encoder
        attn_weights_all = []
        for block in self.blocks:
            x = block(x, mask=None)        # encoder 用 Full mask（双向）
            if return_attn_weights:
                attn_weights_all.append(block.self_attn.attn_weights)

        x = self.norm(x)

        # 3. Pooling：[CLS] 位或 mean pooling
        if hasattr(self, 'cls_token_id') and self.cls_token_id is not None:
            # [CLS] pooling
            cls_pos = (token_ids == self.cls_token_id).nonzero(as_tuple=True)
            pooled = x[cls_pos[0], cls_pos[1], :]
        else:
            # Mean pooling（排除 padding）
            mask_expanded = (~pad_mask).unsqueeze(-1).float()   # (B, T, 1)
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if return_attn_weights:
            return logits, attn_weights_all
        return logits


# ----------------------------------------------------------------------
# 工厂函数：加载 checkpoint，返回 (model, tokenize_fn)
# ----------------------------------------------------------------------
def load_for_eval(ckpt_path: str):
    """加载训练好的模型，供 eval/run.py 调用。

    Returns:
        model: TransformerClassifier（eval 模式）
        tokenize_fn: (text: str) -> LongTensor (T,)
    """
    import os
    from transformers import AutoTokenizer

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})

    # 兼容旧 ckpt（config 未存）
    d_model = cfg.get('d_model', 128)
    n_heads = cfg.get('n_heads', 4)
    n_layers = cfg.get('n_layers', 4)
    d_ff = cfg.get('d_ff', 512)
    vocab_size = cfg.get('vocab_size', 21128)

    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 用 BERT 中文词表做 tokenize（来自 ckpt 所在目录或默认）
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', use_fast=False)

    def tokenize_fn(text: str) -> torch.LongTensor:
        ids = tokenizer.encode(
            text,
            max_length=cfg.get('max_len', 256),
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return ids.squeeze(0)   # (T,)

    return model, tokenize_fn
