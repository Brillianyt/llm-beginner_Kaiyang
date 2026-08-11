"""M5: 采样策略 — greedy / temperature / top-k / top-p。

实现顺序(对齐 HuggingFace Transformers 惯例):
  1. temperature ≤ 0 → 直接 argmax,跳过一切截断
  2. logits /= temperature
  3. top-k 截断:把第 k 名之后的 logits 置为 -inf
  4. top-p 截断:按概率降序累加到阈值,把超出阈值的尾部置为 -inf
  5. softmax → multinomial 采样

注意事项:
- top-k 与 top-p 都要做"保留第一个超过阈值的 token"避免全 -inf
- top-p 内部用 sorted indices 排序,最后 scatter 回原 logits 空间
- batch 内独立处理,逻辑按 batch 维广播
"""
import torch
import torch.nn.functional as F


def sample(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """logits: (B, V);返回 (B,) 采样 token id。"""
    # 1. temperature=0 或 top_k=1(且无 top_p)→ 直接 greedy
    if temperature <= 0 or (top_k == 1 and top_p is None):
        return logits.argmax(dim=-1)

    # 2. 应用 temperature
    logits = logits / temperature

    # 3. top-k 截断
    if top_k is not None and top_k > 0:
        # 取 top-k 阈值
        kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1:]   # (B, 1)
        logits = torch.where(
            logits < kth_vals,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    # 4. top-p (nucleus) 截断
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)
        # 超过阈值的尾部 mask;但保留第一个超过阈值的(否则概率全为 0)
        sorted_mask = cumprobs > top_p
        sorted_mask[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        # scatter 回原 logits 空间
        logits = torch.empty_like(logits).scatter(-1, sorted_idx, sorted_logits)

    # 5. softmax + 采样
    probs = F.softmax(logits, dim=-1)
    # 防止全 -inf 导致的 NaN:把 NaN 替换为均匀分布
    probs = torch.nan_to_num(probs, nan=0.0)
    # 极端情况下 probs 全 0(几乎不可能),退化为 argmax
    if (probs.sum(dim=-1) == 0).any():
        return logits.argmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)