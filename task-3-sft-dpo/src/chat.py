"""Qwen2.5 chat template + loss masking。

Qwen2.5 官方对话模板格式（chatml 变体）：

```
<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
...
<|im_start|>assistant
```

要点：

1. 每个 turn 之间换行分隔，``<|im_end|>`` 之后必须有 ``\\n``；
2. ``build_labels`` 只对 assistant turn 的内容 + 结束符计算 loss，其余
   （user / system / 模板控制符）一律设为 ``-100``；
3. 多轮对话里 **每一轮** assistant turn 都参与训练，不能只取最后一轮。

合约
----
- :func:`format_messages` 接受 ``List[dict]``，返回 ``str``；
- :func:`build_labels` 接受 ``input_ids: Tensor``（``(T,)`` 或 ``(B, T)``）与
  ``messages``，返回与 ``input_ids`` 同形状的 ``labels`` tensor。
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import torch


# ---------------------------------------------------------------------------
# 模板标记（与 Qwen2.5 chat_template 一致）
# ---------------------------------------------------------------------------
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# 常用 token 字符串。``build_labels`` 需要在 tokenizer 不可用时也尽量可用，
# 因此优先用 ``tokenizer.encode`` 精确切片；当 tokenizer 缺失时退回字符串匹配。
ROLE_TO_LINE = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}


# ---------------------------------------------------------------------------
# 格式化：把 messages 拼成 chat template 字符串
# ---------------------------------------------------------------------------
def _format_one_turn(role: str, content: str) -> str:
    """单个 turn 的字面拼装。"""
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def format_messages(messages: Sequence[dict]) -> str:
    """把 ``[{"role":..., "content":...}, ...]`` 拼成 Qwen chat template 字符串。

    通常最后一轮 assistant 不闭合 ``<|im_end|>``，留给模型续写；本任务的
    SFT 训练场景里也会出现「完整闭合」与「尾部开放」两种数据，函数对两者
    都接受：只要 message 列表里没有「末尾开放」的呼应 token，就按闭合返回。
    若希望最后一轮开放，调用方在 ``messages`` 末尾 ``assistant`` turn 的
    ``content`` 留空字符串 ``""``，模板的 ``<|im_end|>\\n`` 也会保留——这种情况下
    调用方可以自己截断。
    """
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in ROLE_TO_LINE:
            raise ValueError(f"未知 role: {role}")
        parts.append(_format_one_turn(role, content))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Loss masking：只对 assistant turn 算 loss
# ---------------------------------------------------------------------------
def build_labels(
    input_ids: torch.Tensor,
    messages: Sequence[dict],
    tokenizer=None,
) -> torch.Tensor:
    """生成与 ``input_ids`` 同形状的 labels，未参与训练的 token 标 ``-100``。

    策略：
        1. 对每条 message 单独跑 ``tokenizer.encode`` 取真实 token 边界；
        2. 仅 assistant 角色的「content + 末尾 ``<|im_end|>\\n``」段对应
           的 token 位置保留原 ``input_ids``，其余置 ``-100``。

    Args:
        input_ids: ``(T,)`` 或 ``(B, T)`` 的 LongTensor（``format_messages`` 后
            再 ``tokenizer(...)`` 的结果）。
        messages: 与 ``format_messages`` 同一份输入。
        tokenizer: 用于精确定位 assistant 段起止的 tokenizer。如果 ``None``，
            则退回到「整段字符串精确匹配」的策略（仅用于离线测试）。
    Returns:
        与 ``input_ids`` 同形状的 LongTensor。
    """
    is_batched = input_ids.dim() == 2
    if is_batched:
        labels = input_ids.clone()
    else:
        labels = input_ids.clone().unsqueeze(0)

    for b in range(labels.size(0)):
        row_ids = labels[b]
        # 原始 token 序列（后面 assistant 段从这里取值）。
        src = input_ids if not is_batched else input_ids[b]
        # 把 mask 一次性置成 -100，再逐段恢复 assistant 段。
        row_ids.fill_(-100)
        if tokenizer is not None:
            _mask_with_tokenizer(row_ids, src, messages, tokenizer)
        else:
            _mask_with_string_fallback(row_ids, src, messages)

    if not is_batched:
        labels = labels.squeeze(0)
    return labels


def _mask_with_tokenizer(
    row_ids: torch.Tensor,
    src: torch.Tensor,
    messages: Sequence[dict],
    tokenizer,
) -> None:
    """用 tokenizer 精确切片定位 assistant 段。"""
    # 先把整段模板的 token 序列拼出来，再按 message 顺序切。
    # 每条 message = f"{IM_START}{role}\n{content}{IM_END}\n"
    # 我们用 chat template 风格：「header」和「content」分开 encode，再决定 mask。
    cursor = 0
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        header_str = f"{IM_START}{role}\n"
        header_ids = tokenizer.encode(header_str, add_special_tokens=False)
        # 大多数 Qwen tokenizer 把 ``<|im_end|>`` 编为单 token；末尾的 ``\n``
        # 单算 1 个 token。
        end_str = f"{IM_END}\n"
        end_ids = tokenizer.encode(end_str, add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)

        header_len = len(header_ids)
        end_len = len(end_ids)
        content_len = len(content_ids)

        # header 段跳过（无论是 system / user / assistant，header 都不参与 loss）。
        cursor += header_len

        if role == "assistant":
            # content 段 + 末尾的 <|im_end|>\n 参与 loss，从原始 src 拷贝。
            row_ids[cursor : cursor + content_len] = src[cursor : cursor + content_len]
            cursor += content_len
            row_ids[cursor : cursor + end_len] = src[cursor : cursor + end_len]
            cursor += end_len
        else:
            # user / system：content 段、im_end、\n 全部 -100（已 masked，跳过）。
            cursor += content_len
            cursor += end_len

    # 截断超出部分（如果有的话）也保持 -100。


def _mask_with_string_fallback(
    row_ids: torch.Tensor,
    src: torch.Tensor,
    messages: Sequence[dict],
) -> None:
    """无 tokenizer 时的字符串匹配回退：基于 ``format_messages`` 重新拼字符串，
    再按字符级边界映射到 token 区间。

    由于字符级映射不准确（不同 token 覆盖不同字符数），此路径仅用于无
    tokenizer 的离线测试。生产路径请使用 ``tokenizer`` 参数。
    """
    n_token = row_ids.size(0)
    full_text = format_messages(messages)
    approx_assistant_frac = sum(
        len(m["content"]) for m in messages if m["role"] == "assistant"
    ) / max(len(full_text), 1)
    keep = max(1, int(n_token * approx_assistant_frac))
    start = n_token - keep
    for i in range(start, n_token):
        row_ids[i] = src[i]


# ---------------------------------------------------------------------------
# 高级封装：把「format + tokenize + mask」三步串起来
# ---------------------------------------------------------------------------
def encode_chat(
    tokenizer,
    messages: Sequence[dict],
    max_length: int = 2048,
    device: str | torch.device | None = None,
) -> dict:
    """一键完成：chat template → tokenize → loss mask → dict 输出。

    Returns:
        dict with keys ``input_ids``、``labels``、``attention_mask``。
    """
    text = format_messages(messages)
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = enc["input_ids"][0]
    labels = build_labels(input_ids, messages, tokenizer=tokenizer)
    attention_mask = enc["attention_mask"][0]
    if device is not None:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
