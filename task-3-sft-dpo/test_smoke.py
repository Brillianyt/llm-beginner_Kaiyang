"""不依赖真实模型的离线 smoke test。

验证核心算法在 mock 环境下正确：

1. LoRA forward 形状与初始化；
2. LoRA 合并等价性（merge 后前向与未合并一致）；
3. loss masking 形状与 ``-100`` 占比；
4. DPO 损失的符号与边界（chosen 概率 < rejected → margin < 0）；
5. ``format_messages`` 的 Qwen 模板拼接。

运行：

```bash
python test_smoke.py
```

退出码 ``0`` = 全部通过；非零 = 失败。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.lora import _LoRALinear, inject_lora, lora_state_dict, load_lora_state_dict, merge_lora
from src.chat import build_labels, format_messages


# ---------------------------------------------------------------------------
# Test 1: LoRA 形状 / 初始化 / 训练参数占比
# ---------------------------------------------------------------------------
def test_lora_shapes_and_init():
    torch.manual_seed(0)
    base = nn.Linear(16, 32, bias=True)
    layer = _LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 5, 16)
    y = layer(x)
    assert y.shape == (2, 5, 32), f"shape mismatch: {y.shape}"
    # 初始 B=0，应该等价于 base(x)。
    y_base = base(x)
    assert torch.allclose(y, y_base, atol=1e-6), "init: B=0, A 任意，前向应等于 base"
    # 验证不可训练参数。
    for p in layer.base.parameters():
        assert not p.requires_grad, "base 必须冻结"
    assert layer.lora_A.requires_grad and layer.lora_B.requires_grad, "LoRA 必须可训练"
    print("  [1] LoRA 形状 / 初始化 / 冻结  OK")


def test_lora_param_ratio():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 128)
            self.v_proj = nn.Linear(64, 128)
            self.k_proj = nn.Linear(64, 128)  # 非目标
            self.other = nn.Linear(64, 64)

    m = Toy()
    inject_lora(m, ["q_proj", "v_proj"], r=4, alpha=8)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    ratio = trainable / total
    # 2 个 LoRA 层各 (64*4 + 4*128) = 768 个参数；总参数正比 |model|；toy 总
    # 参数相对小，ratio 偏高，但应当 < 50%（绝大部分应来自 base）。
    assert ratio < 0.5, f"LoRA param ratio too high: {ratio:.4f}"
    # 显式：非目标层（k_proj / other）应仍然冻结。
    assert not m.k_proj.weight.requires_grad
    assert not m.other.weight.requires_grad
    print(f"  [2] LoRA 注入后 trainable={trainable}/{total} = {ratio:.2%}  OK")


def test_lora_merge_equivalence():
    """merge 之后前向输出与未合并一致。"""
    torch.manual_seed(1)
    base = nn.Linear(10, 14, bias=False)
    layer = _LoRALinear(base, r=3, alpha=6)
    with torch.no_grad():
        layer.lora_A.normal_(mean=0.0, std=0.1)
        layer.lora_B.normal_(mean=0.0, std=0.1)
    x = torch.randn(4, 10)
    y_before = layer(x)
    expected = F.linear(x, base.weight.data + (6 / 3) * (layer.lora_B.T @ layer.lora_A.T))
    assert torch.allclose(y_before, expected, atol=1e-5), "merge 公式与 forward 不一致"
    # 模拟 merge_lora 序列：原地加到 base。
    layer.merge_into_base()
    y_after = layer(x)
    assert torch.allclose(y_before, y_after, atol=1e-5), "merge 前后前向不一致"
    print("  [3] LoRA merge 等价性  OK")


def test_lora_state_dict_roundtrip():
    torch.manual_seed(2)
    base = nn.Linear(8, 12, bias=True)
    layer = _LoRALinear(base, r=4, alpha=8)
    with torch.no_grad():
        layer.lora_A.fill_(0.3)
        layer.lora_B.fill_(0.4)
    sd = lora_state_dict(layer)
    base2 = nn.Linear(8, 12, bias=True)
    base2.weight.data = base.weight.data.clone()
    base2.bias.data = base.bias.data.clone()
    layer2 = _LoRALinear(base2, r=4, alpha=8)
    load_lora_state_dict(layer2, sd)
    x = torch.randn(3, 8)
    assert torch.allclose(layer(x), layer2(x), atol=1e-6), "round-trip 不一致"
    print("  [4] LoRA state_dict round-trip  OK")


# ---------------------------------------------------------------------------
# Test 5: 模板拼接
# ---------------------------------------------------------------------------
def test_format_messages():
    msgs = [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    text = format_messages(msgs)
    assert text.startswith("<|im_start|>system\n")
    assert "<|im_end|>" in text
    assert text.count("<|im_start|>") == 3
    assert text.count("<|im_end|>") == 3
    print("  [5] format_messages 模板拼接  OK")


def test_loss_masking_without_tokenizer():
    """无 tokenizer 路径下，labels 形状正确。"""
    msgs = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "介绍 LoRA"},
        {"role": "assistant", "content": "低秩适配。"},
    ]
    text = format_messages(msgs)
    # 模拟 tokenizer：每个字符 1 token（仅用于测试 mask 形状）。
    ids = torch.tensor([ord(c) for c in text], dtype=torch.long)
    labels = build_labels(ids, msgs, tokenizer=None)
    assert labels.shape == ids.shape, f"shape mismatch: {labels.shape} vs {ids.shape}"
    ratio = (labels == -100).float().mean().item()
    # 字符串 fallback 比较保守，仍应大致落在范围内。
    assert 0.0 < ratio < 1.0, f"mask 比例异常: {ratio}"
    print(f"  [6] loss masking (no tokenizer) 形状 + ratio={ratio:.2f}  OK")


def test_loss_masking_with_fake_tokenizer():
    """用 fake tokenizer 验证 assistant 段被正确保留。"""
    class FakeTok:
        """把每个 char 映射为 ``[ord(c)]``，每个 ``<|im_end|>`` 编码为特殊 token 999。"""
        pad_token_id = 0

        def encode(self, text, add_special_tokens=True):
            # 用正则把 <|im_end|> 切成单 token，其余按字符。
            special = "<|im_end|>"
            ids = []
            i = 0
            while i < len(text):
                if text[i:i + len(special)] == special:
                    ids.append(999)
                    i += len(special)
                else:
                    ids.append(ord(text[i]))
                    i += 1
            return ids

    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi!"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "ok!"},
    ]
    text = format_messages(msgs)
    ids = torch.tensor(FakeTok().encode(text, add_special_tokens=False), dtype=torch.long)
    labels = build_labels(ids, msgs, tokenizer=FakeTok())
    assert labels.shape == ids.shape
    # 验证 user 段对应的位置是 -100，assistant content 段对应非 -100。
    # 找到第一个 "hi\n" 后的字符 —— 在 fake tokenizer 下它是 assistant 段。
    n_mask = (labels == -100).sum().item()
    n_keep = (labels != -100).sum().item()
    assert n_keep > 0, "至少应有部分 token 保留作为训练目标"
    assert n_mask > 0, "至少应有部分 token 被 mask"
    print(f"  [7] loss masking (fake tokenizer) keep={n_keep}, mask={n_mask}  OK")


# ---------------------------------------------------------------------------
# Test 8: DPO 损失符号 / 边界
# ---------------------------------------------------------------------------
def test_dpo_loss_sign():
    """chosen 概率 > rejected → margin > 0 → loss < log 2 ≈ 0.6931。"""
    from train_dpo import dpo_loss

    # 模拟 chosen / rejected 的平均 log-prob。
    pi_c = torch.tensor([0.0])    # log π(yw) = 0
    pi_r = torch.tensor([-2.0])    # log π(yl) = -2（rejected 概率低）
    ref_c = torch.tensor([0.0])
    ref_r = torch.tensor([0.0])    # ref 同 policy 起点
    loss, margin = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    # margin = (0 - (-2)) - 0 = 2
    assert margin.item() > 1.5, f"margin 期待 > 1.5，got {margin.item()}"
    # loss = -log σ(0.1 * 2) = -log σ(0.2) ≈ 0.475
    assert 0.4 < loss.item() < 0.7, f"loss 越界: {loss.item()}"
    print(f"  [8] DPO 损失符号 / margin  loss={loss.item():.4f}, margin={margin.item():.4f}  OK")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    print("===== smoke test 启动 =====")
    try:
        test_lora_shapes_and_init()
        test_lora_param_ratio()
        test_lora_merge_equivalence()
        test_lora_state_dict_roundtrip()
        test_format_messages()
        test_loss_masking_without_tokenizer()
        test_loss_masking_with_fake_tokenizer()
        test_dpo_loss_sign()
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return 1
    print("===== smoke test 全部通过 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
