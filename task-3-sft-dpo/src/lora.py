"""手写 LoRA：低秩适配注入原始 ``nn.Linear`` 层。

本模块不依赖 ``peft``，完全用 PyTorch 原生算子实现 LoRA（[paper](https://arxiv.org/abs/2106.09685)）。

核心做法是把一个 ``nn.Linear`` 替换成一个轻量容器：

* 原权重 ``W`` 形状 ``(out_features, in_features)`` 冻结；
* 旁挂两个低秩矩阵 ``A: (in_features, r)``、``B: (r, out_features)``；
* forward 改为 ``y = W x + scaling * B (A x)``，其中 ``scaling = alpha / r``；
* 合并阶段把 ``scaling * B @ A`` 直接加回 ``W``，为推理节省一次 matmul。

设计要点
--------

1. ``A`` 用 kaiming 初始化（与原始 ``nn.Linear`` 默认行为一致），
   ``B`` 用零初始化，保证训练步 0 时低秩分支对前向为 0；
2. ``W`` 与 ``bias`` 上的梯度永远关闭，可训练参数占比远小于 5%；
3. 用 ``_LoRALinear`` 包裹原 ``Linear``，保留 ``out_features`` / ``in_features``
   之类的属性，外部（HF 模型）无须感知差别。

合约
----
- :func:`inject_lora` 在模型上递归查找名称以 ``target_modules`` 中任一元素结尾
  的 ``nn.Linear``，原地替换为 ``_LoRALinear``；
- :func:`merge_lora` 把所有 ``_LoRALinear`` 的低秩增量合并回主权重，并清理
  ``LoRA`` 命名空间。
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 核心模块：单层 LoRA 适配
# ---------------------------------------------------------------------------
class _LoRALinear(nn.Module):
    """把 ``nn.Linear`` 改造成「主路径 + 低秩旁路」的复合模块。

    Attributes:
        base: 原始 ``nn.Linear``，参数冻结。
        lora_A: ``(in_features, r)``，kaiming 初始化。
        lora_B: ``(r, out_features)``，零初始化。
        scaling: ``alpha / r``，写入模型状态时使用。
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: int,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank 必须为正整数，got r={r}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha 必须为正整数，got alpha={alpha}")

        # 1. 主路径：冻结。
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features

        # 2. 低秩旁路。
        # A 形状 (in, r)，B 形状 (r, out)，
        # 这样 B(A x) 等价于 (B @ A) x，前向仅多一次 in→r 和 r→out 的矩阵乘。
        self.lora_A = nn.Parameter(torch.empty(in_features, r))
        self.lora_B = nn.Parameter(torch.empty(r, out_features))
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False  # 标记：是否已 merge 到 base

        # 3. 初始化：A 用 kaiming_uniform（与 nn.Linear 默认一致），
        # B 用 0 —— 保证训练初始时 B(A x) 为 0，原模型输出完全保留。
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``y = W x + scaling * B (A x)``。

        Args:
            x: 任意形状 ``(*, in_features)`` 的输入。
        Returns:
            形状 ``(*, out_features)`` 的输出。

        Notes:
            - 主路径是不带梯度的，保证冻结权重不参与反向。
            - ``A: (in, r)``、``B: (r, out)``，我们把 ``A`` / ``B`` 看作
              "外部矩阵"，直接 ``x @ A`` 得到 ``(..., r)``，再 ``(中间) @ B``
              得到 ``(..., out)``。
            - merge 之后 ``merged=True``，跳过低秩旁路，避免双计数。
        """
        # 主路径。
        base_out = self.base(x)
        if self.merged:
            return base_out
        # 低秩增量。
        h = torch.matmul(x, self.lora_A)          # (..., in) @ (in, r) → (..., r)
        lora_out = torch.matmul(h, self.lora_B)   # (..., r)  @ (r, out) → (..., out)
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.base.in_features}, out={self.base.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"
        )

    def merge_into_base(self) -> None:
        """把 ``scaling * B @ A`` 加回主权重 ``W``，并禁用 LoRA 旁路。

        推理阶段调用：合并后前向输出与未合并完全一致，但少两次 matmul。

        公式推导：
            y = W x + scaling * B (A x) = (W + scaling * B A) x
            其中 ``A: (in, r)``、``B: (r, out)``、``W: (out, in)``。
            ``B A`` 形状不直接相乘，但写成 ``W += scaling * B.T @ A.T``
            后形状为 ``(out, r) @ (r, in) = (out, in)``，与 ``W`` 同形。

        实现要点：合并后置 ``merged=True``，forward 自动跳过低秩旁路，
        避免再次叠加 delta。
        """
        with torch.no_grad():
            delta_w = self.scaling * (self.lora_B.T @ self.lora_A.T)
            self.base.weight.data.add_(delta_w)
        # 合并后冻结 LoRA 旁路参数，避免后续被误更新。
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
        self.merged = True


# ---------------------------------------------------------------------------
# 注入 / 合并 入口
# ---------------------------------------------------------------------------
def _match_target(name: str, target_modules: Sequence[str]) -> bool:
    """模块名是否以 ``target_modules`` 中任一元素结尾（典型 HF 命名约定）。

    例如 ``target_modules=["q_proj","v_proj"]`` 匹配
    ``model.layers.0.self_attn.q_proj``；``target_modules=["query","value"]`` 匹配
    ``bert.encoder.layer.0.attention.self.query``。
    """
    return any(name.endswith(t) for t in target_modules)


def _replace_module(root: nn.Module, target: str, new: nn.Module) -> None:
    """把 ``root`` 子模块里名为 ``target`` 的子节点替换为 ``new``。"""
    parts = target.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int = 8,
    alpha: int = 16,
) -> nn.Module:
    """在 ``model`` 的目标线性层上注入 LoRA 适配。

    遍历 ``model.named_modules()``，对模块名以 ``target_modules`` 中任一元素结尾
    的 ``nn.Linear`` 替换为 :class:`_LoRALinear`，**不修改其它层**。

    Args:
        model: 任意 ``nn.Module``（典型为 ``AutoModelForCausalLM``）。
        target_modules: 目标线性层名后缀列表（HF 命名）。
        r: 低秩。
        alpha: 缩放系数（``scaling = alpha / r``）。
    Returns:
        原 ``model``（原地修改），方便链式调用。
    """
    target_modules = list(target_modules)
    if not target_modules:
        raise ValueError("target_modules 不能为空")

    # 先收集再替换，避免在迭代 named_modules() 时修改模型结构。
    to_inject: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _match_target(name, target_modules):
            # 已经注入过的 LoRA 层跳过（允许重复调用安全）。
            if not isinstance(module, _LoRALinear):
                to_inject.append(name)

    for name in to_inject:
        old = _get_module(model, name)
        wrapped = _LoRALinear(old, r=r, alpha=alpha)
        _replace_module(model, name, wrapped)

    # 主动把 base 之外的所有参数冻结（保险起见）。
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] 注入 {len(to_inject)} 层，可训练参数 {n_train}/{n_total} "
          f"({n_train / max(n_total, 1):.4%})")
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """把模型内所有 :class:`_LoRALinear` 的低秩增量合并回主权重。

    合并后不再额外存在 LoRA 旁路，base 权重 ``W`` 已经吸收了
    ``scaling * B @ A``。可用于推理 / 部署 / 量化时一步到位。

    Returns:
        原 ``model``（原地修改）。
    """
    n_merged = 0
    for module in model.modules():
        if isinstance(module, _LoRALinear):
            module.merge_into_base()
            n_merged += 1
    print(f"[LoRA] 合并 {n_merged} 层 LoRA 权重到 base")
    return model


def _get_module(root: nn.Module, target: str) -> nn.Module:
    """按点分路径取子模块。"""
    parts = target.split(".")
    m = root
    for p in parts:
        m = getattr(m, p)
    return m


# ---------------------------------------------------------------------------
# 工具：列出当前 LoRA 状态
# ---------------------------------------------------------------------------
def lora_state_dict(model: nn.Module) -> dict:
    """收集所有 LoRA 旁路参数 + 顶层配置，作为轻量 checkpoint 使用。"""
    state: dict = {}
    for name, module in model.named_modules():
        if isinstance(module, _LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            state[f"{name}.scaling"] = torch.tensor(module.scaling)
    return state


def load_lora_state_dict(model: nn.Module, state: dict) -> nn.Module:
    """把 :func:`lora_state_dict` 产生的字典加载回模型。

    顶层模块（名称为空字符串）的命名空间也支持。
    """
    by_prefix: dict = {}
    for key, value in state.items():
        # 去掉尾部 ``.lora_A`` / ``.lora_B`` / ``.scaling``，剩下的就是模块名。
        for tail in ("lora_A", "lora_B", "scaling"):
            suffix = "." + tail
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                by_prefix.setdefault(prefix, {})[tail] = value
                break

    for name, module in model.named_modules():
        if not isinstance(module, _LoRALinear):
            continue
        slot = by_prefix.get(name)
        if slot is None:
            continue
        if "lora_A" in slot:
            module.lora_A.data.copy_(slot["lora_A"].to(module.lora_A.device))
        if "lora_B" in slot:
            module.lora_B.data.copy_(slot["lora_B"].to(module.lora_B.device))
    return model
