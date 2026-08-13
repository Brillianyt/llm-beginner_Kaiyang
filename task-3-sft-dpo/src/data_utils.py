"""数据加载工具：MOSS-003-sft / moss-003-sft-plugin / DPO 偏好数据。

本模块关键点：

1. **路径兼容**：MOSS jsonl 可能为压缩包（``.jsonl.zip``），需先解压；
2. **容错**：环境无数据 / 下载失败时，提供 smoke 模式让脚本仍能跑；
3. **统一接口**：所有数据集最终都 yield ``list[dict]``（messages 风格）或
   ``dict(prompt, chosen, rejected)``（DPO 风格）。

数据字段约定：

- MOSS SFT 一行形如 ``{"conversation": [{"role":..., "content":...}, ...]}``；
- plugin 数据行结构相同，但 assistant content 中包含工具调用 JSON。
- DPO 数据行形如 ``{"chosen": [...msgs], "rejected": [...msgs]}``（OpenAssistant /
  Anthropic 风格），有些数据集会拼成 ``prompt + response`` 字符串，需归一化。
"""
from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


# ---------------------------------------------------------------------------
# 路径与常用文件
# ---------------------------------------------------------------------------
def find_first_existing(paths: Sequence[Path]) -> Path | None:
    """返回 ``paths`` 中第一个存在的文件路径，否则 ``None``。"""
    for p in paths:
        if p.exists():
            return p
    return None


def load_jsonl(path: Path) -> Iterator[dict]:
    """逐行 yield jsonl 内容，空行自动跳过。"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_jsonl_zip(zip_path: Path, member: str | None = None) -> Iterator[dict]:
    """从 ``.jsonl.zip`` 中读取第一份（或指定 ``member``）jsonl。"""
    with zipfile.ZipFile(zip_path) as zf:
        name = member
        if name is None:
            # 取第一个 ``.jsonl`` 成员。
            candidates = [n for n in zf.namelist() if n.endswith(".jsonl")]
            if not candidates:
                raise FileNotFoundError(f"{zip_path} 内未找到 .jsonl 成员")
            name = candidates[0]
        with zf.open(name) as f:
            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# MOSS SFT 加载
# ---------------------------------------------------------------------------
def load_moss_sft(
    data_dir: Path,
    split: str = "no-tools",
    max_samples: int | None = None,
) -> List[dict]:
    """加载 MOSS-003-sft 数据为 ``[{"messages": [...]}, ...]``。

    Args:
        data_dir: 通常为 ``data/moss-sft`` 或 ``data/moss-sft-plugin``。
        split: ``"no-tools"`` / ``"plugin"`` / ``"with-tools"`` 之一，用于选文件。
        max_samples: 截断样本数，``None`` 表示全量。
    """
    candidates: list[Path] = []
    if split == "no-tools":
        candidates = [
            data_dir / "moss-003-sft-no-tools.jsonl",
            data_dir / "moss-003-sft-no-tools.jsonl.zip",
        ]
    elif split in ("plugin", "with-tools"):
        candidates = [
            data_dir / "moss-003-sft-with-tools-no-text2image.jsonl",
            data_dir / "moss-003-sft-with-tools-no-text2image.jsonl.zip",
            data_dir / "moss-003-sft-with-tools-text2image.jsonl",
            data_dir / "moss-003-sft-with-tools-text2image.jsonl.zip",
        ]
    else:
        candidates = [data_dir / f"{split}.jsonl", data_dir / f"{split}.jsonl.zip"]

    found = find_first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            f"未在 {data_dir} 找到 MOSS {split} 数据，请运行 data/download.py"
        )

    if found.suffix == ".zip":
        stream = load_jsonl_zip(found)
    else:
        stream = load_jsonl(found)

    samples: list[dict] = []
    for row in stream:
        msgs = _row_to_messages(row)
        if msgs is None:
            continue
        samples.append({"messages": msgs})
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def _row_to_messages(row: dict) -> list[dict] | None:
    """把 MOSS 行统一为 ``[{"role":..., "content":...}, ...]``。"""
    if "conversation" in row:
        conv = row["conversation"]
        return [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in conv
        ]
    if "messages" in row:
        return list(row["messages"])
    if "prompt" in row and "response" in row:
        # 单轮 SFT。
        return [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    return None


# ---------------------------------------------------------------------------
# DPO 偏好数据
# ---------------------------------------------------------------------------
def load_dpo(
    data_dir: Path,
    max_samples: int | None = None,
    seed: int = 0,
) -> List[dict]:
    """加载 DPO 偏好数据为 ``[{"prompt": ..., "chosen": ..., "rejected": ...}, ...]``。

    支持两种来源：

    1. ``data/dpo/*.jsonl``，每行 ``{"prompt": ..., "chosen": ..., "rejected": ...}``；
    2. ``data/dpo/<dataset>/*.jsonl``（如 hiyouga 的 DPO-En-Zh-20k）。

    任意条目字段缺失则跳过。
    """
    dpo_dir = data_dir / "dpo"
    if not dpo_dir.exists():
        raise FileNotFoundError(f"DPO 目录不存在: {dpo_dir}")

    files: list[Path] = []
    for p in dpo_dir.rglob("*.jsonl"):
        files.append(p)
    if not files:
        raise FileNotFoundError(f"{dpo_dir} 内没有 .jsonl 文件")

    samples: list[dict] = []
    for f in files:
        for row in load_jsonl(f):
            item = _row_to_dpo(row)
            if item is not None:
                samples.append(item)
            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break

    random.Random(seed).shuffle(samples)
    return samples


def _row_to_dpo(row: dict) -> dict | None:
    """把不同源的 DPO 行归一化为 ``prompt / chosen / rejected``。"""
    if "prompt" in row and "chosen" in row and "rejected" in row:
        # 最常见格式（anthropic/hh-rlhf 等）。
        prompt, chosen, rejected = row["prompt"], row["chosen"], row["rejected"]
    elif "question" in row and "chosen" in row and "rejected" in row:
        prompt = row["question"]
        chosen, rejected = row["chosen"], row["rejected"]
    elif "input" in row and "chosen" in row and "rejected" in row:
        prompt = row["input"]
        chosen, rejected = row["chosen"], row["rejected"]
    else:
        return None

    # 归一化：chosen / rejected 若为 list-of-dicts，序列化为对话。
    if isinstance(chosen, list):
        chosen = _flatten_msgs(chosen)
    if isinstance(rejected, list):
        rejected = _flatten_msgs(rejected)
    if isinstance(prompt, list):
        prompt = _flatten_msgs(prompt)
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _flatten_msgs(msgs: list[dict]) -> str:
    """把 ``[{"role":..., "content":...}, ...]`` 拼成单字符串，便于大多数 DPO 模板。"""
    return "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs
    )


# ---------------------------------------------------------------------------
# 兜底：离线 smoke 模式
# ---------------------------------------------------------------------------
SMOKE_SFT_SAMPLES: list[dict] = [
    {"messages": [
        {"role": "user", "content": "你好，请介绍一下自己。"},
        {"role": "assistant", "content": "你好！我是一个中文对话助手，"
                                       "很高兴见到你。"},
    ]},
    {"messages": [
        {"role": "user", "content": "请用一句话解释 LoRA。"},
        {"role": "assistant", "content": "LoRA 通过在原始权重上叠加低秩矩阵 "
                                       "实现参数高效微调。"},
    ]},
    {"messages": [
        {"role": "user", "content": "深度学习里 batch size 过大会有什么影响？"},
        {"role": "assistant", "content": "收敛变慢、显存占用上升，但梯度 "
                                       "更稳定；通常要在 batch size 与 "
                                       "学习率之间做权衡。"},
    ]},
]


def load_sft_smoke() -> List[dict]:
    """无数据时返回少量内置样本，用于 pipeline 验证。"""
    return list(SMOKE_SFT_SAMPLES)


SMOKE_DPO_SAMPLES: list[dict] = [
    {"prompt": "什么是 LoRA？",
     "chosen": "LoRA 通过在原始权重上叠加低秩矩阵实现参数高效微调。",
     "rejected": "LoRA 是一种新型的编程语言。"},
    {"prompt": "解释一下 DPO 的损失函数。",
     "chosen": "DPO 损失直接最大化 chosen 和 rejected 之间的对数概率差。",
     "rejected": "DPO 用 Q-learning 训练。"},
]


def load_dpo_smoke() -> List[dict]:
    """无 DPO 数据时返回少量内置样本。"""
    return list(SMOKE_DPO_SAMPLES)
