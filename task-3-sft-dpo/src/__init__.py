"""任务三：SFT + DPO 两阶段对齐。

子模块：
- :mod:`lora` 手写 LoRA 注入 + merge；
- :mod:`chat` Qwen chat template + loss masking；
- :mod:`data_utils` MOSS / DPO 数据加载；
- :mod:`model_utils` 模型加载与 LoRA 注入的统一入口；
- :mod:`compare` base / SFT / DPO 三方对比生成。
"""
from .lora import inject_lora, merge_lora, lora_state_dict, load_lora_state_dict
from .chat import format_messages, build_labels, encode_chat
from .compare import compare

__all__ = [
    "inject_lora",
    "merge_lora",
    "lora_state_dict",
    "load_lora_state_dict",
    "format_messages",
    "build_labels",
    "encode_chat",
    "compare",
]
