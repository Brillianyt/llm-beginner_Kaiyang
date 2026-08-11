"""配置中心:MiniGPTConfig + TrainConfig。

设计原则:
- dataclass 字段顺序与默认值与训练脚本一致,任何修改都需要同步 ckpt 格式
- to_dict / from_dict 保证 ckpt 重建不会因默认值漂移而错装架构
"""
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class MiniGPTConfig:
    """模型架构配置。序列化为 ckpt['config']。"""
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024           # SwiGLU 中间维度
    block_size: int = 256      # 训练时的最大上下文长度;eval 必读,用于困惑度分块
    dropout: float = 0.1
    rope_base: float = 10000.0
    weight_tying: bool = True  # 共享 token_embed 与 lm_head 的权重
    pos_encoding: str = "rope"  # "rope"(默认)或 "absolute"(sinusoidal,仅供 S2 对比)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MiniGPTConfig":
        # 过滤掉 ckpt 里多余的键,容错性更好
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class TrainConfig:
    """训练超参数配置。device='auto' 时由 train.py 解析为 cuda/cpu。"""
    data_dir: Path
    ckpt_dir: Path
    tokenizer_path: Path
    model_config: MiniGPTConfig

    # 优化
    batch_size: int = 32
    max_iters: int = 3000
    eval_interval: int = 200
    eval_iters: int = 20         # 每次评估采多少 batch 算平均 loss
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"         # "auto" | "cuda" | "cpu"
    log_interval: int = 50