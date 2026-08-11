"""M2/M3/M4: MiniGPT 主体 + load_for_eval 工厂函数 + generate。

设计要点:
- Pre-LN decoder-only,集成 RoPE 与 KV cache
- weight tying(token_embed ↔ lm_head)默认开启
- block_size 必须作为属性暴露,供 eval/run.py 困惑度分块使用
- load_for_eval(ckpt_path) 是 eval 入口,返回 (model, tokenizer)
- forward 接受 kv_cache(per-layer list),position_offset 从 cache 自动推断
- generate 支持 KV cache 自回归 + 4 种采样(由 sampling.py 提供)
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalMultiHeadAttention  # noqa: F401 (供外部 import)
from .block import TransformerBlock, SwiGLU
from .config import MiniGPTConfig
from .rope import RotaryPositionalEmbedding, SinusoidalPositionalEncoding
from .sampling import sample
from .tokenizer import BPETokenizer


class MiniGPT(nn.Module):
    """decoder-only mini-GPT,带 RoPE 与 KV cache。"""

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        # 暴露 block_size,供 eval/run.py 困惑度分块
        self.block_size = config.block_size

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # 位置编码:RoPE(默认)或 Sinusoidal(供 S2 对比)
        if config.pos_encoding == "rope":
            head_dim = config.d_model // config.n_heads
            self.rope = RotaryPositionalEmbedding(
                head_dim=head_dim,
                max_seq_len=config.block_size,
                base=config.rope_base,
            )
            self.pos_embed: nn.Module | None = None
        elif config.pos_encoding == "absolute":
            self.rope = None
            self.pos_embed = SinusoidalPositionalEncoding(
                d_model=config.d_model,
                max_len=config.block_size,
                dropout=0.0,  # 已经在前面用了 self.dropout
            )
        else:
            raise ValueError(f"unknown pos_encoding: {config.pos_encoding!r}")

        # N 层 TransformerBlock
        # 注:absolute PE 时,TransformerBlock 不再使用 rope=None
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    rope=self.rope,  # absolute 模式下 self.rope=None,block 内部会跳过 RoPE
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.weight_tying:
            # 共享权重:lm_head.weight 是 token_embed.weight 的别名
            # state_dict() 只保存一份,load_state_dict(strict=True) 也只 load 一份
            self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2 风格初始化:xavier_uniform for Linear,normal(0.02) for Embedding。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        ids: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """ids: (B, T) LongTensor

        kv_cache: None 或长度为 n_layers 的 list,每个元素是 (K_i, V_i),各 (B, H, T_past, head_dim)
        return_cache: True 时返回 (logits, new_cache)

        返回:logits (B, T, vocab_size);若 return_cache=True 则附带 new_cache
        """
        B, T = ids.shape
        # 从 cache 自动推断 position_offset;非 cache 路径默认 0
        if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0][0] is not None:
            position_offset = kv_cache[0][0].size(-2)
        else:
            position_offset = 0

        x = self.token_embed(ids)
        # absolute PE:加到 embedding;RoPE 模式:不加(在 attention 内部加到 Q/K)
        if self.pos_embed is not None:
            x = self.pos_embed(x)
        x = self.dropout(x)

        new_cache: list | None = [] if return_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache[i]
            if return_cache:
                x, layer_new_cache = block(
                    x,
                    kv_cache=layer_cache,
                    position_offset=position_offset,
                    return_cache=True,
                )
                new_cache.append(layer_new_cache)
            else:
                x = block(
                    x,
                    kv_cache=layer_cache,
                    position_offset=position_offset,
                    return_cache=False,
                )

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if return_cache:
            return logits, new_cache
        return logits

    # ------------------------------------------------------------------
    # generate(M5)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids,
        max_new_tokens: int,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        eos_id: int | None = None,
        device: str | None = None,
        use_kv_cache: bool = True,
    ):
        """自回归生成,可用 KV cache 加速(use_kv_cache=True)或朴素全量前向(False)。

        Args:
            prompt_ids: list[int](单条)或 list[list[int]](batch)。list[int] 自动包成 batch=1。
            max_new_tokens: 最多生成多少 token
            top_k / top_p / temperature: 传给 sampling.sample
            eos_id: 早停 token id(单条生效,batch 模式下未实现)
            device: 推理设备,None 时跟随模型当前 device
            use_kv_cache: True 走增量 KV cache;False 走朴素全量 forward

        Returns:
            - 单条输入 → 1D list[int](prompt + 生成)
            - 批量输入 → list[list[int]],每个元素是 (原 prompt + 生成),已 pad→trim 回去

        batch 模式说明:
            - 不同长度 prompt 自动 right-pad 到 batch 内最长
            - 用 PAD_ID(默认 0)填充
            - 因 causal mask + pad attention,生成质量略有损失(可接受)
            - 不支持 per-sequence eos 早停(需要 attention_mask 改造)
        """
        # 兼容单条输入
        single_input = False
        if prompt_ids and isinstance(prompt_ids[0], int):
            prompt_ids = [list(prompt_ids)]
            single_input = True

        if device is None:
            device = next(self.parameters()).device
        if not prompt_ids:
            return [] if isinstance(prompt_ids, list) and (not prompt_ids or not isinstance(prompt_ids[0], list)) else [[]]

        PAD = 0  # 默认 pad token;若模型用了不同 PAD_ID 可在此调整
        batch_size = len(prompt_ids)
        max_len = max(len(p) for p in prompt_ids)
        # right-pad
        padded = [p + [PAD] * (max_len - len(p)) for p in prompt_ids]
        orig_lens = [len(p) for p in prompt_ids]

        if not use_kv_cache:
            # 朴素路径(每步全量 forward):用于 S3 benchmark 等对照实验
            cur_seqs = [list(p) for p in prompt_ids]
            all_done = [False] * batch_size
            for _ in range(max_new_tokens):
                # 把当前所有序列拼到 batch 内最长
                cur_max = max(len(s) for s in cur_seqs)
                cur_padded = [s + [PAD] * (cur_max - len(s)) for s in cur_seqs]
                chunk = torch.tensor(cur_padded, dtype=torch.long, device=device)
                logits = self.forward(chunk, kv_cache=None, return_cache=False)
                next_logits = torch.stack(
                    [logits[i, len(cur_seqs[i]) - 1, :] for i in range(batch_size)], dim=0
                )
                next_ids = sample(next_logits, top_k=top_k, top_p=top_p, temperature=temperature)
                for i in range(batch_size):
                    if all_done[i]:
                        continue
                    nid = int(next_ids[i].item())
                    cur_seqs[i].append(nid)
                    if eos_id is not None and nid == eos_id:
                        all_done[i] = True
                if all(all_done):
                    break
            return cur_seqs

        # 1. 一次性 forward 所有 prompt,得到 cache 和每条最后位置的 logits
        ids = torch.tensor(padded, dtype=torch.long, device=device)
        logits, cache = self.forward(ids, kv_cache=None, return_cache=True)
        # 每条取 orig_len[i] - 1 位置的 logits
        next_logits = torch.stack(
            [logits[i, orig_lens[i] - 1, :] for i in range(batch_size)], dim=0
        )  # (B, V)
        next_ids = sample(next_logits, top_k=top_k, top_p=top_p, temperature=temperature)
        generated = [[int(next_ids[i].item())] for i in range(batch_size)]

        # 2. 增量循环:每步喂 batch 个 token
        cur = next_ids.unsqueeze(1)  # (B, 1)
        for _ in range(max_new_tokens - 1):
            logits, cache = self.forward(cur, kv_cache=cache, return_cache=True)
            next_logits = logits[:, -1, :]   # (B, V)
            next_ids = sample(next_logits, top_k=top_k, top_p=top_p, temperature=temperature)
            for i in range(batch_size):
                generated[i].append(int(next_ids[i].item()))
            cur = next_ids.unsqueeze(1)

        # 拼回完整序列(prompt + 生成),prompt 部分保持原长
        full = []
        for i in range(batch_size):
            full.append(list(prompt_ids[i]) + generated[i])
        return full[0] if single_input else full

    # ------------------------------------------------------------------
    # 加载 / 保存
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> "MiniGPT":
        """纯模型加载(不含 tokenizer),用于训练 / 内部。

        注意:SinusoidalPE 是确定函数(从 d_model 重建),加载时会忽略 ckpt 中
        pos_embed.pe 的形状(训练中可能因 eval 触发扩展),并按当前 cfg.block_size
        重建。这样 ckpt 里若存的是扩展过的 (1, 65, d) 形状,加载到 (1, 256, d)
        形状的模型上不会 shape mismatch。
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = MiniGPTConfig.from_dict(ckpt["config"])
        model = cls(cfg)
        sd = ckpt["model_state_dict"]
        msd = model.state_dict()
        for k in msd:
            if k in sd and sd[k].shape == msd[k].shape:
                msd[k].copy_(sd[k])
        # strict=False:跳过 shape 不匹配的(主要是 SinusoidalPE 的 pe buffer)
        model.load_state_dict(msd, strict=False)
        model.eval()
        return model

    def save_checkpoint(
        self,
        ckpt_path: str,
        tokenizer_path: str,
        extra: dict | None = None,
    ) -> None:
        """保存 {model_state_dict, config, tokenizer_meta};供 train.py 调用。"""
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "tokenizer_meta": {"path": tokenizer_path},
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, ckpt_path)


def load_for_eval(ckpt_path: str) -> tuple["MiniGPT", "BPETokenizer"]:
    """eval/run.py 的唯一入口:从 ckpt 重建 (model, tokenizer)。

    要求 ckpt_path 所在目录下同时存在 tokenizer.json。
    """
    ckpt_path = Path(ckpt_path)
    tokenizer_path = ckpt_path.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"load_for_eval 需要 {tokenizer_path},但文件不存在"
        )
    # 默认 CPU,推理端可视情况 to('cuda')
    model = MiniGPT.from_checkpoint(str(ckpt_path), device="cpu")
    tok = BPETokenizer.from_pretrained(str(tokenizer_path))
    return model, tok