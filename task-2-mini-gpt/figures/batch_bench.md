# Batch Generation 速度对比

模型:`best.pt`(block_size=256)
对比:N× 串行 batch=1 vs 一次 batch=N(后者用 KV cache 在 batch 维独立)

| n | 串行 (ms) | batch (ms) | speedup |
|---|---|---|---|
| 1 | — | 449.7 | inf× |
| 2 | 883.6 | 469.7 | 1.88× |
| 4 | 1795.9 | 549.0 | 3.27× |
| 8 | 1828.5 | 566.3 | 3.23× |

## 实现要点

- `MiniGPT.generate(prompts, ...)` 接受 list[int] 或 list[list[int]]
- 不同长度 prompt 自动 right-pad 到 batch 内最长(用 PAD_ID=0)
- KV cache 的 batch 维天然独立,无需修改 attention
- 返回值:单条 → list[int];批量 → list[list[int]]