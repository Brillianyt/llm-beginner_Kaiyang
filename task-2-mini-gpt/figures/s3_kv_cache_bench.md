# S3: KV cache 推理速度对比

模型:`best.pt`(block_size=256, vocab=8192)  
设备:CPU  
生成方式:greedy(temperature=0),每组取 2 次中位数  

## 结果

| prompt_len | max_new | cache (ms) | no-cache (ms) | speedup | cache tok/s | no-cache tok/s |
|---|---|---|---|---|---|---|
| 128 | 32 | 253.3 | 771.1 | **3.04x** | 126.4 | 41.5 |
| 128 | 64 | 513.8 | 1560.1 | **3.04x** | 124.6 | 41.0 |
| 128 | 128 | 987.4 | 3500.8 | **3.55x** | 129.6 | 36.6 |
| 256 | 32 | 276.7 | 1149.8 | **4.16x** | 115.7 | 27.8 |
| 256 | 64 | 536.8 | 2474.6 | **4.61x** | 119.2 | 25.9 |
| 256 | 128 | 1034.5 | 5574.2 | **5.39x** | 123.7 | 23.0 |
| 512 | 32 | 312.4 | 1916.9 | **6.14x** | 102.4 | 16.7 |
| 512 | 64 | 564.3 | 4039.9 | **7.16x** | 113.4 | 15.8 |
| 512 | 128 | 1090.9 | 8437.2 | **7.73x** | 117.3 | 15.2 |

## 解读

- KV cache 加速比随 **prompt + max_new** 总长度线性增长(理论上 O(T²) → O(T))
- 朴素路径每步重算整个 (prompt + 已生成),复杂度 O(T²);cache 路径每步 O(1)
- 实测在 prompt=512, max_new=128 时,加速比 ≈ 7.7x

## 实现要点

- `MiniGPT.forward(ids, kv_cache=None, return_cache=False)`:接受可选 cache
- `MiniGPT.generate(..., use_kv_cache=True)`:新增参数,True 走增量、False 走朴素
- cache 拼接在 K/V 的 seq 维(dim=-2)
- 增量解码时新 token 的 position_offset 自动从 cache 长度推断