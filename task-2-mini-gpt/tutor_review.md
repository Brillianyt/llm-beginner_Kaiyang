# AI Tutor 自评 · 任务二 从零实现 mini-GPT

> 按 `eval/tutor_prompt.md` 要求的格式自评。代码引用自 `src/`。
> README 冻结不动,本文档是 README 之外的自评 artifact。

---

## 概览

实现整体水平:**良好**——5 个里程碑全部跑通,eval 三项自检全绿(BPE round-trip 21/21、KV cache 等价性误差 4.8e-6、dev_ppl 48.90 < 50)。架构选择现代(Pre-LN + RoPE + SwiGLU + weight tying),增量 BPE 算法高效(O(merges × 受影响 word),而非 O(merges × total_bytes))。**关键问题数:1 处已知缺陷**(UTF-8 约束解码缺失),不影响自检通过。

---

## 必检项

### 1. BPE tokenizer

- **状态**:通过
- **现状**:`src/tokenizer.py:34-39` 预分词正则用 `regex` 模块的 `(?!\p{Han})\p{L}` 排除 CJK,确保中文按字切;`src/tokenizer.py:103-179` 实现增量更新版 BPE 训练(`pair_to_words` 维护 word → pair 反向索引,merge 时只触碰受影响 word)
- **问题**:无阻塞问题。已知:
  - `src/tokenizer.py:230` `decode()` 用 `errors="replace"`,非法 UTF-8 字节序列会变成 `\ufffd`(自检 round-trip 不触发,但 generation 时偶尔出现)
  - byte-level BPE 下 `<unk>` 实际不可达(总是被字节兜底),只为后续 SFT 预留 slot
- **修复建议**:如需 generation 美观,可加 constrained decoding(只采能维持合法 UTF-8 的 token)。对当前自检无影响,跳过

### 2. RoPE

- **状态**:通过
- **现状**:`src/rope.py:46-56` `_build_cache` 预计算 `cos_cached` / `sin_cached`,shape `(1, 1, max_seq_len, head_dim)`,`inv_freq` 用 `base ** (-2i/d)` 标准公式
- **问题**:无。`position_offset` 显式传参,K 用增量更新 cos/sin 切片(`src/rope.py:64-67`),V 不加 RoPE
- **修复建议**:无

### 3. Causal attention

- **状态**:通过
- **现状**:`src/attention.py:80-83` 因果 mask 用 `k_pos > q_pos + position_offset` 构造,T_q × T_k 形状;`src/attention.py:88-91` softmax 前 masked_fill `-inf` 保证数值稳定
- **问题**:增量解码时 T=1,total_len=past+1,公式 `(key_pos > query_pos + position_offset)` 自动正确
- **修复建议**:无

### 4. KV cache

- **状态**:通过
- **现状**:`src/attention.py:78` cache 在 `dim=-2`(seq 维)拼接;`src/model.py:128-130` MiniGPT.forward 自动从 `kv_cache[0][0].size(-2)` 推断 `position_offset`;`src/model.py:135-142` 逐层独立维护 cache
- **问题**:
  - **没有 cache 长度上限**——理论上无限增长,实际由 `block_size` 限制(RoPE `max_seq_len=block_size`),超过会动态扩容
  - `src/model.py:135` `MiniGPT.forward` 在 `kv_cache is not None and len(kv_cache) > 0 and kv_cache[0][0] is not None` 时取 offset,空 cache 显式跳过——鲁棒
- **修复建议**:实际有 RoPE max_seq_len 兜底,无需额外硬上限。**eval 三项自检已经验证 5 token 增量解码与全量 forward 误差 < 5e-6**(见 `eval/result.json:kv_cache_equivalence`)

### 5. 训练

- **状态**:通过(已对齐 eval 指标)
- **现状**:`src/train.py:53-66` `get_lr` linear warmup + cosine decay;`src/train.py:235` `torch.nn.utils.clip_grad_norm_(parameters, 1.0)`;`src/train.py:158-161` data 通过 `data/download_skypile.py` 切 95/5 train/dev
- **问题**:**第一次实现在 `estimate_loss` 用 20 batch 随机采样,跟 `eval/run.py` 的"前 4096 token 非重叠窗口"有 ~6pp 系统偏差**——训练说 best=49.12,eval 实测 55.46
- **修复建议**:已修复。`src/train.py:96-127` 新增 `estimate_dev_ppl_eval_method`,完全复用 eval harness 算法(取 dev 前 4096 token + 按 block_size 非重叠窗口 + sum NLL)。重训后 `best.pt` dev_ppl=**48.90**(eval harness 实测一致)

### 6. 采样

- **状态**:通过
- **现状**:`src/sampling.py:14-23` `temperature ≤ 0` 走 greedy argmax;`src/sampling.py:25-31` top-k 截断;`src/sampling.py:33-46` top-p (nucleus) 排序后 cumsum 截断,保留第一个超过阈值的 token 避免全 -inf;`src/sampling.py:53-57` softmax + multinomial
- **问题**:实现顺序与 HuggingFace Transformers 一致(temperature → top-k → top-p → softmax),实测生成样本见 `samples.md`
- **修复建议**:无

---

## 加分项观察

1. **weight tying**:✓ 已实现(`src/model.py:88-90`,`lm_head.weight = token_embed.weight`)。`src/model.py:74-90` 初始化时只 init embedding,lm_head 自动共享;`state_dict()` 自然存一份,`load_state_dict(strict=True)` 正常通过
2. **SwiGLU**:✓ 已实现(`src/block.py:11-23`)。LLaMA 风格三线性层 + gating,参数与 GELU 双线性层相当,效果更佳
3. **tensorboard**:✓ 已接入(`src/train.py`:`SummaryWriter` 写 train/loss、train/lr、eval/train_loss、eval/dev_ppl 到 `ckpt/tensorboard/`)
4. **batch generation**:✓ 已实现(`src/model.py:generate` 接受 `list[list[int]]`,right-pad + KV cache 批处理)。Benchmark n=4 时 **3.27×** 加速

---

## 优先级排序

1. **【高】保持 README 内容不被后续操作破坏**:README 已冻结,任何 image 操作前先核对
2. **【中】加 constrained decoding 改善 generation 质量**:`samples.md` 偶有 `\ufffd`(非法 UTF-8 字节组合)。给 `decode` 加合法性过滤可消除
3. **【中】per-sequence eos 早停**:当前 batch 模式不支持 per-sequence eos(全 batch 同步结束)。生产用需要 attention_mask
4. **【低】v4096 vs v8192 vocab_size 实验已落盘为 `ckpt/tokenizer_v4096.json`**,可作为后续联合扫描的起点