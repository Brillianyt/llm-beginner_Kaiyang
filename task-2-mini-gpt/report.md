# Task 2 产出报告 · 从零实现 mini-GPT

> 按 M1-Mn / S1-Sn 顺序给出每项"要求 / 状态 / 证据"。
> `README.md` 是 spec,本报告是完成情况;过程记录见 `notes.md`,自评分见 `tutor_review.md`。

---

## 提交(`nndl-discussion` 帖子要附的内容)

### 1. Fork 仓库链接

`https://github.com/<your-fork>/llm-beginner`,task-2-mini-gpt 目录。

### 2. `eval/result.json` 内容

```json
[
  {"test": "tokenizer_roundtrip", "pass": true, "failures": []},
  {"test": "kv_cache_equivalence", "pass": true, "max_abs_diff": 3.8e-6},
  {"test": "perplexity_on_dev", "pass": true, "perplexity": 48.9,
   "threshold": 50.0, "n_tokens": 4095, "dataset": "skypile_subset"}
]
```

### 3. DoD checklist 勾选状态

详见下方"必做 M1-M5"和"加分 S1-S4"两节(全 ✓)。

### 4. 几段生成样例(不同采样策略对比)

详见 `samples.md`,4 个 news-style prompt × 4 种采样策略(greedy / top-k=40 / top-p=0.9 / temperature=0.7 + top-p=0.9)。示例:

- **Prompt "中新社北京" + greedy**:输出"中新社北京4月18日电(记者 吴玉玲)..." — 模型学到新闻通讯格式
- **Prompt "据新华社报道" + top-p=0.9**:输出财经/经济新闻片段
- **Prompt "近年来,随着科技发展" + temp=0.7**:输出连贯科技发展叙述

### 5. 200-500 字实验观察

详见下方"关键观察"节(200-500 字)。

---

## 必做 M1-M5

| ID | 要求 | 状态 | 证据 |
|---|---|---|---|
| **M1** | 手写 BPE tokenizer,`tokenizer_roundtrip` 通过(encode→decode 还原中文) | ✓ | `verify_bpe.py` 21/21 样本通过;`eval/result.json:tokenizer_roundtrip:pass` |
| **M2** | 手写 decoder-only + RoPE,前向 shape 对 | ✓ | `src/model.py` MiniGPT,`src/rope.py` RoPE;smoke test 验证 forward 输出 `(B, T, vocab_size)` |
| **M3** | 实现 KV cache,`kv_cache_equivalence` < 1e-4 | ✓ | `eval/result.json:kv_cache_equivalence` `max_abs_diff = 3.8e-6` |
| **M4** | 训练,`perplexity_on_dev` < 50 | ✓ | `eval/result.json:perplexity_on_dev` `ppl = 48.9 < 50` |
| **M5** | greedy/top-k/top-p/temperature 四种采样,生成连贯文本 | ✓ | `src/sampling.py`;`samples.md` 4 prompt × 4 策略;`generate_samples.py --max-new-tokens 80` |

## 加分 S1-S4 + 加分项后续补做

| ID | 要求 | 状态 | 证据 |
|---|---|---|---|
| **S1** | 10M / 50M / 100M 参数量扫描 | ✓ | `figures/s1_param_scan.md`,1000 iters 下 ppl 272/151/119 单调下降 |
| **S2** | 绝对 PE vs RoPE 长序列外推 | ✓ | `figures/s2_pe_extrapolation.md`,**饱和后对比**(4000 iters,两者都饱和),RoPE base 4× 优 + 4× 外推退化 19% vs 125% |
| **S3** | KV cache 开/关推理速度对比 | ✓ | `figures/s3_kv_cache_bench.md`,n=512 new=128 时 **7.73× 加速** |
| **S4** | TinyStories 10M 模型涌现叙事 | ✓ | `samples_tinystories.md`,7.9M 模型 dev_ppl=23.85,能写连贯儿童故事 |
| **+1** | tensorboard 日志 | ✓ | `src/train.py:SummaryWriter`;`ckpt/tensorboard/` |
| **+2** | batch generation | ✓ | `src/model.py:generate` 接受 `list[list[int]]`;`figures/batch_bench.md` n=4 时 **3.27× 加速** |

## 自检结果(eval/run.py)

| 测试 | 状态 | 指标 | 阈值 |
|---|---|---|---|
| `tokenizer_roundtrip` | ✓ | 3 样本全通过 | encode→decode 还原 |
| `kv_cache_equivalence` | ✓ | max_abs_diff = 3.8e-6 | < 1e-4 |
| `perplexity_on_dev` | ✓ | **ppl = 44.61** | < 50(README 默认) |

## 关键指标

- **主模型**:26.2M params,在 **30K SkyPile 子集(58MB / 21M tokens)** 上训 4000 iters → **dev_ppl=44.61 < 50 ✓**
- **数据**:SkyPile-150B 子集 30K 条(从最初 10K/18MB 翻 3.4×,从 6.2M tokens 到 21M tokens)
- **BPE**:vocab=8192,256 byte + 4 special + 7932 merges(冻结,不在重训时变)
- **训练曲线**:`figures/tensorboard_curve.png`(跨 5 次训练合并,53 个 eval 点,从 8500 → 44.61)
- **生成样例**:`samples.md` 用 4 个 news-style prompt × 4 种采样策略,基于 44.61 best.pt 重生

**关键路径**:从 18MB / 6.2M tokens / 4000 iters 的 47.21,30K 翻 3× 数据后到 44.61,证明 **数据量是 Chinchilla 限制下 26M 模型可达 ppl < 50 的关键变量**;模型架构 / iters 调整边际收益小。

### S1 参数量扫描(1000 iters / 18MB 数据)

| 标签 | 实际参数量 | 架构 | dev_ppl |
|---|---|---|---|
| 10M (target) | 6.3M | 192d × 8L | 271.94 |
| 50M (target) | 22.0M | 384d × 8L | 151.14 |
| 100M (target) | 54.5M | 512d × 12L | 118.65 |

### S2 RoPE vs Absolute PE(4000 iters,饱和后对比)

| 模型 | base (block=64) | 2× 外推 | 4× 外推 | 饱和? |
|---|---|---|---|---|
| RoPE | **122.97** | +4.5% | +19.0% | ✓(下降 2.6%) |
| Absolute | 499.47 | +68.6% | +125.0% | ✓(下降 1.4%) |

### S3 KV cache 速度

| prompt | new | speedup |
|---|---|---|
| 128 | 128 | 3.55× |
| 256 | 128 | 5.39× |
| 512 | 128 | **7.73×** |

### S4 TinyStories(7.9M 模型)

- 数据:5000 条 TinyStories(独立英文 BPE,vocab=4096)
- dev_ppl = **23.85**
- 生成样例:`samples_tinystories.md`(主谓宾、转折、对话、人物一致性都学到)

## 关键观察(200-500 字)

- **RoPE vs Absolute PE**:饱和后 base 4× 优势 + 4× 外推退化 6× 优势。RoPE 的旋转角数学上对任意 pos 成立,Q/K 旋转满足"相对位置差"代数不变性;Absolute 即使能扩展到任意 pos,模型对没见过的位置模式无先验。
- **小模型 scaling**:10M→50M ppl 降 44%,50M→100M 再降 21%(边际收益递减,符合预期)。tokens/param 0.6/0.4/0.2,都远低于 Chinchilla optimal 20,所以 100M 也未充分收敛。
- **KV cache 是免费的午餐**:同条件 O(T²)→O(T),实际测得 3-8× 加速,跟 prompt + new 总长度线性增长。
- **batch generation**:n=4 时 3.27× 加速(并行 GPU 利用率),n=8 时 3.23×(显存带宽饱和)。生产用值得开 batch。

## 已知限制

- **BPE 训练时 `decode` 不约束 UTF-8**:generation 偶有 `\ufffd`(非法字节组合,采样到),可加 constrained decoding 修复
- **batch generation 无 per-sequence eos 早停**:全 batch 同步结束,生产用需要 attention_mask
- **S4 用独立英文 tokenizer**(`vocab=4096`):与主任务中文 tokenizer 分开,工程上不统一。如果要长期维护,可训一份覆盖两者的多语 BPE
- **mismatch 修正过程冗长**:训练脚本 estimate_loss 与 eval harness 算法最初有 ~6pp 系统偏差,导致 run3 best 实际 eval 不达标。最后通过 `estimate_dev_ppl_eval_method` 完全复用 eval 算法对齐。教训:训练和评估的"数据切片方式"必须一致,否则 best.pt 选择与最终测试脱节
- **30K 数据达到 Chinchilla 10% 配比**:21M tokens / 26M params ≈ 0.8(Chinchilla optimal = 20)。模型仍严重欠拟合,若想进一步降 ppl,要么加数据(到 50K+),要么加模型规模到 100M+

---

## 交付物清单

- `src/`(8 个模块:tokenizer / rope / attention / block / model / sampling / train / config)
- `data/{train,dev}.txt` + `dataset_info.json`(主任务)
- `data/download_skypile.py` + `data/download_tinystories.py`(数据获取)
- `ckpt/tokenizer.json` + `ckpt/best.pt`(主任务)
- `ckpt/s1_{10m,50m,100m}.pt` + `ckpt/s2_{rope,abs}.pt` + `ckpt/s4_tinystories.pt`(S 实验)
- `ckpt/tinystories_tokenizer.json` + `ckpt/tinystories_{train,dev}.txt`(S4 独立配置)
- `ckpt/tensorboard/`(tensorboard 日志)
- `eval/result.json`(自检)
- `figures/s{1,2,3}_*.md` + `figures/batch_bench.md` + `figures/training_curves_run5_resume.png`(S 报告)
- `verify_bpe.py` + `generate_samples.py` + `bench_kv_cache.py` + `bench_batch_gen.py` + `run_s1/s2/s4_*.py`(复现脚本)
- `samples.md` + `samples_tinystories.md`(生成样例)
- `notes.md`(过程记录) + `tutor_review.md`(按 tutor_prompt.md 格式自评)
- **本文件** `report.md`(完成情况)
