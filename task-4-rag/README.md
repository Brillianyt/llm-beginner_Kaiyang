# 任务四：RAG 文档问答

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 一句话目标

搭一套端到端中文 RAG（PDF chunking → BGE embedding + FAISS → reranker → Qwen 生成），在随任务提供的 30 条 NNDL gold QA 上做到 Recall@10 > 0.6，并把检索质量和生成忠实性拆开量化。

## 任务情境

假装你刚入职某团队，要做一个「拿 PDF 资料库回答问题」的内部问答工具。组长的要求是：

- 知识库就是《神经网络与深度学习（第二版）》PDF，必须从 PDF 抽文本建索引，不许偷懒去索引 LaTeX 源文件
- 不能只看「答案读起来顺不顺」，要拿固定 gold QA 报 Recall@k / MRR
- 两周后汇报：召回指标 + 生成忠实性 + 你对 chunking / rerank / prompt 拼接每一环的提升量化

这就是本任务。

## 输入 / 输出

| | 内容 |
|---|---|
| **给你** | NNDL v2 PDF（`data/kb.pdf`，`data/download.py` 自动拉）/ 30 条 gold QA（`data/gold_qa.jsonl`，每条含 `source_file` 和 `gold_anchors`）/ BGE embedding + reranker 模型 / 本地 Qwen2.5-7B-Instruct（自行下载，Ollama / vLLM / llama.cpp 均可）/ 单卡 GPU 建议，embedding/检索 CPU 也能跑 |
| **交付** | 1. 可复现的 RAG 索引与流水线（`src/` 下 chunker / indexer / retriever / reranker / generator / rag） 2. 检索评测产物（Recall@1/3/5/10 + MRR） 3. `eval/result.json`（自检结果） 4. 一段 200–500 字实验观察 |

## Definition of Done

必做 4 项，缺一不算完成：

- [ ] **M1** 实现 `chunk_text`，自检 `chunking_sanity` 通过（chunk 数 > 10，平均长度落在 chunk_size 的 0.5–1.2 倍）
- [ ] **M2** 从 `data/kb.pdf`（而非 LaTeX 源）建 embedding + FAISS 索引，实现 `Retriever.retrieve`
- [ ] **M3** 在 `data/gold_qa.jsonl` 上自检 `nndl_gold_recall_at_10` 通过（Recall@10 > 0.6，同时报 Recall@1/3/5/10 与 MRR）
- [ ] **M4** 串起端到端 `answer`，自检 `rag_end_to_end` 通过（返回非空 answer + 非空 sources）

加分（任选）：

- [ ] **S1** chunk_size 扫描（128 / 256 / 512 / 1024 字符）对 Recall 的影响
- [ ] **S2** 加 / 不加 reranker 的 Recall + faithfulness 对比
- [ ] **S3** Query rewriting / HyDE 的有效性
- [ ] **S4** 用 RAGAS 给端到端打 faithfulness / answer relevancy 分

## 实施步骤（建议节奏：2 周）

### 第 1-2 天：环境 + 数据

```bash
pip install -r requirements.txt

# 下载 embedding / reranker 模型、NNDL v2 PDF，并检查随任务提供的 gold QA
python data/download.py

# 如果只想先准备 PDF 并检查 gold QA，可跳过模型下载
python data/download.py --skip-models
```

默认知识库是《神经网络与深度学习（第二版）》PDF，下载到 `data/kb.pdf`：

```text
https://github.com/nndl/nndl/releases/download/book-pdf/nndl-v2.pdf
```

评测题目保存在 `data/gold_qa.jsonl`，共 30 条。这些题目基于工作区中的 `../神经网络与深度学习2/` LaTeX 正文设计，每条题目都包含 `source_file` 和 `gold_anchors`；下载脚本会检查 LaTeX 来源锚点，并在可提取 PDF 文本时确认每题至少一个 anchor 能在 `data/kb.pdf` 中命中。学生的 RAG 索引仍应从 `data/kb.pdf` 构建，而不是直接索引 LaTeX 源文档。

生成模型 Qwen2.5-7B-Instruct 自行下载（量化版省显存），下载脚本结尾会打印 Ollama / vLLM / llama.cpp 三种起服务的命令。

### 第 3-5 天：Chunking + 索引（M1 + M2）

**输入**：`data/kb.pdf`
**输出**：`src/chunker.py`、`src/indexer.py` 完整，建好的 FAISS 索引，通过 `chunking_sanity` 自检

实现内容：

1. `src/chunker.py`：从 PDF 提取文本，做固定大小 / 递归 / 语义切分
2. `src/indexer.py`：BGE embedding + FAISS 索引建立

**常见坑**：

- `chunk_text` 的 `chunk_size` / `overlap` **以字符计**，不是词元数；自检按字符平均长度核验，平均长度要落进 `(chunk_size * 0.5, chunk_size * 1.2)`，写成按词元数切会直接挂
- PDF 抽文本会丢版式：表格、公式、代码块抽出来常是乱序碎片，切分前最好按段落/章节边界归并，别在句子中间硬切
- FAISS 用内积索引等价于 cosine，前提是 embedding 已 L2 normalize；忘了归一化检索分数全乱
- 索引必须来自 `data/kb.pdf` 抽出的文本；直接索引 LaTeX 源文件能让召回虚高但违背任务，gold anchor 命中口径也会失真

### 第 6-9 天：检索 + Rerank（M3）

**输入**：建好的索引 + `data/gold_qa.jsonl`
**输出**：`src/retriever.py`、`src/reranker.py` 完整，通过 `nndl_gold_recall_at_10` 自检

实现内容：

1. `src/retriever.py`：query embedding + top-k 召回，`retrieve` 返回的每个 dict 含 `text` / `score` / `source`
2. `src/reranker.py`：bge-reranker 精排

**常见坑**：

- 自检的命中口径是「召回 chunk 文本（去掉所有空白后）包含任一 gold anchor」——anchor 落在两个相邻 chunk 的交界处时，overlap 太小会两边都切断，谁都命中不了
- BGE 查询要加检索前缀（query 加 `"为这个句子生成表示以用于检索相关文章："`），文档侧不加；加错或漏加都掉召回
- reranker 吃的是 `[query, doc]` 文本对，不是再算一遍 embedding；别把它当成第二个 embedding 模型
- 召回数要 >> 最终返回数（如召回 20 再 rerank 取 top 几），召回阶段 k 给太小，rerank 救不回来
- 自检按 k=10 调 `retrieve(query, k=10)`，确保 `k` 参数真的生效、不要写死成别的值

### 第 10-12 天：生成 + 端到端（M4）

**输入**：检索结果 + 本地 Qwen 服务
**输出**：`src/generator.py`、`src/rag.py` 完整，通过 `rag_end_to_end` 自检

实现内容：

1. `src/generator.py`：把检索结果拼成 prompt，调本地 Qwen2.5-7B-Instruct
2. `src/rag.py`：`answer(query)` 串起检索 + 生成，返回 `{answer, sources}`

**常见坑**：

- `answer` 必须返回 `dict`，且 `answer` 是非空字符串、`sources` 非空；自检里特意 `bool(...)` 强转判断（直接拿 list 当哈希键会让运行壳在写 `result.json` 前崩），所以返回结构对不上整条自检会挂在最后一步
- prompt 要明确「只能用提供的上下文回答，不知道就说不知道」，否则模型会拿预训练知识硬答，掩盖检索失败
- 上下文要去重 + 按长度截断，召回片段有重叠，原样全拼进去既浪费词元消耗又稀释关键信息
- 别让流畅答案掩盖召回失败：评估时检索质量和生成忠实性分开看，最终仍以 gold anchor 召回为稳定核心指标

### 第 13-14 天：消融 + 写报告

**输入**：跑通的流水线
**输出**：实验表格 + 报告文字

按「实验建议」做几组消融（chunk_size 扫描 / 有无 reranker），量化每一环带来的 Recall 或 faithfulness 提升，写进观察。

## 实现约定

| 文件 | 必须导出 |
|---|---|
| `src/chunker.py` | `chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]`（`chunk_size` / `overlap` 以字符计，自检按字符平均长度核验） |
| `src/retriever.py` | `class Retriever` 含 `retrieve(query: str, k: int) -> List[dict]`（每个 dict 含 `text`、`score`、`source`） |
| `src/rag.py` | `answer(query: str) -> dict(answer: str, sources: List[dict])` |

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 | 对应 DoD |
|---|---|---|
| `chunking_sanity` | chunk 数 > 10；平均长度在 (chunk_size * 0.5, chunk_size * 1.2) | M1 |
| `nndl_gold_recall_at_10` | 在 `data/gold_qa.jsonl` 上 Recall@10 > 0.6；同时输出 Recall@1/3/5/10 与 MRR；召回 chunk 需命中至少一个 gold anchor | M3 |
| `rag_end_to_end` | 端到端能返回 answer 且 sources 非空（手动验证语义） | M4 |

结果写入 `eval/result.json`，提交时附上。

## 客观评价口径

把检索和生成拆开评估。检索部分使用固定的 `data/gold_qa.jsonl`：每个问题都有从 LaTeX 正文抽取的 `gold_anchors`，但系统只能索引 `data/kb.pdf`；如果 top-k chunk 中出现任一 gold anchor，就记为该题命中。这样可以稳定报告 Recall@1/3/5/10 和 MRR，指标不依赖生成模型的表达能力。

生成部分不要让流畅答案掩盖检索失败。建议基于同一批题目检查：答案是否覆盖 `answer` 中的关键点、是否能被返回 sources 支持、是否出现上下文外断言，以及不知道时是否拒答。RAGAS 或 LLM-as-judge 可以作为辅助，但 judge 提示必须只允许依据检索证据打分，最终仍以 gold anchor 召回作为稳定核心指标。

## AI Tutor 反馈

把 [eval/tutor_prompt.md](eval/tutor_prompt.md) 整段贴给 Claude / Qwen / DeepSeek，连同你的代码。模型会按统一格式（必检 / 加分 / 优先级）给你针对性 review。

## 实验建议

- Chunk size 扫描（128 / 256 / 512 / 1024 字符）
- 加 / 不加 reranker 的 Recall + faithfulness
- Query rewriting / HyDE 的有效性
- 用 RAGAS 打端到端分数

## 前置阅读（非必需）

- [RAG 综述](https://arxiv.org/abs/2312.10997)
- [BGE Embedding 系列](https://huggingface.co/BAAI)
- [RAGAS 评测框架](https://github.com/vibrantlabsai/ragas)
- 实践书 v2《大语言模型与智能体》「检索增强生成（RAG）」一节

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 「llm-beginner 实践成果」分类发帖，附：

1. 你的 fork 仓库链接
2. `eval/result.json` 内容（贴文本即可）
3. DoD checklist 勾选状态
4. 检索评测产物（Recall@1/3/5/10 + MRR）
5. 200-500 字实验观察：你做了哪些消融（chunk_size / reranker / query 改写）、每一环提升多少、看到了什么有意思的现象

## 时间

约 2 周。如果在 M3（gold 召回）卡住，先固定一档 chunk_size 把 Recall@10 跑过 0.6，再回头扫描 chunk 策略和加 reranker 提升上限。
