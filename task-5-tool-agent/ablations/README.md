# Ablations · 任务五

四个消融实验脚本。所有脚本都依赖 `data/tasks.json`（`python data/download.py` 生成）和一个能联通的 OpenAI 兼容 endpoint。

## 公共前置

```bash
# 1. 生成任务集 + 检索夹具
python data/download.py

# 2. 启动本地 LLM（任选其一）
# Ollama
ollama pull qwen2.5:7b-instruct && ollama serve

# SGLang (本次实跑)
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/models')"
sglang serve --model-path /root/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master \
  --host 0.0.0.0 --port 30000 --trust-remote-code --context-length 8192 --mem-fraction-static 0.85

# 3. 设环境变量（指向 endpoint）
export OPENAI_BASE_URL=http://localhost:30000/v1
export OPENAI_API_KEY=EMPTY
export OPENAI_MODEL=/root/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master
```

## S1 · `qwen_agent_baseline.py`

**目标**：手写 ReAct vs Qwen-Agent 原生 function calling 成功率对比

**输入**：10 题任务集 + 4 工具 + 同一 LLM
**输出**：`eval/s1_qwen_agent_result.json`

```bash
# 仅跑 Qwen-Agent
python ablations/qwen_agent_baseline.py

# 同时跑自写 ReActAgent 做对比
python ablations/qwen_agent_baseline.py --compare-with-self

# 接口冒烟（只跑 1 题）
python ablations/qwen_agent_baseline.py --smoke
```

**关键点**：
- 把自写 `Tool` 包成 `qwen_agent.tools.BaseTool` 子类（共享 4 个工具实现）
- `ReActChat` 走 OpenAI tools API，跟手写 prompt 风格不同
- `_answer_matches` 与 `eval/run.py` 同一套 normalize 逻辑，结果可比

## S2 · `model_size_compare.py`

**目标**：不同模型尺寸（1.5B / 7B / 14B）的成功率对比

**输入**：同一任务集 + 同一 prompt
**输出**：`eval/s2_model_size_result.json`

```bash
# 自动 probe Ollama/vLLM/SGLang 端口探测可达模型
python ablations/model_size_compare.py
```

**注意**：脚本会探测 `qwen2.5:1.5b-instruct` / `qwen2.5:7b-instruct` / `qwen2.5:14b-instruct` 三个模型名。本地只有哪个就只跑哪个；都不可达 → 写占位 JSON。

## S3 · `prompt_ablation.py`

**目标**：不同 prompt 模板（few-shot 数量、错误恢复提示）对工具调用准确率影响

**参数化**：
- `few_shot_count ∈ {0, 1, 3}`（S3-1）
- `include_error_hint ∈ {True, False}`（S3-2）

跑出 3×2 = 6 组。

**输出**：`eval/s3_prompt_ablation_result.json`

```bash
python ablations/prompt_ablation.py
```

**两组消融**：
- 不依赖 LLM：prompt 长度消融（messages / chars / 估算 tokens）
- 依赖 LLM：跑 6 组真模型，命中率对比

## S4 · `error_injection.py`

**目标**：错误注入消融，验证 `inject_error` 钩子 + agent 自纠错

**参数化**：`error_rate ∈ {0.0, 0.2, 0.5, 0.8}`

**输出**：`eval/s4_error_injection_result.json`

```bash
python ablations/error_injection.py
```

**两组消融**：
- Stub 模式（不依赖 LLM）：用 `_RepeatFakeLLM` 验证注入逻辑
- 真模型模式：跑 10 题，看命中率随错误率的衰减曲线

## 共用依赖

```bash
pip install qwen-agent json5 dashscope  # 仅 S1 需要
```

无依赖脚本（S2/S3/S4 stub 部分）任何环境都能跑。

## 结果对照表（本次实跑 · Qwen2.5-7B-Instruct via SGLang）

| 实验 | 关键指标 | 自写结果 | 对照结果 |
|---|---|---|---|
| M4 主评测 | 10 题命中率 | **70-80%** (7-8/10,非确定性) | — |
| S1 | Qwen-Agent 命中率 | 80% (8/10) | 90% (9/10) Qwen-Agent |
| S2 | 1.5B vs 7B vs 14B | 仅 7B 实跑 | — |
| S3 | 0/1/3 shot × err hint on/off | 6 组 (50-70%) | 见 `s3_prompt_ablation_result.json` |
| S4 | error_rate 0/0.2/0.5/0.8 | stub 通过 | 见 `s4_error_injection_result.json` |

### S3 实跑结果速查（6 组对比）

| Config | 命中率 |
|---|---:|
| 0-shot, no hint | 60% |
| 0-shot, +hint | 60% |
| 1-shot, no hint | **50%** ← 最差（局部最优陷阱）|
| 1-shot, +hint | 60% |
| **3-shot, no hint** | **70%** ← 最佳 |
| 3-shot, +hint | 60% |

**关键发现**:3-shot no hint 是甜区;`include_error_hint` 在 3-shot 上反而拖 10pp(prompt 过长分散注意力)。