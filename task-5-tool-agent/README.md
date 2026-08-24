# 任务五：工具调用 Agent

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 一句话目标

手写约 200 行 ReAct 循环，让本地 Qwen2.5-7B-Instruct 自主调度 calculator / python_sandbox / file_search / wiki 四类工具，在自建 10 题任务集上答案关键词命中率 > 60%（按关键词校验，不信任 agent 自报 `success`）。

## 任务情境

假装你刚入职某 agent 团队，组长让你「先把 ReAct 自己撸一遍」。规则：

- 不许直接用框架的 agent 封装，循环自己写
- 工具要齐 4 类，每类一个文件、各带 OpenAI function calling schema
- 两周后周会汇报：10 题成功率 + 几条完整 trace + 你对工具路由、错误恢复、prompt 模板的理解

这就是本任务。**扩展**实践书 v2 ReAct 节的单工具示例：扩展到 4 类工具、错误恢复、与 Qwen-Agent 框架对照。

## 输入 / 输出

| | 内容 |
|---|---|
| **给你** | 本地 Qwen2.5-7B-Instruct（Ollama / vLLM / llama.cpp 任选，OpenAI 兼容 API）/ `data/download.py` 自动生成的 10 题任务集（`data/tasks.json`）+ 本地检索夹具（`data/agent-fixtures/`）/ 7B 模型建议 8GB+ 显存，量化版可更低 |
| **交付** | 1. `src/tools/` 下 4 个工具模块 2. `src/agent.py`（手写 ReAct 循环） 3. `eval/result.json`（自检结果，含每题 trace 预览） 4. 几条完整 ReAct trace（Thought / Action / Observation 全文） 5. 一段 200–500 字实验观察 |

## Definition of Done

必做 4 项，缺一不算完成：

- [ ] **M1** 实现 4 个工具（calculator / python_sandbox / file_search / wiki），各带 `TOOL_SCHEMA` 与 `run(args)`，自检 `tools_individual` 通过
- [ ] **M2** 手写 ReAct 循环（Thought / Action / Action Input / Observation），含工具路由、步数上限、Final Answer 终止
- [ ] **M3** 工具抛异常时捕获并把错误消息塞回 Observation 让 agent 自我纠错，不让单次工具失败 crash 整个循环
- [ ] **M4** 在自建 10 题任务集上自检 `multi_tool_success_rate` 通过（关键词命中率 > 60%）

加分（任选）：

- [ ] **S1** 用 Qwen-Agent 写一版功能相同的，对比成功率
- [ ] **S2** 不同模型尺寸（1.5B / 7B / 14B）的成功率对比
- [ ] **S3** 不同 prompt 模板（few-shot 条数、工具描述写法）对工具调用准确率的影响
- [ ] **S4** 实现 `inject_error` 钩子跑通 `error_recovery`，或对比是否用任务三 plugin SFT 后的模型 vs zero-shot

## 实施步骤（建议节奏：2 周）

### 第 1-2 天：环境 + 模型 + 数据

```bash
pip install -r requirements.txt

# 部署本地模型（Qwen2.5-7B-Instruct 经 Ollama 提供 OpenAI 兼容 API）
ollama pull qwen2.5:7b-instruct
ollama serve  # 默认 http://localhost:11434/v1

# 生成自建评测任务集和本地文件检索夹具
python data/download.py
```

**输入**：本地推理后端 + 网络（wiki 工具用）
**输出**：`data/tasks.json`（10 题）、`data/agent-fixtures/`（含 `README.md`、`todo_note.md`）、能连通的 OpenAI 兼容 endpoint

显存吃紧可改 vLLM 量化版（`Qwen2.5-7B-Instruct-AWQ`）或 llama.cpp GGUF（`q4_k_m`），`data/download.py` 末尾打印了三种部署方式。

**本次实跑使用 SGLang**（推荐 PPU/A100 等大显存场景）：

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/models')"
sglang serve \
  --model-path /root/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code --context-length 8192 --mem-fraction-static 0.85
```

然后跑 eval（环境变量指向 SGLang endpoint）：

```bash
OPENAI_BASE_URL=http://localhost:30000/v1 \
OPENAI_API_KEY=EMPTY \
OPENAI_MODEL=/root/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master \
python eval/run.py
```

**常见坑**：

- 客户端没指对 `OPENAI_BASE_URL` / `OPENAI_API_KEY`：Ollama 走 `http://localhost:11434/v1`、key 随便填（如 `ollama`），不是真去调 openai.com
- 没先 `ollama serve` 就跑 agent，连接被拒
- wiki 题需要联网；离线/被墙时该工具单测按「跳过」处理（见自检表），别误以为代码错了

### 第 3-7 天：4 个工具（M1）

**输入**：工具调用参数（见下方固定参数键）
**输出**：`src/tools/` 下 4 个模块，各能通过 `tools_individual` 单测

每个工具一个文件，导出 `TOOL_SCHEMA`（OpenAI function calling 格式）和 `run(args: dict) -> str`：

- `calculator.py`：四则运算 + 高级函数
- `python_sandbox.py`：受限 exec（限制 import、超时、stdout 捕获）。⚠️ 黑名单 import + 白名单 builtins + 超时只是**教学级**防护，不是真正隔离——仍可经 `().__class__.__bases__` / `__globals__` 等路径逃逸，超时也挡不住内存耗尽；只对可信 / 自产代码用，别对真正不可信输入直接 exec（进阶可上子进程 + resource 限制 / RestrictedPython / 容器）
- `file_search.py`：本地目录文件名 / 内容检索（评测任务集既要按名/内容定位，也要能返回匹配文件的内容片段）
- `wiki.py`：维基百科 API 查询（评测任务集含中英文条目，需支持中文查询）

**常见坑**：

- `calculator` 直接 `eval` 表达式 = 任意代码执行；要限定到算术 / 数学函数白名单
- `python_sandbox` 把上面那条 ⚠️ 当真：黑白名单只是教学级，路径越界 / `__globals__` 逃逸 / 内存耗尽都挡不住，别拿它跑真正不可信输入
- `file_search` 不做路径越界保护：`dir` 收到 `../../` 能读到工作区外文件；先把路径 resolve 后校验落在允许根内
- `file_search` 只匹配文件名不返回内容片段：任务集里第 10 题要「第一段写了什么」，得真读文件内容
- 自检按固定参数键调 `run`（`{"expression"}` / `{"code"}` / `{"pattern","dir"}` / `{"query"}`），键名对不上直接 KeyError

### 第 8-11 天：ReAct 循环（M2 + M3）

**输入**：自然语言 task + 工具列表
**输出**：`src/agent.py` 完整，`class ReActAgent` 的 `run(task)` 返回结构化 trace

实现内容：

1. prompt 模板：清楚区分 Thought / Action / Action Input / Observation 四种 turn，给 few-shot 示例让小模型理解格式，工具列表动态拼到 system prompt
2. 工具路由 + 调用 + 错误捕获：解析模型输出里的 Action / Action Input，路由到对应工具，异常塞回 Observation
3. 终止条件：模型输出 Final Answer 或步数达上限

**常见坑**：

- 没设步数上限：模型反复 Action 不收尾，死循环
- Action 解析靠脆弱正则：模型偶尔多输出一行解释就 parse 失败，要么容错、要么检测到解析失败时要求重试
- 工具抛异常直接冒泡 crash 整个 `run`：必须 catch 后把错误消息当 Observation 喂回去，让 agent 自我纠错（这就是 M3）
- 多步任务（如第 5 题 wiki→calculator）把上一步 Observation 漏出上下文：模型拿不到中间结果

### 第 12-14 天：评测 + 对照 + 写报告（M4）

**输入**：训练/部署好的 agent + `data/tasks.json`
**输出**：`eval/result.json`、几条完整 trace、报告文字

跑 `python eval/run.py`，看 10 题命中率。自检按 `expected_answer_contains` 校验最终答案、同义变体（如 Turing / 图灵）按可接受变体处理，**不信任** agent 自报的 `success`。加分项可再用 Qwen-Agent 写一版对照成功率。

**常见坑**：

- 用 agent 自报 `success` 当成绩：自检故意忽略它，按答案关键词判定，自报 success 不算数
- 答案藏在 trace 里没写进 `final_answer`：自检只看 `final_answer` 字段
- 数值题答案带千分位逗号 / 全角逗号：自检会 normalize 掉逗号和空白，但你的工具输出格式也得让关键词能命中（如第 8 题要 `45.011110` 这类精度）

## 实现约定

`eval/run.py` 会自动检测以下接口；接口对上就能跑自检：

| 文件 | 必须导出 |
|---|---|
| `src/tools/{name}.py` | `TOOL_SCHEMA: dict`（OpenAI function calling 格式）、`def run(args: dict) -> str`；自检按名导入这 4 个工具模块，并以固定参数键调用 `run`：`calculator` → `{"expression": str}`、`python_sandbox` → `{"code": str}`、`file_search` → `{"pattern": str, "dir": str}`、`wiki` → `{"query": str}` |
| `src/agent.py` | `class ReActAgent` 含 `run(task: str) -> AgentTrace`；`AgentTrace` 是 dict 含 `steps`、`final_answer`、`success: bool` |

接口可以改，但改了请同步调整 `eval/run.py`。

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 | 对应 DoD |
|---|---|---|
| `tools_individual` | 4 个工具各自跑一组单元测试全部通过 | M1 |
| `multi_tool_success_rate` | 在自建 10 题任务集（`data/tasks.json`）上答案关键词 / 同义词组命中率 > 60%；不信任 agent 自报 `success` | M4 |
| `error_recovery` | 可选实验（默认跳过）：需自行实现 `src/agent.py` 的 `inject_error` 钩子，注入 1 次错误工具响应后 agent 仍能完成任务的比例 > 40% | S4 |

> `tools_individual` 中 `wiki` 依赖网络：离线 / 被墙时该工具按「跳过」处理，不拖累其余工具判定；其余 3 个工具任一报错即不通过。

结果写入 `eval/result.json`，提交时附上。

## AI Tutor 反馈

把 [eval/tutor_prompt.md](eval/tutor_prompt.md) 整段贴给 Claude / Qwen / DeepSeek，连同你的代码。模型会按统一格式（必检 / 加分 / 优先级）给你针对性 review。

## 实验建议

- 手写 ReAct vs Qwen-Agent 原生 function calling
- 不同模型尺寸（1.5B / 7B / 14B）的成功率
- 不同 prompt 模板对工具调用准确率的影响
- 是否使用任务三 plugin SFT 后的模型，对比 zero-shot

## 前置阅读（非必需）

- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
- 实践书 v2《大语言模型与智能体》「ReAct 智能体」一节

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 「llm-beginner 实践成果」分类发帖，附：

1. 你的 fork 仓库链接
2. `eval/result.json` 内容（贴文本即可）
3. DoD checklist 勾选状态
4. 几条完整 ReAct trace（Thought / Action / Observation 全文）
5. 200-500 字实验观察：你做了哪些消融、看到了什么有意思的现象（如哪类任务最易失败、错误恢复是否真起作用）

## 时间

约 2 周。如果在 M4（成功率）卡住，先确认 4 个工具单测全过、再逐题看 trace 定位是工具调错还是 Action 解析失败。
