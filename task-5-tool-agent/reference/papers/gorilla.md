# Gorilla: Large Language Model Can Connect to Massive APIs

## 来源
- 链接：https://arxiv.org/abs/2305.15334
- 作者/组织：Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez, Raluca Ada Popa, Ion Stoica（UC Berkeley）
- 发布时间：2023 年 5 月
- 核心定位：让 LLM 能针对大规模、版本频繁变化的 API 集合准确生成调用指令

## 关键要点

### 1. 核心问题
现实 API 数量巨大（数千个）且版本频繁更新（API 调用形式会变）。手工写 few-shot 无法覆盖，需要让模型从 API 文档中"读懂"调用方法。

### 2. 方法
- 微调 LLaMA-7B：在 (自然语言指令, API 调用) 数据对上微调。
- 数据生成：从 API 文档（自建 / HuggingFace / PyTorch 等）合成 instruction-API 训练对。
- 检索增强：推理时先把候选 API 文档拼到 prompt，再让模型生成调用。

### 3. 与本任务的距离
本任务的 4 个工具是固定的，且都是 OpenAI function calling schema（结构化 JSON），所以 Gorilla 路线（API 检索 + 文档理解）不是必需的。但 Gorilla 揭示了一个普适原则：**工具调用模型的瓶颈往往不是"知道用哪个工具"，而是"准确填对参数"**。

### 4. 启发
- 工具的 description 写得越具体、给出示例值，模型调用越准。
- 参数 schema 用 enum 比 free-form 字符串好。
- "参数错误"是工具调用失败的首要原因，比"选错工具"更常见。

## 与我们任务的关联

- **M1 工具 schema**：Gorilla 提示我们 `TOOL_SCHEMA` 的 `description` 字段不能只写一句"搜索文件"——要给典型 pattern（如"匹配 .md 文件名"）、示例值、错误情况下的 fallback 行为。
- **M3 错误恢复**：Gorilla 路线里"参数错"占多数失败，我们的错误恢复 prompt 里也应该专门提示"如果参数不对，下一个 Thought 里改参数再试"。
- **可选 S1（Qwen-Agent 对照）**：Qwen-Agent 内部应该用了某种 Gorilla-style 思路（把工具描述塞进 prompt），可以观察它和我们手写版本的差异。

## 代码片段（论文摘要里的 API 调用格式）

```json
{
  "api_name": "search_file",
  "parameters": {
    "pattern": "*.md",
    "dir": "data/agent-fixtures"
  }
}
```

我们的 `TOOL_SCHEMA` 应该长成这个样子（OpenAI function calling 格式）：

```python
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": "在指定目录下按文件名或内容片段搜索文件，返回匹配的文件路径或片段。pattern 支持 glob（如 '*.md'）或纯字符串。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "文件名/内容模式，例如 '*.md' 或 'TODO'"},
                "dir": {"type": "string", "description": "搜索根目录，相对于工作区，例如 'data/agent-fixtures'"},
            },
            "required": ["pattern", "dir"],
        },
    },
}
```

## 我们应该怎么借鉴

1. **description 要写得像 API doc**：包含典型输入、典型输出、典型错误模式。7B 模型读得懂人话。
2. **enum 能用就用**：例如 file_search 的 `dir` 可以从几个固定候选里选，避免 7B 模型胡乱拼路径。
3. **必填字段要明确**：required 字段标错，模型就漏参数 → KeyError。
4. **不要追求工具数多**：Gorilla 关注"上千 API 中选对"，我们只要"4 个里选对"——这点比 Gorilla 简单得多。

## 不需要借鉴的
- 检索增强（向量召回候选 API）：工具数 ≤ 10 时全量塞进 prompt 更稳。
- 大规模微调：本任务不动模型权重。
