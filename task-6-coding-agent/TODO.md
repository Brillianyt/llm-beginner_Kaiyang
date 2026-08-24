# TODO

本文件记录已知的技术债、待验证的想法和未来改进方向。格式：一行摘要 + 一句上下文 + （可选）验证方式。

## 技术债

- **[已记录] 文本模式 tool-call fallback 是模型组合的固有路径，不是异常**
  - Qwen2.5-Coder-7B-Instruct 经 SGLang 从不产生原生 `tool_calls`——它把工具调用写成 `<function_call>`/JSON 文本，SGLang 也不做后处理。所以 `_parse_text_tool_calls` 兜底是每轮必走的正常路径，而非偶发回退。
  - 已做：fallback 从"每轮一条 THOUGHT"改为"计数器 `text_tool_call_fallback` + 首次一次性说明"，避免 trace 刷屏（`src/agent.py`，2026-08-24）。
  - 待做（可选）：若想彻底消除 fallback，二选一——
    1. 换模型：Qwen2.5-Coder-7B 不是 function-calling 微调版；换 Qwen2.5-7B-Instruct 系列（含工具调用训练）或 Qwen3 系列，可拿到原生 `tool_calls`；
    2. 换推理引擎 / chat template：给 SGLang 指定 Qwen 的 function-calling template，看模型是否直接走原生 `tool_calls`。
  - 验证方式：裸 API 发带 `tools` 的请求，看 `finish_reason` 是否为 `tool_calls` 且 `message.tool_calls` 非空。

## 待验证

- **SGLang 换 Qwen function-calling template 能否消除 fallback**（见上）。若可行，`_parse_text_tool_calls` 可降级为纯防御性代码。

## 未来改进（低优先级）

- 多 agent 并行时 `CODING_AGENT_AUDIT_LOG` 默认路径会争抢，建议 per-instance 显式传路径（已在 PRODUCT.md FAQ 提及）。
- trace 完全 1.0 replay 需要自带时间戳 monkeypatch（状态链 diff 是正确行为，非 bug）。
