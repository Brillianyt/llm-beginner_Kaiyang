# CodeAct: Executable Code Actions Elicit Better LLM Agents

## 来源
- 论文链接：https://arxiv.org/abs/2402.01030
- 作者：Xingyao Wang, Zihan Wang, Jiateng Liu, et al.（UIUC / xingyaoww 团队）
- 发布时间：2024-02-02（v1），v2 修订于 2024
- 代码仓库：https://github.com/xingyaoww/code-act
- 核心定位：**用可执行 Python 代码作为统一动作空间的 LLM Agent 范式**

## 关键要点
1. **统一动作空间**：把「调工具」「读环境」「向用户提问」「提交答案」等原本分散为 JSON / 文本 / 自然语言格式的动作，统一成一段可被 Python 解释器执行的代码。Agent 每一步生成一段 Python，主循环直接 `exec()`。
2. **数据规模**：从 68 个 LLM 与 27 个任务的多轮交互中收集 7,364 条样本，用于 CodeAct-Instruct 微调。
3. **效果**：在 HumanEval+ 等基准上，CodeAct-Instruct 微调后的 LLaMA-2 7B 提升 14.0%，LLaMA-2 34B 提升 16.4%（已接近 GPT-4 的 19.6%）。Python 版在 27 个任务里 24 个优于 JSON / 文本版。
4. **完整 agentic 框架**：仓库里包含 CodeActAgent（多轮 ReAct + 代码执行 sandbox）和评测脚本。
5. **沙箱安全**：执行用 subprocess + 受限环境（默认 5 秒超时），防止 Agent 写出破坏性代码。

## 与我们任务的关联
- **M3（agent loop）**：CodeAct 的 ReAct 循环（thought → code → observation）正是我们 `CodingAgent.run` 要实现的范式。
- **M4（Trace 结构）**：CodeAct 的 trace 也是「thought / action / observation」三段式，与 README 里要求的 `steps` 字段直接对应。
- **执行方式选择**：我们不一定要走纯 code-action（Qwen2.5-Coder-7B 也支持 function calling JSON），但 CodeAct 的设计提醒我们：**工具调用 schema 应尽量声明为函数签名**，让 LLM 生成「调用 read_file(path='calculator.py')」这种比纯 JSON 更自然。

## 代码片段（CodeAct 的最小 ReAct 循环）

```python
# 来自 CodeAct 论文附录的伪代码风格
while step < max_steps:
    prompt = build_prompt(history, tools)  # 注入可用工具签名 + 历史
    code = llm(prompt)                      # LLM 生成可执行 Python
    observation = sandboxed_exec(code, timeout=5)
    history.append({"thought": ..., "code": code, "observation": observation})
    if "FINISH" in code:                    # LLM 显式声明 done
        break
```

## 我们应该怎么借鉴
1. **prompt 设计**：在 system prompt 里以「Python 函数签名」的方式列出所有工具（`read_file(path: str) -> str`、`write_file(path: str, content: str) -> None`），让 Qwen 习惯这种格式——比纯 JSON schema 更容易让它一次性生成正确的调用。
2. **执行 vs 调用的边界**：本地简单工具用 `exec` code-action 风格会更灵活（一次调用可组合多步），但 MCP 协议要求 JSON-RPC 风格的 `tools/call`。**建议**：在 CodingAgent 主循环里两种都支持——Tool 描述可被模型选 JSON 工具调用，而 Tool 内部实现可借助 code-style 模板拼接（折中方案）。
3. **沙箱思路**：跑 `pytest` 一定要带 timeout；CodeAct 的 5 秒超时太短，我们的玩具 repo 可以给 60 秒，但要在 tool 实现里强制设 `subprocess.run(timeout=60)`。
4. **FINISH 信号**：CodeAct 里的 done 信号是 LLM 在代码里写 `FINISH`，我们也应给 Qwen 一个明确的「我已经修好且测试通过，现在生成 patch 结束」信号——可以专门定义 `submit_patch(diff_text)` 作为「伪工具」来触发终结。

## 主要参考来源
- arXiv 论文：https://arxiv.org/abs/2402.01030
- GitHub 代码：https://github.com/xingyaoww/code-act
- Gradient Flow 解读：https://gradientflow.com/codeact-executable-python-code-actions/