# ReAct: Synergizing Reasoning and Acting in Language Models

## 来源
- 链接：https://arxiv.org/abs/2210.03629
- 作者/组织：Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, Yuan Cao（普林斯顿 / Google Research）
- 发布时间：2022 年 10 月（ICLR 2023 接收）
- 核心定位：提出 Thought-Action-Observation 交替提示方法，把 Chain-of-Thought 的推理与环境交互能力结合起来

## 关键要点

### 1. 核心思想：交错推理与行动
ReAct 把任务求解轨迹写成 `Thought_i → Action_i → Observation_i` 的循环：
- **Thought**：自由形式的"内心独白"，可以拆解任务、汇总信息、决定下一步动作、处理异常。类似于 CoT，但是是面向"行动决策"的。
- **Action**：调用外部工具/API 的指令，语法上是 `tool_name[input]`。在原论文中工具包括 Wikipedia Search、Lookup、Finish 等。
- **Observation**：环境（Wikipedia / 搜索引擎 / 模拟器）返回的字符串结果。

每个 Thought 不需要"自然语言回答用户"，而是"给模型自己的备忘"——这是和纯 CoT 的关键区别。

### 2. Few-shot 提示而非微调
ReAct 用 in-context 演示让模型学会格式：人写若干个完整任务轨迹作为示例塞进 prompt，推理时模型按相同格式续写。这正好契合本任务"不微调、用 Qwen2.5-7B-Instruct 原生能力"的设定。

### 3. 终止与"Final Answer"
原论文 HotpotQA 设置中，最后一个动作是 `Finish[answer]`——动作即终止信号。本任务 README 沿用类似设定但拆成显式的 `Final Answer:` 字段，方便评测脚本按字符串匹配最终答案。

### 4. 错误恢复
Thought 中允许"反思上一步是否合理"，如果 Observation 不对/工具出错，下一个 Thought 可以决定换工具、重试、修改输入。这是 M3（错误恢复）的算法基础。

### 5. 实验结论（与本任务相关）
- 在 HotpotQA / FEVER 上比纯 CoT / 只 Action 更稳健，因为推理可以引用 Observation 校验。
- 在 ALFWorld / WebShop 这种交互式任务上显著优于 imitation learning / RL baseline。
- **失败模式**：循环步数过多、解析失败、Thought 与 Action 脱节——这正是 README 列出的"常见坑"。

## 与我们任务的关联

- **M2（手写 ReAct 循环）**：ReAct 论文是直接蓝本，照搬 Thought/Action/Action Input/Observation 四段格式即可。
- **M3（错误恢复）**：ReAct 的 Thought 自然支持"上一个工具失败 → 换一个 / 改参数"——把异常字符串作为 Observation 喂回去就行。
- **S3（prompt 模板消融）**：原论文做了 ReAct vs CoT vs Act-only 的消融，我们可参考它的消融维度（few-shot 数量、是否包含 Thought）。
- **M4（命中率 > 60%）**：原论文 HotpotQA 上 7B 左右模型可达 30-40%，我们用 Qwen2.5-7B（指令微调版），任务难度也更低，60% 是合理目标。

## 代码片段

原论文 HotpotQA 风格的 prompt 片段（来自公开复现的格式）：

```
Question: 除了苹果公司，1976 年还有哪家公司成立？
Thought 1: 我需要查 1976 年成立的公司。
Action 1: Search[Companies founded in 1976]
Observation 1: Results: Apple Computer, ... , Microsoft was founded in 1975 ...
Thought 2: 微软是 1975 年不是 1976 年。我需要更精确的搜索。
Action 2: Search[List of companies founded in 1976]
Observation 2: ...
Thought 3: 我已经找到了答案。
Action 3: Finish[Microsoft, Genentech, ...]
```

我们要做的不是 100% 复用，而是把 `Action` 换成 OpenAI function calling 的 JSON 形态，但 Thought / Observation 的语义保持不变。

## 我们应该怎么借鉴

1. **prompt 模板至少 3 个 few-shot**：覆盖单工具、多工具、计算+查 wiki 三种轨迹，否则模型在第 5/9 题这种组合场景容易卡住。
2. **Thought 必须显式**：不要省去 Thought 直接出 Action，7B 模型没了 Thought 几乎一定跑偏。
3. **解析失败的兜底**：正则匹配 `Action: xxx` / `Action Input: xxx` 时若失败，把整段输出作为 Observation 喂回去并要求"按格式重写"，不要直接 break。
4. **终止判定靠关键词扫描**：检测 `Final Answer:` 子串比检测模型说"我完成了"更稳。
5. **不要照搬论文的 `[Finish[xxx]]` 语法**：用自然语言 `Final Answer: xxx` + 解析器提取，eval 脚本按字符串匹配最终答案更友好。

## 局限性（我们要注意的）
- 原论文模型是 PaLM-540B / GPT-3，我们在 7B 上跑，可能需要更严格的格式约束（如 markdown 代码块包 JSON）。
- ReAct 的步数上限在论文里默认 7 步，对 wiki→calculator 这种两阶段任务勉强够用；对更长的链（如第 5 题先 wiki 再算）建议放到 10 步。
