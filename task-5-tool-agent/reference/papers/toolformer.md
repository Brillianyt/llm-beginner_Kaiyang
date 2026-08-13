# Toolformer: Language Models Can Teach Themselves to Use Tools

## 来源
- 链接：https://arxiv.org/abs/2302.04761
- 作者/组织：Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom（Meta AI）
- 发布时间：2023 年 2 月（NeurIPS 2023）
- 核心定位：用自监督方式把工具调用能力"烘焙"进模型权重，不需要手写大量 prompt 工程

## 关键要点

### 1. 训练方法：自监督采样 + 过滤
- 给定无标注文本，用已有的 few-shot 提示让模型在合适位置插入 API 调用（如 `[Calculator(2+3) → 5]`），得到候选训练样本。
- 对每个候选样本，计算"调用工具后 vs 直接续写"对模型 loss 的下降量；只有 loss 下降超过阈值的样本才保留下来训练。
- 这样一个 7B 模型自动学会了在合适位置插 API 调用，不依赖昂贵的人工标注轨迹。

### 2. 涵盖的工具
- 问答系统（搜索引擎 / QA 模型）
- 计算器
- 翻译系统
- 日历查询
- Wikipedia 搜索

与我们任务的 calculator / wiki / python_sandbox / file_search 高度重合。

### 3. 与 ReAct 的对比
| 维度 | ReAct | Toolformer |
|---|---|---|
| 引入方式 | prompt（in-context） | 训练（自监督微调） |
| 适用模型 | 任何 LLM | 需要微调权限 |
| 推理开销 | 每步都要生成 Thought | 模型"原生"知道何时调用 |
| 错误恢复 | Thought 中可反思 | 仍是模型自决 |

### 4. 关键发现
- 即使只用几千个自动筛选的样本，工具调用能力也能显著注入模型。
- 在 WikiSQL / MathQA / Temporal 等基准上明显优于纯 LLM 和 ReAct 风格的 prompting。
- 模型学会"何时不该用工具"——不是每次都查，而是当且仅当能降低 loss 时才调。

## 与我们任务的关联

- **M2 启示**：ReAct 的 Thought 不只是"思考"，更本质是"判断该不该调、调哪个"——这恰好是 Toolformer 学到的能力。我们要让 prompt 明确告诉模型：先用 Thought 评估，再决定 Action。
- **S4（plugin SFT 对照）**：本任务 README 加分项 S4 提到"对比任务三 plugin SFT 后的模型 vs zero-shot"——这正是 Toolformer 路线的简化版。
- **M3 错误恢复**：Toolformer 没有显式错误恢复机制，错误靠模型自决。我们的手写 agent 应该显式 catch 异常，比 Toolformer 更鲁棒。
- **prompt 工程可借鉴**：Toolformer 论文附录里有工具调用位置的格式定义（如 `[Calculator(2+3) → 5]`），可以作为我们 prompt 中 Action / Action Input 格式的参考。

## 代码片段

Toolformer 风格工具调用的"位置标记"思想：

```python
# 训练样本构造示例（伪代码）
text = "巴黎是法国的首都，人口约 2.2 百万。"
augmented = "巴黎是法国的首都[Search(Population of Paris)]人口约 2.2 百万。"
# 计算 loss 差：去掉 [Search(...)] 后是否显著更差？
# 差得多 → 保留为正样本
```

我们在 prompt 里要做的等价物：明确告诉模型"输出格式必须包含 `Action:` 和 `Action Input:` 两行"——这相当于把位置标记从隐式变显式。

## 我们应该怎么借鉴

1. **工具描述要"自包含"**：Toolformer 论文里工具 schema 写得很细，模型不需要外部文档。我们也要在 system prompt 里写清楚每个工具的输入格式、输出格式、典型用例。
2. **不要"事事都调工具"**：Toolformer 关键贡献是学会"不该调时别调"。我们的 prompt 可以给反例（"已知 2+3=5 就不用 calculator"），避免模型在简单题上浪费步数。
3. **微调版 vs 提示版是两个独立路线**：README 里 S4 的对照实验实际上就是在比较这两条路线，建议两边都跑，论文里 Toolformer 在多数任务上比 ReAct prompting 高 5-10 个百分点。
4. **错误处理**：Toolformer 不显式处理工具错误——我们既然要写工程级 agent，就该补上。

## 我们不需要借鉴的
- 自监督 loss-based 过滤：太重，本任务用不上。
- 工具调用位置标记：换成 OpenAI function calling 风格后就不需要了。
