# 责任链模式与错误恢复

## 来源
- 模式分类：经典 OO 设计模式（Chain of Responsibility）
- 应用领域：HTTP 中间件链、日志系统、异常处理链、agent 工具错误 fallback
- 相关参考：Express.js middleware、Python try/except 嵌套、LangChain `handle_parsing_errors`

## 关键要点

### 1. 责任链模式定义
> "Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle it. Chain the receiving objects and pass the request along until an object handles it."

在 agent 错误恢复里，"请求"是一次工具调用，"接收者"是若干个错误处理策略：重试、改参数、换工具、回退到 LLM 自行回答、终止。

### 2. ReAct 错误恢复的层次

工具调用失败的可能原因：
1. **解析失败**：模型没按格式输出（action 名缺失、参数错位）→ 重试，要求重新生成。
2. **工具抛异常**：代码 bug / 参数非法 / 网络超时 → 把异常字符串当 Observation。
3. **结果为空**：工具返回空字符串或 "Not Found" → 让 LLM 决定换工具或修改输入。
4. **步数耗尽**：模型反复 Action 不收尾 → 强制终止，给出 best-effort 答案。
5. **连续无进展**：连续 N 步 Observation 相同 / Thought 重复 → 检测后强制终止。

这五层可以组成一条链：每层处理失败 → 把"半成品结果"或"错误信息"传给下一层。

### 3. 实现：错误信息作为合法 Observation

这是 ReAct 比传统软件错误处理更优雅的地方——**没有真正的异常传播**。所有错误都变成 LLM 能读的字符串，喂回去让它自己判断：

```python
def safe_tool_call(tool, args):
    try:
        return tool.run(args)
    except KeyError as e:
        return f"[ERROR: 缺少必填参数 {e}]"
    except TimeoutError:
        return f"[ERROR: {tool.name} 执行超时]"
    except Exception as e:
        return f"[ERROR: {tool.name} 失败：{type(e).__name__}: {e}]"
```

下一轮 prompt：
```
Thought 3: 上一步 calculator 报"缺少必填参数 'expression'"，我应该把表达式写完整。
Action 3: calculator
Action Input 3: {"expression": "(123 + 456) * 789"}
Observation 3: 456831
```

### 4. 责任链的"分支决策"
不是所有错误都要"塞回 Observation 让模型重试"。可以分轻重：

| 错误类型 | 策略 |
|---|---|
| 解析失败（语法错） | 让模型重新生成（追加"请按格式"指令） |
| 工具抛异常 | 塞 Observation，让模型换工具或改参数 |
| 同一工具连续 2 次失败 | 强制换工具，或跳过这一步 |
| 连续 N 步无进展 | 终止，给 best-effort |
| 模型反复说"我不知道" | 终止，记录为失败 |

### 5. 错误注入（与 S4 加分项呼应）
`inject_error` 钩子是责任链模式的天然应用：在 tool 调用前后插入一层 middleware，按概率 / 模式注入错误：

```python
class ErrorInjector:
    def __init__(self, error_rate=0.0, error_type=ValueError):
        self.error_rate = error_rate
        self.error_type = error_type
    
    def wrap(self, tool):
        original_run = tool.run
        def wrapped(args):
            if random.random() < self.error_rate:
                raise self.error_type(f"[注入错误] {tool.name} 故意失败")
            return original_run(args)
        tool.run = wrapped
        return tool
```

S4 加分项就是让 `ErrorInjector` 接进主循环，看 agent 在多少错误率下还能保持命中率。

## 与我们任务的关联

- **M3（必做）**：错误恢复的核心就是"异常字符串当 Observation"。这是手写 agent 比框架 agent 更可控的地方——可以精细到每一类异常的处理策略。
- **S4（加分）**：`inject_error` 钩子的实现就是责任链。
- **README 常见坑**："工具抛异常直接冒泡 crash 整个 run" —— 责任链模式正好解决这个问题。
- **trace 设计**：trace 里应该有 `error` 字段，区分"成功 Observation"和"错误 Observation"，方便调试。

## 我们应该怎么借鉴

1. **所有异常统一 catch 在主循环里**：工具内部不要 raise 出去。
2. **错误字符串必须包含工具名和异常类型**：模型才知道是哪个工具错了。
3. **解析失败单独走 retry 路径**：和工具抛异常分两种 Observation 格式。
4. **连续无进展检测**：记录最近 3 步的 Thought/Action，如果重复就强制终止。
5. **错误注入做测试钩子**：不要写在生产代码里，单独放 `src/agent.py` 的 `inject_error` 方法或 `src/middleware.py`。
6. **trace 区分成功 / 失败 Observation**：debug 时一眼看出哪步失败。

## 代码片段（错误恢复的核心 5 行）

```python
def call_tool_safely(self, name: str, args: dict) -> str:
    """统一的安全工具调用入口，所有异常转 Observation。"""
    if name not in self.tools:
        return f"[ERROR: 未知工具 '{name}'，可用：{list(self.tools.keys())}]"
    try:
        return self.tools[name].run(args)
    except Exception as e:
        return f"[ERROR: {name} 抛 {type(e).__name__}: {e}]"
```

这 5 行覆盖了 M3 80% 的需求。剩下 20% 是"连续失败处理"、"步数耗尽兜底"。

## 不需要借鉴的
- 完整的责任链抽象（Handler 基类 + 链式调用）：4 个错误类型用 4 个 if 就够了。
- 重试退避（exponential backoff）：单次任务用不上，本地调用也不需要。
