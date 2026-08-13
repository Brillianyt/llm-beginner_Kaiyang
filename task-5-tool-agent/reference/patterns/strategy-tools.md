# 策略模式与工具路由器

## 来源
- 模式分类：经典 OO 设计模式（Strategy）
- 原始定义：Gang of Four《Design Patterns》"Define a family of algorithms, encapsulate each one, and make them interchangeable."
- 应用领域：支付方式、压缩算法、排序策略、agent 工具调用路由

## 关键要点

### 1. 经典策略模式结构
- **Context**：持有 Strategy 引用，按需调用。
- **Strategy 接口**：抽象方法 `execute(input)`。
- **ConcreteStrategy**：具体实现（多个）。

### 2. 在 agent 工具调用中的映射
| 角色 | 对应本任务 |
|---|---|
| Context | `ReActAgent` 主循环 |
| Strategy 接口 | `Tool` 基类的 `run(args: dict) -> str` |
| ConcreteStrategy | `Calculator`、`PythonSandbox`、`FileSearch`、`Wiki` |
| 切换逻辑 | 从模型输出解析出 action 名 → 查表 |

### 3. 关键设计：策略注册表
策略模式的核心是"运行时选哪个策略"。在 agent 里就是"模型说要用 calculator 时调 calculator，说要用 wiki 时调 wiki"。注册表用 dict 实现 O(1) 查找：

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    
    def call(self, name: str, args: dict) -> str:
        if name not in self._tools:
            raise KeyError(f"未知工具：{name}，可用：{list(self._tools.keys())}")
        return self._tools[name].run(args)
    
    def names(self) -> list[str]:
        return list(self._tools.keys())
    
    def schema_list(self) -> list[dict]:
        return [t.parameters for t in self._tools.values()]
```

主循环里：
```python
registry = ToolRegistry()
registry.register(Calculator())
registry.register(PythonSandbox())
# ...
obs = registry.call(parsed_action, parsed_args)
```

### 4. 策略模式 vs 工厂模式
容易混淆。区别：
- 工厂："创建什么"——根据输入返回对象。
- 策略："怎么算"——运行时选算法，行为差异在算法本身。

本任务工具调用属于策略模式：每个工具的 `run` 是"算法"，注册表是"调度器"。

### 5. 策略模式的额外好处
- **单元测试独立**：每个工具一个 `tests/test_calculator.py`，互不影响。
- **运行时替换**：测试时可以注入一个 mock 工具替换 wiki，验证错误恢复。
- **配置驱动**：未来想做"工具白名单/黑名单"，直接在 registry 上加方法。

## 与我们任务的关联

- **M1 工具实现**：4 个工具用同一个 `Tool` 基类，`run` 是统一入口——这是策略模式的教科书用法。
- **M2 工具路由**：解析出 action 名后查 `registry[action]`——查不到就当错误塞 Observation。
- **M3 错误恢复**：registry 抛 `KeyError` / `Exception` 都 catch 住。
- **S1 对照 Qwen-Agent**：Qwen-Agent 内部用的也是类似 dict 注册表，方便对照。
- **S4 错误注入**：可以注入"工具执行结果替换"中间件，符合策略模式的"运行时替换"特性。

## 我们应该怎么借鉴

1. **`Tool` 基类必须有**：`name / description / parameters / run` 四件套，子类实现 `run`。
2. **注册表用 dict 而非 if-elif**：4 个工具看似可以写 `if action == "calculator": ...`，但加第 5 个工具时容易出错；dict 注册表天然支持插件式扩展。
3. **Tool 类放 `src/tools/base.py`**：所有具体工具 `from .base import Tool` 后继承，方便后续重构。
4. **registry 的 schema 列表导出**：让 prompt builder 直接遍历生成"工具描述"段落，无需硬编码。
5. **错误信息统一格式**：抛异常时 `raise ValueError(f"calculator 表达式错误：{expr}")`，Observation 里就写成 `calculator: 表达式错误：...`，模型能看出是哪个工具错了。

## 代码片段

```python
# src/tools/base.py
from abc import ABC, abstractmethod

class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict = {}  # OpenAI function calling schema
    
    @abstractmethod
    def run(self, args: dict) -> str:
        ...
    
    def to_openai_schema(self) -> dict:
        """导出 OpenAI function calling 格式的 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
```

```python
# src/tools/calculator.py
from .base import Tool

class Calculator(Tool):
    name = "calculator"
    description = "执行四则运算和常见数学函数（+ - * / sqrt sin cos log abs 等）"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "算术表达式，例如 '(123+456)*789' 或 'sqrt(2026)'",
            },
        },
        "required": ["expression"],
    }
    
    def run(self, args: dict) -> str:
        expr = args["expression"]
        # ... safe eval ...
        return str(result)

# src/tools/__init__.py
from .base import Tool
from .calculator import Calculator
from .python_sandbox import PythonSandbox
from .file_search import FileSearch
from .wiki import Wiki

ALL_TOOLS = [Calculator(), PythonSandbox(), FileSearch(), Wiki()]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
```

## 不需要借鉴的
- 复杂的策略上下文对象（Context 类）：本任务 registry 已足够。
- 策略的优先级 / 排序：模型选哪个就是哪个，不需要策略层再做选择。
