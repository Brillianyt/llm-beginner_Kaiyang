# 插件架构与工具注册

## 来源
- 模式分类：软件架构模式（Plugin Architecture）
- 应用领域：VS Code 扩展、webpack 插件、Django middleware、agent tool plugin
- 相关参考：OpenAI 的 Plugin 协议（已弃用但概念沿用）、MCP（Model Context Protocol）

## 关键要点

### 1. 核心思想
定义稳定的核心接口，让第三方/后续代码能"即插即用"扩展功能，而无需修改核心代码。本任务的"核心"是 ReActAgent 主循环，"插件"是 4 个工具。

### 2. 插件三要素
- **接口契约**：`Tool` 基类定义 `run(args)` 签名。
- **注册机制**：dict registry 或 decorator 自动注册。
- **发现机制**：启动时遍历某个目录、import 某个模块列表、或显式 `registry.register(...)`。

### 3. 三种发现机制的对比

| 方式 | 实现 | 优缺点 |
|---|---|---|
| 显式注册 | `registry.register(MyTool())` | 简单、可控，但每加一个工具要改一处代码 |
| 装饰器自动注册 | `@register` | 简洁，但要小心 import 顺序（必须先 import 才能注册） |
| 目录扫描 | `pkgutil.iter_modules(['src/tools'])` | 真正"即插即用"，但引入隐式行为 |

本任务 4 个工具固定不变，**显式注册**最合适（最易调试）。

### 4. 插件架构与策略模式的关系
- 策略模式是 OO 层面的"算法可替换"。
- 插件架构是系统层面的"功能可扩展"。

本任务的 4 个工具**同时是策略（不同算法）也是插件（不同模块）**。这是合理的——视角不同。

### 5. 插件元数据
现代插件系统通常带元数据：
- `name`、`version`：用于日志、调试
- `description`、`author`：可读性
- `parameters schema`：模型可见

我们 `Tool` 基类已经有 name / description / parameters，相当于把元数据和接口契约合在一起。

## 与我们任务的关联

- **M1**：4 个工具模块就是 4 个插件，符合架构预期。
- **加分 S1**：Qwen-Agent 也是插件架构，加新工具不用改 agent 类——可以验证我们的设计是否对得上。
- **加分 S3**：prompt 模板里"工具列表"是从 registry 动态拼出来的，正是插件架构的体现。
- **加分 S4**：`inject_error` 钩子就是"中间件式插件"——拦截工具调用、注入错误。

## 我们应该怎么借鉴

1. **`src/tools/` 目录 + `__init__.py` 集中导出**：每个工具一个文件，主 agent 只 `from src.tools import ALL_TOOLS`，不关心具体哪些工具。
2. **`Tool` 基类定义契约**：`name`、`description`、`parameters`、`run(args) -> str` 四个字段/方法必须都有。
3. **`run` 必须 `args: dict` 入口**：内部自己 `args["expression"]` 取参数，符合 README 要求的固定参数键。
4. **不引入自动扫描**：4 个工具显式列在 `__init__.py` 里。
5. **预留扩展点**：未来要加"web_search"、"email"工具，只要 `src/tools/web_search.py` 定义 + `__init__.py` 加一行——这是插件架构的承诺。
6. **trace 里记录工具来源模块**：调试时能定位"是 calculator 抛的还是 python_sandbox 抛的"。

## 代码片段（插件注册中心模式）

```python
# src/tools/__init__.py —— 插件注册中心
from .base import Tool, ToolRegistry
from .calculator import Calculator
from .python_sandbox import PythonSandbox
from .file_search import FileSearch
from .wiki import Wiki

# 默认注册：4 个内置工具
DEFAULT_TOOLS = [Calculator(), PythonSandbox(), FileSearch(), Wiki()]

def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for t in DEFAULT_TOOLS:
        reg.register(t)
    return reg
```

## 不需要借鉴的
- 运行时动态加载（`importlib.import_module`）：4 个工具用不到。
- 插件签名校验：固定参数键已在 README 里规定，无需 plugin manifest。
- MCP 协议：over-engineering。
