# 六边形架构（Hexagonal / Ports & Adapters）

## 来源
- 起源：Alistair Cockburn 2005 年提出（https://alistair.cockburn.eu/hexagonal-architecture/）
- 业界演化：Vaughn Vernon《Implementing Domain-Driven Design》、Uncle Bob《Clean Architecture》
- Agent 应用：Microsoft Azure Architecture Center「Agentic Design Patterns」、Martin Fowler 文章

## 关键要点
1. **核心思想**：领域核心（domain core）独立于外部技术细节；外部通过「端口（ports）」与核心交互；具体实现是「适配器（adapters）」
2. **三要素**：
   - **Domain Core**（领域核心）：业务规则、纯逻辑，不依赖任何外部库
   - **Ports**（端口）：核心对外的接口契约（abstract base class / Protocol）
   - **Adapters**（适配器）：实现端口的具体技术（DB、HTTP、MCP、LLM provider）
3. **依赖方向**：Adapters → Ports ← Domain Core（核心不依赖适配器）
4. **优势**：可测试（mock 适配器）、可替换（换 LLM provider 不改核心）、可演化（加新适配器不影响业务）
5. **在 Agent 系统的映射**：
   - Domain Core = Agent 推理循环（"该调什么 tool"）
   - Ports = Tool 接口（name / description / schema / call）
   - Adapters = MCP client / Ollama / vLLM / OpenAI client / 文件系统

## 与我们任务的关联
- **M1 / M3 的接口设计**：Tool 接口就是 Port；MCP server / OpenAI client 都是 Adapter
- **M2 (Skill 加载器)**：SkillLoader 是 Domain Core 的能力扩展（也是 Port）；YAML 解析器、文件系统读写是 Adapter
- **M4 (Trace)**：Trace 数据结构是 Domain Core 的输出，序列化到 JSON / Parquet 是 Adapter
- **测试**：单元测试可以用 InMemoryAdapter（MockToolRegistry），不需要真起 MCP server

## 文字版架构图

```
                    ┌──────────────────────────┐
                    │      Domain Core         │
                    │  (CodingAgent.run 循环)  │
                    │  决策逻辑 + 状态机        │
                    └────────────┬─────────────┘
                                 │
                          ┌──────┴───────┐
                          │ Ports        │
                          │  - Tool       │
                          │  - Skill      │
                          │  - Subagent   │
                          │  - LLM client │
                          └──────┬───────┘
                                 │
        ┌─────────────────┬──────┼──────┬─────────────────┐
        │                 │      │      │                 │
        ▼                 ▼      ▼      ▼                 ▼
┌──────────────┐  ┌─────────────┐ ┌────────────┐ ┌──────────────┐
│ MCPClient    │  │ OpenAI API  │ │ FileSystem │ │ InMemory     │
│ (stdio/HTTP) │  │ (Ollama/    │ │ (read/     │ │ (mock for    │
│              │  │  vLLM)      │ │  write)    │ │  unit test)  │
└──────────────┘  └─────────────┘ └────────────┘ └──────────────┘
    Adapters
```

## 代码片段（Python Protocol 实现 Ports）

```python
# src/ports.py —— 端口定义（领域核心需要的接口）
from typing import Protocol, runtime

class Tool(Protocol):
    name: str
    description: str
    input_schema: dict
    def call(self, args: dict, ctx: "ToolContext") -> str: ...

class Skill(Protocol):
    name: str
    description: str
    def load(self) -> str: ...  # 完整 markdown 正文

class Subagent(Protocol):
    def run(self, task: str) -> str: ...  # 返回摘要

class LLMClient(Protocol):
    def chat(self, messages: list, tools: list[dict]) -> "ChatResponse": ...

class FileSystemAdapter(Protocol):
    def read(self, path: str) -> str: ...
    def write(self, path: str, content: str) -> None: ...


# src/adapters/real_fs.py —— 文件系统适配器（实现 port）
class RealFileSystem:
    def __init__(self, repo_root: Path):
        self.root = repo_root.resolve()
    def read(self, path: str) -> str:
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root):
            raise PermissionError("path traversal")
        return p.read_text(encoding="utf-8")
    def write(self, path: str, content: str) -> None:
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root):
            raise PermissionError("path traversal")
        p.write_text(content, encoding="utf-8")

# src/adapters/mock_fs.py —— 内存适配器（单元测试用）
class InMemoryFileSystem:
    def __init__(self): self._files = {}
    def read(self, path): return self._files[path]
    def write(self, path, content): self._files[path] = content

# src/agent.py —— 领域核心（只依赖 Port，不依赖具体 Adapter）
class CodingAgent:
    def __init__(self, llm: LLMClient, tools: list[Tool], fs: FileSystemAdapter):
        self.llm = llm; self.tools = tools; self.fs = fs  # 通过构造函数注入
    def run(self, issue: str) -> Trace: ...
```

## 我们应该怎么借鉴
1. **Ports 必须先于 Adapters 定义**：在 `src/ports.py` 里写 `class Tool(Protocol)`、`class LLMClient(Protocol)`；具体实现在 `src/adapters/`
2. **CodingAgent 注入而非 import**：`__init__(self, llm, tools, fs)`，避免 `from openai import OpenAI` 这种硬依赖
3. **单测用 InMemory**：写 `class FakeLLM(LLMClient)`，预录 response JSON，断言 tool call 顺序
4. **路径校验放在 Adapter**：`RealFileSystem.read/write` 内部做 `is_relative_to` 校验；不让 Domain Core 关心路径安全
5. **MCP 也是 Adapter**：CodingAgent 不应该知道「有 MCP server」这回事；只看到 `list[Tool]` 接口；MCP client 是个 Adapter，把 JSON-RPC 调用包成 Tool 调用
6. **演进路径**：v1 可以只有 RealFileSystem + RealOpenAIClient；v2 加 InMemoryFileSystem 给单测；v3 加 MCPClientAdapter 替代直接调 SDK
7. **不要过度抽象**：Protocol 适用于多实现场景（FS / LLM）；Tool 不需要 Protocol——它就是普通类（有 call 方法即可）

## 主要参考来源
- Alistair Cockburn 原文：https://alistair.cockburn.eu/hexagonal-architecture/
- Uncle Bob Clean Architecture：https://blog.cleancoder.com/uncle-bob/2012/08/13/clean-architecture.html
- Microsoft Azure Agentic Design Patterns：https://learn.microsoft.com/en-us/azure/architecture/agentic-design/agentic-design-patterns
- Claude Code `packages/mcp-client/interfaces.ts`（用 TypeScript interface 实现依赖倒置）