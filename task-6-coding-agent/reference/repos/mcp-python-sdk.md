# MCP Python SDK 学习笔记

## 来源
- 仓库：https://github.com/modelcontextprotocol/python-sdk
- 文档：https://modelcontextprotocol.io/
- 维护者：Anthropic + 社区
- 核心定位：**Python 实现 MCP 协议的官方 SDK**，含 client、server、stdio/HTTP transport

## 关键要点
1. **三层 API**：
   - 底层 `mcp.server.Server`：手工处理 JSON-RPC，灵活但啰嗦
   - 中层 `mcp.server.stdio.stdio_server`：用 stdio 与 client 通信的 transport
   - 高层 `mcp.server.fastmcp.FastMCP`：装饰器风格，自动从类型注解和 docstring 生成 schema（**最推荐**）
2. **FastMCP 用法核心**：实例化 → `@server.tool()` 装饰器注册工具 → `server.run(transport="stdio")` 启动。
3. **schema 自动生成**：工具函数的 Python 类型注解 + docstring 被转成 JSON Schema。`def read_file(path: str) -> str` 自动得到 `{"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}`。
4. **错误处理**：`@server.tool()` 装饰的函数抛异常时，server 自动捕获并包装成 JSON-RPC error 响应（不 crash）。但抛之前最好在函数内部 `try/except` 转成更有信息量的字符串。
5. **异步 vs 同步**：FastMCP 同时支持 `def` 和 `async def`；stdio server 默认跑在 asyncio 事件循环里，但同步工具也能直接注册（内部用 `asyncio.to_thread` 跑）。

## 与我们任务的关联
- **M1（手写 MCP server）**：直接用 FastMCP 能省 80% 样板代码；但 README 要求模块顶层导出 `list_tools()`，所以我们要在 `mcp_server.py` 里**另写一个静态字典**（同步维护工具元数据），供自检 import。
- **stdio transport**：CodingAgent 启动时把 MCP server 作为子进程拉起（`subprocess.Popen(["python", "src/mcp_server.py"], stdin=PIPE, stdout=PIPE)`），用 JSON-RPC 消息与之通信。

## 代码片段（FastMCP 的极简 server）

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("coding-agent-server")

@mcp.tool()
def read_file(path: str) -> str:
    """Read the full content of a file in the working repository."""
    p = (REPO_ROOT / path).resolve()
    assert p.is_relative_to(REPO_ROOT), "path traversal blocked"
    return p.read_text(encoding="utf-8")

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Overwrite a file in the working repository."""
    p = (REPO_ROOT / path).resolve()
    assert p.is_relative_to(REPO_ROOT)
    p.write_text(content, encoding="utf-8")
    return "ok"

# 也维护一个静态 list_tools() 供自检 import
def list_tools() -> list[dict]:
    return [
        {"name": "read_file", "description": "...", "inputSchema": {...}},
        {"name": "write_file", ...},
        ...
    ]

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## 我们应该怎么借鉴
1. **优先 FastMCP**：减少重复样板，专注于工具逻辑本身。
2. **`list_tools()` 与装饰器分离**：用装饰器注册的工具有 SDK 自己的查询函数（比如 `mcp._tool_manager.list_tools()`），但**不要在自检里 import 这个 SDK 内部对象**——直接维护一个本地 list/dict，自检只看你模块顶层的导出。这样也方便你自定义 description。
3. **tool 函数必须是「纯」调用**：不要在装饰器函数里 print 到 stdout（污染 JSON-RPC）；用 logging 到 stderr。
4. **path 参数必须 resolve**：FastMCP 不帮你做沙箱校验，得自己加 `Path.resolve().is_relative_to(ROOT)`。
5. **错误透传**：工具内部出错时 `raise ValueError(f"read_file failed: {e}")`，SDK 会自动包装为 JSON-RPC error payload。
6. **启动模式双轨**：
   - 自检：`from src.mcp_server import list_tools` —— 不应启动子进程
   - 实战：`python src/mcp_server.py` —— 启动 stdio server；用 `if __name__ == "__main__"` 分隔

## 主要参考来源
- SDK 仓库：https://github.com/modelcontextprotocol/python-sdk
- 官方文档：https://modelcontextprotocol.io/
- FastMCP API 参考：`/src/mcp/server/fastmcp.py`（在 SDK 源码内）