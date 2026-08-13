# MCP 协议细节（JSON-RPC 2.0 over stdio）

## 来源
- 官方文档：https://modelcontextprotocol.io/
- JSON-RPC 2.0 规范：https://www.jsonrpc.org/specification
- Anthropic SDK：https://github.com/modelcontextprotocol/python-sdk
- Claude Code 源码：`packages/mcp-client/`

## 关键要点
1. **基础**：MCP = JSON-RPC 2.0 + transport 层 + 预定义方法
2. **JSON-RPC 消息三种**：
   - **Request**：`{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {...}}`
   - **Response**：`{"jsonrpc": "2.0", "id": 1, "result": {...}}` 或 `{"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "..."}}`
   - **Notification**：`{"jsonrpc": "2.0", "method": "notifications/initialized"}`（无 id，无需回复）
3. **核心方法**：
   - `initialize` —— 握手，client 发 capabilities，server 回 serverInfo
   - `tools/list` —— client 列 server 暴露的工具，返回 `[{name, description, inputSchema}, ...]`
   - `tools/call` —— client 调工具，`params: {name, arguments}`，server 执行后返回 `CallToolResult` 或 error
   - `notifications/initialized` —— client 确认握手完成
4. **stdio transport 协议**：
   - client 起 server 子进程：`subprocess.Popen(["python", "mcp_server.py"], stdin=PIPE, stdout=PIPE, stderr=PIPE)`
   - 每个 JSON-RPC 消息占一行，以 `\n` 结尾（line-delimited JSON）
   - stdout 只能写 JSON-RPC 消息；log 写到 stderr
5. **HTTP/SSE transport**（不用于本任务）：
   - server 暴露 HTTP endpoint
   - client 发 POST → server 通过 SSE 流式返回
6. **错误码**：
   - `-32700` ParseError —— JSON 解析失败
   - `-32600` InvalidRequest
   - `-32601` MethodNotFound
   - `-32602` InvalidParams
   - `-32603` InternalError
   - `-32000` 到 `-32099` —— Server error（自定义）

## 与我们任务的关联
- **M1（手写 MCP server）**：最简实现用 stdio + FastMCP；自检只调 `list_tools()`；实战要能跑 JSON-RPC 握手
- **stdio 陷阱**：server 不能 print 到 stdout；log 必须到 stderr
- **工具 schema 完整性**：每个 tool 必须有 `name`/`description`/`inputSchema` 三个字段；缺一自检失败

## 文字版协议交互流

```
Client                                Server (MCP)
  │                                          │
  │ ─── {"method":"initialize","params":{...}} ───> │
  │ <── {"result":{"serverInfo":{...}}} ──── │
  │                                          │
  │ ─── {"method":"notifications/initialized"} ───> │
  │                                          │
  │ ─── {"method":"tools/list","id":2} ───> │
  │ <── {"result":{"tools":[{name, description, inputSchema}, ...]}} ──── │
  │                                          │
  │ ─── {"method":"tools/call","params":{"name":"read_file","arguments":{"path":"x.py"}}} ───> │
  │ <── {"result":{"content":[{"type":"text","text":"文件内容..."}]}} ──── │
```

## 代码片段（Python MCP Client + Server 最小实现）

```python
# src/mcp_server.py —— 用 FastMCP 简化
from mcp.server.fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP("coding-agent")
REPO_ROOT = Path(__file__).parent.parent / "data" / "toy-repo"

@mcp.tool()
def read_file(path: str) -> str:
    """读取仓库内文件全文。"""
    p = (REPO_ROOT / path).resolve()
    if not p.is_relative_to(REPO_ROOT):
        raise PermissionError(f"path traversal blocked: {path}")
    return p.read_text(encoding="utf-8")

# 模块顶层导出 list_tools() 供自检 import
def list_tools() -> list[dict]:
    return [
        {"name": "read_file", "description": "...", "inputSchema": {...}},
        {"name": "write_file", ...},
        ...
    ]

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

```python
# src/mcp_client.py —— 与 server 通信（简化版）
import json
import subprocess
from typing import Any

class MCPStdioClient:
    def __init__(self, server_cmd: list[str]):
        self.proc = subprocess.Popen(
            server_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self._id = 0

    def _send(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params: msg["params"] = params
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        # 读一行响应
        return json.loads(self.proc.stdout.readline())

    def initialize(self):
        return self._send("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})

    def list_tools(self):
        return self._send("tools/list")["result"]["tools"]

    def call_tool(self, name: str, args: dict):
        resp = self._send("tools/call", {"name": name, "arguments": args})
        if "error" in resp:
            raise RuntimeError(f"tool call failed: {resp['error']}")
        return resp["result"]
```

## 我们应该怎么借鉴
1. **优先用 FastMCP**：避免手写 JSON-RPC；装饰器自动生成 schema
2. **`list_tools()` 与装饰器分离**：自检只 import `list_tools()`；不要触发 SDK 启动
3. **stdio server 不 print 到 stdout**：log 走 stderr；否则 JSON-RPC 流被污染
4. **错误用 JSON-RPC error 对象**：不要让 server 崩；抛异常让 SDK 包装成 error payload
5. **路径校验放在 tool 函数内**：`(REPO_ROOT / path).resolve().is_relative_to(REPO_ROOT)`，**双重断言**
6. **超时与 timeout**：每个 tool 内部用 `subprocess.run(timeout=60)`；MCP 协议层可加额外超时
7. **测试时不启 server**：自检走 `import list_tools`；不真起 stdio；端到端测试才真起

## 主要参考来源
- MCP 官方：https://modelcontextprotocol.io/
- JSON-RPC 2.0：https://www.jsonrpc.org/specification
- Python SDK：https://github.com/modelcontextprotocol/python-sdk
- 中文 MCP 完全指南：https://www.cnblogs.com/2678066103hs/p/20065344
- Claude Code `packages/mcp-client/`：生产级实现参考