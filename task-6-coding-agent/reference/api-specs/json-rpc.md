# JSON-RPC 2.0 协议规范（精简版）

## 来源
- 官方规范：https://www.jsonrpc.org/specification
- MCP 协议基于此：https://modelcontextprotocol.io/

## 关键要点
1. **JSON-RPC 2.0 是无状态、轻量级 RPC 协议**——用 JSON 编码、单方法调用、批量支持
2. **请求**（必须有 `id`）：
   ```json
   {"jsonrpc": "2.0", "id": 1, "method": "subtract", "params": {"minuend": 42, "subtrahend": 23}}
   ```
3. **响应**：
   - 成功：`{"jsonrpc": "2.0", "id": 1, "result": 19}`
   - 失败：`{"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Method not found"}}`
4. **Notification**（无 `id`，无需响应）：
   ```json
   {"jsonrpc": "2.0", "method": "update", "params": [1,2,3,4,5]}
   ```
5. **错误码**：
   - `-32700` ParseError —— JSON 解析失败
   - `-32600` InvalidRequest —— JSON 合法但不符合 JSON-RPC 规范
   - `-32601` MethodNotFound
   - `-32602` InvalidParams
   - `-32603` InternalError
   - `-32000` 到 `-32099` —— Server error（自定义应用层错误）
6. **批量**：`[]` 里放多个请求；server 可并发处理
7. **stdin/stdout 传输约定**（line-delimited JSON）：
   - 每个 JSON 对象占一行（`\n` 结尾）
   - stdout 只能写 JSON 消息；其他输出走 stderr

## 与我们任务的关联
- **M1（手写 MCP server）**：MCP 用 JSON-RPC over stdio；要理解请求/响应格式才能调试协议层错误
- **自检只调 `list_tools()`**：实战才走完整协议（initialize → list → call）
- **错误处理**：工具抛异常要包装成 JSON-RPC error 对象（`{"code": -32603, "message": str(e)}`）

## 代码片段（Python JSON-RPC 极简实现）

```python
import json
import sys
from typing import Any, Callable

class JSONRPCServer:
    """简化版 JSON-RPC over stdio 服务器（理解协议用，不用于生产）。"""
    def __init__(self):
        self.handlers: dict[str, Callable] = {}

    def register(self, method: str, handler: Callable):
        self.handlers[method] = handler

    def handle_request(self, req: dict) -> dict | None:
        if req.get("jsonrpc") != "2.0":
            return self._error(None, -32600, "Invalid Request")
        if "id" not in req:
            # notification 不响应
            self._dispatch(req)
            return None
        try:
            result = self._dispatch(req)
            return {"jsonrpc": "2.0", "id": req["id"], "result": result}
        except Exception as e:
            return self._error(req["id"], -32603, str(e))

    def _dispatch(self, req):
        method = req["method"]
        if method not in self.handlers:
            raise KeyError(f"method not found: {method}")
        return self.handlers[method](req.get("params", {}))

    def _error(self, req_id, code, message):
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    def serve_forever(self, stdin=sys.stdin, stdout=sys.stdout):
        for line in stdin:
            line = line.strip()
            if not line: continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                if resp is not None:
                    stdout.write(json.dumps(resp) + "\n")
                    stdout.flush()
            except json.JSONDecodeError:
                err = self._error(None, -32700, "Parse error")
                stdout.write(json.dumps(err) + "\n")
                stdout.flush()


# 使用示例
server = JSONRPCServer()
server.register("tools/list", lambda params: [{"name": "read_file", ...}])
server.register("tools/call", lambda params: "file contents...")
server.serve_forever()
```

## 我们应该怎么借鉴
1. **理解协议但不自己实现**：用官方 SDK 的 FastMCP；自己实现 JSON-RPC 层是 over-engineering
2. **错误透传**：工具抛异常 → SDK 包装为 JSON-RPC error（`code: -32603`）；不要让 server 进程退出
3. **Notification vs Request 区分**：无 `id` 字段的是 notification；不响应即可
4. **协议版本**：MCP 协议版本（如 `2024-11-05`）在 `initialize` 里声明；server 端不必强制校验
5. **stdio 注意事项**：
   - stdout 严格只写 JSON
   - log 走 stderr
   - 用 `bufsize=1` 行缓冲；`flush()` 后等待
6. **批量请求**：MCP 不要求支持；如有可忽略

## 主要参考来源
- JSON-RPC 2.0 规范：https://www.jsonrpc.org/specification
- MCP 协议（基于 JSON-RPC）：https://modelcontextprotocol.io/
- Claude Code `packages/mcp-client/transport/`：生产级 transport 实现