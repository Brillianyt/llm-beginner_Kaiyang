# Model Context Protocol (MCP) 协议规范笔记

## 来源
- 官方文档：https://modelcontextprotocol.io/
- Python SDK 仓库：https://github.com/modelcontextprotocol/python-sdk
- 作者/组织：Anthropic 主导，多家公司共建（2024-11 公开）
- 核心定位：**让 LLM 以统一协议接入任意数据源 / 工具的标准**，被称为「AI 应用的 USB-C 接口」

## 关键要点
1. **三层角色**：Host（运行 LLM 的进程，如 Claude Desktop）/ Client（Host 内部的连接组件）/ Server（暴露资源/工具/提示的服务）。每个 Host 可启多个 Client，每个 Client 独立连接一个 Server。
2. **JSON-RPC 2.0 底座**：所有 client-server 消息严格遵循 JSON-RPC 2.0，三种消息：Request（带 id + method）、Response（id 匹配的结果或错误）、Notification（单向、不需回复）。
3. **传输方式**：
   - `stdio`（标准输入输出）：本地进程，最简单，**最适合本任务**
   - `HTTP + SSE`（Server-Sent Events 长连接）：远程多客户端
   - 新版规范在引入 Streamable HTTP 替代旧 SSE。
4. **核心方法（tools 相关）**：
   - `tools/list`：client 询问 server 暴露的工具，server 返回 `[{name, description, inputSchema}, ...]`
   - `tools/call`：client 用 `name + arguments` 调工具，server 执行并返回结果或 JSON-RPC 错误对象。
   - 配套方法：`initialize`（握手）、`notifications/initialized`、`resources/list`、`prompts/list` 等。
5. **工具 schema 规范**：每个工具必须含 `name`（字符串，唯一）、`description`（自然语言，帮助 LLM 决策何时调用）、`inputSchema`（JSON Schema 对象，描述参数 type/properties/required）。

## 与我们任务的关联
- **M1（手写 MCP server）**：我们要么用官方 SDK 的 `FastMCP` 简化实现，要么手写 JSON-RPC over stdio。README 提示用 `list_tools()` 暴露给自检，所以我们必须保证 `from src.mcp_server import list_tools` 能拿到一个含 name 字段的列表。
- **协议选型**：本任务最小化选 stdio——MCP server 与 agent 同进程启动，agent 通过 stdin/stdout 与之对话；后续要远程化再切 HTTP+SSE。
- **测试兼容**：自检只调 `list_tools()`，但 server 本身必须能 `python src/mcp_server.py` 独立启动、跑 JSON-RPC 握手。两条路都得通。

## 代码片段（FastMCP 简化版定义工具）

```python
from mcp.server.fastmcp import FastMCP

server = FastMCP("coding-agent-server")

@server.tool()
def read_file(path: str) -> str:
    """读取工作目录内文件的全文内容。"""
    ...

@server.tool(
    name="run_tests",
    description="在工作目录里跑 pytest 并返回最后 800 字输出",
    inputSchema={
        "type": "object",
        "properties": {
            "cwd": {"type": "string", "description": "工作目录绝对路径"},
        },
        "required": ["cwd"],
    },
)
def run_tests(cwd: str) -> str:
    ...
```

## 我们应该怎么借鉴
1. **list_tools() 必须独立于 SDK**：自检从外部 `from src.mcp_server import list_tools` 枚举工具；所以即使 SDK 内部用装饰器注册，我们也要在文件顶层维护一个静态字典（name + description + input_schema），避免 `import` 副作用触发 SDK 启动。
2. **错误必须结构化**：工具抛异常时不要让 server 崩，应捕获并返回 JSON-RPC 的 `{"error": {"code": -32603, "message": str(e)}}`。
3. **路径双重校验**：每个文件工具的 `path` 参数都要先 `Path(p).resolve()`，再断言 `resolved.is_relative_to(REPO_ROOT)`，否则 LLM 输出 `../../../etc/passwd` 就能越界。
4. **stdio 启动要点**：`python src/mcp_server.py` 默认走 stdio transport；不要 print 到 stdout（污染 JSON-RPC 流），用 logging 到 stderr。
5. **可观测性**：每次 tools/call 都写一行 `log.info("call %s args=%s", name, args)`，trace 出问题时能反推。

## 主要参考来源
- MCP 官方：https://modelcontextprotocol.io/
- Python SDK：https://github.com/modelcontextprotocol/python-sdk
- 中文综述：https://www.cnblogs.com/2678066103hs/p/20065344
- 协议流程详解：https://www.cnblogs.com/jmcui/p/archive/2025/09/23