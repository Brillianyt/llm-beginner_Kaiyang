"""MCP server entry — stdio JSON-RPC bridge for the agent's tool layer.

Two modes (blueprint Part I ):

1. ``python src/mcp_server.py`` — runs an MCP server over stdio using
   the official ``mcp`` Python SDK (``FastMCP``).
2. ``from src.mcp_server import list_tools`` — synchronous enumeration
   used by ``eval/run.py``. This path must NOT start the SDK transport.

Top-level exports (consumed by ``eval/run.py``):
* ``list_tools() -> list[dict]``  — schema list, each with ``name``.
* ``call_tool(name, args, repo_root) -> ToolResult`` — direct in-process
  invocation used by ``CodingAgent`` to skip JSON-RPC.

Safety rails:
* path tools go through :func:`safe_resolve`,
* subprocess calls use ``args=[...], shell=False``,
* no print to stdout (would corrupt JSON-RPC) — log to stderr only.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("mcp_server")

# Make sure ``src.*`` is importable when this file is run directly.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from src.tools import ALL_TOOLS, BaseTool, ToolResult  # noqa: E402

# Lazy FastMCP import — only needed when running as a server.

_TOOL_BY_NAME: Dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}


# ---------------------------------------------------------------------------
# In-process API (used by eval harness + CodingAgent)
# ---------------------------------------------------------------------------

def list_tools() -> List[Dict[str, Any]]:
    """Return MCP-style tool descriptors."""
    return [t.to_dict() for t in ALL_TOOLS]


def call_tool(name: str, args: Dict[str, Any], repo_root: str | Path) -> ToolResult:
    """Direct tool dispatch — bypasses JSON-RPC."""
    tool = _TOOL_BY_NAME.get(name)
    if tool is None:
        return ToolResult(content=f"[ERROR] unknown tool: {name}", is_error=True)
    root = Path(repo_root).resolve()
    return tool(args, root)


# ---------------------------------------------------------------------------
# FastMCP server entry point
# ---------------------------------------------------------------------------

def _build_mcp_server():
    """Build an ``mcp.server.Server`` instance with every tool registered.

    Uses the public SDK types — ``mcp.Tool`` for descriptors and the
    ``list_tools`` / ``call_tool`` request handlers. Survives SDK
    internal renames because we only touch documented attributes.
    """
    import asyncio
    from mcp import Tool
    from mcp.server import Server

    server: Server = Server("coding-agent-tools")

    import os as _os
    _default_repo = Path(_os.environ.get(
        "CODING_AGENT_REPO_ROOT", _os.getcwd()
    )).resolve()

    @server.list_tools()
    async def _handle_list_tools() -> list[Tool]:
        return [
            Tool(
                name=t.name,
                description=t.description,
                inputSchema=t.input_schema,
            )
            for t in _TOOL_BY_NAME.values()
        ]

    @server.call_tool()
    async def _handle_call_tool(name: str, arguments: dict):
        tool = _TOOL_BY_NAME.get(name)
        if tool is None:
            return [{"type": "text", "text": f"[ERROR] unknown tool: {name}"}]
        # Boundary schema validation — surface bad arguments at the MCP
        # layer instead of letting them reach the tool and explode
        # mid-execution. The tool's own ``inputSchema`` is the contract.
        try:
            import jsonschema  # type: ignore
            jsonschema.validate(instance=arguments or {}, schema=tool.input_schema)
        except ImportError:
            pass  # jsonschema is optional — tool will validate defensively
        except jsonschema.ValidationError as e:
            return [{
                "type": "text",
                "text": f"[ERROR] invalid arguments for {name}: {e.message}",
            }]
        result = tool(arguments or {}, _default_repo)
        return [{"type": "text", "text": result.content}]

    return server


def main(argv: Optional[List[str]] = None) -> None:
    """stdio entry point — run via ``python src/mcp_server.py``.

    We avoid ``print`` entirely (would corrupt JSON-RPC); the SDK handles
    its own handshake on stdin/stdout.
    """
    import asyncio
    from mcp.server.stdio import stdio_server

    log.info("starting coding-agent MCP server over stdio")
    server = _build_mcp_server()

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("server interrupted, exiting")
    except Exception:  # noqa: BLE001
        log.exception("server crashed")
        sys.exit(1)