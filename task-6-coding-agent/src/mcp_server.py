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

def _build_fast_mcp_server():
    """Build a ``FastMCP`` instance with every tool registered."""
    from mcp.server.fastmcp import FastMCP  # imported lazily
    from mcp.server.fastmcp.tools import Tool

    server = FastMCP("coding-agent-tools")

    import os as _os
    _default_repo = Path(_os.environ.get(
        "CODING_AGENT_REPO_ROOT", _os.getcwd()
    )).resolve()

    for tool in _TOOL_BY_NAME.values():
        def _make_handler(t: BaseTool, repo_root: Path):
            def handler(**kwargs: Any) -> str:
                args = {k: v for k, v in kwargs.items() if v is not None}
                return t(args, repo_root).content
            handler.__name__ = t.name
            return handler

        handler = _make_handler(tool, _default_repo)
        mcp_tool = Tool.from_function(
            fn=handler,
            name=tool.name,
            description=tool.description,
        )
        mcp_tool.parameters = tool.input_schema  # type: ignore[attr-defined]
        server._tool_manager._tools[mcp_tool.name] = mcp_tool  # type: ignore[attr-defined]

    return server


def main(argv: Optional[List[str]] = None) -> None:
    log.info("starting coding-agent MCP server over stdio")
    server = _build_fast_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("server interrupted, exiting")
    except Exception:  # noqa: BLE001
        log.exception("server crashed")
        sys.exit(1)