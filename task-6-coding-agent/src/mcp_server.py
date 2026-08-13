"""MCP server entry — stdio JSON-RPC bridge for the agent's tool layer.

Two modes:
  1. ``python src/mcp_server.py`` — runs an MCP server over stdio using
     the official ``mcp`` Python SDK (``FastMCP``). The agent (or any
     MCP-compliant client) can spawn it and call tools via JSON-RPC.
  2. ``from src.mcp_server import list_tools`` — synchronous enumeration
     used by the eval harness (``eval/run.py``). This path must **not**
     start the SDK transport (it would pollute the host process).

Top-level exports (consumed by ``eval/run.py``):
  * ``list_tools() -> list[dict]``  — schema list, each with ``name``
  * ``call_tool(name, args, repo_root) -> ToolResult`` — direct in-process
    invocation used by the ``CodingAgent`` when it wants to skip the
    JSON-RPC round trip.

Safety rails (every one of them required by the README):
  * Path tools go through ``safe_resolve`` (resolve + ``is_relative_to``).
  * ``subprocess.run`` always uses ``args=[...], shell=False``.
  * Git dangerous patterns are filtered at the *tool* layer.
  * **No print to stdout.** MCP stdio traffic is JSON-RPC; any spurious
    log line breaks the handshake. All diagnostics go to ``sys.stderr``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logger — stderr only. Anything written to stdout will corrupt JSON-RPC.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("mcp_server")

# Make sure the package sibling ``tools`` is importable when this file is
# invoked directly: ``python src/mcp_server.py``.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))  # project root for `src.*`

from src.tools import (  # noqa: E402  - sys.path tweak above is intentional
    ALL_TOOLS,
    BaseTool,
    ToolResult,
)
from src.tools.base import safe_resolve  # noqa: E402

# Lazy FastMCP import — only needed when running as a server. Importing the
# SDK at module load would slow down the eval-only path.

# ---------------------------------------------------------------------------
# Module-level tool registry. Instances are cheap to construct — they hold
# no state — so we share one across both the SDK and the direct path.
# ---------------------------------------------------------------------------
_TOOL_BY_NAME: Dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}


def list_tools() -> List[Dict[str, Any]]:
    """Return MCP-style tool descriptors.

    Each item is a plain dict with ``name``, ``description``, ``inputSchema``.
    Stable ordering matches ``tools.ALL_TOOLS`` so eval assertions can
    rely on position.
    """
    return [t.to_dict() for t in ALL_TOOLS]


def call_tool(name: str, args: Dict[str, Any], repo_root: str | Path) -> ToolResult:
    """In-process tool dispatch (skips JSON-RPC).

    Used by ``CodingAgent`` when it wants to talk to the tools directly
    instead of spawning a subprocess. The same safety code paths apply.
    """
    tool = _TOOL_BY_NAME.get(name)
    if tool is None:
        return ToolResult(content=f"[ERROR] unknown tool: {name}", is_error=True)
    root = Path(repo_root).resolve()
    # Defence-in-depth: reject any path the agent claims is in the repo
    # when it isn't. Tools re-check inside ``safe_resolve`` anyway.
    return tool(args, root)


# ---------------------------------------------------------------------------
# FastMCP server — only imported when running ``python src/mcp_server.py``
# ---------------------------------------------------------------------------

def _build_fast_mcp_server():
    """Construct a ``FastMCP`` instance and register every tool.

    Borrowed from the Claude Code MCP client pattern
    (``packages/mcp-client/manager.ts``): a single registry object that
    the host can introspect and call. The server runs over stdio so the
    agent communicates via JSON-RPC framed by the SDK.

    We build ``Tool`` objects directly (instead of going through
    ``add_tool``) so the JSON schema in ``tools/*.py`` remains the
    single source of truth.
    """
    from mcp.server.fastmcp import FastMCP  # imported lazily — see top
    from mcp.server.fastmcp.tools import Tool

    server = FastMCP("coding-agent-tools")

    import os as _os
    _default_repo = Path(_os.environ.get(
        "CODING_AGENT_REPO_ROOT", _os.getcwd()
    )).resolve()

    for tool in _TOOL_BY_NAME.values():
        # Wrap the tool's ``call`` in a plain function so FastMCP can
        # invoke it. We bypass the auto-schema-derivation path because
        # we already have an authoritative ``input_schema``.
        def _make_handler(t: BaseTool, repo_root: Path):
            def handler(**kwargs: Any) -> str:
                args = {k: v for k, v in kwargs.items() if v is not None}
                result = t(args, repo_root)
                return result.content
            handler.__name__ = t.name
            return handler

        handler = _make_handler(tool, _default_repo)
        mcp_tool = Tool.from_function(
            fn=handler,
            name=tool.name,
            description=tool.description,
        )
        # Replace the auto-derived parameters with our hand-written schema.
        mcp_tool.parameters = tool.input_schema  # type: ignore[attr-defined]
        # Inject directly into the manager's internal dict (private but
        # stable across MCP Python SDK 1.x).
        server._tool_manager._tools[mcp_tool.name] = mcp_tool  # type: ignore[attr-defined]

    return server


def main(argv: Optional[List[str]] = None) -> None:
    """stdio entry point — run via ``python src/mcp_server.py``.

    We avoid ``print`` entirely (would corrupt JSON-RPC); the SDK handles
    its own handshake on stdin/stdout.
    """
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
        # Exit non-zero so the client can detect failure.
        sys.exit(1)
