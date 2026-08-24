"""System prompt assembly.

Per blueprint Part IV §4.2 step 1, the prompt is *deterministic*:

* today's UTC date,
* MCP tool catalogue,
* Level-1 skill index (built via ``SkillLoader.system_prompt_section``),
* subagent catalogue,
* termination protocol.

Host-side composition makes the prompt cache-friendly across runs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Sequence

from src.mcp_server import list_tools
from src.skill_loader import SkillLoader

TERMINATION_PROTOCOL = (
    "Termination protocol:\n"
    "1. When you believe the fix is complete and tests are green, call "
    "`submit_patch` exactly once with the unified diff and a one-line summary.\n"
    "2. If you cannot make progress, call `submit_text` to stop.\n"
    "3. Never edit test files (`test_*.py`, `*_test.py`, `*/tests/*`).\n"
    "\n"
    "How to call a tool (IMPORTANT — read carefully):\n"
    "- Every tool invocation MUST be a single JSON object with exactly two "
    "  keys: `\"name\"` (the tool name) and `\"arguments\"` (an object).\n"
    "- Example: {\"name\": \"read_file\", \"arguments\": {\"file_path\": \"/abs/path\"}}\n"
    "- Do NOT wrap the JSON in a fenced ``` ``` block — output it raw on its own line.\n"
    "- Do NOT output Python code that calls the tool — output the JSON above.\n"
    "- Do NOT use markdown headings or prose between tool calls; the next "
    "  message you send should be the JSON for the next tool call.\n"
    "- For `write_file`, the `content` argument is a JSON string — escape "
    "  internal newlines as `\\n` and internal quotes as `\\\"`.\n"
    "- After observing the result, decide whether to call another tool or "
    "  call `submit_patch` / `submit_text`.\n"
)

SUBAGENT_PROTOCOL = (
    "Subagent delegation:\n"
    "- `dispatch_subagent(name, task)` launches a sub-agent with its own "
    "message history. You will receive only its final plain-text summary.\n"
    "- Available sub-agents:\n"
    "    * `search_executor` — read-only code search (read_file, grep).\n"
    "    * `test_executor`  — runs pytest + reports structured failures.\n"
)


def build_system_prompt(
    *,
    repo_root: str,
    skill_loader: SkillLoader | None = None,
    subagent_names: Sequence[str] = (),
    max_turns: int = 50,
) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    tool_lines = "\n".join(
        f"- `{t['name']}` — {t['description']}" for t in list_tools()
    )
    skills_block = (
        skill_loader.system_prompt_section() if skill_loader else "(no skill loader)"
    )
    subagent_block = ", ".join(subagent_names) if subagent_names else "(none)"
    return (
        f"You are a Mini Coding Agent operating in `{repo_root}` on {today} (UTC).\n\n"
        f"## Available tools (call via JSON)\n{tool_lines}\n\n"
        f"## Available skills (Level-1 index — load body on demand via `load_skill`)\n"
        f"{skills_block}\n\n"
        f"## Subagents you can dispatch\n{subagent_block}\n\n"
        f"## Hard rules\n"
        f"- Always use absolute file paths.\n"
        f"- Read before you write: `Edit` / `Write` will fail if you haven't read the file.\n"
        f"- Don't modify test files — the hook will reject the call.\n"
        f"- Independent tool calls go in one message (parallel).\n"
        f"- Prefer `Edit` (small diff) over `Write` (full overwrite).\n\n"
        f"{SUBAGENT_PROTOCOL}"
        f"{TERMINATION_PROTOCOL}"
        f"Turn budget: {max_turns}. After every tool call you'll receive the "
        f"plain-text result. When done, call `submit_patch` exactly once.\n"
    )