"""System prompt assembly.

Per blueprint Part IV §4.2 step 1, the prompt is *deterministic*:

* today's UTC date,
* MCP tool catalogue (terse descriptions),
* Level-1 skill index (built via ``SkillLoader.system_prompt_section``),
* subagent catalogue,
* termination protocol.

Host-side composition makes the prompt cache-friendly across runs.
The prompt is intentionally concise: long prompts cause Qwen2.5-Coder-7B
to spin (re-call the same tool) instead of progressing.  See
``reference/repos/claude-code.md`` for the patterns we mirror — terse
worker prompt, terse tool descriptions, clear termination.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from src.mcp_server import list_tools
from src.skill_loader import SkillLoader

# Tool descriptions live in ``src/tools/*.py`` and can be verbose.  For
# the system prompt we want short summaries so the model doesn't get
# distracted.  Override the verbose defaults here.
_TERSE_TOOL_DESC: dict[str, str] = {
    "read_file": (
        "Read a file. Header is 1-based; if truncated, re-call with "
        "`offset=K` to continue."
    ),
    "write_file": (
        "Write a UTF-8 file. Returns a unified diff. Must have been "
        "read in the same session."
    ),
    "edit": (
        "Exact string replacement. `old_string` must match uniquely "
        "unless `replace_all=true`. File must have been read first."
    ),
    "list_files": "List files under a directory (skip .git).",
    "grep": "ripgrep search for a pattern across the repo.",
    "run_tests": (
        "Run pytest and report pass/fail counts plus a failures[]. "
        "Args: cmd, cwd, extra_args, timeout_s."
    ),
    "git_diff": "`git diff` working tree vs HEAD. Per-file unified diffs.",
    "git_apply": (
        "`git apply` a unified diff. Default dry_run=true. Refuses "
        "paths outside the repo."
    ),
}


def _tool_line(t: dict) -> str:
    name = t["name"]
    desc = _TERSE_TOOL_DESC.get(name) or t["description"].split(".")[0] + "."
    return f"- `{name}` — {desc}"


TERMINATION_PROTOCOL = (
    "Termination:\n"
    "1. Fix complete? → call `submit_patch(diff, summary)` exactly once.\n"
    "2. Stuck? → call `submit_text(text)` to stop.\n"
    "3. Never edit test files (`test_*.py`, `*_test.py`, `*/tests/*`).\n"
)

SUBAGENT_PROTOCOL = (
    "Subagents: `dispatch_subagent(name, task)` runs a sub-agent and "
    "returns only its final plain-text summary. Available: "
    "`search_executor` (read-only exploration), `test_executor` (pytest).\n"
)


def build_system_prompt(
    *,
    repo_root: str,
    skill_loader: SkillLoader | None = None,
    subagent_names: Sequence[str] = (),
    max_turns: int = 50,
) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    tool_lines = "\n".join(_tool_line(t) for t in list_tools())
    skills_block = (
        skill_loader.system_prompt_section() if skill_loader else "(none)"
    )
    subagent_block = ", ".join(subagent_names) if subagent_names else "(none)"
    return (
        f"You are a Mini Coding Agent operating in `{repo_root}` on {today} (UTC).\n"
        f"Workflow: explore → read → edit → test → submit_patch. "
        f"Use absolute paths. ONE tool call per assistant message.\n\n"
        f"## Tools\n{tool_lines}\n\n"
        f"## Skills (Level-1 — load body via `load_skill`)\n{skills_block}\n\n"
        f"## Subagents\n{subagent_block}\n\n"
        f"{SUBAGENT_PROTOCOL}"
        f"{TERMINATION_PROTOCOL}"
        f"Turn budget: {max_turns}. When done, call `submit_patch` once.\n"
    )