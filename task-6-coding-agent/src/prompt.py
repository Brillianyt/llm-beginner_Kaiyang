"""System prompt construction.

Aligned with Claude Code ``src/context.ts``:

* Inject today's date so the model doesn't hallucinate old API versions.
* Inject the tool catalogue (the LLM should see every available tool).
* Inject the **Level-1 skill list** — full bodies are loaded on demand.
* Include a short reminder of the termination protocol.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Sequence

from src.mcp_server import list_tools
from src.skill_loader import SkillLoader, format_skill_list

TERMINATION_PROTOCOL = (
    "Termination protocol:\n"
    "1. Call `submit_patch` with the unified diff once you believe the fix is "
    "complete and tests are green.\n"
    "2. If you cannot make progress, call `submit_patch` with an empty "
    "string and a summary explaining why.\n"
    "3. Otherwise keep calling tools until tests pass or you hit the turn "
    "limit.\n"
)

SUBAGENT_PROTOCOL = (
    "Subagent delegation:\n"
    "- You can call `dispatch_subagent(name, task)` to delegate a focused "
    "sub-task. Two helpers are available: `code_search` (read-only) and "
    "`test_runner` (read + run tests). You will only receive the "
    "subagent's final summary string — never its internal trace.\n"
)


def build_system_prompt(
    *,
    repo_root: str,
    skill_loader: SkillLoader | None = None,
    subagent_names: Sequence[str] = (),
    max_turns: int = 30,
) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    tool_lines = "\n".join(
        f"- `{t['name']}` — {t['description']}" for t in list_tools()
    )
    skills_block = format_skill_list(skill_loader.list_skills()) if skill_loader else "(no skill loader)"
    subagent_block = (
        ", ".join(subagent_names) if subagent_names else "(none)"
    )
    return (
        f"You are a Mini Coding Agent operating in `{repo_root}` "
        f"on {today} (UTC).\n\n"
        f"You have access to the following tools (call via JSON):\n{tool_lines}\n\n"
        f"Skills available (Level 1 descriptions only; load body via "
        f"`load_skill(name)` when relevant):\n{skills_block}\n\n"
        f"Subagents you can dispatch: {subagent_block}\n\n"
        f"{SUBAGENT_PROTOCOL}"
        f"{TERMINATION_PROTOCOL}"
        f"Turn budget: {max_turns}. After every tool execution you will "
        f"receive a plain-text observation. When the work is finished (or "
        f"you decide to stop), call `submit_patch` exactly once.\n"
    )
