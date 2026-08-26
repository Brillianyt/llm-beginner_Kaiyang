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
    "list_files": (
        "List files under a directory (skip .git). Use when you need to "
        "see what files exist in a directory without knowing specific "
        "filenames."
    ),
    "grep": (
        "ripgrep search tool. **ALWAYS use `grep` for search tasks — never "
        "shell out to `rg` via `run_bash`.** Supports full regex; literal "
        "braces need escaping (`interface\\{\\}`). Output modes: "
        "`files_with_matches` (default, paths only — best for locating the "
        "buggy file), `content` (matching lines with `context=N`), `count`. "
        "Filter with `glob='*.py'`/`type='py'` or narrow `path`. Pass "
        "`-i=true` for case-insensitive (useful for case-sensitivity bugs). "
        "`head_limit` defaults to 250; tool reports when results truncated."
    ),
    "run_tests": (
        "Run pytest. Pass the FAIL_TO_PASS test path in `extra_args` "
        "(e.g. ['/abs/path/to/test_foo.py::test_specific']); otherwise "
        "it defaults to the package's `tests/` dir and may run "
        "unrelated tests that fail for unrelated reasons."
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


SUBAGENT_PROTOCOL = (
    "Subagents: `dispatch_subagent(name, task)` runs a sub-agent and "
    "returns only its final plain-text summary. Available: "
    "`search_executor` (read-only exploration), `test_executor` (pytest).\n"
)


TERMINATION_PROTOCOL = (
    "Termination:\n"
    "1. Fix complete? → call `submit_patch(diff, summary)` exactly once.\n"
    "2. Stuck? → call `submit_text(text)` to stop.\n"
    "3. Never edit test files (`test_*.py`, `*_test.py`, `*/tests/*`).\n"
    "\n"
    "Edit discipline:\n"
    "- Make the MINIMUM change that fixes the bug. NEVER rename\n"
    "  existing functions, NEVER add recursive variants, NEVER\n"
    "  refactor surrounding code. The fix is almost always a\n"
    "  one-line change to a buggy existing line.\n"
    "- If `edit` fails with 'old_string not found', DO NOT keep\n"
    "  retrying the same edit. Re-read the file and find the EXACT\n"
    "  line (including leading whitespace) you want to change.\n"
    "- After your edit succeeds, call `run_tests` to verify. If\n"
    "  the FAIL_TO_PASS test passes, STOP editing immediately and\n"
    "  call `submit_patch`. Do not make further edits.\n"
    "- Bug-location heuristics: read the file; the issue text often\n"
    "  names the buggy function. For matrix / numeric bugs check\n"
    "  indexing and `np.zeros` initial-value assignment; for\n"
    "  case-sensitivity bugs prefer `re.IGNORECASE` on `re.compile`.\n"
    "\n"
    "Locate-before-test:\n"
    "- Never call `run_tests` until you've located the buggy file. If the\n"
    "  issue text names a file, `read_file` it. If it only names a symbol,\n"
    "  feature, or error string, call `grep(pattern='<keyword>',\n"
    "  output_mode='files_with_matches')` first to find the file. The traceback\n"
    "  at the bottom of a stack usually points to a dispatch / wrapper — the\n"
    "  real fix is often in the deeper caller, not the topmost frame.\n"
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
        f"Workflow: locate → read → edit → test → submit_patch. "
        f"Use absolute paths. ONE tool call per assistant message.\n\n"
        f"## Tools\n{tool_lines}\n\n"
        f"## Skills (Level-1 — load body via `load_skill`)\n{skills_block}\n"
        f"\n**Before deciding an approach, scan the Level-1 list above.** "
        f"If a skill's \"Use when ...\" description matches your task, "
        f"call `load_skill(name)` to load its workflow (Level-2 body). "
        f"The skill may prescribe specific tools, sequences, or scripts. "
        f"After loading, follow its steps; do not skip directly to read/edit.\n\n"
        f"## Subagents\n{subagent_block}\n\n"
        f"{SUBAGENT_PROTOCOL}"
        f"{TERMINATION_PROTOCOL}"
        f"Turn budget: {max_turns}. When done, call `submit_patch` once.\n"
    )