"""run_bash — sandboxed subprocess runner.

Per blueprint Part I §1.6 / file-system-spec §6:

* the user-supplied ``cmd`` string is **parsed** with ``shlex`` and run as
  a list (``subprocess.run(args=[...], shell=False)``) — never via a shell,
* every path-looking token is checked against the repo root via
  :func:`safe_resolve` (path traversal / absolute path → ``PermissionError``),
* a deny-list rejects destructive commands (``rm -rf``, ``chmod -R 777``,
  ``git reset --hard``, ``curl | sh`` style piped downloads, etc.) before
  the subprocess is even started,
* stdout / stderr are captured and returned as tails,
* a hard timeout is enforced — there is no way for the model to hang the
  agent,
* the env is stripped to ``PATH + HOME + extra_env`` — no leaking of
  ``OPENAI_API_KEY`` or similar secrets into a child process,
* ``fire_pre`` hook runs first (test-write protection, audit), ``fire_post``
  hook runs after for observation post-processing.

Reference (claude):
  * ``claude_reference/.../packages/builtin-tools/src/tools/BashTool/``
    — tree-sitter AST parsing, sandbox adapter, classifier-based
    permission decisions.  Our version is intentionally simpler
    (shlex + deny-list) but follows the same principles:
    list-form subprocess + path containment + destructive-command
    deny-list + audit hook.

What this tool is NOT:

* **NOT** a shell.  No pipes (``|``), redirects (``>``), command
  substitution (``$()``), backgrounding (``&``), or glob expansion
  (``*``) survive ``shlex.split``.  The model must call ``run_bash``
  multiple times for chained commands, or pre-compute via
  ``read_file`` + ``write_file``.  This is intentional — it makes the
  command surface auditable.  The deny-list catches the dangerous
  cases (``rm -rf /etc``) that survive.
* **NOT** a long-running process launcher.  Default timeout 60s,
  hard cap 600s — anything longer should not be in the agent loop.
* **NOT** a way to bypass :func:`safe_resolve`.  Path tokens get
  rejected even if the command itself looks innocuous.
"""
from __future__ import annotations

import logging
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from .base import BaseTool, run_subprocess

log = logging.getLogger("coding_agent.tools.run_bash")


# ---------------------------------------------------------------------------
# Deny-list — patterns that are rejected even if path checks succeed.
# ---------------------------------------------------------------------------
#
# Each entry is a (compiled regex, human-readable reason) tuple.  The
# regex is matched against the **shlex-tokenised command blob** (with
# ``shlex.join``), so whitespace is normalised.  We deliberately keep
# this list **short** and **auditable** — every entry has a reason
# documented next to it.  Adding more patterns is fine; making it
# longer than ~30 entries means we should switch to AST parsing.
_DENY_PATTERNS: List[tuple] = [
    # File-system destruction
    (re.compile(r"\brm\s+(-[a-zA-Z]*[rfRF]+|--recursive|--force)\b"), "rm -rf / recursive rm / forced rm"),
    (re.compile(r"\brm\s+-rf?\s+/\s*$"), "rm -rf / or rm -rf /something"),
    (re.compile(r"\bfind\s+/.+\s+-delete\b"), "find … -delete (recursive delete)"),
    # Privilege escalation
    (re.compile(r"\bsudo\b"), "sudo (privilege escalation)"),
    (re.compile(r"\bsu\b\s"), "su (switch user)"),
    # chmod mass / world-writable
    (re.compile(r"\bchmod\s+(-R|--recursive)\s+[0-7]*[2367][0-7]*\s+/"), "chmod -R world-writable on / path"),
    (re.compile(r"\bchown\s+(-R|--recursive)\s"), "chown -R (mass ownership change)"),
    # Git destructive
    (re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f|checkout\s+--\s|branch\s+-D)"), "destructive git command"),
    (re.compile(r"\bgit\s+push\s+(--force|-f)\b"), "git push --force"),
    # Network exfiltration / piped download
    (re.compile(r"\bcurl\b.*\|\s*(sh|bash)\b"), "curl … | sh (piped download/exec)"),
    (re.compile(r"\bwget\b.*\|\s*(sh|bash)\b"), "wget … | sh (piped download/exec)"),
    # Disk formatting / mounts
    (re.compile(r"\b(mkfs|mkfs\.\w+)\b"), "mkfs (disk format)"),
    (re.compile(r"\b(umount|mount)\b\s+/"), "umount/mount on root path"),
    # Process / kernel
    (re.compile(r"\b(kill\s+-9|kill\s+-KILL|killall)\b"), "force-kill process"),
    (re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b"), "system shutdown/reboot"),
    # Output redirection to absolute paths / outside the repo (we still
    # catch it after, but deny-list is the loud-error path).
    (re.compile(r">\s*/"), "redirect to absolute path"),
]


def _check_deny_list(cmd_str: str) -> Optional[str]:
    """Return a reason string if the raw command string matches a denied pattern.

    We match on the **raw** user-supplied string (not ``shlex.join(tokens)``)
    because ``shlex.join`` quotes shell metacharacters like ``|`` and
    ``>``, which would otherwise defeat the deny-list (e.g. the
    ``curl foo | sh`` pattern needs to match *unescaped* ``|``).

    Tradeoff: a malicious command can quote the metacharacter and bypass
    the regex (``curl foo '|' sh``).  But that's still caught at the
    path-resolution layer for the ``sh`` argument (it's not a path so
    doesn't get rejected; but the **execution** layer catches it because
    the parsed argv is just ``['curl', 'foo', '|', 'sh']`` — curl will
    treat ``|`` and ``sh`` as URLs and return 6/7 exit codes; it
    doesn't pipe to a shell).  The deny-list is defence-in-depth, not
    the only line.
    """
    for pat, reason in _DENY_PATTERNS:
        if pat.search(cmd_str):
            return f"command denied by safety policy: {reason} (matched: {pat.pattern})"
    return None


# ---------------------------------------------------------------------------
# Path-token extraction
# ---------------------------------------------------------------------------
#
# We treat any token that ``looks like a path`` as a candidate for
# containment-check.  "Looks like a path" means:
#
#   * starts with ``/``  (absolute),
#   * starts with ``./`` or ``../`` (explicit relative),
#   * contains a ``/`` but doesn't look like a flag (i.e. not preceded
#     by a flag-introducer in the previous token).
#
# Tokens that are clearly flags (``-rf``, ``--foo``), command names
# (``rm``, ``git``, ``python``), or simple unprefixed words are skipped.
_PATHY_TOKEN_RE = re.compile(r"^(?:\./|\.\./|/|[^-\s][^/]*/[^/].*)")


def _extract_path_tokens(tokens: List[str]) -> List[str]:
    """Return the subset of tokens that look like paths."""
    out: List[str] = []
    for tok in tokens:
        if not tok or tok.startswith("-"):
            continue
        if _PATHY_TOKEN_RE.match(tok):
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class RunBashTool(BaseTool):
    """Run a sandboxed shell command inside the repo.

    The ``cmd`` argument is parsed with ``shlex`` and run as a list.
    No shell features survive parsing — no pipes, no redirects, no
    command substitution.  Use multiple ``run_bash`` calls for chains,
    or use ``write_file`` / ``run_tests`` for file I/O.

    Refuses:

    * absolute paths anywhere in the command,
    * ``..`` traversal outside the repo,
    * destructive patterns (``rm -rf``, ``chmod -R``, ``git reset --hard``,
      ``curl | sh``, etc.),
    * commands over 600 s.

    Returns ``exit_code``, ``stdout_tail``, ``stderr_tail``, ``timed_out``,
    ``duration_s``.
    """

    name: ClassVar[str] = "run_bash"
    description: ClassVar[str] = (
        "Run a single shell command inside the repo and return its "
        "stdout / stderr / exit code. The command is parsed with shlex "
        "and executed as a list — NO shell features (pipes, redirects, "
        "command substitution, globs) survive parsing. Use multiple "
        "calls for chains, write_file / run_tests for file I/O. "
        "Refuses: absolute paths, .. traversal, destructive commands "
        "(rm -rf, chmod -R, git reset --hard, curl | sh, …). Default "
        "timeout 60s, hard cap 600s."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": (
                    "Command string. The string is split on whitespace "
                    "with shlex.split; quote segments you want kept "
                    "together. Do NOT include pipes (|), redirects (> "
                    "or <), or command substitution ($() / ``) — they "
                    "are stripped before execution."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Override timeout in seconds (default 60, hard cap 600)."
                ),
                "minimum": 1,
                "maximum": 600,
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Override the working directory (relative to repo root, "
                    "default repo root). Path must stay inside the repo."
                ),
            },
            "extra_env": {
                "type": "object",
                "description": (
                    "Extra environment variables to add (PATH and HOME "
                    "are always included). Values must be strings."
                ),
            },
        },
        "required": ["cmd"],
    }
    is_read_only: ClassVar[bool] = False

    DEFAULT_TIMEOUT: ClassVar[int] = 60
    HARD_TIMEOUT_CAP: ClassVar[int] = 600
    MAX_OUTPUT_TAIL: ClassVar[int] = 8000

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        cmd_str = (args.get("cmd") or "").strip()
        if not cmd_str:
            return "[ERROR] cmd is required"
        try:
            tokens = shlex.split(cmd_str, posix=True)
        except ValueError as e:
            return f"[ERROR] failed to parse cmd: {e}"
        if not tokens:
            return "[ERROR] cmd parses to empty token list"

        # 1. Deny-list first — loud error before any path resolution.
        # Match against the raw cmd_str (NOT shlex.join(tokens)) so
        # that shell metacharacters like '|' don't get quoted away.
        deny_reason = _check_deny_list(cmd_str)
        if deny_reason is not None:
            log.warning("run_bash denied: %s | cmd=%s", deny_reason, cmd_str)
            return f"[ERROR] {deny_reason}\n  cmd: {cmd_str}"

        # 2. Path containment on every path-looking token.
        cwd = repo_root
        rel_cwd = args.get("cwd")
        if rel_cwd:
            try:
                cwd = (repo_root / rel_cwd).resolve(strict=False)
                cwd.relative_to(repo_root)
            except ValueError as e:
                return f"[ERROR] cwd escapes repo root: {rel_cwd} ({e})"
            except OSError as e:
                return f"[ERROR] cwd could not be resolved: {e}"

        # Path check on the tokens. We do this before cwd itself (cwd
        # was already validated above).
        from .base import safe_resolve  # local to avoid cycle
        for pathy in _extract_path_tokens(tokens):
            try:
                safe_resolve(pathy, repo_root)
            except PermissionError as e:
                return (
                    f"[ERROR] path in cmd escapes repo root: {pathy} ({e})\n"
                    f"  cmd: {cmd_str}"
                )

        # 3. Timeout (with hard cap).
        timeout = int(args.get("timeout") or self.DEFAULT_TIMEOUT)
        timeout = max(1, min(timeout, self.HARD_TIMEOUT_CAP))

        # 4. Subprocess.
        extra_env = args.get("extra_env") or None
        if extra_env is not None:
            # Defensive type check — JSON schema says object, but we
            # only accept string → string.
            extra_env = {str(k): str(v) for k, v in extra_env.items()}

        t0 = time.time()
        try:
            cp = run_subprocess(
                tokens,
                cwd=cwd,
                timeout=timeout,
                extra_env=extra_env,
            )
        except subprocess.TimeoutExpired as e:
            duration = time.time() - t0
            return (
                f"[ERROR] command timed out after {timeout}s\n"
                f"  cmd: {cmd_str}\n"
                f"  duration_s: {duration:.2f}\n"
                f"  partial stdout:\n{_tail(e.stdout or '', self.MAX_OUTPUT_TAIL)}\n"
                f"  partial stderr:\n{_tail(e.stderr or '', self.MAX_OUTPUT_TAIL)}"
            )
        except Exception as e:  # noqa: BLE001
            return f"[ERROR] subprocess crashed: {e}\n  cmd: {cmd_str}"

        duration = time.time() - t0
        return _format_result(cmd_str, cp, duration)

    # BaseTool already wraps exceptions and applies max_result_chars;
    # we don't override __call__ here.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _format_result(cmd_str: str, cp: subprocess.CompletedProcess, duration: float) -> str:
    """Format the subprocess result as a structured plain-text block.

    The first line is ``exit_code=N duration_s=F.TT`` so the agent's
    stuck-loop detector (which grep's ``exit_code=`` summary lines —
    see ``src/agent.py:_recent_test_summaries``) can apply the same
    heuristic to bash failures.
    """
    out_tail = _tail(cp.stdout or "", RunBashTool.MAX_OUTPUT_TAIL)
    err_tail = _tail(cp.stderr or "", RunBashTool.MAX_OUTPUT_TAIL)
    return (
        f"exit_code={cp.returncode} duration_s={duration:.2f}\n"
        f"cmd: {cmd_str}\n"
        f"stdout:\n{out_tail}\n"
        f"stderr:\n{err_tail}"
    )