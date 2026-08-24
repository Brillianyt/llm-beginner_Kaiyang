"""run_tests — structured pytest runner.

Per file-system-spec §3 / blueprint Part I §1.3:

* default command: ``python -m pytest -x --tb=short``,
* parses pytest output into ``passed / failed / errors`` and a
  ``failures[]`` array of ``{file, line, test, msg}``,
* truncates stdout/stderr to a tail,
* returns structured text with all of the above.

Return shape (blueprint §1.3):

    {
      "exit_code": int,
      "passed": int, "failed": int, "errors": int,
      "duration_s": float,
      "stdout_tail": "...", "stderr_tail": "...",
      "failures": [{file, line, test, msg}, ...],
      "truncated": bool
    }
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from .base import BaseTool, run_subprocess


class RunTestsTool(BaseTool):
    name: ClassVar[str] = "run_tests"
    description: ClassVar[str] = (
        "Runs the project's test suite and returns structured pass/fail counts "
        "plus a per-test failure list. Defaults to `python -m pytest -x --tb=short`. "
        "Never executes a shell — commands are always passed as argv lists."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": "Override command (default: 'python -m pytest -x --tb=short').",
            },
            "cwd": {
                "type": "string",
                "description": "Override working directory (default: repo root).",
            },
            "timeout_s": {
                "type": "integer",
                "minimum": 5,
                "maximum": 3600,
                "description": "Hard cap in seconds (default 300).",
            },
        },
    }
    is_read_only: ClassVar[bool] = False  # may touch pytest cache
    max_result_chars: ClassVar[int] = 80_000

    TAIL_LINES = 50  # how many lines of stdout/stderr to keep in the tail
    _FAILED_RE = re.compile(
        r"^FAILED\s+(?P<file>[^\s:]+)::(?P<test>\S+)(?:\s+-\s+(?P<msg>.+))?$",
        re.MULTILINE,
    )
    _SUMMARY_RE = re.compile(
        r"=(?P<sep>=+)\s*(?P<body>\d+\s+(?:passed|failed|error)[^\n]*)\s*=+",
    )

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        cmd_str = args.get("cmd") or "python -m pytest -x --tb=short"
        cwd_str = args.get("cwd") or str(repo_root)
        timeout = int(args.get("timeout_s") or 300)
        cwd = Path(cwd_str)
        try:
            cwd.relative_to(repo_root)
        except ValueError as e:
            raise PermissionError(f"cwd escapes repo root: {cwd_str}") from e
        if not cwd.exists():
            return f"[ERROR] cwd not found: {cwd_str}"
        # argv split (very simple — supports quoted args).
        argv = _split(cmd_str)
        argv = [sys.executable, "-m", "pytest", "-x", "--tb=short"] if argv == ["pytest"] else argv
        if not argv:
            argv = [sys.executable, "-m", "pytest", "-x", "--tb=short"]

        start = time.time()
        try:
            cp = run_subprocess(argv, cwd=cwd, timeout=timeout)
            duration = time.time() - start
        except FileNotFoundError as e:
            return f"[ERROR] command not found: {e}"
        stdout = cp.stdout or ""
        stderr = cp.stderr or ""
        failures = self._parse_failures(stdout + "\n" + stderr)
        summary = self._parse_summary(stdout + "\n" + stderr)
        return _render({
            "exit_code": cp.returncode,
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "errors": summary.get("errors", 0),
            "duration_s": round(duration, 2),
            "stdout_tail": _tail(stdout, self.TAIL_LINES),
            "stderr_tail": _tail(stderr, self.TAIL_LINES),
            "failures": failures,
            "truncated": False,
        })

    @classmethod
    def _parse_failures(cls, blob: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in cls._FAILED_RE.finditer(blob):
            entry: Dict[str, Any] = {
                "file": m.group("file"),
                "line": 0,
                "test": m.group("test"),
                "msg": (m.group("msg") or "").strip(),
            }
            # Try to extract a `> assert ...` line number from the block below.
            tail = blob[m.end(): m.end() + 2000]
            line_match = re.search(r"\.py:(\d+):", tail)
            if line_match:
                entry["line"] = int(line_match.group(1))
            out.append(entry)
        return out

    @classmethod
    def _parse_summary(cls, blob: str) -> Dict[str, int]:
        # Pytest's final summary line looks like:
        #   "=== 3 passed, 1 failed in 0.05s ==="
        m = cls._SUMMARY_RE.search(blob)
        if not m:
            return {"passed": 0, "failed": 0, "errors": 0}
        text = m.group("body")
        out: Dict[str, int] = {"passed": 0, "failed": 0, "errors": 0}
        for tok in re.findall(r"(\d+)\s+(passed|failed|error)", text):
            n, name = int(tok[0]), tok[1]
            if name == "passed":
                out["passed"] = n
            elif name == "failed":
                out["failed"] = n
            elif name == "error":
                out["errors"] = n
        return out


def _tail(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _split(cmd: str) -> List[str]:
    """Very small argv splitter: honours single/double quotes, otherwise splits on whitespace."""
    out: List[str] = []
    cur = []
    quote = None
    for ch in cmd:
        if quote:
            if ch == quote:
                quote = None
            else:
                cur.append(ch)
        elif ch in ("'", '"'):
            quote = ch
        elif ch.isspace():
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _render(d: Dict[str, Any]) -> str:
    head = (
        f"exit_code={d['exit_code']} "
        f"passed={d['passed']} failed={d['failed']} errors={d['errors']} "
        f"duration_s={d['duration_s']}\n"
    )
    if d["failures"]:
        head += "--- failures ---\n"
        for f in d["failures"]:
            head += f"  - {f['file']}::{f['test']}  (line {f['line']}) — {f['msg']}\n"
    head += "--- stdout (tail) ---\n" + d["stdout_tail"]
    if d["stderr_tail"].strip():
        head += "\n--- stderr (tail) ---\n" + d["stderr_tail"]
    return head