"""run_tests — structured pytest runner.

Per file-system-spec §3 / blueprint Part I §1.3:

* default command: ``python -m pytest --tb=short`` — deliberately NOT
  ``-x`` (fail-fast): on SWE-bench-sized repos an unrelated early failure
  would hide the target test's result from the agent.
* parses pytest output into ``passed / failed / errors`` and a
  ``failures[]`` array of ``{file, line, test, msg}``,
* truncates stdout/stderr to a tail,
* returns structured text with all of the above.

Wheel-mirror mode
-----------------
For SWE-bench-style tasks where the cloned source repo is NOT
buildable in-place (e.g. astropy's setuptools_scm broken in the
clone, missing compiled `.so` files), running pytest against
``repo_root`` produces ``ImportError: setuptools_scm broken`` and
the model can't see real test results — it confuses ``passed=0
failed=0`` with "no tests collected" and gives up.

When ``WHEEL_MIRROR_ROOT`` env var is set, before running pytest we:

1. Sync every ``*.py`` under ``<repo_root>/<package>`` (default
   ``astropy``) to ``<WHEEL_MIRROR_ROOT>/<package>``.  The mirror
   is a wheel-installed copy of the same package that already has
   the compiled extensions.
2. Run pytest with ``--rootdir=<WHEEL_MIRROR_ROOT>`` so it uses
   the mirror's tests and imports.

The mirror is **read-only** from the model's perspective: the model
edits files in the cloned repo, the mirror reflects those edits,
and pytest runs against the mirror.

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

import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from .base import BaseTool, run_subprocess


class RunTestsTool(BaseTool):
    name: ClassVar[str] = "run_tests"
    description: ClassVar[str] = (
        "Runs the project's test suite and returns structured pass/fail counts "
        "plus a per-test failure list. Defaults to `python -m pytest --tb=short` "
        "(NOT -x: on big repos an unrelated first failure would hide the "
        "target test's result). Never executes a shell — commands are "
        "always passed as argv lists."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": (
                    "Override command (default: 'python -m pytest --tb=short'). "
                    "The string is split on whitespace; quote segments you "
                    "want kept together."
                ),
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Override working directory (default: repo root). "
                    "Must stay inside the repo root."
                ),
            },
            "extra_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Extra pytest flags, e.g. ['-x', '-k', 'test_add_positive_numbers']. "
                    "Appended after `cmd`."
                ),
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
        # Match either ``==== 3 passed, 1 failed in 0.05s ====`` (pytest
        # default verbose) or the bare ``3 passed, 1 failed in 0.05s``
        # form that pytest emits under ``-q`` after ``[100%]``.  We
        # anchor the body so it must start with a digit followed by a
        # status word, end with ``in <time>s`` (optional ``s``), and
        # contain no equals signs in between.  We match the LAST such
        # line in the blob so we prefer the actual summary over earlier
        # ``FAILED ... in`` lines.
        r"(?P<body>\d+\s+(?:passed|failed|error)(?:[^\n=]*?\d+\s+(?:passed|failed|error))*\s+in\s+[0-9.]+\s*s?)",
    )

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        cmd_str = args.get("cmd") or "python -m pytest --tb=short"
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
        argv = [sys.executable, "-m", "pytest", "--tb=short"] if argv == ["pytest"] else argv
        if not argv:
            argv = [sys.executable, "-m", "pytest", "--tb=short"]
        # Validate extra_args and append.
        extra = args.get("extra_args") or []
        if not isinstance(extra, list) or not all(isinstance(s, str) for s in extra):
            return "[ERROR] extra_args must be a list of strings"
        # Reject any extra arg starting with '-' that has '=' or contains shell
        # metacharacters — keeps the argv-only contract honest.
        banned = {";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\r"}
        for s in extra:
            if any(c in s for c in banned):
                return f"[ERROR] extra_arg contains shell metacharacter: {s!r}"
        argv = argv + extra

        # Wheel-mirror sync: when WHEEL_MIRROR_ROOT is set, copy `*.py`
        # files from <repo>/<pkg> to <mirror>/<pkg> and run pytest against
        # the mirror.  This makes the cloned source's edits visible to a
        # properly-installed astropy at test time, instead of running
        # pytest against the unbuildable clone.
        wheel_mirror = ""
        files_synced = 0
        mirror_root_str = os.environ.get("WHEEL_MIRROR_ROOT")
        mirror_pkg = os.environ.get("WHEEL_MIRROR_PKG", "astropy")
        # Optional ``WHEEL_TEST_PATCH_FILE``: path to a ``test_patch``
        # file (unified diff) that the SWE-bench instance expects to be
        # applied to the wheel's tests BEFORE running pytest.  Without
        # this, FAIL_TO_PASS tests like ``test_roundtrip[True]`` are not
        # even collected because the parametrization is added by the
        # patch.
        wheel_test_patch = os.environ.get("WHEEL_TEST_PATCH_FILE")
        # Detect whether the model passed an explicit test path in
        # ``extra_args``.  If not, we scope pytest to the package's
        # ``tests/`` directory instead of the whole mirror — otherwise
        # pytest tries to collect numpy/scipy tests too and hits a
        # network error during astropy's data download.
        has_explicit_test_path = any(
            not a.startswith("-") for a in extra
        ) or any(
            not a.startswith("-") for a in _split(cmd_str)[3:]
        )
        if mirror_root_str:
            mirror_root = Path(mirror_root_str)
            src_pkg = repo_root / mirror_pkg
            dst_pkg = mirror_root / mirror_pkg
            # Files we MUST NOT copy from the broken cloned source to
            # the wheel: build artifacts that the wheel already has
            # correctly (its own _version.py), and the source's
            # broken version.py fallback.  Copying these would
            # break the wheel's import.
            SKIP_SYNC = {
                f"{mirror_pkg}/version.py",  # source has broken
                                              # setuptools_scm fallback
                f"{mirror_pkg}/_version.py",  # wheel's build artifact
                f"{mirror_pkg}/_dev/scm_version.py",
            }
            if src_pkg.is_dir() and dst_pkg.is_dir():
                for py in src_pkg.rglob("*.py"):
                    rel = py.relative_to(src_pkg)
                    rel_str = str(rel).replace("\\", "/")
                    if rel_str in SKIP_SYNC:
                        continue
                    target = dst_pkg / rel
                    if not target.exists() or py.read_bytes() != target.read_bytes():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(py, target)
                        files_synced += 1
                # Apply test_patch (unified diff) to the wheel's tests.
                # We do this against the mirror copy so the cloned
                # source's test files are untouched (the model must
                # never edit tests anyway).
                if wheel_test_patch:
                    patch_path = Path(wheel_test_patch)
                    if patch_path.is_file():
                        import subprocess as _sp
                        patch_text = patch_path.read_text()
                        applied = False
                        for variant in (["apply", "--3way"], ["apply"]):
                            args = ["git"] + variant + ["-"]
                            cp = _sp.run(
                                args,
                                cwd=str(mirror_root),
                                input=patch_text,
                                shell=False,
                                capture_output=True,
                                text=True,
                                timeout=15,
                            )
                            if cp.returncode == 0:
                                files_synced += 1  # count the patch
                                applied = True
                                break
                        if not applied:
                            return (
                                f"[ERROR] WHEEL_TEST_PATCH failed to "
                                f"apply (last stderr: "
                                f"{cp.stderr[-500:] if cp.stderr else ''})"
                            )
                wheel_mirror = str(mirror_root)
                # Re-target pytest to the mirror so its test-collection
                # and import resolution use the installed package.
                argv = [sys.executable, "-m", "pytest", "--tb=short",
                        f"--rootdir={mirror_root}"] + [
                    a for a in argv[3:] if not a.startswith("--rootdir=")
                ]
                # Scope to <pkg>/tests/ by default so we don't collect
                # numpy/scipy tests too.
                if not has_explicit_test_path:
                    tests_dir = dst_pkg / "tests"
                    if tests_dir.is_dir():
                        argv.append(str(tests_dir))
                else:
                    # The model specified a test path.  Resolve any
                    # RELATIVE path against the package's directory
                    # (mirror_root/<pkg>/), since the wheel-mirror's
                    # top level has no `tests/` subdir — the test files
                    # live under the package.
                    new_argv = []
                    for a in argv:
                        if (
                            not a.startswith("-")
                            and not Path(a).is_absolute()
                            and not Path(mirror_root / a).exists()
                        ):
                            candidate = dst_pkg / a
                            if candidate.exists():
                                new_argv.append(str(candidate))
                                continue
                        new_argv.append(a)
                    argv = new_argv
                cwd = mirror_root

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
            "wheel_mirror": wheel_mirror,
            "files_synced": files_synced,
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
        # Pytest's summary line can be:
        #   "=== 3 passed, 1 failed in 0.05s ==="  (verbose)
        #   "3 passed in 0.05s"                   (-q bare)
        # Pick the LAST match so we prefer the real summary over
        # intermediate ``FAILED ... in ...`` lines.
        matches = list(cls._SUMMARY_RE.finditer(blob))
        text = matches[-1].group("body") if matches else None
        if not text:
            return {"passed": 0, "failed": 0, "errors": 0}
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
    if d["wheel_mirror"]:
        head += f"[wheel_mirror={d['wheel_mirror']} synced={d['files_synced']}]\n"
    if d["failures"]:
        head += "--- failures ---\n"
        for f in d["failures"]:
            head += f"  - {f['file']}::{f['test']}  (line {f['line']}) — {f['msg']}\n"
    head += "--- stdout (tail) ---\n" + d["stdout_tail"]
    if d["stderr_tail"].strip():
        head += "\n--- stderr (tail) ---\n" + d["stderr_tail"]
    return head