"""git_diff — read-only repo diff with per-file patches.

Per file-system-spec §4 / blueprint Part I §1.4:

* supports ``staged`` (``--cached``), per-file filter,
* default context_lines=3,
* returns ``files[]`` of ``{path, additions, deletions, status, patch}``
  plus a ``total_files`` count and ``truncated`` flag,
* redacts secrets (``*.env``, ``*credentials*``, ``*secret*``),
* patches > 100 KB are truncated.

Output:

    {
      "files": [
        {path, additions, deletions, status: modified|added|deleted, patch}
      ],
      "total_files": N,
      "truncated": bool
    }
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from .base import BaseTool, check_blocked_git, run_subprocess


class GitDiffTool(BaseTool):
    name: ClassVar[str] = "git_diff"
    description: ClassVar[str] = (
        "Shows `git diff` (working tree vs HEAD) for the repository. "
        "Supports `--cached` (staged) and per-file filters. Returns "
        "per-file unified diffs. Does not modify repo state."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repo (default: <repo_root>).",
            },
            "staged": {
                "type": "boolean",
                "description": "If true, show staged diff (--cached).",
                "default": False,
            },
            "file_path": {
                "type": "string",
                "description": "Restrict diff to a single file (relative to repo).",
            },
            "context_lines": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Unified-diff context lines (default 3).",
                "default": 3,
            },
        },
    }
    is_read_only: ClassVar[bool] = True

    SECRET_RE = re.compile(r"(\.env|credentials?|secret)", re.IGNORECASE)
    PATCH_LIMIT_BYTES = 100 * 1024
    _FILE_HEADER_RE = re.compile(r"^diff --git a/(?P<path>\S+) b/(?P<path2>\S+)$", re.MULTILINE)

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        repo_path_str = args.get("repo_path") or str(repo_root)
        repo_path = Path(repo_path_str)
        try:
            repo_path.relative_to(repo_root)
        except ValueError as e:
            raise PermissionError(f"repo_path escapes repo root: {repo_path_str}") from e
        if not (repo_path / ".git").exists():
            return _render({"files": [], "total_files": 0, "truncated": False})

        staged = bool(args.get("staged", False))
        file_filter = args.get("file_path")
        ctx = int(args.get("context_lines") or 3)

        cmd = ["git", "--no-pager", "diff", f"--unified={ctx}"]
        if staged:
            cmd.append("--cached")
        if file_filter:
            cmd.extend(["--", file_filter])

        blocked = check_blocked_git(cmd)
        if blocked:
            return f"[ERROR] {blocked}"

        cp = run_subprocess(cmd, cwd=repo_path, timeout=30)
        body = cp.stdout or ""
        if not body.strip():
            return _render({"files": [], "total_files": 0, "truncated": False})

        files: List[Dict[str, Any]] = []
        truncated_any = False
        for m in self._FILE_HEADER_RE.finditer(body):
            start = m.end()
            nxt = self._FILE_HEADER_RE.search(body, start)
            end = nxt.start() if nxt else len(body)
            chunk = body[start:end]
            path = m.group("path")
            additions = sum(1 for ln in chunk.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
            deletions = sum(1 for ln in chunk.splitlines() if ln.startswith("-") and not ln.startswith("---"))
            status = self._infer_status(chunk, path)
            if self.SECRET_RE.search(path):
                chunk = "[REDACTED — secret-like file path]\n"
            if len(chunk) > self.PATCH_LIMIT_BYTES:
                chunk = chunk[: self.PATCH_LIMIT_BYTES] + "\n...[truncated]..."
                truncated_any = True
            files.append({
                "path": path,
                "additions": additions,
                "deletions": deletions,
                "status": status,
                "patch": chunk,
            })
        return _render({"files": files, "total_files": len(files), "truncated": truncated_any})

    @staticmethod
    def _infer_status(chunk: str, path: str) -> str:
        if "new file mode" in chunk:
            return "added"
        if "deleted file mode" in chunk:
            return "deleted"
        return "modified"


def _render(d: Dict[str, Any]) -> str:
    if not d["files"]:
        return "(no changes)"
    head = f"files_changed={d['total_files']} truncated={d['truncated']}\n"
    for f in d["files"]:
        head += (
            f"\n--- {f['path']} ({f['status']}: +{f['additions']}/-{f['deletions']}) ---\n"
            f"{f['patch']}"
        )
    return head