"""list_files tool — directory listing helper."""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import BaseTool, safe_resolve


class ListFilesTool(BaseTool):
    name: ClassVar[str] = "list_files"
    description: ClassVar[str] = (
        "List files under a repo-relative directory (default '.'). Skips "
        "the `.git` directory and any path that escapes the repo root."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repo-relative directory, default '.'.",
                "default": ".",
            },
            "max_depth": {
                "type": "integer",
                "description": "Recursion depth (0 = just immediate children).",
                "default": 2,
                "minimum": 0,
                "maximum": 6,
            },
        },
    }
    is_read_only: ClassVar[bool] = True

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        path_str = args.get("path", ".")
        max_depth = int(args.get("max_depth", 2))
        # Accept BOTH repo-relative ("rules") and absolute paths inside
        # the repo ("/abs/repo/rules"). read_file / write_file / edit
        # require absolute, so the model will naturally pass absolute
        # paths here too — reject those and it loops forever.
        if Path(path_str).is_absolute():
            p = Path(path_str).resolve()
            try:
                rel = p.relative_to(repo_root)
            except ValueError as e:
                return f"[ERROR] path escapes repo root: {path_str}"
            base = repo_root / rel
        else:
            try:
                base = safe_resolve(path_str, repo_root)
            except PermissionError as e:
                return f"[ERROR] {e}"
        if not base.exists():
            return f"[ERROR] path not found: {path_str}"
        if not base.is_dir():
            return f"[ERROR] not a directory: {path_str}"
        lines: list = []
        self._walk(base, base, lines, depth=0, max_depth=max_depth)
        if not lines:
            return "(empty)"
        return "\n".join(lines[:2000])

    def _walk(
        self,
        root: Path,
        cur: Path,
        out: list,
        *,
        depth: int,
        max_depth: int,
    ) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(cur.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except (PermissionError, FileNotFoundError):
            return
        for entry in entries:
            if entry.name.startswith(".git"):
                continue
            rel = entry.relative_to(root)
            out.append(f"{'  ' * depth}{entry.name}{'' if entry.is_file() else '/'}")
            if entry.is_dir():
                self._walk(root, entry, out, depth=depth + 1, max_depth=max_depth)