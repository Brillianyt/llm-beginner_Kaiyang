"""git_apply — apply a unified diff to the working tree.

Per file-system-spec §5 / blueprint Part I §1.5:

* accepts the patch as a string (not a path),
* default ``dry_run=true`` → only ``git apply --check``,
* on real apply, parse conflicts into ``conflicts[]``,
* never touches paths outside the repo,
* cleans up the temp patch file even on error.

Output:

    {
      "applied": bool,
      "files_touched": [...],
      "fuzzy": bool,
      "conflicts": [{path, hunk}, ...],
      "error": str | None
    }
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from .base import BaseTool, run_subprocess


class GitApplyTool(BaseTool):
    name: ClassVar[str] = "git_apply"
    description: ClassVar[str] = (
        "Applies a unified diff to the working tree using `git apply`. "
        "By default runs in dry_run mode (`--check` only). Set `dry_run=false` "
        "to actually write the changes. Supports `three_way=true` for merge "
        "fallback. Refuses to touch paths outside the repo root."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repo (default: <repo_root>).",
            },
            "patch": {
                "type": "string",
                "description": "Unified diff text (output of `git_diff` or hand-written).",
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true (default), only run `git apply --check`.",
                "default": True,
            },
            "three_way": {
                "type": "boolean",
                "description": "Pass `--3way` to fall back to merge-style apply.",
                "default": False,
            },
        },
        "required": ["patch"],
    }
    is_read_only: ClassVar[bool] = False
    _CONFLICT_RE = re.compile(
        r"error:\s*patch failed:\s*(?P<path>[^:]+):(?P<hunk>\d+)",
        re.IGNORECASE,
    )

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        repo_path_str = args.get("repo_path") or str(repo_root)
        repo_path = Path(repo_path_str)
        try:
            repo_path.relative_to(repo_root)
        except ValueError as e:
            raise PermissionError(f"repo_path escapes repo root: {repo_path_str}") from e
        patch = args.get("patch", "")
        if not patch.strip():
            return _render({"applied": False, "files_touched": [], "fuzzy": False,
                            "conflicts": [], "error": "empty patch"})
        dry_run = bool(args.get("dry_run", True))
        three_way = bool(args.get("three_way", False))

        # Verify all paths in the patch stay inside the repo.
        bad = _paths_outside_patch(patch, repo_path)
        if bad:
            return _render({
                "applied": False, "files_touched": [], "fuzzy": False, "conflicts": [],
                "error": f"patch touches paths outside repo: {bad}",
            })

        with tempfile.NamedTemporaryFile(
            "w", suffix=".patch", delete=False, encoding="utf-8"
        ) as fp:
            fp.write(patch)
            patch_path = fp.name

        try:
            if not dry_run and not (repo_path / ".git").exists():
                # No git repo → fall back to a best-effort patch via the
                # `patch` library if available, else manual hunk application.
                return _render({
                    "applied": False, "files_touched": [], "fuzzy": False, "conflicts": [],
                    "error": "no .git directory; cannot git apply",
                })

            check_cmd = ["git", "apply", "--check"]
            if three_way:
                check_cmd.append("--3way")
            check_cmd.append(patch_path)
            check_cp = run_subprocess(check_cmd, cwd=repo_path, timeout=15)
            if check_cp.returncode != 0:
                conflicts = self._parse_conflicts(check_cp.stderr or "")
                return _render({
                    "applied": False, "files_touched": [], "fuzzy": False,
                    "conflicts": conflicts,
                    "error": (check_cp.stderr or "").strip()[:1500] or "git apply --check failed",
                })

            if dry_run:
                return _render({
                    "applied": False, "files_touched": [], "fuzzy": False,
                    "conflicts": [], "error": None,
                })

            # Snapshot the files we're about to touch. If the real
            # ``git apply`` (or the OS) corrupts them mid-write — disk
            # full, killed by signal, half-written hunk — we can roll
            # back. This is the tool-layer equivalent of the per-file
            # snapshot in swebench_sample.
            touched = _paths_in_patch(patch)
            snapshot: Dict[str, bytes] = {}
            try:
                for rel in touched:
                    p = repo_path / rel
                    if p.exists() and p.is_file():
                        try:
                            snapshot[rel] = p.read_bytes()
                        except OSError:
                            pass
            except Exception:
                snapshot = {}

            apply_cmd = ["git", "apply"]
            if three_way:
                apply_cmd.append("--3way")
            apply_cmd.append(patch_path)
            apply_cp = run_subprocess(apply_cmd, cwd=repo_path, timeout=30)
            if apply_cp.returncode != 0:
                # Roll back anything that did get written before the
                # failure.
                for rel, data in snapshot.items():
                    target = repo_path / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        target.write_bytes(data)
                    except OSError:
                        pass
                conflicts = self._parse_conflicts(apply_cp.stderr or "")
                return _render({
                    "applied": False, "files_touched": touched, "fuzzy": three_way,
                    "conflicts": conflicts,
                    "error": (apply_cp.stderr or "").strip()[:1500] or "git apply failed",
                })
            touched = _paths_in_patch(patch)
            return _render({
                "applied": True, "files_touched": touched, "fuzzy": False,
                "conflicts": [], "error": None,
            })
        finally:
            try:
                Path(patch_path).unlink()
            except OSError:
                pass

    @classmethod
    def _parse_conflicts(cls, stderr: str) -> List[Dict[str, Any]]:
        return [
            {"path": m.group("path"), "hunk": int(m.group("hunk"))}
            for m in cls._CONFLICT_RE.finditer(stderr)
        ]


def _paths_in_patch(patch: str) -> List[str]:
    return [m.group(1) for m in re.finditer(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)]


def _paths_outside_patch(patch: str, repo_root: Path) -> List[str]:
    bad: List[str] = []
    repo = repo_root.resolve()
    for rel in _paths_in_patch(patch):
        # Reject absolute paths and `..` traversal.
        if rel.startswith("/") or ".." in Path(rel).parts:
            bad.append(rel)
            continue
        resolved = (repo / rel).resolve()
        try:
            resolved.relative_to(repo)
        except ValueError:
            bad.append(rel)
    return bad


def _render(d: Dict[str, Any]) -> str:
    if d["error"]:
        return f"[ERROR] {d['error']}"
    if not d["applied"]:
        return "(dry run only — patch would apply cleanly)"
    files = ", ".join(d["files_touched"])
    head = f"applied OK  files=[{files}]\n"
    if d["fuzzy"]:
        head += "(with --3way fallback)\n"
    return head