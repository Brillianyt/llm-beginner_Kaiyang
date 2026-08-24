#!/usr/bin/env python3
"""diff_stats.py — produce a quick summary of a unified diff.

Called by the ``code-review`` skill (see ``SKILL.md``). The agent invokes
this script via ``run_bash`` (or any shell tool) to get a quick
"lines added / removed / files touched" breakdown before doing a full
review.

Usage:
    python diff_stats.py <path-to-diff-file>
    # or
    git diff HEAD | python diff_stats.py -
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_diff(text: str) -> dict:
    """Parse a unified diff and return a summary dict."""
    files: dict = defaultdict(lambda: {"add": 0, "del": 0})
    cur_file = None
    # `diff --git a/foo b/foo` — capture the new path.
    file_header = re.compile(r"^\+\+\+\s+b/(.+)$", re.MULTILINE)
    for match in file_header.finditer(text):
        cur_file = match.group(1)
    # Fall back: some diffs use `+++ b/...` only.
    if not cur_file:
        cur_file = "(unknown)"
    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            files[cur_file]["add"] += 1
        elif line.startswith("-"):
            files[cur_file]["del"] += 1
    return dict(files)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: diff_stats.py <diff-file|->", file=sys.stderr)
        return 2
    arg = sys.argv[1]
    if arg == "-":
        text = sys.stdin.read()
    else:
        text = Path(arg).read_text(encoding="utf-8")
    summary = parse_diff(text)
    total_add = sum(f["add"] for f in summary.values())
    total_del = sum(f["del"] for f in summary.values())
    print(f"files={len(summary)}  +{total_add}  -{total_del}")
    for path, stats in sorted(summary.items()):
        print(f"  {path}  +{stats['add']}/-{stats['del']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())