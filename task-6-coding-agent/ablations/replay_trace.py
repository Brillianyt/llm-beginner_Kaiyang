"""Replay a saved Trace end-to-end and report a summary.

Usage::

    python -m ablations.replay_trace eval/traces/m3_toy_repo.json

Prints a JSON report and exits 0 if the replay matches the original
trace on every step, 1 if any step diverged.
"""
import json
import sys
from pathlib import Path

from src.replay import TraceReplay


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m ablations.replay_trace <trace.json> [repo_root]",
              file=sys.stderr)
        return 2
    trace_path = Path(argv[1])
    if len(argv) >= 3:
        repo_root = Path(argv[2])
    else:
        # Default to the toy-repo (matches eval/run.py).
        repo_root = Path("/mnt/workspace/llm-beginner_Kaiyang/task-6-coding-agent/data/toy-repo")
    trace = json.loads(trace_path.read_text())
    report = TraceReplay(repo_root).replay(trace)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.match_rate >= 1.0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))