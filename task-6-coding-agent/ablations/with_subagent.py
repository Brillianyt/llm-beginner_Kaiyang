"""S2 — Single agent vs Subagent.

Compares the token cost of running the toy-repo fix with and without
delegating exploration to a ``code_search`` subagent.

Usage:

.. code-block:: bash

   python ablations/with_subagent.py --smoke       # offline mock
   python ablations/with_subagent.py --mode sub    # with subagent
   python ablations/with_subagent.py --mode single # without subagent

Each mode writes one line to ``ablations/results/with_subagent.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import CodingAgent  # noqa: E402
from src.llm_client import LLMError  # noqa: E402

RESULTS_DIR = ROOT / "ablations" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _reset(repo: Path) -> bool:
    orig = repo / "calculator.py.orig"
    if not orig.exists():
        return False
    shutil.copy(orig, repo / "calculator.py")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=["single", "sub"], default="sub")
    args = ap.parse_args()

    repo = ROOT / "data" / "toy-repo"
    if not _reset(repo):
        print("[skip] toy-repo missing")
        return 0

    issue = (repo / "ISSUE.md").read_text(encoding="utf-8")

    try:
        agent = CodingAgent()
    except LLMError as e:
        print(f"[skip] LLM init failed: {e}")
        return 0

    if args.mode == "single":
        # Remove subagent tools entirely so the LLM cannot delegate.
        # The dispatch_subagent tool still exists in the schema but
        # we filter its usage by removing it from the schema.
        agent._tool_schemas = [
            s for s in agent._tool_schemas if s.get("name") != "dispatch_subagent"
        ]

    start = time.time()
    try:
        trace = agent.run(str(repo), issue)
    except LLMError as e:
        record = {"mode": args.mode, "error": str(e)}
        print(json.dumps(record, indent=2))
        return 0
    duration = int((time.time() - start) * 1000)

    # Token accounting — we only see prompt/completion totals per turn,
    # so accumulate from the trace.
    total_prompt = 0
    total_completion = 0
    last_usage = trace.get("last_usage", {})
    if isinstance(last_usage, dict):
        total_prompt = last_usage.get("prompt_tokens", 0)
        total_completion = last_usage.get("completion_tokens", 0)

    record = {
        "mode": args.mode,
        "tests_passed": bool(trace.get("tests_passed")),
        "done_reason": trace.get("done_reason"),
        "turn_count": trace.get("turn_count"),
        "tool_call_count": trace.get("tool_call_count"),
        "subagent_invocations": len(trace.get("subagent_invocations") or []),
        "prompt_tokens_last_turn": total_prompt,
        "completion_tokens_last_turn": total_completion,
        "duration_ms": duration,
        "smoke": args.smoke,
    }
    out = RESULTS_DIR / "with_subagent.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
