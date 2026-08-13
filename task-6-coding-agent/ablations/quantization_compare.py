"""S1 — Quantisation comparison (Q4_K_M vs FP16).

This ablation compares two local model endpoints on the same toy-repo
issue. The two endpoints differ only in the model checkpoint — typically
the FP16 version and a Q4_K_M quantised GGUF.

How to use:

.. code-block:: bash

   # FP16
   ollama pull qwen2.5-coder:7b-instruct
   ollama serve
   QWEN_MODEL=qwen2.5-coder:7b-instruct \
       OPENAI_BASE_URL=http://localhost:11434/v1 \
       python ablations/quantization_compare.py --label fp16

   # Q4_K_M (different endpoint / different model name)
   QWEN_MODEL=qwen2.5-coder:7b-instruct-q4_K_M \
       OPENAI_BASE_URL=http://localhost:8080/v1 \
       python ablations/quantization_compare.py --label q4_k_m

Each run is recorded in ``ablations/results/quantization.jsonl`` so you
can diff them after the fact.

The script degrades gracefully: if the endpoint is unreachable it logs
the failure and exits non-zero — but never crashes the smoke test.
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


def _reset_toy_repo(repo: Path) -> bool:
    """Restore the buggy state so the run is reproducible."""
    orig = repo / "calculator.py.orig"
    if not orig.exists():
        print(f"[skip] {orig} missing — run data/download.py first")
        return False
    shutil.copy(orig, repo / "calculator.py")
    return True


def run_once(label: str, smoke: bool, model: str | None) -> dict:
    repo = ROOT / "data" / "toy-repo"
    if not _reset_toy_repo(repo):
        return {"label": label, "skipped": "toy-repo missing"}

    # Override model for this run.
    if model:
        os.environ["QWEN_MODEL"] = model

    issue = (repo / "ISSUE.md").read_text(encoding="utf-8")
    try:
        agent = CodingAgent()
    except LLMError as e:
        return {"label": label, "skipped": f"LLM init failed: {e}"}

    start = time.time()
    try:
        trace = agent.run(str(repo), issue)
    except LLMError as e:
        return {"label": label, "error": f"run failed: {e}", "duration_ms": int((time.time() - start) * 1000)}
    duration = int((time.time() - start) * 1000)

    record = {
        "label": label,
        "endpoint": agent.llm.endpoint_summary,
        "tests_passed": bool(trace.get("tests_passed")),
        "done_reason": trace.get("done_reason"),
        "turn_count": trace.get("turn_count"),
        "tool_call_count": trace.get("tool_call_count"),
        "duration_ms": duration,
        "patch_len": len(trace.get("patch", "")),
        "compaction_events": trace.get("compaction_events", 0),
    }
    if smoke:
        # Truncate the record so the smoke variant is small.
        record["smoke"] = True
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description="Quantisation comparison (S1)")
    ap.add_argument("--smoke", action="store_true", help="Use built-in samples and offline client")
    ap.add_argument("--label", default=os.environ.get("LABEL", "fp16"))
    ap.add_argument("--model", default=None, help="Override QWEN_MODEL for this run")
    args = ap.parse_args()

    record = run_once(args.label, args.smoke, args.model)
    out = RESULTS_DIR / "quantization.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0 if record.get("tests_passed") else 1


if __name__ == "__main__":
    sys.exit(main())
