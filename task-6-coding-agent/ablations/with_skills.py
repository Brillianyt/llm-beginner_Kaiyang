"""S3 — With vs without Skills.

Compares prompt-only mode (no SkillLoader) against skill-loaded mode.

In *prompt-only* mode:
* No Level-1 list in the system prompt
* ``load_skill`` is removed from the tool schema

In *with-skills* mode:
* Level-1 list is injected into the system prompt
* The LLM can call ``load_skill(name)`` to fetch full bodies
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import CodingAgent  # noqa: E402
from src.llm_client import LLMError  # noqa: E402
from src.skill_loader import SkillLoader  # noqa: E402

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
    ap.add_argument("--mode", choices=["prompt", "skills"], default="skills")
    args = ap.parse_args()

    repo = ROOT / "data" / "toy-repo"
    if not _reset(repo):
        print("[skip] toy-repo missing")
        return 0
    issue = (repo / "ISSUE.md").read_text(encoding="utf-8")

    loader = None
    if args.mode == "skills":
        loader = SkillLoader(str(ROOT / "src" / "skills"))

    try:
        agent = CodingAgent(skill_loader=loader)
    except LLMError as e:
        print(f"[skip] LLM init failed: {e}")
        return 0

    if args.mode == "prompt":
        # Remove load_skill from the schema entirely.
        agent._all_schemas = [
            s for s in agent._all_schemas if s.get("name") != "load_skill"
        ]

    start = time.time()
    try:
        trace = agent.run(str(repo), issue)
    except LLMError as e:
        record = {"mode": args.mode, "error": str(e)}
        print(json.dumps(record, indent=2))
        return 0
    duration = int((time.time() - start) * 1000)

    record = {
        "mode": args.mode,
        "tests_passed": bool(trace.get("tests_passed")),
        "done_reason": trace.get("done_reason"),
        "turn_count": trace.get("turn_count"),
        "skill_loads": trace.get("skill_loads") or [],
        "compaction_events": trace.get("compaction_events", 0),
        "usage": trace.get("token_usage") or {},
        "duration_ms": duration,
        "smoke": args.smoke,
    }
    out = RESULTS_DIR / "with_skills.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
