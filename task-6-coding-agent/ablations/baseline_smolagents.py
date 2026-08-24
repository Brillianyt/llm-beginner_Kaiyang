"""S-baseline — smolagents.CodeAgent on the same toy-repo task.

This is a **baseline comparison** (per the review recommendation): we run
``smolagents.CodeAgent`` against the toy-repo ISSUE.md using the same
Qwen2.5-Coder-7B-Instruct endpoint that powers our ``CodingAgent``.
The point is not to beat smolagents — it's to measure whether our
SkillsLoader / Subagent / two-stage system brings any net benefit over
a well-known off-the-shelf agent.

Output: ``ablations/results/baseline_smolagents.jsonl`` — one record per
run with pass rate + token usage.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    args = ap.parse_args()

    repo = ROOT / "data" / "toy-repo"
    if not _reset(repo):
        print("[skip] toy-repo missing")
        return 0

    issue = (repo / "ISSUE.md").read_text(encoding="utf-8")

    # The smolagents default is OpenAI's gpt-4o; point it at the same
    # local SGLang endpoint the rest of the harness uses.
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:30000/v1")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    from smolagents import CodeAgent, OpenAIServerModel, tool

    # Two tools only — read + write — to mirror the minimum CodingAgent
    # needs to fix the toy-repo bug. We deliberately don't pass run_tests
    # because smolagents can spawn a subprocess from code it writes.
    @tool
    def read_file(file_path: str) -> str:
        """Read a UTF-8 text file from disk.

        Args:
            file_path: Absolute path of the file to read. Must be inside the repo.
        """
        target = Path(file_path).resolve()
        if not target.exists():
            return f"[ERROR] file not found: {file_path}"
        return target.read_text(encoding="utf-8", errors="replace")

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Overwrite a UTF-8 text file with new content.

        Args:
            file_path: Absolute path of the file to write. Must be inside the repo.
            content: New file contents (UTF-8).
        """
        target = Path(file_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return f"wrote {len(content)} bytes to {file_path}"

    model_id = os.environ.get("QWEN_MODEL", "./models/Qwen2.5-Coder-7B-Instruct")
    model = OpenAIServerModel(
        model_id=model_id,
        api_base=os.environ.get("OPENAI_BASE_URL", "http://localhost:30000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )

    agent = CodeAgent(
        tools=[read_file, write_file],
        model=model,
        max_steps=12,
        verbosity_level=0,  # quiet — we capture output via final answer
    )

    start = time.time()
    try:
        # CodeAgent's run takes a single string task. We paste the issue
        # so the agent gets exactly the same prompt our CodingAgent sees.
        answer = agent.run(issue)
    except Exception as e:  # noqa: BLE001
        return _record({"error": str(e)[:200], "duration_ms": int((time.time() - start) * 1000)},
                       "smolagents")

    duration = int((time.time() - start) * 1000)

    # Re-run pytest to verify the bug is fixed.
    cp = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(repo), shell=False, capture_output=True, text=True, timeout=60,
    )
    passed = cp.returncode == 0

    record = {
        "tests_passed": passed,
        "duration_ms": duration,
        "final_answer": str(answer)[:500],
    }
    _record(record, "smolagents")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def _record(record: dict, label: str) -> None:
    out = RESULTS_DIR / f"baseline_{label}.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())