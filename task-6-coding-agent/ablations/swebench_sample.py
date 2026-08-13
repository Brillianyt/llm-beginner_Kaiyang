"""S4 — SWE-bench Lite sample (3 instances).

This is the *advanced* optional bonus — even on a 7B model getting ≥ 1
instance green is hard. The script is structured so:

* If ``data/swebench-lite-sample.parquet`` is missing, exit cleanly.
* If ``data/repos/<repo>`` is missing for any instance, that instance
  is skipped (not failed).
* Each instance is run in its own copy of the repo so failures don't
  pollute the working tree.

Usage:

.. code-block:: bash

   # 1) download metadata
   python data/download.py --with-swebench

   # 2) clone the repos into data/repos/
   mkdir -p data/repos
   cd data/repos && git clone https://github.com/<owner>/<repo>.git

   # 3) run the ablation
   python ablations/swebench_sample.py --smoke
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import CodingAgent  # noqa: E402
from src.llm_client import LLMError  # noqa: E402

SAMPLE = ROOT / "data" / "swebench-lite-sample.parquet"
REPOS = ROOT / "data" / "repos"
RESULTS = ROOT / "ablations" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _checkout(repo: Path, commit: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "checkout", commit, "--"],
        check=False,
        shell=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_pytest(repo: Path, timeout: int = 300) -> int:
    cp = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(repo),
        check=False,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return cp.returncode


def _attempt_instance(row: dict, smoke: bool) -> dict:
    repo_name = row["repo"].split("/")[-1]
    repo = REPOS / repo_name
    if not repo.exists():
        return {"instance_id": row["instance_id"], "skipped": f"repo missing: {repo}"}
    workdir = Path(tempfile.mkdtemp(prefix=f"swebench_{row['instance_id']}_"))
    shutil.copytree(repo, workdir / repo_name)
    work_repo = workdir / repo_name
    _checkout(work_repo, row["base_commit"])

    try:
        agent = CodingAgent()
    except LLMError as e:
        return {"instance_id": row["instance_id"], "skipped": f"LLM init failed: {e}"}

    if smoke:
        # Don't actually call the model in smoke mode.
        return {
            "instance_id": row["instance_id"],
            "smoke": True,
            "tests_passed": False,
            "duration_ms": 0,
        }

    try:
        trace = agent.run(str(work_repo), row["problem_statement"])
    except Exception as e:  # noqa: BLE001
        return {"instance_id": row["instance_id"], "error": str(e)[:120]}
    passed = bool(trace.get("tests_passed"))
    return {
        "instance_id": row["instance_id"],
        "tests_passed": passed,
        "done_reason": trace.get("done_reason"),
        "turn_count": trace.get("turn_count"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not SAMPLE.exists():
        print(f"[skip] {SAMPLE} missing — run data/download.py --with-swebench")
        return 0
    try:
        import pandas as pd  # noqa: WPS433
        df = pd.read_parquet(SAMPLE)
    except Exception as e:  # noqa: BLE001
        print(f"[skip] cannot read parquet: {e}")
        return 0

    results = []
    for _, row in df.iterrows():
        results.append(_attempt_instance(dict(row), args.smoke))

    out = RESULTS / "swebench.jsonl"
    with out.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    passed = sum(1 for r in results if r.get("tests_passed"))
    return 0 if passed >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
