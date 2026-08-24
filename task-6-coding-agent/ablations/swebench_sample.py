"""S4 — SWE-bench Lite sample (3 instances).

This is the *advanced* optional bonus — even on a 7B model getting ≥ 1
instance green is hard. The script is structured so:

* If ``data/swebench-lite-sample.parquet`` is missing, exit cleanly.
* If ``data/repos/<repo>`` is missing for any instance, that instance
  is skipped (not failed).
* For each instance we ``git checkout <base_commit>`` directly inside
  the cloned repo (no copytree — the repo is large). The agent then
  runs against that working tree and we restore the original HEAD
  afterwards.

Usage:

.. code-block:: bash

   # 1) download metadata (via ModelScope or HF Hub)
   python data/download.py --with-swebench
   # OR manually with the ModelScope SDK:
   # python -c "from modelscope import snapshot_download; snapshot_download('princeton-nlp/SWE-bench_Lite', repo_type='dataset', cache_dir='./data/cache')"

   # 2) clone the repos into data/repos/ (full history so checkout works)
   mkdir -p data/repos
   cd data/repos && git clone https://github.com/sqlfluff/sqlfluff.git
   cd sqlfluff && git fetch --unshallow

   # 3) run the ablation
   python ablations/swebench_sample.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import CodingAgent  # noqa: E402
from src.llm_client import LLMError  # noqa: E402
from src.tools.base import clear_read_registry  # noqa: E402

SAMPLE = ROOT / "data" / "swebench-lite-sample.parquet"
REPOS = ROOT / "data" / "repos"
RESULTS = ROOT / "ablations" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Generous defaults — SWE-bench instances may take many minutes to fix.
AGENT_MAX_TURNS = int(os.environ.get("SWE_BENCH_MAX_TURNS", "10"))


def _git(repo: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        cwd=str(repo), check=False, shell=False,
        capture_output=True, text=True, timeout=timeout,
    )


def _original_head(repo: Path) -> str:
    out = _git(repo, "rev-parse", "HEAD", timeout=30)
    return (out.stdout or "").strip()


def _restore_head(repo: Path, original_head: str) -> None:
    _git(repo, "reset", "--hard", original_head, timeout=60)
    _git(repo, "clean", "-fd", timeout=60)


def _checkout(repo: Path, commit: str) -> bool:
    cp = _git(repo, "checkout", commit, "--", timeout=60)
    return cp.returncode == 0


# Snapshot every file under ``repo`` so we can roll back any state the
# agent (or git_apply) leaves behind. Even if ``_restore_head`` + ``git
# clean`` lose track (e.g. .gitignored files, files the agent created),
# this per-file snapshot catches them.
_SNAPSHOT_EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".tox", "node_modules"}


def _snapshot_dir(repo: Path) -> dict:
    """Return ``{rel_path: bytes}`` for every regular file under ``repo``."""
    snap: dict = {}
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SNAPSHOT_EXCLUDE_DIRS for part in p.relative_to(repo).parts):
            continue
        try:
            snap[str(p.relative_to(repo))] = p.read_bytes()
        except OSError:
            continue
    return snap


def _restore_dir(repo: Path, snapshot: dict) -> None:
    """Restore files to a prior snapshot, deleting anything not in it."""
    # Remove any current file that's not in the snapshot.
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(repo)
        if any(part in _SNAPSHOT_EXCLUDE_DIRS for part in rel.parts):
            continue
        if str(rel) not in snapshot:
            try:
                p.unlink()
            except OSError:
                pass
    # Re-write snapshot bytes.
    for rel, data in snapshot.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_bytes(data)
        except OSError:
            pass


def _attempt_instance(row: dict, smoke: bool) -> dict:
    repo_name = row["repo"].split("/")[-1]
    repo = REPOS / repo_name
    if not repo.exists():
        return {"instance_id": row["instance_id"], "skipped": f"repo missing: {repo}"}

    original_head = _original_head(repo)
    clear_read_registry()

    # Per-file snapshot — rollback guarantee even if ``_restore_head``
    # loses track of files the agent (or git_apply) created or modified
    # outside the working tree (e.g. .gitignored, or untracked files).
    snapshot = _snapshot_dir(repo)

    try:
        if not _checkout(repo, row["base_commit"]):
            return {"instance_id": row["instance_id"],
                    "skipped": f"checkout failed at {row['base_commit']}"}

        try:
            agent = CodingAgent(max_turns=AGENT_MAX_TURNS, bootstrap_explore=True)
        except LLMError as e:
            return {"instance_id": row["instance_id"],
                    "skipped": f"LLM init failed: {e}"}

        if smoke:
            return {
                "instance_id": row["instance_id"],
                "smoke": True,
                "tests_passed": False,
                "duration_ms": 0,
            }

        try:
            # Override verify_tests to use a longer timeout — sqlfluff's
            # full pytest suite takes ~100s; 180s gives us headroom.
            original_verify = agent._verify_tests
            agent._verify_tests = lambda *a, **kw: original_verify(*a, timeout=180)
            trace = agent.run(str(repo), row["problem_statement"])
            agent._verify_tests = original_verify
        except Exception as e:  # noqa: BLE001
            return {"instance_id": row["instance_id"], "error": str(e)[:200]}

        passed = bool(trace.get("tests_passed"))
        usage = trace.get("token_usage") or {}
        return {
            "instance_id": row["instance_id"],
            "tests_passed": passed,
            "done_reason": trace.get("done_reason"),
            "turn_count": trace.get("turn_count"),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        }
    finally:
        # Restore HEAD first (cheap), then restore from per-file snapshot
        # in case anything slipped through (agent-written files outside
        # git, .gitignored files, etc.).
        try:
            _restore_head(repo, original_head)
        except Exception:
            pass
        try:
            _restore_dir(repo, snapshot)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only attempt the first N instances")
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
    for i, (_, row) in enumerate(df.iterrows()):
        if args.limit is not None and i >= args.limit:
            break
        print(f"\n=== [{i+1}/{len(df)}] {row['instance_id']} ===", flush=True)
        r = _attempt_instance(dict(row), args.smoke)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        results.append(r)

    out = RESULTS / "swebench.jsonl"
    with out.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n=== Summary ===")
    attempted = [r for r in results if not r.get("skipped") and not r.get("smoke")]
    passed = sum(1 for r in attempted if r.get("tests_passed"))
    print(f"attempted={len(attempted)} passed={passed} skipped={len(results)-len(attempted)}")
    return 0 if passed >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())