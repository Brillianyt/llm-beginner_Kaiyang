"""Task-6 self-check — exercises the 4 contract points from blueprint Part VI.

Run with ``python eval/run.py``.

* ``mcp_server_lists_tools`` — list_tools() yields ≥ 5 tools (M1).
* ``skill_loader_metadata``   — SkillLoader.list_skills() returns items
  with ``name`` and ``description`` (M2).
* ``toy_repo_patch``         — agent.run() fixes ``calculator.add`` and
  ``pytest`` exits 0 (M3). Requires a live LLM endpoint
  (``OPENAI_BASE_URL`` + ``QWEN_MODEL``).
* ``swebench_lite_sample``   — optional: 1+ pass on 3 sampled SWE-bench
  Lite instances (S4). Skipped unless the parquet + repos are present.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from _eval_harness import run_tests  # noqa: E402


def test_mcp_server_lists_tools():
    try:
        from src.mcp_server import list_tools
    except ImportError as e:
        return {"test": "mcp_server_lists_tools", "pass": False,
                "error": f"src/mcp_server.py must export list_tools(): {e}"}
    tools = list_tools()
    names = [t.get("name") if isinstance(t, dict) else str(t) for t in tools]
    return {
        "test": "mcp_server_lists_tools",
        "pass": isinstance(tools, list) and len(tools) >= 5,
        "tools": names,
    }


def test_skill_loader_metadata():
    from src.skill_loader import SkillLoader
    loader = SkillLoader(str(ROOT / "src" / "skills"))
    skills = loader.list_skills()
    if not skills:
        return {"test": "skill_loader_metadata", "pass": False,
                "error": "no skills scanned"}
    missing = [s for s in skills
               if not (isinstance(s, dict) and s.get("name") and s.get("description"))]
    return {
        "test": "skill_loader_metadata",
        "pass": len(skills) >= 2 and not missing,
        "count": len(skills),
        "missing_meta": [s.get("name", "?") for s in missing],
    }


def test_toy_repo_patch():
    from src.agent import CodingAgent
    from src.skill_loader import SkillLoader
    toy_repo = ROOT / "data" / "toy-repo"
    issue_path = toy_repo / "ISSUE.md"
    if not issue_path.exists():
        return {"test": "toy_repo_patch", "pass": None,
                "skip": "data/toy-repo missing — run data/download.py"}

    # Reset to the buggy snapshot so we always start from scratch.
    buggy = toy_repo / "calculator.py.orig"
    if not buggy.exists():
        return {"test": "toy_repo_patch", "pass": None,
                "skip": "calculator.py.orig snapshot missing"}
    shutil.copy(buggy, toy_repo / "calculator.py")
    # Clear the in-process read-before-write registry so the agent must
    # observe the buggy file via read_file before editing it.
    from src.tools.base import clear_read_registry
    clear_read_registry()

    agent = CodingAgent(skill_loader=SkillLoader(str(ROOT / "src" / "skills")))
    issue = issue_path.read_text(encoding="utf-8")
    trace = agent.run(repo_path=str(toy_repo), issue=issue)

    test_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=toy_repo, shell=False, text=True, capture_output=True, timeout=60,
    )
    return {
        "test": "toy_repo_patch",
        "pass": test_run.returncode == 0,
        "tests_passed": test_run.returncode == 0,
        "patch_present": bool(isinstance(trace, dict) and trace.get("patch")),
        "steps": len(trace.get("steps", [])),
        "done_reason": trace.get("done_reason"),
        "pytest_output_tail": (test_run.stdout + test_run.stderr)[-800:],
    }


def test_swebench_lite_sample():
    sample = ROOT / "data" / "swebench-lite-sample.parquet"
    if not sample.exists():
        return {"test": "swebench_lite_sample", "pass": None,
                "skip": "data/swebench-lite-sample.parquet missing — run data/download.py --with-swebench"}
    try:
        import pandas as pd
        df = pd.read_parquet(sample)
    except Exception as e:
        return {"test": "swebench_lite_sample", "pass": False, "error": str(e)}

    from src.agent import CodingAgent
    agent = CodingAgent()
    passed, attempted = 0, 0
    results = []
    for _, row in df.iterrows():
        repo_path = ROOT / "data" / "repos" / row["repo"].split("/")[-1]
        if not repo_path.exists():
            results.append({"id": row["instance_id"],
                            "skip": f"repo not cloned: {repo_path.relative_to(ROOT)}"})
            continue
        attempted += 1
        try:
            trace = agent.run(repo_path=str(repo_path), issue=row["problem_statement"])
            ok = bool(trace.get("tests_passed"))
            passed += int(ok)
            results.append({"id": row["instance_id"], "tests_passed": ok})
        except Exception as e:
            results.append({"id": row["instance_id"], "error": str(e)[:120]})
    if attempted == 0:
        return {"test": "swebench_lite_sample", "pass": None,
                "skip": "metadata present but data/repos/ missing", "details": results}
    return {
        "test": "swebench_lite_sample",
        "pass": passed >= 1,
        "passed": passed,
        "attempted": attempted,
        "details": results,
    }


if __name__ == "__main__":
    run_tests([
        test_mcp_server_lists_tools,
        test_skill_loader_metadata,
        test_toy_repo_patch,
        test_swebench_lite_sample,
    ], ROOT)