"""Single-instance SWE driver for astropy-14365 with FULL debug output.

Uses ``LLM_DEBUG=2`` to print every message / tool call / argument the
agent sends to vLLM, plus the model's responses.  Verifies the agent's
edit against the FAIL_TO_PASS test (``test_roundtrip[True]``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["QWEN_MODEL"] = "models/Qwen2.5-Coder-7B-Instruct"
os.environ["OPENAI_BASE_URL"] = "http://localhost:30000/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_MAX_TOKENS"] = "4096"
os.environ.setdefault("LLM_DEBUG", "2")

from src.agent import CodingAgent
from src.tools.base import clear_read_registry


WHEEL_QDP = "/tmp/astropy_lib/astropy/io/ascii/qdp.py"
CLONED_QDP = ROOT / "data" / "repos" / "astropy" / "astropy" / "io" / "ascii" / "qdp.py"
CLONED_ASTROPY = ROOT / "data" / "repos" / "astropy"


def reset_cloned_to_head():
    subprocess.run(["git", "checkout", "HEAD", "--", "."],
                   cwd=CLONED_ASTROPY, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "clean", "-fd"],
                   cwd=CLONED_ASTROPY, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_fail_to_pass_test(qdp_source: Path) -> bool:
    """Test against wheel's compiled astropy with qdp_source as the
    patched file."""
    shutil.copy(qdp_source, WHEEL_QDP)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header",
         "--rootdir=/tmp/astropy_lib",
         "-p", "no:cacheprovider", "--noconftest",
         "-W", "ignore::DeprecationWarning",
         "/tmp/astropy_lib/astropy/io/ascii/tests/test_qdp.py::test_roundtrip"],
        capture_output=True, text=True, timeout=120,
    )
    return r.returncode == 0


def main():
    print("=" * 78)
    print("LIVE Coder vLLM · astropy-14365 (qdp case-insensitive)")
    print("LLM_DEBUG=2 — every wire message dumped below")
    print("=" * 78)

    print("\n[1/5] Reset cloned astropy to HEAD pre-fix ...")
    reset_cloned_to_head()
    shutil.copy(CLONED_QDP, WHEEL_QDP)

    print("\n[2/5] Verify pre-fix state (FAIL_TO_PASS expected to FAIL) ...")
    pre = run_fail_to_pass_test(CLONED_QDP)
    print(f"  pre-fix test passed? {pre} (expected False)")

    print("\n[3/5] Running CodingAgent against live Coder vLLM (LLM_DEBUG=2) ...")
    t0 = time.time()
    agent = CodingAgent(max_turns=12, bootstrap_explore=False)
    print(f"  endpoint: {agent.llm.endpoint_summary}")
    clear_read_registry()
    issue = (
        "ascii.qdp assumes QDP commands are upper case. Make them case-insensitive. "
        "Fix astropy/io/ascii/qdp.py so that 'read serr 1 2' (lowercase) parses "
        "the same as 'READ SERR 1 2'."
    )
    trace = agent.run(repo_path=str(CLONED_ASTROPY), issue=issue)
    dt = time.time() - t0
    print(f"\n[4/5] Agent done in {dt:.0f}s")
    print(f"  done_reason: {trace.get('done_reason')}")
    print(f"  turn_count: {trace.get('turn_count')}")
    print(f"  tool_call_native_rate: {trace.get('tool_call_native_rate')}")
    print(f"  fallback_markers: {[k for k in trace if 'fallback' in k or 'parser_miss' in k]}")
    print(f"  tool_calls_dispatched: "
          f"{[s['payload'].get('name') for s in trace['steps'] if s.get('kind') == 'tool_call']}")

    print("\n[5/5] Testing agent's edit against FAIL_TO_PASS ...")
    post = run_fail_to_pass_test(CLONED_QDP)
    agent_qdp = CLONED_QDP.read_text()
    has_ignorecase = "re.IGNORECASE" in agent_qdp
    has_upper_no = '.upper() == "NO"' in agent_qdp
    print(f"  post-fix test passed? {post}")
    print(f"  agent has re.IGNORECASE flag: {has_ignorecase}")
    print(f"  agent has v.upper() == 'NO':  {has_upper_no}")

    print()
    print("=" * 78)
    print(f"VERDICT: {'PASS' if post else 'FAIL'}")
    print("=" * 78)


if __name__ == "__main__":
    main()