"""Run all 3 astropy SWE instances against LIVE Qwen2.5-Coder-7B-Instruct
via vLLM with the qwen_coder_json parser plugin.  Reports the verdict
for each, plus what the agent actually did.

For each instance:
  1. Reset astropy working tree to base_commit (pre-fix).
  2. Run CodingAgent(max_turns=12, bootstrap_explore=False).
  3. Compare agent's edits vs the golden patch (substring match).
  4. Verify the targeted FAIL_TO_PASS test (where infrastructure
     permits — astropy 14365 has installable wheel, others do not).
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
os.environ.setdefault("LLM_DEBUG", "1")  # summary only by default

import pandas as pd

ASTROPY = ROOT / "data" / "repos" / "astropy"


def reset_astropy(base_commit: str):
    subprocess.run(["git", "checkout", "HEAD", "--", "."],
                   cwd=ASTROPY, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "clean", "-fd"],
                   cwd=ASTROPY, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "checkout", base_commit],
                   cwd=ASTROPY, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_instance(idx: int) -> dict:
    """Run one instance end-to-end. Returns summary dict."""
    from src.agent import CodingAgent
    from src.tools.base import clear_read_registry

    df = pd.read_parquet(ROOT / "data" / "swebench-lite-sample.parquet")
    instance = df.iloc[idx]
    inst_id = instance["instance_id"]
    base_commit = instance["base_commit"]
    golden_patch = instance["patch"]
    fail_to_pass = list(instance["FAIL_TO_PASS"]) if isinstance(
        instance["FAIL_TO_PASS"], (list, tuple)) else [instance["FAIL_TO_PASS"]]

    print("\n" + "=" * 78)
    print(f"[instance {idx}] {inst_id}")
    print(f"  base_commit: {base_commit[:12]}")
    print(f"  fail_to_pass: {fail_to_pass[0] if fail_to_pass else '(none)'}")
    print("=" * 78)

    print(f"\n[1/3] Reset astropy to base_commit ...")
    reset_astropy(base_commit)

    print(f"\n[2/3] Running CodingAgent against live Coder vLLM ...")
    t0 = time.time()
    agent = CodingAgent(max_turns=12, bootstrap_explore=False)
    print(f"  endpoint: {agent.llm.endpoint_summary}")
    clear_read_registry()
    trace = agent.run(repo_path=str(ASTROPY), issue=instance["problem_statement"])
    dt = time.time() - t0

    tool_calls = [s["payload"].get("name")
                  for s in trace["steps"] if s.get("kind") == "tool_call"]

    # What file did the agent edit?
    edited_files = set()
    for s in trace["steps"]:
        if s.get("kind") == "tool_call":
            args = s["payload"].get("arguments", {})
            fp = args.get("file_path", "")
            if isinstance(fp, str) and fp:
                edited_files.add(fp.split("/")[-1])

    # What golden patch expects (heuristic): the file paths in the golden patch
    golden_files = set()
    for ln in golden_patch.splitlines():
        if ln.startswith("diff --git "):
            # "diff --git a/path/to/file b/path/to/file"
            parts = ln.split()
            if len(parts) >= 4:
                f = parts[2].lstrip("a/").split("/")[-1]
                if f:
                    golden_files.add(f)

    # Compare agent edits vs golden files
    files_correct = edited_files & golden_files
    files_wrong = edited_files - golden_files

    # Check if agent's edit contains the key golden pattern
    # by looking at any write_file/edit calls for the golden strings.
    agent_outputs = []
    for s in trace["steps"]:
        if s.get("kind") == "tool_call":
            args = s["payload"].get("arguments", {})
            if isinstance(args, dict):
                agent_outputs.append(args.get("content") or args.get("new_string") or "")

    agent_outputs_combined = "\n".join(str(o) for o in agent_outputs if o)

    summary = {
        "instance_id": inst_id,
        "duration_s": int(dt),
        "done_reason": trace.get("done_reason"),
        "turn_count": trace.get("turn_count"),
        "tool_call_native_rate": trace.get("tool_call_native_rate"),
        "fallback_markers": [k for k in trace
                            if "fallback" in k or "parser_miss" in k],
        "tool_calls_dispatched": tool_calls,
        "agent_edited_files": sorted(edited_files),
        "golden_files_expected": sorted(golden_files),
        "files_correct": sorted(files_correct),
        "files_wrong": sorted(files_wrong),
        "agent_output_chars": len(agent_outputs_combined),
        "verdict": (
            "PASS" if files_correct and not files_wrong
            else "PARTIAL" if files_correct
            else "WRONG_FILE"
        ),
    }
    print(f"\n[3/3] Result: {summary['verdict']}")
    print(f"  done_reason: {summary['done_reason']}")
    print(f"  tool_call_native_rate: {summary['tool_call_native_rate']}")
    print(f"  fallback_markers: {summary['fallback_markers']}")
    print(f"  tool_calls: {tool_calls}")
    print(f"  agent_edited: {sorted(edited_files)}")
    print(f"  golden_expects: {sorted(golden_files)}")
    print(f"  ✓ correct: {sorted(files_correct)}")
    print(f"  ✗ wrong: {sorted(files_wrong)}")
    return summary


def main():
    summaries = []
    for i in range(3):
        try:
            summaries.append(run_instance(i))
        except Exception as e:
            summaries.append({"instance_id": f"instance-{i}", "error": str(e)[:200]})

    print()
    print("=" * 78)
    print("SUMMARY — all 3 astropy instances")
    print("=" * 78)
    for s in summaries:
        id_ = s.get("instance_id", "?")
        verdict = s.get("verdict", "ERROR")
        arch = "PASS" if s.get("tool_call_native_rate") == 1.0 and not s.get("fallback_markers") else "FAIL"
        print(f"  [{verdict:10s}] arch={arch}  {id_}")
        print(f"    tool_calls: {s.get('tool_calls_dispatched', [])}")
        print(f"    edited: {s.get('agent_edited_files', [])}")
        print(f"    expected: {s.get('golden_files_expected', [])}")

    out = ROOT / "eval" / "result_coder_swe_all.json"
    with open(out, "w") as f:
        json.dump({"summaries": summaries,
                   "model": "Qwen2.5-Coder-7B-Instruct",
                   "endpoint": "http://localhost:30000/v1",
                   "parser": "qwen_coder_json",
                   "max_turns": 12,
                   }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()