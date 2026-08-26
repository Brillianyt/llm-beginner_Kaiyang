"""Single-instance SWE driver with full wire capture.

Runs ONE astropy instance against live vLLM, dumping every HTTP request/response
to disk so we can inspect what the chat_template produced + how the parser
parsed the model output.

Defaults to astropy-12907 (was the prior PASS). Override via --instance-idx.
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

os.environ["QWEN_MODEL"] = "Qwen2.5-Coder-7B-Instruct"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:30000/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_MAX_TOKENS"] = "4096"
os.environ.setdefault("LLM_DEBUG", "2")  # verbose: print every wire message


def install_capture_hook():
    """Monkey-patch httpx to record every POST /v1/chat/completions body."""
    import httpx, threading
    captured: list = []
    lock = threading.Lock()
    orig_send = httpx.Client.send

    def _send(self, request, **kwargs):
        try:
            body = request.content
            parsed = None
            if body:
                try:
                    parsed = json.loads(body)
                except Exception:
                    pass
            with lock:
                captured.append({
                    "url": str(request.url),
                    "method": request.method,
                    "request_body": parsed,
                    "request_body_bytes": len(body) if body else 0,
                })
        except Exception:
            pass
        resp = orig_send(self, request, **kwargs)
        try:
            body_bytes = resp.read()
            with lock:
                if captured:
                    captured[-1]["response_status"] = resp.status_code
                    try:
                        captured[-1]["response_body"] = json.loads(body_bytes)
                    except Exception:
                        captured[-1]["response_body_bytes"] = len(body_bytes)
            # Replay the body back into the response so upstream consumers
            # (e.g. the OpenAI SDK's raise_for_status path) can still read it.
            resp.read = lambda: body_bytes
        except Exception:
            pass
        return resp

    httpx.Client.send = _send
    return captured, lock


def reset_astropy(base_commit: str):
    astropy = ROOT / "data" / "repos" / "astropy"
    subprocess.run(["git", "checkout", "HEAD", "--", "."],
                   cwd=astropy, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "clean", "-fd"],
                   cwd=astropy, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "checkout", base_commit],
                   cwd=astropy, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-idx", type=int, default=0,
                    help="0=12907 (PASS), 1=14182 (WRONG_FILE), 2=14365 (PASS)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(ROOT / "data" / "swebench-lite-sample.parquet")
    instance = df.iloc[args.instance_idx]
    inst_id = instance["instance_id"]
    base_commit = instance["base_commit"]

    captured, lock = install_capture_hook()
    print(f"[capture_wire] hook installed.")

    print("=" * 78)
    print(f"LIVE Coder vLLM · {inst_id}")
    print(f"  base_commit: {base_commit[:12]}")
    print("=" * 78)

    print("\n[1/3] Reset astropy to base_commit ...")
    reset_astropy(base_commit)

    print(f"\n[2/3] Running CodingAgent (max_turns={args.max_turns}) ...")
    from src.agent import CodingAgent
    from src.tools.base import clear_read_registry
    t0 = time.time()
    agent = CodingAgent(max_turns=args.max_turns, bootstrap_explore=False)
    print(f"  endpoint: {agent.llm.endpoint_summary}")
    clear_read_registry()
    trace = agent.run(repo_path=str(ROOT / "data" / "repos" / "astropy"),
                      issue=instance["problem_statement"])
    dt = time.time() - t0

    print(f"\n[3/3] Agent done in {dt:.0f}s")
    print(f"  done_reason: {trace.get('done_reason')}")
    print(f"  turn_count: {trace.get('turn_count')}")
    print(f"  tool_call_native_rate: {trace.get('tool_call_native_rate')}")
    print(f"  fallback_markers: {[k for k in trace if 'fallback' in k or 'parser_miss' in k]}")
    tool_calls = [s["payload"].get("name")
                  for s in trace.get("steps", []) if s.get("kind") == "tool_call"]
    print(f"  tool_calls_dispatched: {tool_calls}")

    edited_files = sorted({s["payload"].get("arguments", {}).get("file_path", "").split("/")[-1]
                           for s in trace.get("steps", []) if s.get("kind") == "tool_call"
                           and s["payload"].get("arguments", {}).get("file_path")})
    print(f"  edited_files: {edited_files}")

    # Golden patch expected files
    golden_files = []
    for ln in instance["patch"].splitlines():
        if ln.startswith("diff --git "):
            parts = ln.split()
            if len(parts) >= 4:
                f = parts[2].lstrip("a/").split("/")[-1]
                if f:
                    golden_files.append(f)
    print(f"  golden_files: {sorted(set(golden_files))}")

    # Save everything
    out_dir = ROOT / "eval" / "wire_captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = out_dir / f"{args.label}__{ts}.json"
    payload = {
        "summary": {
            "label": args.label,
            "instance_id": inst_id,
            "endpoint": agent.llm.endpoint_summary,
            "max_turns": args.max_turns,
            "duration_s": int(dt),
            "done_reason": trace.get("done_reason"),
            "turn_count": trace.get("turn_count"),
            "tool_call_native_rate": trace.get("tool_call_native_rate"),
            "fallback_markers": [k for k in trace if "fallback" in k or "parser_miss" in k],
            "tool_calls_dispatched": tool_calls,
            "edited_files": edited_files,
            "golden_files": sorted(set(golden_files)),
            "verdict": (
                "PASS" if edited_files and set(edited_files) & set(golden_files) and not (set(edited_files) - set(golden_files))
                else "PARTIAL" if set(edited_files) & set(golden_files)
                else "WRONG_FILE"
            ),
        },
        "captured_http_requests": captured,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nWrote: {out_path}")
    print(f"VERDICT: {payload['summary']['verdict']}")


if __name__ == "__main__":
    main()