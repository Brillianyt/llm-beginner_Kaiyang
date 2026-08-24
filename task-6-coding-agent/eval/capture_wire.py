"""Capture the EXACT HTTP request bodies that CodingAgent sends to vLLM.

Monkey-patches ``httpx.Client.send`` (which the openai SDK uses) to
record every POST /v1/chat/completions body before it goes on the wire.

Use this whenever you need ground truth on what the agent is actually
sending — system prompt, tool schemas, message history, sampling
params.  Trust nothing else: summaries like "content_chars: N" lose
information.

Run::

    python eval/capture_wire.py --label my_run --max-turns 20

It will:
1. Set QWEN_MODEL / OPENAI_BASE_URL to point at the running vLLM
   (defaults: model ``models/Qwen2.5-Coder-7B-Instruct``,
   URL ``http://localhost:30000/v1``).
2. Install the httpx hook.
3. Run ``CodingAgent(max_turns=...).run(repo_path, issue)``.
4. Save the captured bodies to
   ``eval/wire_captures/<label>__<timestamp>.json``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Default LLM endpoint (Coder vLLM with my parser plugin).
DEFAULT_BASE_URL = "http://localhost:30000/v1"
DEFAULT_MODEL = "models/Qwen2.5-Coder-7B-Instruct"


def install_capture_hook():
    """Install an httpx hook that records every /v1/chat/completions
    request body.  Returns ``(captured_list, lock)``.

    The captured list contains FULL request bodies (not truncated),
    plus the response status for cross-checking.
    """
    import httpx
    captured: list = []
    lock = threading.Lock()
    orig_send = httpx.Client.send

    def _send(self, request, **kwargs):
        try:
            body = request.content
            if body:
                try:
                    parsed = json.loads(body)
                except Exception:
                    parsed = None
                with lock:
                    captured.append({
                        "url": str(request.url),
                        "method": request.method,
                        "request_body_full": parsed,
                        "request_body_bytes": len(body) if body else 0,
                    })
        except Exception:
            pass
        resp = orig_send(self, request, **kwargs)
        try:
            with lock:
                if captured:
                    captured[-1]["response_status"] = resp.status_code
                    body_bytes = resp.read()
                    resp.read = lambda: body_bytes  # restore for upstream
                    try:
                        captured[-1]["response_body"] = json.loads(body_bytes)
                    except Exception:
                        captured[-1]["response_body_bytes"] = len(body_bytes)
        except Exception:
            pass
        return resp

    httpx.Client.send = _send
    return captured, lock


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True,
                   help="label for this run, e.g. 'toy_repo_native'")
    p.add_argument("--repo-path", default=str(ROOT / "data" / "toy-repo"))
    p.add_argument("--issue", default=None,
                   help="issue text; default reads ISSUE.md from repo-path")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--bootstrap-explore", action="store_true")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    os.environ["QWEN_MODEL"] = args.model
    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_KEY"] = "EMPTY"

    issue = args.issue
    if issue is None:
        issue_path = Path(args.repo_path) / "ISSUE.md"
        if issue_path.exists():
            issue = issue_path.read_text(encoding="utf-8")
        else:
            issue = "(no issue provided)"

    captured, lock = install_capture_hook()
    print(f"[capture_wire] hook installed. running agent...")
    print(f"  endpoint : {args.base_url}")
    print(f"  model    : {args.model}")
    print(f"  repo     : {args.repo_path}")
    print(f"  max_turns: {args.max_turns}")

    from src.agent import CodingAgent
    from src.tools.base import clear_read_registry
    clear_read_registry()

    agent = CodingAgent(
        max_turns=args.max_turns,
        bootstrap_explore=args.bootstrap_explore,
    )
    t0 = time.time()
    trace = agent.run(repo_path=args.repo_path, issue=issue)
    dt = time.time() - t0

    summary = {
        "label": args.label,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "endpoint": agent.llm.endpoint_summary,
        "max_turns": args.max_turns,
        "duration_s": int(dt),
        "done_reason": trace.get("done_reason"),
        "tests_passed": bool(trace.get("tests_passed")),
        "turn_count": trace.get("turn_count"),
        "step_count": len(trace.get("steps", [])),
        "tool_call_native_rate": trace.get("tool_call_native_rate"),
        "fallback_markers": [k for k in trace
                            if "fallback" in k or "parser_miss" in k],
        "tool_calls_dispatched": [
            s["payload"].get("name")
            for s in trace.get("steps", [])
            if s.get("kind") == "tool_call"
        ],
        "n_captured_requests": len(captured),
    }

    out_dir = ROOT / "eval" / "wire_captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{args.label}__{ts}.json"

    payload = {
        "summary": summary,
        "trace_keys": list(trace.keys()),
        "captured_http_requests": captured,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print()
    print("=" * 70)
    print(f"CAPTURE SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()