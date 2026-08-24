"""Render a captured-wire JSON into a fully-readable Markdown trace.

Each HTTP request the agent sent to vLLM gets its own section with:
- Sampling params (model, temperature, max_tokens, tool_choice)
- ALL messages (system, user, assistant, tool) — full content verbatim
- Each assistant turn's tool_calls — full arguments dumped
- Each tool response — full content dumped
- Usage tokens
- finish_reason / response snippet

This is the audit-grade view of what the agent actually sent.  No
truncation.  No filtering.  Just full text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fmt_msg(m: dict, idx: int) -> str:
    role = m.get("role", "?")
    out = [f"#### Message [{idx}] role=`{role}`"]
    if "content" in m and m["content"] is not None:
        c = m["content"]
        out.append("```text")
        out.append(c if c else "(empty)")
        out.append("```")
    if "tool_calls" in m and m["tool_calls"]:
        out.append("")
        out.append("**Tool calls:**")
        for j, tc in enumerate(m["tool_calls"]):
            fn = tc.get("function", {})
            tcid = tc.get("id", "?")
            name = fn.get("name", "?")
            args_raw = fn.get("arguments", "")
            out.append(f"- `tool_call_id={tcid}`  **{name}**")
            out.append("")
            out.append("```json")
            try:
                out.append(json.dumps(json.loads(args_raw),
                                      indent=2, ensure_ascii=False))
            except Exception:
                out.append(args_raw)
            out.append("```")
    if "tool_call_id" in m:
        out.append(f"\n*tool_call_id: `{m['tool_call_id']}`*")
    out.append("")
    return "\n".join(out)


def render_capture(json_path: Path, out_path: Path) -> None:
    with open(json_path) as f:
        data = json.load(f)
    summary = data.get("summary", {})
    caps = data.get("captured_http_requests", [])

    lines = []
    lines.append(f"# Wire capture trace — {json_path.stem}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k in ("label", "timestamp", "endpoint", "max_turns",
              "duration_s", "done_reason", "tests_passed",
              "turn_count", "step_count", "tool_call_native_rate",
              "fallback_markers", "n_captured_requests"):
        if k in summary:
            v = summary[k]
            lines.append(f"- **{k}**: `{v}`")
    lines.append(f"- **tool_calls_dispatched**: "
                 f"{summary.get('tool_calls_dispatched', [])}")
    lines.append("")
    lines.append(f"Total HTTP requests captured: **{len(caps)}**")
    lines.append("")

    for i, cap in enumerate(caps, 1):
        req = cap["request_body_full"]
        resp = cap.get("response_body", {})
        lines.append("---")
        lines.append("")
        lines.append(f"## Request {i}")
        lines.append("")
        lines.append("### Sampling params")
        lines.append("")
        lines.append(f"- model: `{req.get('model')}`")
        lines.append(f"- temperature: `{req.get('temperature')}`")
        lines.append(f"- max_tokens: `{req.get('max_tokens')}`")
        lines.append(f"- tool_choice: `{req.get('tool_choice')}`")
        lines.append(f"- n_tools: `{len(req.get('tools', []))}`")
        lines.append(f"- n_messages: `{len(req.get('messages', []))}`")
        lines.append(f"- request_body_bytes: `{cap.get('request_body_bytes')}`")
        if "response_status" in cap:
            lines.append(f"- response_status: `{cap['response_status']}`")
        lines.append("")

        tools = req.get("tools", [])
        if tools:
            lines.append("### Tools schema (function names only)")
            lines.append("")
            for t in tools:
                lines.append(f"- `{t.get('function', {}).get('name')}`")
            lines.append("")

        lines.append("### Messages (full, in order)")
        lines.append("")
        for j, m in enumerate(req.get("messages", [])):
            lines.append(fmt_msg(m, j))
            lines.append("")

        if resp:
            lines.append("### Response")
            lines.append("")
            lines.append(f"- finish_reason: `{resp.get('choices',[{}])[0].get('finish_reason')}`")
            usage = resp.get("usage", {})
            if usage:
                lines.append(f"- usage: prompt={usage.get('prompt_tokens')}, "
                             f"completion={usage.get('completion_tokens')}, "
                             f"total={usage.get('total_tokens')}")
            choice = resp.get("choices", [{}])[0]
            msg = choice.get("message", {})
            if msg.get("content"):
                lines.append("")
                lines.append("**Assistant content:**")
                lines.append("")
                lines.append("```text")
                lines.append(msg["content"])
                lines.append("```")
                lines.append("")
            tcs = msg.get("tool_calls", [])
            if tcs:
                lines.append("**Response tool_calls (with full arguments):**")
                lines.append("")
                for j, tc in enumerate(tcs):
                    fn = tc.get("function", {})
                    lines.append(f"- [{j}] **{fn.get('name')}**  id=`{tc.get('id')}`")
                    lines.append("")
                    lines.append("```json")
                    try:
                        lines.append(json.dumps(
                            json.loads(fn.get("arguments", "")),
                            indent=2, ensure_ascii=False))
                    except Exception:
                        lines.append(fn.get("arguments", ""))
                    lines.append("```")
                    lines.append("")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture", help="path to a captured JSON (or glob)")
    p.add_argument("--out", help="output .md path (default: alongside with .md)")
    args = p.parse_args()

    src = Path(args.capture)
    if src.is_dir():
        candidates = sorted(src.glob("*.json"))
    elif "*" in args.capture:
        candidates = sorted(Path(".").glob(args.capture))
    else:
        candidates = [src]
    if not candidates:
        print(f"No captures matched: {args.capture}", file=sys.stderr)
        sys.exit(1)

    for c in candidates:
        out = Path(args.out) if args.out else c.with_suffix(".md")
        render_capture(c, out)


if __name__ == "__main__":
    main()