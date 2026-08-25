"""Cross-capture analytics over eval/wire_captures/*.json.

Runs in-process — no LLM calls, no vLLM.  Produces aggregate stats
that answer the data-questions without re-running anything:

* tool-call frequency per verdict
* sequence motifs (e.g. read_file -> edit -> run_tests)
* stuck detector precision: when done_reason=stuck, was the agent
  actually stuck?
* token cost per verdict
* skill_load rate across all runs
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

CAPTURES = Path("eval/wire_captures")


def load_all() -> List[Dict[str, Any]]:
    out = []
    for p in sorted(CAPTURES.glob("*.json")):
        try:
            out.append((p.name, json.loads(p.read_text())))
        except Exception as e:
            print(f"  skip {p.name}: {e}", file=sys.stderr)
    return out


def main() -> None:
    runs = load_all()
    print(f"=== {len(runs)} captures loaded ===\n")

    # ------------------------------------------------------------------
    # A. Tool-call frequency
    # ------------------------------------------------------------------
    print("## A. Tool-call frequency (across all turns of all runs)")
    tool_counter: Counter = Counter()
    per_run_tools: Dict[str, Counter] = {}
    for name, d in runs:
        c: Counter = Counter()
        # Walk response_message.tool_calls for every turn
        for req in d.get("captured_http_requests", []):
            try:
                resp = req["response_body"]
                choices = resp.get("choices") or []
                for ch in choices:
                    msg = ch.get("message") or {}
                    for tc in (msg.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        tname = fn.get("name", "<none>")
                        c[tname] += 1
                        tool_counter[tname] += 1
            except Exception:
                pass
        per_run_tools[name] = c
    for t, n in tool_counter.most_common():
        print(f"  {t:18s}  {n:5d}")
    print()

    # ------------------------------------------------------------------
    # B. Sequence motifs (3-tool sliding window)
    # ------------------------------------------------------------------
    print("## B. Sequence motifs (3-tool sliding window, top 10)")
    motif_counter: Counter = Counter()
    for name, d in runs:
        seq: List[str] = []
        for req in d.get("captured_http_requests", []):
            try:
                resp = req["response_body"]
                for ch in (resp.get("choices") or []):
                    for tc in ((ch.get("message") or {}).get("tool_calls") or []):
                        seq.append((tc.get("function") or {}).get("name", "?"))
            except Exception:
                pass
        for i in range(len(seq) - 2):
            motif_counter[tuple(seq[i:i + 3])] += 1
    for motif, n in motif_counter.most_common(10):
        print(f"  {' -> '.join(motif):50s}  {n:4d}")
    print()

    # ------------------------------------------------------------------
    # C. Stuck-detector precision
    # ------------------------------------------------------------------
    print("## C. Stuck-detector precision")
    stuck_runs = []
    for name, d in runs:
        s = d.get("summary", {})
        if s.get("done_reason") == "stuck":
            stuck_runs.append((name, s, d))
    print(f"  {len(stuck_runs)} runs hit done_reason=stuck:")
    for name, s, d in stuck_runs:
        # Was the agent actually stuck?  Check if the last 3 turns had
        # identical summary lines.
        summary_lines = []
        for req in d.get("captured_http_requests", []):
            try:
                resp = req["response_body"]
                for ch in (resp.get("choices") or []):
                    for tc in ((ch.get("message") or {}).get("tool_calls") or []):
                        if (tc.get("function") or {}).get("name") == "run_tests":
                            # The observation comes back in the NEXT
                            # request's tool-role message.
                            pass  # we'll grep the request bodies
            except Exception:
                pass
        # Quick proxy: turn count vs tool-call uniqueness
        unique_calls = set()
        for req in d.get("captured_http_requests", []):
            try:
                resp = req["response_body"]
                for ch in (resp.get("choices") or []):
                    msg = ch.get("message") or {}
                    for tc in (msg.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        unique_calls.add((fn.get("name"), json.dumps(fn.get("arguments"), sort_keys=True)))
            except Exception:
                pass
        n_calls = sum(per_run_tools[name].values())
        # Read the last 3 request bodies to see last run_tests summary lines
        summaries = []
        for req in d.get("captured_http_requests", [])[-6:]:
            try:
                rb = req.get("request_body_full")
                if isinstance(rb, dict):
                    rb = json.dumps(rb)
                for line in rb.splitlines():
                    if "exit_code=" in line and "passed=" in line:
                        summaries.append(line.strip())
            except Exception:
                pass
        last_3 = summaries[-3:]
        verdict = s.get("verdict", "?")
        truly_stuck = (
            len(unique_calls) < n_calls  # at least one repeat
            and (len(set(last_3)) == 1 if len(last_3) >= 3 else False)
        )
        print(
            f"  {name:50s}  turns={s.get('turn_count')} "
            f"unique_calls={len(unique_calls)}/{n_calls} "
            f"last_3_run_tests={'IDENTICAL' if len(set(last_3)) == 1 and len(last_3) == 3 else 'mixed'} "
            f"-> detector={'CORRECT' if truly_stuck else 'PREMATURE'}"
        )
    print()

    # ------------------------------------------------------------------
    # D. Skill loading rate
    # ------------------------------------------------------------------
    print("## D. Skill loading rate")
    n_total = 0
    n_loaded = 0
    for name, d in runs:
        # Cheap signal: does any tool_call in this run have name=load_skill?
        loaded = False
        for req in d.get("captured_http_requests", []):
            try:
                resp = req["response_body"]
                for ch in (resp.get("choices") or []):
                    for tc in ((ch.get("message") or {}).get("tool_calls") or []):
                        if (tc.get("function") or {}).get("name") == "load_skill":
                            loaded = True
            except Exception:
                pass
        n_total += 1
        if loaded:
            n_loaded += 1
    print(f"  {n_loaded} / {n_total} runs ever called load_skill")
    print()

    # ------------------------------------------------------------------
    # E. Token cost per verdict
    # ------------------------------------------------------------------
    print("## E. Token cost per verdict")
    cost_by_verdict = defaultdict(lambda: {"runs": 0, "prompt": 0, "completion": 0})
    for name, d in runs:
        # Aggregate usage from request bodies
        total_prompt = 0
        total_completion = 0
        for req in d.get("captured_http_requests", []):
            try:
                resp = req.get("response_body", {})
                usage = resp.get("usage") or {}
                total_prompt += int(usage.get("prompt_tokens") or 0)
                total_completion += int(usage.get("completion_tokens") or 0)
            except Exception:
                pass
        verdict = d.get("summary", {}).get("verdict", "?")
        cost_by_verdict[verdict]["runs"] += 1
        cost_by_verdict[verdict]["prompt"] += total_prompt
        cost_by_verdict[verdict]["completion"] += total_completion
    for verdict, c in sorted(cost_by_verdict.items()):
        if c["runs"] == 0:
            continue
        avg_p = c["prompt"] // c["runs"]
        avg_co = c["completion"] // c["runs"]
        print(
            f"  verdict={verdict:12s}  runs={c['runs']}  "
            f"avg_prompt={avg_p:6d}  avg_completion={avg_co:5d}  "
            f"avg_total={avg_p + avg_co:6d}"
        )
    print()

    # ------------------------------------------------------------------
    # F. Tool-call distribution by verdict (PASS vs FAIL)
    # ------------------------------------------------------------------
    print("## F. Tool-call mix by verdict (turn-weighted avg)")
    verdict_tool_counts = defaultdict(lambda: defaultdict(int))
    verdict_turn_counts = defaultdict(int)
    for name, d in runs:
        verdict = d.get("summary", {}).get("verdict", "?")
        n_turns = d.get("summary", {}).get("turn_count") or 0
        verdict_turn_counts[verdict] += n_turns
        for t, c in per_run_tools[name].items():
            verdict_tool_counts[verdict][t] += c
    for verdict, tcounts in verdict_tool_counts.items():
        n_turns = verdict_turn_counts[verdict] or 1
        sorted_tools = sorted(tcounts.items(), key=lambda x: -x[1])
        print(f"  verdict={verdict} ({n_turns} total turns):")
        for t, c in sorted_tools[:8]:
            pct = 100 * c / n_turns
            print(f"    {t:18s}  {c:4d}  ({pct:5.1f}/turn)")
        print()


if __name__ == "__main__":
    main()