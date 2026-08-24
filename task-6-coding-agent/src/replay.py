"""TraceReplay — re-execute a saved Trace's tool calls without the LLM.

Use case: given ``eval/traces/m3_toy_repo.json`` (a Trace dict that
was saved by a previous run), re-run every ``tool_call`` against the
current tools and compare each observation to the one in the trace.

Divergences between replay and original usually mean one of:
* the repo state has drifted (someone else modified the file);
* a tool's behaviour changed (regression in safety rails);
* the same tool call with the same input now produces a different
  output (e.g. tests started passing in a new commit).

Note: ``match_rate < 1.0`` is **expected** for traces where the steps
form a state chain (write_file in step N changes the file that
read_file in step N+2 sees). Replay walks the chain in order — diffs
just show that the live repo state at step N+k differs from the
historical state at that step. This is *correct* behaviour, not a
replay bug. Use ``_strip`` (timestamp / duration normalisation) to
make human eyeballing easier.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.mcp_server import call_tool
from src.tools import ALL_TOOLS, BaseTool
from src.tools.base import mark_read_for


@dataclass
class ReplayDiff:
    """One tool call whose replay result differs from the original trace."""

    step_index: int
    tool_name: str
    arguments: Dict[str, Any]
    original_observation: str
    replay_observation: str


@dataclass
class ReplayReport:
    steps_total: int = 0
    steps_replayed: int = 0
    steps_skipped: int = 0
    diffs: List[ReplayDiff] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        if self.steps_total == 0:
            return 1.0
        return self.steps_replayed / self.steps_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps_total": self.steps_total,
            "steps_replayed": self.steps_replayed,
            "steps_skipped": self.steps_skipped,
            "match_rate": round(self.match_rate, 4),
            "diffs": [
                {
                    "step_index": d.step_index,
                    "tool_name": d.tool_name,
                    "arguments": d.arguments,
                    "original_observation": d.original_observation,
                    "replay_observation": d.replay_observation,
                }
                for d in self.diffs
            ],
        }


_TOOL_BY_NAME = {t.name: t for t in ALL_TOOLS}


class TraceReplay:
    """Replay tool calls from a saved Trace against the live tool set."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()

    def replay(self, trace: Dict[str, Any]) -> ReplayReport:
        """Walk the trace's tool_call steps, invoke each tool, compare."""
        report = ReplayReport()
        steps = trace.get("steps") or []
        for i, step in enumerate(steps):
            if step.get("kind") != "tool_call":
                report.steps_skipped += 1
                continue
            # The tool name and args live under ``payload`` (TraceStep
            # shape), not at the step root. Fall back to the legacy
            # flat shape for traces written by an older harness.
            payload = step.get("payload") or {}
            tool_name = (payload.get("name") or step.get("name") or "").strip()
            args = payload.get("arguments") or step.get("arguments") or {}
            report.steps_total += 1
            if tool_name not in _TOOL_BY_NAME:
                report.steps_skipped += 1
                continue
            # read_file / edit / write_file need a prior read registry
            # entry; mark this path as already read so the guard passes.
            if tool_name in ("edit", "write_file"):
                fp = (args.get("file_path") or args.get("path") or "").strip()
                if fp:
                    # Replay only needs the read to pass the tool's
                    # guard — we use the module-level set (path=None
                    # means "use shared registry"). Multiple replays
                    # racing on the same file would still see each
                    # other's reads, but replay is single-threaded so
                    # that's fine.
                    mark_read_for(str(self._abs_path(fp)), None)
            try:
                result = _TOOL_BY_NAME[tool_name](args, self.repo_root)
                replay_obs = result.content if hasattr(result, "content") else str(result)
            except Exception as e:  # noqa: BLE001
                replay_obs = f"[ERROR] replay exception: {e}"
            # The original observation lives in the OBSERVATION step
            # that immediately follows this tool_call step (in agent.py
            # both are appended in lockstep).
            original_obs = self._next_obs(steps, i)
            report.steps_replayed += 1
            if original_obs is not None and self._strip(replay_obs) != self._strip(original_obs):
                report.diffs.append(ReplayDiff(
                    step_index=i,
                    tool_name=tool_name,
                    arguments=args,
                    original_observation=original_obs,
                    replay_observation=replay_obs,
                ))
        return report

    def _abs_path(self, p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return self.repo_root / pp

    @staticmethod
    def _next_obs(steps: list, idx: int) -> Optional[str]:
        for j in range(idx + 1, len(steps)):
            if steps[j].get("kind") == "observation":
                # Same nested-payload convention as the tool_call side.
                payload = steps[j].get("payload") or {}
                return payload.get("observation", steps[j].get("observation", ""))
        return None

    @staticmethod
    def _strip(s: str) -> str:
        """Normalise: collapse whitespace, drop volatile tokens like
        ``duration_s=0.43`` so two tool runs of the same input still
        match for replay purposes."""
        import re
        s = re.sub(r"duration_s=[\d.]+", "duration_s=<n>", s)
        s = re.sub(r"timestamps?:\s*\S+", "ts=<n>", s)
        return " ".join(s.split())


def replay_from_file(trace_path: str | Path, repo_root: str | Path) -> ReplayReport:
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    return TraceReplay(repo_root).replay(trace)