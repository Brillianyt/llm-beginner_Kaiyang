"""Trace data structures for the CodingAgent.

Per blueprint Part IV §4.1:

* ``TraceStep(kind, payload)`` — kind ∈ {thought, tool_call, tool_result,
  observation, summary}, payload is a dict.
* ``Trace(task, steps)`` — a ``dict`` subclass so the eval harness can
  read ``trace.get("steps") / "patch" / "tests_passed"`` while internal
  code may use attribute-style access.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class StepKind(str, Enum):
    """Mirrors the kinds in blueprint Part IV §4.2."""

    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    SUMMARY = "summary"


class DoneReason(str, Enum):
    COMPLETED = "completed"
    TESTS_PASSED = "tests_passed"
    MAX_TURNS = "max_turns"
    ERROR = "error"
    ABORTED = "aborted"
    # Set when the agent's stuck-loop detector fires (3+ consecutive
    # identical tool signatures).  Mirrors Claude Code's 5-step
    # no-insight heuristic but is tighter because Qwen2.5-Coder-7B
    # spins faster than Opus.
    STUCK = "stuck"


@dataclass
class TraceStep:
    kind: StepKind
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value if isinstance(self.kind, StepKind) else self.kind,
            "payload": self.payload,
            "ts": self.ts,
        }


class Trace(dict):
    """A dict subclass for agent traces.

    Holds:
      * ``steps`` — list of step dicts (each from :meth:`TraceStep.to_dict`).
      * ``patch`` — final unified diff string (empty if none).
      * ``tests_passed`` — bool (set during ``finalize``).
      * ``done_reason`` — str (set during ``finalize``).
    """

    def __init__(self, task: str = "") -> None:
        super().__init__(
            task=task,
            steps=[],
            patch="",
            tests_passed=False,
            done_reason=None,
            turn_count=0,
            tool_call_count=0,
            started_at=datetime.utcnow().isoformat() + "Z",
            finished_at=None,
            duration_ms=0,
            summary="",
            skill_loads=[],
            subagent_invocations=[],
            error=None,
            token_usage={},
        )
        self._steps_internal: List[TraceStep] = []

    # -- step helpers ------------------------------------------------------

    def append(self, step: TraceStep) -> None:
        self._steps_internal.append(step)
        self["steps"] = [s.to_dict() for s in self._steps_internal]
        if step.kind == StepKind.TOOL_CALL:
            self["tool_call_count"] = int(self.get("tool_call_count", 0)) + 1

    @property
    def steps_internal(self) -> List[TraceStep]:
        return self._steps_internal

    # -- attribute-style proxies for ergonomic internal use -----------------

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self[key] = value

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(key)

    # -- finalisation ------------------------------------------------------

    def finalize(
        self,
        *,
        done_reason: DoneReason,
        tests_passed: bool,
        patch: str,
        summary: str = "",
        error: Optional[str] = None,
    ) -> None:
        self["done_reason"] = done_reason.value
        self["tests_passed"] = bool(tests_passed)
        self["patch"] = patch or ""
        self["summary"] = summary
        self["error"] = error
        self["finished_at"] = datetime.utcnow().isoformat() + "Z"

    def to_json(self) -> str:
        return json.dumps(self, ensure_ascii=False, indent=2, default=str)