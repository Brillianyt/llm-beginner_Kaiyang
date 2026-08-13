"""Trace data structures for the CodingAgent.

The contract with ``eval/run.py``:

* ``trace`` is a dict (or ``dict`` subclass).
* ``trace.get("steps")`` returns a list of step dicts.
* ``trace.get("patch")`` returns a unified diff (str) — may be empty.
* ``trace.get("tests_passed")`` returns ``bool``.

We expose a :class:`Trace` dict-subclass so the eval harness can also
``trace.attribute_access`` in tests, but ``get`` is the canonical path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DoneReason(str, Enum):
    """Termination enum — mirrors Claude Code's ``Terminal`` type."""

    COMPLETED = "completed"          # LLM emitted no tool call
    TESTS_PASSED = "tests_passed"    # special: agent submitted a patch and tests are green
    MAX_TURNS = "max_turns"          # turn budget exhausted
    ERROR = "error"                  # unrecoverable error
    ABORTED = "aborted"              # user / external abort


@dataclass
class Step:
    thought: str = ""
    tool_call: Optional[Dict[str, Any]] = None
    observation: str = ""
    duration_ms: int = 0
    error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "tool_call": self.tool_call,
            "observation": self.observation,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class Trace(dict):
    """A ``dict`` subclass that records every agent action.

    Supports attribute-style access for ergonomic internal use
    (``self._trace.patch = "..."``) while remaining a plain dict for the
    eval harness.
    """

    def __init__(self) -> None:
        super().__init__(
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
        )
        self._steps_internal: List[Step] = []

    # -- steps helpers -----------------------------------------------------

    @property
    def steps_internal(self) -> List[Step]:
        return self._steps_internal

    def append_step(self, step: Step) -> None:
        self._steps_internal.append(step)
        self["steps"] = [s.to_dict() for s in self._steps_internal]
        if step.tool_call:
            self["tool_call_count"] = int(self.get("tool_call_count", 0)) + 1

    # -- attribute-style proxies for ergonomic internal code ---------------

    def __setattr__(self, key, value):
        if key in {"_steps_internal"} or key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self[key] = value

    def __getattr__(self, key):
        # Called only when the attribute is missing on the instance.
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
        self["patch"] = patch
        self["summary"] = summary
        self["error"] = error
        self["finished_at"] = datetime.utcnow().isoformat() + "Z"
        steps = self["steps"] if isinstance(self.get("steps"), list) else []
        total = sum(int(s.get("duration_ms", 0)) for s in steps)
        self["duration_ms"] = total

    # -- JSON helpers ------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(self, ensure_ascii=False, indent=2, default=str)
