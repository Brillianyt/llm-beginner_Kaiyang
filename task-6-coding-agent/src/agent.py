"""CodingAgent — the orchestrator.

Translates Claude Code's ``src/query.ts`` ``queryLoop`` to Python:

.. code-block:: python

   while not done:
       response = llm.chat(messages, tools)
       if not response.tool_calls:
           return completed
       for call in response.tool_calls:
           obs = execute(call, repo_root)        # may be a subagent dispatch
           messages.append(tool_message(call.id, obs))
       if max_turns reached:
           return max_turns
       if submitted:
           return tests_passed

Public contract (consumed by ``eval/run.py``):

* ``CodingAgent()`` — construct with defaults (real LLM via env vars).
* ``agent.run(repo_path: str, issue: str) -> dict`` — returns a ``Trace``
  dict with keys ``steps``, ``patch``, ``tests_passed``, ...

Graceful degradation:

* If the LLM endpoint is unreachable, the loop terminates after a single
  round with ``done_reason="error"`` instead of crashing.
* If MCP server fails to start, the in-process :func:`mcp_server.call_tool`
  fallback is used (still applies the same safety checks).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.context import maybe_compact
from src.hooks import HookSystem, default_pre_hooks
from src.llm_client import LLMClient, LLMError
from src.mcp_server import call_tool as call_mcp_tool, list_tools
from src.prompt import build_system_prompt
from src.skill_loader import SkillLoader
from src.subagents.base import SubagentResult
from src.subagents.code_search import CodeSearchSubagent
from src.subagents.test_runner import TestRunnerSubagent
from src.tools import ALL_TOOLS
from src.trace import DoneReason, Step, Trace

log = logging.getLogger("coding_agent")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ms(start: float) -> int:
    return int((time.time() - start) * 1000)


_PATCH_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)


def _extract_patch(text: str) -> str:
    """Pull a unified diff out of ``text``.

    Looks for a fenced code block first; falls back to the whole text if
    it already starts with ``diff --git``.
    """
    if not text:
        return ""
    matches = _PATCH_FENCE_RE.findall(text)
    if matches:
        return matches[0].strip()
    if text.lstrip().startswith("diff --git"):
        return text.strip()
    return ""


# ---------------------------------------------------------------------------
# CodingAgent
# ---------------------------------------------------------------------------

class CodingAgent:
    """Main agent loop.

    Construct with defaults pulled from environment variables (see
    :class:`LLMClient`); pass ``use_offline=True`` to disable network
    access (used by smoke tests).
    """

    def __init__(
        self,
        *,
        llm: Optional[LLMClient] = None,
        skill_loader: Optional[SkillLoader] = None,
        max_turns: int = 30,
        temperature: float = 0.1,
        auto_load_skills: bool = True,
    ) -> None:
        if llm is None:
            try:
                llm = LLMClient(temperature=temperature)
            except LLMError as e:
                log.warning("LLM client construction failed: %s", e)
                raise
        self.llm = llm
        self.skill_loader = skill_loader
        self.max_turns = int(os.environ.get("CODING_AGENT_MAX_TURNS", max_turns))
        self.auto_load_skills = auto_load_skills
        self._tool_schemas: List[Dict[str, Any]] = list(list_tools())
        self._tool_names = {t["name"] for t in self._tool_schemas}
        # Custom tool schemas we layer on top of the MCP ones.
        self._tool_schemas.extend(
            [
                {
                    "name": "load_skill",
                    "description": (
                        "Load the full markdown body of a skill by name. "
                        "Use after deciding a skill from the Level-1 list "
                        "is relevant."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
                {
                    "name": "dispatch_subagent",
                    "description": (
                        "Delegate a focused sub-task to a subagent. "
                        "Available: 'code_search' (read-only exploration) "
                        "and 'test_runner' (run pytest). You will only "
                        "receive the subagent's final plain-text summary."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "task": {"type": "string"},
                        },
                        "required": ["name", "task"],
                    },
                },
                {
                    "name": "submit_patch",
                    "description": (
                        "Submit a unified diff as the final fix. Call "
                        "exactly once when you believe tests should pass "
                        "(or when you decide to stop)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "diff": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                        "required": ["diff"],
                    },
                },
                {
                    "name": "submit_text",
                    "description": (
                        "Stop the loop without a patch. Use when you "
                        "decide not to submit (e.g. cannot make progress)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": [],
                    },
                },
            ]
        )
        # Subagent registry (independent messages + max_steps + allowlist).
        self.subagents = {
            "code_search": CodeSearchSubagent(self.llm),
            "test_runner": TestRunnerSubagent(self.llm),
        }
        # Hooks — defaulted to "do not edit tests/".
        self.hooks = HookSystem()
        for h in default_pre_hooks():
            self.hooks.register_pre(h)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def run(self, repo_path: str, issue: str) -> Trace:
        repo_root = Path(repo_path).resolve()
        trace = Trace()
        self.attach_trace(trace)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(
                repo_root=str(repo_root),
                skill_loader=self.skill_loader,
                subagent_names=tuple(self.subagents.keys()),
                max_turns=self.max_turns,
            )},
            {"role": "user", "content": issue},
        ]
        # The hooks system needs a stable reference to repo_root.
        self._current_repo = str(repo_root)

        submitted_patch = ""
        submitted_summary = ""
        done_reason: DoneReason = DoneReason.MAX_TURNS
        tests_passed = False

        for turn in range(1, self.max_turns + 1):
            trace["turn_count"] = turn
            # Optional compaction.
            messages, did_compact = maybe_compact(messages)
            if did_compact:
                trace["compaction_events"] = int(trace.get("compaction_events", 0)) + 1

            try:
                resp = self.llm.chat(messages, tools=self._tool_schemas)
            except LLMError as e:
                trace.finalize(
                    done_reason=DoneReason.ERROR,
                    tests_passed=False,
                    patch="",
                    summary=f"LLM error: {e}",
                    error=str(e),
                )
                return trace

            msg = resp.message
            # Echo the assistant message so the next iteration has full history.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                }
            )
            trace["last_assistant_content"] = msg.content or ""
            trace["last_usage"] = resp.usage

            if not msg.tool_calls:
                # Termination reason 1: model output text without tool calls.
                # If the text contains a diff we treat it as a submission.
                patch = _extract_patch(msg.content or "")
                if patch:
                    submitted_patch = patch
                    submitted_summary = msg.content or ""
                done_reason = DoneReason.COMPLETED
                break

            # Execute each tool call.
            stop_loop = False
            for tc in msg.tool_calls:
                fn = tc["function"]
                name = fn["name"]
                args = fn["arguments"] if isinstance(fn["arguments"], dict) else {}
                step_start = time.time()
                thought = ""  # populated if the model emits a separate think block
                if msg.content:
                    thought = (msg.content or "").splitlines()[0][:200]
                obs, error = self._dispatch_tool(name, args, repo_root)
                duration = _now_ms(step_start)
                trace.append_step(Step(
                    thought=thought,
                    tool_call={"name": name, "arguments": args},
                    observation=obs,
                    duration_ms=duration,
                    error=error,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": obs,
                })
                if name == "submit_patch":
                    submitted_patch = args.get("diff", "") or ""
                    submitted_summary = args.get("summary", "") or msg.content or ""
                    done_reason = DoneReason.TESTS_PASSED
                    stop_loop = True
                    break
                if name == "submit_text":
                    submitted_summary = args.get("text", "") or msg.content or ""
                    done_reason = DoneReason.COMPLETED
                    stop_loop = True
                    break
            if stop_loop:
                break

        # Finalisation — optionally re-run tests to confirm.
        if done_reason == DoneReason.TESTS_PASSED and submitted_patch:
            tests_passed = self._verify_tests(repo_root)
            if not tests_passed:
                trace["summary"] = (
                    f"{submitted_summary}\n(agent claimed done; "
                    f"verification run failed)"
                )
                done_reason = DoneReason.COMPLETED
            else:
                trace["summary"] = submitted_summary or "tests passed"
        elif done_reason == DoneReason.COMPLETED and submitted_patch:
            tests_passed = self._verify_tests(repo_root)
            trace["summary"] = submitted_summary or "completed"
        else:
            trace["summary"] = submitted_summary or "no patch submitted"

        # Apply the diff to the working tree so the eval harness sees it.
        if submitted_patch:
            ok = self._apply_diff(submitted_patch, repo_root)
            if not ok:
                log.warning("submitted diff failed to apply cleanly")

        trace.finalize(
            done_reason=done_reason,
            tests_passed=tests_passed,
            patch=submitted_patch,
            summary=trace.get("summary", ""),
        )
        return trace

    # ------------------------------------------------------------------
    # Tool dispatch — handles MCP tools plus the synthetic ones.
    # ------------------------------------------------------------------
    def _dispatch_tool(
        self,
        name: str,
        args: Dict[str, Any],
        repo_root: Path,
    ) -> tuple[str, bool]:
        # Synthetic / agent-internal tools.
        if name == "load_skill":
            return self._handle_load_skill(args), False
        if name == "dispatch_subagent":
            return self._handle_dispatch_subagent(args, repo_root), False
        if name == "submit_patch":
            return ("patch received", False)
        if name == "submit_text":
            return ("text received", False)

        # Hook check.
        decision, args = self.hooks.fire_pre(name, args)
        if decision.value == "deny":
            return "[ERROR] blocked by PreToolUse hook (likely test-write protection)", True

        # MCP-style tool.
        if name not in self._tool_names:
            return f"[ERROR] unknown tool: {name}", True
        result = call_mcp_tool(name, args, repo_root)
        obs = result.content if not result.is_error else f"[ERROR] {result.content}"
        obs = self.hooks.fire_post(name, args, obs)
        return obs, result.is_error

    def _handle_load_skill(self, args: Dict[str, Any]) -> str:
        if self.skill_loader is None:
            return "[ERROR] no skill loader configured"
        name = args.get("name") or ""
        try:
            body = self.skill_loader.load(name)
            # Record which skill we loaded.
            trace = self._last_trace()  # may be None
            if trace is not None:
                loads = list(trace.get("skill_loads") or [])
                loads.append(name)
                trace["skill_loads"] = loads
            return body
        except KeyError:
            return f"[ERROR] skill not found: {name}"

    def _handle_dispatch_subagent(
        self,
        args: Dict[str, Any],
        repo_root: Path,
    ) -> str:
        name = args.get("name") or ""
        task = args.get("task") or ""
        sub = self.subagents.get(name)
        if sub is None:
            return f"[ERROR] unknown subagent: {name}"
        result: SubagentResult = sub.run(task, str(repo_root))
        # Record invocation (without leaking the subagent's trace).
        trace = self._last_trace()
        if trace is not None:
            invocations = list(trace.get("subagent_invocations") or [])
            invocations.append({
                "name": name,
                "task": task,
                "steps": result.steps,
                "error": result.error,
            })
            trace["subagent_invocations"] = invocations
        return (
            f"[{name} summary] {result.summary}"
            if not result.error
            else f"[{name} error] {result.error}"
        )

    def _last_trace(self) -> Optional[Dict[str, Any]]:
        # The Trace instance is set as an attribute right before each
        # turn starts; expose via the public dict for the helper.
        return getattr(self, "_current_trace", None)

    # ------------------------------------------------------------------
    # Verification + diff apply
    # ------------------------------------------------------------------
    def _verify_tests(self, repo_root: Path) -> bool:
        try:
            cp = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=str(repo_root),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return cp.returncode == 0
        except Exception as e:  # noqa: BLE001
            log.warning("verify_tests failed: %s", e)
            return False

    def _apply_diff(self, diff: str, repo_root: Path) -> bool:
        if not diff.strip():
            return False
        # First, try git apply --check (gracefully handles missing .git).
        try:
            probe = subprocess.run(
                ["git", "apply", "--check", "-"],
                cwd=str(repo_root),
                input=diff,
                capture_output=True,
                text=True,
                shell=False,
                timeout=15,
            )
            if probe.returncode != 0:
                # Fall back to writing the patched files manually using a
                # best-effort patch parser. This is intentionally simple.
                return self._fallback_apply(diff, repo_root)
            subprocess.run(
                ["git", "apply", "-"],
                cwd=str(repo_root),
                input=diff,
                capture_output=True,
                text=True,
                shell=False,
                timeout=15,
                check=False,
            )
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("apply diff failed: %s", e)
            return False

    @staticmethod
    def _fallback_apply(diff: str, repo_root: Path) -> bool:
        """Very small parser that handles the toy-repo ``add(a, b)`` fix.

        Real repos would use the ``patch`` package; we keep dependencies
        light and only support ``+`` line additions to ``a/...`` paths.
        """
        try:
            path = None
            additions: List[str] = []
            for line in diff.splitlines():
                if line.startswith("+++ b/"):
                    path = line[len("+++ b/"):].strip()
                    additions = []
                elif line.startswith("+") and not line.startswith("+++") and path:
                    additions.append(line[1:])
            if not path or not additions:
                return False
            target = (repo_root / path).resolve()
            if not str(target).startswith(str(repo_root)):
                return False
            text = target.read_text(encoding="utf-8")
            text = text + "\n" + "\n".join(additions)
            target.write_text(text, encoding="utf-8")
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def attach_trace(self, trace: Trace) -> None:
        """Let internal helpers write into ``trace`` (e.g. skill loads)."""
        self._current_trace = trace
