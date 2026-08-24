"""CodingAgent — main orchestrator.

Per blueprint Part IV §4.2:

* while not done:
      response = llm.chat(messages, tools)
      if response has tool_calls:
          for call in tool_calls:
              obs = dispatch(call)         # may be subagent / skill / mcp
              messages.append(tool_message(call.id, obs))
      else:
          # model emitted text — try to extract a patch, or stop
          ...

Termination signals (any one ends the loop):
* model calls ``submit_patch`` with a non-empty diff,
* model calls ``submit_text`` (gives up),
* model emits plain text without tool calls (final answer),
* ``max_turns`` exceeded.

Public contract with :mod:`eval.run`:

* ``CodingAgent().run(repo_path: str, issue: str) -> Trace``
* ``Trace`` is a dict-subclass with ``steps / patch / tests_passed``.
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
from src.mcp_server import call_tool as mcp_call_tool, list_tools
from src.prompt import build_system_prompt
from src.skill_loader import SkillLoader
from src.subagents.search_executor import SearchExecutorSubagent
from src.subagents.test_executor import TestExecutorSubagent
from src.trace import DoneReason, StepKind, Trace, TraceStep

log = logging.getLogger("coding_agent")

MAX_TURNS_DEFAULT = 50

_PATCH_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
_FINALISE_RE = re.compile(r"^\s*(?:##\s*Summary|Done|<answer>)", re.IGNORECASE | re.MULTILINE)


class CodingAgent:
    """Main agent loop."""

    def __init__(
        self,
        *,
        llm: Optional[LLMClient] = None,
        skill_loader: Optional[SkillLoader] = None,
        max_turns: int = MAX_TURNS_DEFAULT,
        auto_load_skills: bool = True,
        enable_subagents: bool = True,
    ) -> None:
        if llm is None:
            llm = LLMClient()
        self.llm = llm
        self.skill_loader = skill_loader
        self.max_turns = int(os.environ.get("CODING_AGENT_MAX_TURNS", max_turns))
        self.auto_load_skills = auto_load_skills
        self.enable_subagents = enable_subagents

        # MCP tool catalogue + meta-tools.
        self._mcp_tool_schemas = list(list_tools())
        self._mcp_tool_names = {t["name"] for t in self._mcp_tool_schemas}
        self._meta_schemas = self._build_meta_tool_schemas()
        self._all_schemas = self._mcp_tool_schemas + self._meta_schemas
        self._meta_names = {t["name"] for t in self._meta_schemas}

        # Subagents (independent message history + step budget + allowlist).
        self.subagents = {
            "search_executor": SearchExecutorSubagent(self.llm),
            "test_executor": TestExecutorSubagent(self.llm),
        }

        # Hooks: default to "no test writes".
        self.hooks = HookSystem()
        for h in default_pre_hooks():
            self.hooks.register_pre(h)

        # current_trace (set per run) — meta-tools write into it.
        self._current_trace: Optional[Trace] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, repo_path: str, issue: str) -> Trace:
        repo_root = Path(repo_path).resolve()
        trace = Trace(task=issue)
        self._current_trace = trace

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(
                repo_root=str(repo_root),
                skill_loader=self.skill_loader,
                subagent_names=tuple(self.subagents.keys()),
                max_turns=self.max_turns,
            )},
            {"role": "user", "content": issue},
        ]

        submitted_patch = ""
        submitted_summary = ""
        done_reason: DoneReason = DoneReason.MAX_TURNS

        for turn in range(1, self.max_turns + 1):
            trace["turn_count"] = turn
            messages, did_compact = maybe_compact(messages)
            if did_compact:
                trace["compaction_events"] = int(trace.get("compaction_events", 0)) + 1

            try:
                resp = self.llm.chat(messages, tools=self._all_schemas)
            except LLMError as e:
                log.warning("llm error: %s", e)
                trace.append(TraceStep(
                    kind=StepKind.OBSERVATION,
                    payload={"error": True, "message": f"LLM error: {e}"},
                ))
                done_reason = DoneReason.ERROR
                trace.finalize(done_reason=done_reason, tests_passed=False,
                               patch="", summary=f"LLM error: {e}", error=str(e))
                return trace

            msg = resp.message
            # Echo assistant turn.
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": msg.tool_calls,
            })
            trace["last_assistant_excerpt"] = (msg.content or "")[:400]
            trace["token_usage"] = resp.usage

            # Track skill loads across turns.
            thought_text = (msg.content or "").splitlines()[0][:200] if msg.content else ""

            if not msg.tool_calls:
                # Model emitted plain text. Try to extract a patch from a
                # fenced code block; otherwise treat it as a final answer.
                patch = _extract_patch(msg.content or "")
                if patch:
                    submitted_patch = patch
                    submitted_summary = msg.content or ""
                    done_reason = DoneReason.COMPLETED
                    trace.append(TraceStep(kind=StepKind.SUMMARY, payload={
                        "via": "fenced_patch",
                        "text_excerpt": thought_text,
                    }))
                else:
                    done_reason = DoneReason.COMPLETED
                    trace.append(TraceStep(kind=StepKind.SUMMARY, payload={
                        "via": "text_only",
                        "text": (msg.content or "")[:1000],
                    }))
                    submitted_summary = msg.content or ""
                break

            stop_loop = False
            for tc in msg.tool_calls:
                fn = tc["function"]
                name = fn["name"]
                args = fn["arguments"] if isinstance(fn["arguments"], dict) else {}
                t0 = time.time()
                obs, error = self._dispatch(name, args, repo_root)
                duration_ms = int((time.time() - t0) * 1000)

                trace.append(TraceStep(kind=StepKind.TOOL_CALL, payload={
                    "name": name, "arguments": args, "duration_ms": duration_ms,
                }))
                trace.append(TraceStep(kind=StepKind.OBSERVATION, payload={
                    "name": name, "observation": obs[:4000], "error": error,
                }))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": obs,
                })

                if name == "submit_patch":
                    submitted_patch = args.get("diff", "") or ""
                    submitted_summary = args.get("summary", "") or msg.content or ""
                    done_reason = DoneReason.TESTS_PASSED if submitted_patch.strip() else DoneReason.COMPLETED
                    stop_loop = True
                    break
                if name == "submit_text":
                    submitted_summary = args.get("text", "") or msg.content or ""
                    done_reason = DoneReason.COMPLETED
                    stop_loop = True
                    break
            if stop_loop:
                break
        # end for turn

        # Verify the patch (re-run tests) before finalising.
        tests_passed = False
        if submitted_patch.strip():
            self._apply_patch(submitted_patch, repo_root)
            tests_passed = self._verify_tests(repo_root)

        trace.finalize(
            done_reason=done_reason,
            tests_passed=tests_passed,
            patch=submitted_patch,
            summary=submitted_summary or ("tests passed" if tests_passed else "no patch"),
        )
        self._current_trace = None
        return trace

    # ------------------------------------------------------------------
    # Tool dispatch (MCP tools + synthetic meta-tools)
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        name: str,
        args: Dict[str, Any],
        repo_root: Path,
    ) -> tuple:
        # Meta-tools first.
        if name == "load_skill":
            return self._handle_load_skill(args), False
        if name == "dispatch_subagent":
            if not self.enable_subagents:
                return "[ERROR] subagents disabled", True
            return self._handle_dispatch_subagent(args, repo_root), False
        if name == "submit_patch":
            return ("patch queued; will verify after loop ends", False), False  # never reached (handler short-circuits)
        if name == "submit_text":
            return ("text queued; will stop after loop ends", False), False

        # Pre-tool hook.
        decision, args = self.hooks.fire_pre(name, args)
        if decision.value == "deny":
            return "[ERROR] blocked by PreToolUse hook (likely test-write protection)", True

        # MCP tool.
        if name not in self._mcp_tool_names:
            return f"[ERROR] unknown tool: {name}", True
        result = mcp_call_tool(name, args, repo_root)
        obs = result.content if not result.is_error else f"[ERROR] {result.content}"
        obs = self.hooks.fire_post(name, args, obs)
        return obs, result.is_error

    def _handle_load_skill(self, args: Dict[str, Any]) -> str:
        if self.skill_loader is None:
            return "[ERROR] no skill loader configured"
        name = args.get("name") or ""
        try:
            body = self.skill_loader.load(name)
            if self._current_trace is not None:
                loads = list(self._current_trace.get("skill_loads") or [])
                loads.append(name)
                self._current_trace["skill_loads"] = loads
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
        result = sub.run(task, str(repo_root))
        if self._current_trace is not None:
            invocations = list(self._current_trace.get("subagent_invocations") or [])
            invocations.append({
                "name": name,
                "task": task,
                "summary": result.summary[:500],
                "steps": result.steps,
                "error": result.error,
            })
            self._current_trace["subagent_invocations"] = invocations
        if result.error:
            return f"[{name} error] {result.error}"
        return f"[{name} summary] {result.summary}"

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

    def _apply_patch(self, patch: str, repo_root: Path) -> bool:
        if not patch.strip():
            return False
        # Try `git apply --check` first.
        try:
            check = subprocess.run(
                ["git", "apply", "--check", "-"],
                cwd=str(repo_root),
                input=patch,
                shell=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if check.returncode == 0:
                subprocess.run(
                    ["git", "apply", "-"],
                    cwd=str(repo_root),
                    input=patch,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        # Fall back to writing the patched content directly.
        return _fallback_apply(patch, repo_root)

    # ------------------------------------------------------------------
    # Meta-tool schema
    # ------------------------------------------------------------------

    @staticmethod
    def _build_meta_tool_schemas() -> List[Dict[str, Any]]:
        return [
            {
                "name": "load_skill",
                "description": (
                    "Load the full markdown body of a skill by name. Use this "
                    "after deciding a skill from the Level-1 index is relevant "
                    "to the current task."
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
                    "Delegate a focused sub-task to a subagent. Available: "
                    "'search_executor' (read-only exploration) and "
                    "'test_executor' (run pytest + report failures). You will "
                    "only receive the subagent's final plain-text summary."
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
                    "Submit a unified diff as the final fix. Call exactly once "
                    "when you believe the fix is complete (or when you decide "
                    "to stop)."
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
                    "Stop the loop without a patch. Use when you decide not "
                    "to submit (e.g. cannot make progress)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            },
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_patch(text: str) -> str:
    if not text:
        return ""
    matches = _PATCH_FENCE_RE.findall(text)
    if matches:
        return matches[0].strip()
    if text.lstrip().startswith("diff --git"):
        return text.strip()
    return ""


def _fallback_apply(diff: str, repo_root: Path) -> bool:
    """Very small parser — only handles toy-repo-style ``a+b`` rewrites."""
    try:
        path = None
        new_lines: List[str] = []
        for raw in diff.splitlines():
            if raw.startswith("+++ b/"):
                path = raw[len("+++ b/"):].strip()
                new_lines = []
            elif raw.startswith("+") and not raw.startswith("+++") and path:
                new_lines.append(raw[1:])
        if not path or not new_lines:
            return False
        target = (repo_root / path).resolve()
        try:
            target.relative_to(repo_root)
        except ValueError:
            return False
        body = target.read_text(encoding="utf-8")
        body = body.rstrip("\n") + "\n" + "\n".join(new_lines) + "\n"
        target.write_text(body, encoding="utf-8")
        return True
    except Exception:  # noqa: BLE001
        return False