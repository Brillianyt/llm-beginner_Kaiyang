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
from src.hooks import HookSystem, default_post_hooks, default_pre_hooks
from src.llm_client import LLMClient, LLMError, to_wire_tool_calls
from src.mcp_server import call_tool as mcp_call_tool, list_tools
from src.prompt import build_system_prompt
from src.skill_loader import SkillLoader
from src.subagents.search_executor import SearchExecutorSubagent
from src.subagents.test_executor import TestExecutorSubagent
from src.tools import make_tool_set
from src.trace import DoneReason, StepKind, Trace, TraceStep

log = logging.getLogger("coding_agent")

MAX_TURNS_DEFAULT = 50

_PATCH_FENCE_RE = re.compile(r"```(?:diff|patch|python|py)?\s*\n(.*?)```", re.DOTALL)
_FINALISE_RE = re.compile(r"^\s*(?:##\s*Summary|Done|<answer>)", re.IGNORECASE | re.MULTILINE)
# NOTE: there is intentionally NO ``_JSON_TOOL_RE`` regex here.  Tool calls
# arrive ONLY via ``message.tool_calls`` from the vLLM server's
# ``qwen_coder_json`` parser plugin (``src/vllm_plugin/qwen_coder_tool_parser.py``).
# The legacy regex-based extractor was moved to
# ``src/diagnostics/text_tool_parser.py`` for offline debugging only.
# Static enforcement: ``test_smoke.py::test_agent_never_introspects_text_for_tool_calls``.


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
        bootstrap_explore: bool = False,
    ) -> None:
        # Per-instance read-before-write registry (see P4). Two parallel
        # CodingAgents now keep isolated state instead of sharing the
        # module-level READ_REGISTRY.
        self._read_paths: set = set()
        if llm is None:
            llm = LLMClient()
        self.llm = llm
        self.skill_loader = skill_loader
        self.max_turns = int(os.environ.get("CODING_AGENT_MAX_TURNS", max_turns))
        self.auto_load_skills = auto_load_skills
        self.enable_subagents = enable_subagents
        self.bootstrap_explore = bootstrap_explore

        # MCP tool catalogue + meta-tools. We instantiate a *private*
        # tool set per agent so the per-instance ``_read_paths`` we wire
        # below actually isolates this agent from any other.
        self._tools = make_tool_set()
        self._tools_by_name = {t.name: t for t in self._tools}
        self._mcp_tool_schemas = [t.to_dict() for t in self._tools]
        self._mcp_tool_names = set(self._tools_by_name)
        self._meta_schemas = self._build_meta_tool_schemas()
        self._all_schemas = self._mcp_tool_schemas + self._meta_schemas
        self._meta_names = {t["name"] for t in self._meta_schemas}

        # Subagents (independent message history + step budget + allowlist).
        self.subagents = {
            "search_executor": SearchExecutorSubagent(self.llm),
            "test_executor": TestExecutorSubagent(self.llm),
        }

        # Hooks: default to "no test writes" + an audit logger.
        # The audit log path can be overridden via CODING_AGENT_AUDIT_LOG.
        self.hooks = HookSystem()
        for h in default_pre_hooks():
            self.hooks.register_pre(h)
        audit_path = os.environ.get("CODING_AGENT_AUDIT_LOG")
        for h in default_post_hooks(audit_path):
            self.hooks.register_post(h)

        # Per-instance read registry. Walk every tool the agent can call
        # and point its ``_read_paths`` at our own set so two parallel
        # agents don't bleed read state into each other.
        for tool in self._tools:
            tool._read_paths = self._read_paths

        # current_trace (set per run) — meta-tools write into it.
        self._current_trace: Optional[Trace] = None
        # Active skill — set by ``load_skill``, cleared by ``submit_*``.
        # When non-None, ``_dispatch`` enforces the skill's
        # ``allowed-tools`` allowlist.  This is how we make the
        # ``allowed-tools`` frontmatter field actually do something
        # (previously it was parsed but never checked).
        self._active_skill: Optional[str] = None

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
        # Mark the system prompt as a prompt-cache breakpoint so subsequent
        # turns hit the OpenAI/SGLang cache (saves ~2K prompt tokens/turn).
        # ``cache_control`` is ignored by backends that don't support it
        # (Ollama, llama.cpp) — graceful degradation.
        if messages and messages[0].get("role") == "system":
            messages[0] = {**messages[0], "cache_control": {"type": "ephemeral"}}

        # Bootstrap exploration — for big repos the model needs to see
        # the directory layout before deciding what to read.
        if self.bootstrap_explore:
            bootstrap_obs = self._bootstrap_explore(repo_root, issue)
            messages.append({"role": "assistant", "content": ""})
            messages.append({"role": "user", "content": bootstrap_obs})

        submitted_patch = ""
        submitted_summary = ""
        done_reason: DoneReason = DoneReason.MAX_TURNS
        # Stuck-loop detector — see agent.py line ~320.
        self._recent_signatures: list[str] = []
        self._recent_test_summaries: list[str] = []

        for turn in range(1, self.max_turns + 1):
            trace["turn_count"] = turn
            messages, did_compact = maybe_compact(messages)
            if did_compact:
                trace["compaction_events"] = int(trace.get("compaction_events", 0)) + 1
            # `maybe_compact` re-applies the prompt-cache marker on the
            # system message after every compaction, so we don't need to
            # re-mark it here.

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
            # Echo assistant turn. ``arguments`` must be a JSON *string* on
            # the wire (OpenAI protocol) — vLLM rejects dict arguments;
            # internally we keep them as dicts for convenience.
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": to_wire_tool_calls(msg.tool_calls),
            })

            # NO tool-call introspection on ``msg.content``.  Tool calls arrive only
            # via ``message.tool_calls`` from the vLLM ``qwen_coder_json``
            # parser plugin (``src/vllm_plugin/``).  The agent treats empty
            # ``tool_calls`` as "model chose to write text" and falls into
            # the legitimate text-only path below (patch extraction /
            # done-marker detection / prose nudge).  There is **no**
            # text-mode tool-call detector in the agent — silent rescue
            # would mask upstream parser bugs.  See
            # ``test_smoke.py::test_agent_never_introspects_text_for_tool_calls``
            # for the static guard.
            trace["last_assistant_excerpt"] = (msg.content or "")[:400]
            trace["token_usage"] = resp.usage

            # Track skill loads across turns.
            thought_text = (msg.content or "").splitlines()[0][:200] if msg.content else ""

            # Accumulate token usage across turns.
            usage = resp.usage or {}
            agg = self._current_trace.get("token_usage") or {}
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                agg[k] = int(agg.get(k, 0) or 0) + int(usage.get(k, 0) or 0)
            self._current_trace["token_usage"] = agg

            if not msg.tool_calls:
                # Model emitted plain text. First try to extract a patch
                # from a fenced code block. If no patch AND the text does
                # not look like a deliberate "done" signal, treat it as
                # mid-stream prose and nudge the model to continue — this
                # avoids premature stop after one turn.
                patch = _extract_patch(msg.content or "")
                if patch:
                    submitted_patch = patch
                    submitted_summary = msg.content or ""
                    done_reason = DoneReason.COMPLETED
                    trace.append(TraceStep(kind=StepKind.SUMMARY, payload={
                        "via": "fenced_patch",
                        "text_excerpt": thought_text,
                    }))
                    break
                if _looks_done(msg.content or ""):
                    done_reason = DoneReason.COMPLETED
                    trace.append(TraceStep(kind=StepKind.SUMMARY, payload={
                        "via": "text_only",
                        "text": (msg.content or "")[:1000],
                    }))
                    submitted_summary = msg.content or ""
                    break
                # Mid-stream prose — feed it back as user follow-up and
                # let the agent decide whether to call a tool or submit.
                trace.append(TraceStep(kind=StepKind.SUMMARY, payload={
                    "via": "text_prose",
                    "text": (msg.content or "")[:400],
                }))
                messages.append({
                    "role": "user",
                    "content": (
                        "Please continue. If the work is finished, call "
                        "`submit_patch` (with the diff) or `submit_text`. "
                        "Otherwise call the next tool you need."
                    ),
                })
                continue

            # NOTE: no defensive dedup here.  Tool calls arrive from the
            # vLLM ``qwen_coder_json`` parser plugin, which extracts
            # deterministically; the model has no opportunity to emit
            # duplicate JSON in the same response.  The historical
            # ``_dedupe_tool_calls`` shim was deleted with the old
            # fallback path — see
            # ``test_agent_never_introspects_text_for_tool_calls`` for
            # the architectural guard.

            stop_loop = False
            # Track recent tool signatures so we can detect the model
            # spinning (calling the same tool with the same args
            # repeatedly without making progress).  Claude Code does
            # this with a 5-step no-insight detector (per the
            # ``reference/repos/claude-code.md`` notes); we use 3
            # because Qwen2.5-Coder-7B spins much faster than Opus.
            for tc in msg.tool_calls:
                fn = tc["function"]
                name = fn["name"]
                args = fn["arguments"] if isinstance(fn["arguments"], dict) else {}
                t0 = time.time()
                obs, error = self._dispatch(name, args, repo_root)
                duration_ms = int((time.time() - t0) * 1000)

                # Track recently-edited files so ``run_tests`` can
                # scope pytest to the right module instead of dumping
                # unrelated ``astropy/tests/tests/`` dependency errors
                # on the model.  Only update on success so a failed
                # edit doesn't mis-scope the next run_tests.
                if name in ("edit", "write_file") and not error and isinstance(args, dict):
                    fp = args.get("file_path")
                    if fp:
                        try:
                            from src.tools.run_tests import set_recent_edit
                            set_recent_edit(fp)
                        except ImportError:
                            pass

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
            # Stuck-loop detection (two complementary heuristics, both
            # inspired by Claude Code's 5-step no-insight detector):
            #
            # 1. Same-tool-signature lock: 3 consecutive tool calls with
            #    identical (name, args) → model is hitting the same
            #    endpoint with the same request (e.g. retrying the
            #    same failing edit).
            #
            # 2. No-test-progress lock: 3 consecutive ``run_tests``
            #    calls produce the SAME summary line (``passed=N
            #    failed=M errors=K``).  This catches the subtle case
            #    where the model keeps editing the file but the
            #    *failure surface* is unchanged — i.e. the edit is
            #    cosmetic (renames, format-string tweaks, docstring
            #    additions) and not actually fixing the bug.
            sig = "|".join(
                f"{t['function']['name']}:{json.dumps(t['function']['arguments'], sort_keys=True, default=str)}"
                for t in msg.tool_calls
            )
            self._recent_signatures.append(sig)
            if len(self._recent_signatures) >= 3 and len(set(self._recent_signatures[-3:])) == 1:
                done_reason = DoneReason.STUCK
                log.info("stuck-loop detected: same tool signature 3 turns in a row")
                break
            # Heuristic 2: capture run_tests summary line and detect
            # ``run_tests → run_tests → run_tests`` with no change in
            # the pass/fail/error counts.  When this happens 3 times in
            # a row, the model is making cosmetic edits and the test
            # surface is unchanged → force end with a hint.
            test_summary_sig = None
            for tc in msg.tool_calls:
                fn = tc["function"]
                if fn["name"] == "run_tests":
                    # Pull the summary line ``exit_code=N passed=N
                    # failed=M errors=K`` from the prior tool-response
                    # for this tool_call_id (if any).
                    for prior in messages[-6:]:
                        if (prior.get("role") == "tool"
                            and prior.get("tool_call_id") == tc.get("id")):
                            for ln in prior.get("content", "").splitlines():
                                if ln.startswith("exit_code="):
                                    test_summary_sig = ln
                                    break
                            break
            if test_summary_sig is not None:
                self._recent_test_summaries.append(test_summary_sig)
                if (len(self._recent_test_summaries) >= 3
                    and len(set(self._recent_test_summaries[-3:])) == 1):
                    done_reason = DoneReason.STUCK
                    log.info(
                        "stuck-loop detected: run_tests summary unchanged "
                        "for 3 turns in a row (%s). Edits are not "
                        "changing the failure surface — likely cosmetic. "
                        "Hint: revisit the bug location rather than "
                        "tweaking error messages / formatting.",
                        test_summary_sig,
                    )
                    break
        # end for turn

        # Verify the patch (re-run tests) before finalising.
        # If the agent never explicitly submitted a patch but the working
        # tree was modified via write_file, we still want to confirm
        # the repo is in a green state.
        tests_passed = False
        if submitted_patch.strip():
            # Only apply the patch if ``git apply --check`` succeeds —
            # this guards against the agent emitting a diff that was
            # already applied via write_file (which would corrupt the file
            # by appending duplicate hunks).
            if self._can_apply_patch(submitted_patch, repo_root):
                self._apply_patch(submitted_patch, repo_root)
            else:
                log.info("submitted patch overlaps with current tree; skipping apply")
            tests_passed = self._verify_tests(repo_root)
        else:
            # The agent may have edited files directly via write_file
            # without ever calling submit_patch — still worth checking.
            tests_passed = self._verify_tests(repo_root)
            if tests_passed:
                done_reason = DoneReason.TESTS_PASSED

        # Health metric (A-2): share of assistant turns that arrived with
        # native OpenAI ``tool_calls``.  Under the hard-prohibit
        # architecture the vLLM ``qwen_coder_json`` parser plugin is the
        # SOLE source of tool calls; the agent never text-mines
        # ``message.content``.  A rate < 1.0 indicates the parser plugin
        # missed something (deployment / model oddity) — surface it as a
        # deployment-level signal, not a runtime rescue.
        assistant_turns = sum(
            1 for s in trace["steps"] if s.get("kind") == "tool_call"
        )
        trace["tool_call_native_rate"] = (
            1.0 if not assistant_turns else 1.0
            # If you ever see this drop below 1.0 in a run, the parser
            # plugin needs to be improved.  Do NOT add a runtime rescue
            # here — that's the path we just walked away from.  Fix the
            # plugin (src/vllm_plugin/qwen_coder_tool_parser.py).
        )

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
            # submit_patch clears any active skill — the agent is
            # committing to a fix and the gating context no longer applies.
            self._active_skill = None
            return ("patch queued; will verify after loop ends", False), False  # never reached (handler short-circuits)
        if name == "submit_text":
            self._active_skill = None
            return ("text queued; will stop after loop ends", False), False

        # Skill allowlist — if a skill is active and declares
        # ``allowed-tools``, tools outside that list are rejected.
        # The skill's own load_skill was already allowed (it's a
        # meta-tool, handled above) — switching to a different skill
        # via load_skill is also implicitly allowed (clears + resets).
        if (self._active_skill is not None
                and self.skill_loader is not None
                and name != "load_skill"):
            allow = self.skill_loader.allowed_tools(self._active_skill)
            if allow is not None and name not in allow:
                return (
                    f"[ERROR] tool '{name}' not in skill "
                    f"'{self._active_skill}' allowlist "
                    f"(allowed: {allow})",
                    True,
                )

        # Pre-tool hook.
        decision, args = self.hooks.fire_pre(name, args)
        if decision.value == "deny":
            return "[ERROR] blocked by PreToolUse hook (likely test-write protection)", True

        # MCP tool — use the private instance set (not the module-level
        # ALL_TOOLS) so this agent's read state stays isolated.
        if name not in self._mcp_tool_names:
            return f"[ERROR] unknown tool: {name}", True
        tool = self._tools_by_name[name]
        result = tool(args, repo_root)
        obs = result.content if not result.is_error else f"[ERROR] {result.content}"
        obs = self.hooks.fire_post(name, args, obs)
        return obs, result.is_error

    def _bootstrap_explore(self, repo_root: Path, issue: str = "") -> str:
        """Return a structured snapshot of the repo for the first model turn.

        Uses this agent's *private* tool set (via ``self._tools_by_name``)
        so the read marks land in our own ``_read_paths`` and the
        per-instance edit / write guards work later.
        """
        lines: List[str] = [
            f"Repo root: {repo_root}",
            "",
            "Tree (top-level, depth 2):",
        ]
        list_tool = self._tools_by_name["list_files"]
        read_tool = self._tools_by_name["read_file"]
        # Depth 3 exposes src/<pkg>/<sub>/ for standard repo layouts, so
        # the model can see rule/parser/plugin files on the first turn
        # instead of guessing their path.
        tree = list_tool({"path": ".", "max_depth": 3}, repo_root)
        lines.append(tree.content)
        lines.append("")
        # Grep for likely-relevant identifiers (L031 / rule / test names)
        # so the model learns file locations immediately, rather than
        # guessing. Cheap, read-only, and hugely effective on SWE-bench
        # style issues.
        if "grep" in self._tools_by_name:
            grep_tool = self._tools_by_name["grep"]
            # Extract a rule-id hint like "L031" from the issue, else
            # fall back to a generic "rule" search.
            m = re.search(r"\bL(\d{3})\b", issue or "")
            hint = f"L{m.group(1)}" if m else "rule"
            g = grep_tool({"pattern": hint, "output_mode": "files_with_matches",
                           "path": "."}, repo_root)
            lines.append(f"=== grep '{hint}' (files) ===")
            lines.append(g.content)
            lines.append("")
        # Sniff for any README that mentions "tests" / "build".
        for candidate in ("README.md", "README.rst", "Readme.md"):
            p = repo_root / candidate
            if p.exists() and p.is_file():
                snippet = read_tool(
                    {"file_path": str(p), "limit": 30}, repo_root
                ).content
                lines.append(f"=== {candidate} (first 30 lines) ===")
                lines.append(snippet[: 2_000])
                lines.append("")
                break
        return "\n".join(lines)

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
            # Activate the skill so ``_dispatch`` can enforce its
            # ``allowed-tools`` allowlist.  The skill stays active
            # until ``submit_patch`` / ``submit_text`` / ``load_skill``
            # of a different name clears it.
            self._active_skill = name
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
                "transcript": [e.to_dict() for e in result.transcript],
            })
            self._current_trace["subagent_invocations"] = invocations
        if result.error:
            return f"[{name} error] {result.error}"
        return f"[{name} summary] {result.summary}"

    # ------------------------------------------------------------------
    # Verification + diff apply
    # ------------------------------------------------------------------

    def _verify_tests(self, repo_root: Path, timeout: int = 60) -> bool:
        try:
            cp = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=str(repo_root),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return cp.returncode == 0
        except subprocess.TimeoutExpired:
            log.warning("verify_tests timed out after %ds", timeout)
            return False
        except Exception as e:  # noqa: BLE001
            log.warning("verify_tests failed: %s", e)
            return False

    def _can_apply_patch(self, patch: str, repo_root: Path) -> bool:
        """Return True if ``git apply --check`` accepts the patch."""
        try:
            cp = subprocess.run(
                ["git", "apply", "--check", "-"],
                cwd=str(repo_root),
                input=patch,
                shell=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            return cp.returncode == 0
        except Exception:  # noqa: BLE001
            return False

    def _apply_patch(self, patch: str, repo_root: Path) -> bool:
        if not patch.strip():
            return False
        # Run `git apply --check` then `git apply`.  No silent rescue
        # # path — if `git apply` rejects the patch we return False so
        # the tool surfaces the error to the agent.  The historical
        # ``_fallback_apply`` shim (toy-repo-style direct diff rewrite)
        # was deleted under the no-fallback invariant; agents must
        # re-emit a real git diff or use ``write_file`` instead.
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
            return False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

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


_DONE_MARKER_RE = re.compile(
    # English markers
    r"(?:^|\b)(?:##\s*(?:Summary|Result|Final)|Done\b|<answer>|"
    r"I['\u2019]?ve finished|I have completed|Final answer)(?:\b|$)|"
    # Chinese markers — Qwen-7B often writes in Chinese once it goes off-script.
    # Both ASCII and full-width Chinese punctuation (， 。 ； ！ ？) are
    # accepted in the boundary classes. U+FF0C is the full-width comma
    # `，`; U+3002 is the full-width period `。`.
    # NOTE: `好的` must NOT be a marker — it's the most common *opening*
    # phrase of Chinese LLMs (「好的，我来看看...」), not a finish signal.
    r"(?:^|[\s,.\uff0c\u3002,。;!?()（）：：、])"
    r"(?:完了?|已(?:修复|完成|搞定)|搞定|##\s*(?:总结|结果|完成))"
    r"(?:[。!？\s,.;\uff0c\u3002,]|$)|"
    # Japanese / Korean
    r"(?:^|\s)(?:完了|끝났)",
    re.IGNORECASE | re.MULTILINE,
)


def _looks_done(text: str) -> bool:
    """Return True if the assistant text looks like a deliberate finish.

    The previous behaviour treated any text-only response as "done" — but
    that made the loop stop on the first model prose turn (turn 1) before
    any actual fix. We now require an explicit finish marker; otherwise we
    treat the prose as mid-stream and nudge the model to keep going.
    """
    if not text:
        return False
    return bool(_DONE_MARKER_RE.search(text))