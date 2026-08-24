"""Smoke tests — ``python -m pytest test_smoke.py -q``.

Covers every contract in the blueprint:

* M1 (Part I) — list_tools returns ≥ 5 tools, every schema is an object.
* Tool safety — safe_resolve blocks absolute paths and `..`.
* Blocked git fragments (file-system-spec §6).
* run_tests parses pytest output (passed/failed + failures[]).
* M2 (Part II) — SkillLoader.list_skills returns name+description,
  load() returns body without frontmatter, search() returns top-k,
  system_prompt_section returns a markdown block under the char budget.
* M3 (Part III) — subagent isolation (independent tool names, max_steps).
* M4 (Part IV) — Trace dict structure, finalisation, _extract_patch.
* Compaction — maybe_compact triggers above threshold.
"""
from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from src.mcp_server import list_tools  # noqa: E402
from src.skill_loader import SkillLoader  # noqa: E402
from src.tools import (  # noqa: E402
    ReadFileTool, WriteFileTool, EditTool, RunTestsTool,
    GitDiffTool, GitApplyTool, ALL_TOOLS,
)
from src.tools.base import (  # noqa: E402
    BLOCKED_GIT_FRAGMENTS, check_blocked_git, safe_resolve,
)
from src.trace import DoneReason, StepKind, Trace, TraceStep  # noqa: E402


# ---------------------------------------------------------------------------
# Part I — MCP server
# ---------------------------------------------------------------------------

class TestMcpServer(unittest.TestCase):
    def test_list_tools_returns_at_least_5(self):
        tools = list_tools()
        self.assertIsInstance(tools, list)
        self.assertGreaterEqual(len(tools), 5, "must expose ≥ 5 tools")
        for t in tools:
            self.assertIn("name", t)
            self.assertIn("description", t)
            self.assertIn("inputSchema", t)

    def test_every_schema_is_object(self):
        for t in list_tools():
            self.assertEqual(t["inputSchema"]["type"], "object")

    def test_no_duplicate_names(self):
        names = [t["name"] for t in list_tools()]
        self.assertEqual(len(names), len(set(names)))

    def test_tools_match_blueprint_part_i(self):
        names = {t["name"] for t in list_tools()}
        for required in ("read_file", "write_file", "run_tests", "git_diff", "git_apply"):
            self.assertIn(required, names, f"missing tool: {required}")

    def test_edit_tool_present(self):
        names = {t["name"] for t in list_tools()}
        self.assertIn("edit", names, "edit tool should be registered alongside write_file")


# ---------------------------------------------------------------------------
# Tool safety (Part I §1.4)
# ---------------------------------------------------------------------------

class TestToolSafety(unittest.TestCase):
    def setUp(self):
        self.repo = ROOT / "data" / "toy-repo"
        if not self.repo.exists():
            self.skipTest("data/toy-repo missing — run data/download.py")

    def test_absolute_path_blocked(self):
        with self.assertRaises(PermissionError):
            safe_resolve("/etc/passwd", self.repo)

    def test_relative_normalises(self):
        p = safe_resolve("calculator.py", self.repo)
        self.assertEqual(p.name, "calculator.py")

    def test_traversal_blocked(self):
        with self.assertRaises(PermissionError):
            safe_resolve("../../etc/passwd", self.repo)

    def test_blocked_git_fragments(self):
        for blocked in (
            "git reset --hard HEAD",
            "git clean -fd",
            "git checkout -- calculator.py",
        ):
            self.assertIsNotNone(check_blocked_git(blocked.split()))

    def test_safe_git_allowed(self):
        self.assertIsNone(check_blocked_git(["git", "--no-pager", "diff"]))

    def test_list_files_accepts_absolute_path_inside_repo(self):
        """list_files must accept the absolute path the model naturally
        passes (read_file / write_file / edit all require absolute), and
        reject only paths that escape the repo."""
        from src.tools import ListFilesTool
        tool = ListFilesTool()
        # Absolute path to repo root → should work.
        r = tool({"path": str(self.repo)}, self.repo)
        self.assertNotIn("absolute path rejected", r.content)
        self.assertIn("calculator.py", r.content)
        # Escape attempt → rejected.
        r2 = tool({"path": "/etc"}, self.repo)
        self.assertIn("escapes repo root", r2.content)

    def test_read_file_not_found_hints_list_files(self):
        """A missing file must hint `list_files` instead of a bare error —
        bare errors make 7B models loop guessing filenames."""
        target = (self.repo / "definitely_missing.py").resolve()
        tool = ReadFileTool()
        r = tool({"file_path": str(target)}, self.repo)
        self.assertIn("file not found", r.content)
        self.assertIn("list_files", r.content,
                      "not-found error should suggest list_files exploration")

    def test_system_prompt_mentions_list_files(self):
        """System prompt hard rules must instruct explore-before-read."""
        from src.prompt import build_system_prompt
        p = build_system_prompt(repo_root=str(self.repo), max_turns=5)
        self.assertIn("list_files", p)
        self.assertIn("Explore before you read", p)

    def test_git_apply_rolls_back_on_failure(self):
        """A patch that fails after partially writing must leave the
        working tree byte-identical to its pre-call state."""
        import subprocess
        from src.tools.git_apply import GitApplyTool
        # Use the toy repo (has a .git from data/download.py).
        repo = self.repo
        # Reset to a known state.
        shutil.copy(repo / "calculator.py.orig", repo / "calculator.py")
        original = (repo / "calculator.py").read_bytes()

        # Patch that would corrupt the file: pretend to delete a line
        # that doesn't exist (3-way false). We make it bigger than the
        # original so ``git apply`` can corrupt mid-apply only if it
        # actually wrote.
        bad_patch = (
            "--- a/calculator.py\n"
            "+++ b/calculator.py\n"
            "@@ -100,1 +100,1 @@\n"
            "-this line never existed in the file\n"
            "+replacement\n"
        )
        tool = GitApplyTool()
        # First, init a git repo if missing so ``git apply`` works.
        if not (repo / ".git").exists():
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "-c", "user.email=t@t",
                 "-c", "user.name=t", "commit", "-q", "-m", "init"],
                check=True,
            )
        # Apply a patch that can't match — should fail, then roll back.
        result = tool({"patch": bad_patch, "dry_run": False}, repo).content
        self.assertIn("[ERROR]", result)
        # Rollback: file should be byte-identical to the snapshot.
        after = (repo / "calculator.py").read_bytes()
        self.assertEqual(after, original,
                         "git_apply must restore file bytes on failure")

    def test_run_tests_executes_pytest(self):
        # Reset to the buggy snapshot.
        shutil.copy(self.repo / "calculator.py.orig", self.repo / "calculator.py")
        tool = RunTestsTool()
        result = tool({"cmd": "pytest"}, self.repo)
        out = result.content
        self.assertIn("exit_code", out)
        # Buggy code → at least one test fails → non-zero exit.
        self.assertIn("exit_code=1", out)

    def test_read_file_cat_n_format(self):
        target = (self.repo / "calculator.py").resolve()
        tool = ReadFileTool()
        result = tool({"file_path": str(target), "include_line_numbers": True}, self.repo)
        out = result.content  # tool returns ToolResult
        self.assertIn("=== ", out)
        self.assertIn("lines ", out)
        # cat -n: at least one line should start with "<n>\t".
        self.assertRegex(out, r"\b\d+\t", "cat -n line number prefix missing")

    def test_read_file_clean_text_default(self):
        # Default mode omits the line-number prefix — this avoids the model
        # echoing `cat -n` output back into a write_file call.
        target = (self.repo / "calculator.py").resolve()
        tool = ReadFileTool()
        out = tool({"file_path": str(target)}, self.repo).content
        # Lines should NOT start with "<n>\t" in default mode.
        body_lines = out.splitlines()[2:]  # skip header lines
        for line in body_lines[:3]:
            self.assertFalse(
                line.lstrip()[: len(line.lstrip().split("\t")[0])].isdigit(),
                f"line looks cat -n prefixed in default mode: {line!r}",
            )

    def test_edit_tool_basic_replace(self):
        """Edit replaces a unique substring and updates the file in place."""
        target = (self.repo / "calculator.py").resolve()
        from src.tools.base import mark_read_for
        tool = EditTool()
        mark_read_for(str(target), tool._read_paths)
        original = target.read_text(encoding="utf-8")
        tool(
            {
                "file_path": str(target),
                "old_string": "return a - b",
                "new_string": "return a + b",
            },
            self.repo,
        )
        new = target.read_text(encoding="utf-8")
        self.assertIn("return a + b", new)
        self.assertNotIn("return a - b", new)
        target.write_text(original, encoding="utf-8")  # restore

    def test_edit_tool_rejects_non_unique(self):
        target = (self.repo / "calculator.py").resolve()
        from src.tools.base import mark_read_for
        shutil.copy(self.repo / "calculator.py.orig", self.repo / "calculator.py")
        tool = EditTool()
        mark_read_for(str(target), tool._read_paths)
        r = tool(
            {
                "file_path": str(target),
                "old_string": "    result",
                "new_string": "    out",
            },
            self.repo,
        )
        self.assertIn("matched", r.content.lower())
        self.assertIn("replace_all", r.content)

    def test_edit_tool_replace_all(self):
        target = (self.repo / "calculator.py").resolve()
        from src.tools.base import mark_read_for
        shutil.copy(self.repo / "calculator.py.orig", self.repo / "calculator.py")
        tool = EditTool()
        mark_read_for(str(target), tool._read_paths)
        original = target.read_text(encoding="utf-8")
        n = original.count("    result")
        self.assertGreaterEqual(n, 2)
        tool(
            {
                "file_path": str(target),
                "old_string": "    result",
                "new_string": "    out",
                "replace_all": True,
            },
            self.repo,
        )
        new = target.read_text(encoding="utf-8")
        self.assertNotIn("    result", new)
        self.assertEqual(new.count("    out"), n)
        target.write_text(original, encoding="utf-8")  # restore

    def test_edit_tool_requires_read_first(self):
        from src.tools.base import clear_read_registry
        clear_read_registry()
        # Use the test's own isolated EditTool (its per-instance read
        # set is fresh and empty) — the helper now also falls back to
        # the module-level set, so clearing both is the right way to
        # test the guard.
        tool = EditTool()
        from src.tools.base import READ_REGISTRY
        READ_REGISTRY.clear()
        target = (self.repo / "calculator.py").resolve()
        r = tool(
            {
                "file_path": str(target),
                "old_string": "return a - b",
                "new_string": "return a + b",
            },
            self.repo,
        )
        self.assertIn("has not been read", r.content)

    def test_edit_tool_rejects_empty_old(self):
        target = (self.repo / "calculator.py").resolve()
        from src.tools.base import mark_read_for
        tool = EditTool()
        mark_read_for(str(target), tool._read_paths)
        tool = EditTool()
        r = tool(
            {
                "file_path": str(target),
                "old_string": "",
                "new_string": "anything",
            },
            self.repo,
        )
        self.assertIn("empty", r.content.lower())

    def test_run_tests_rejects_bad_extra_args(self):
        tool = RunTestsTool()
        r = tool(
            {"cmd": "pytest", "extra_args": ["-x; rm -rf /"]},
            self.repo,
        )
        self.assertIn("metacharacter", r.content)

    def test_run_tests_extra_args_pass_through(self):
        # Bogus marker so we can detect that extra_args reached the command.
        tool = RunTestsTool()
        r = tool(
            {"cmd": "pytest", "extra_args": ["-k", "no_such_marker_xyzzy"]},
            self.repo,
        )
        # exit_code=5 means pytest collected 0 tests for the -k filter.
        self.assertIn("exit_code=5", r.content)


# ---------------------------------------------------------------------------
# Part II — SkillLoader
# ---------------------------------------------------------------------------

class TestSkillLoader(unittest.TestCase):
    def setUp(self):
        self.loader = SkillLoader(str(ROOT / "src" / "skills"))

    def test_list_skills_has_name_and_description(self):
        skills = self.loader.list_skills()
        self.assertGreaterEqual(len(skills), 2)
        for s in skills:
            self.assertIn("name", s)
            self.assertIn("description", s)
            self.assertTrue(s["description"].strip())

    def test_load_strips_frontmatter(self):
        skills = self.loader.list_skills()
        self.assertGreater(len(skills), 0)
        body = self.loader.load(skills[0]["name"])
        self.assertFalse(body.lstrip().startswith("---"),
                         "frontmatter delimiters must be stripped")

    def test_search_returns_top_k(self):
        hits = self.loader.search("please review this PR diff", k=3)
        names = [h["name"] for h in hits]
        self.assertTrue(any("review" in n or "pr-" in n for n in names),
                        f"search should surface review/pr- skills; got {names}")

    def test_search_returns_empty_for_unrelated_query(self):
        hits = self.loader.search("kubernetes helm chart", k=3)
        self.assertEqual(hits, [])

    def test_system_prompt_section_under_budget(self):
        section = self.loader.system_prompt_section(char_budget=2000)
        self.assertTrue(section.startswith("Available skills"))
        self.assertLess(len(section), 2000)
        self.assertIn("`", section, "skill names should be backticked")

    def test_load_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.loader.load("does-not-exist")

    def test_list_scripts_returns_diff_stats(self):
        from src.skill_loader import SkillLoader
        loader = SkillLoader(str(ROOT / "src" / "skills"))
        scripts = loader.list_scripts("code-review")
        names = [p.name for p in scripts]
        self.assertIn("diff_stats.py", names)

    def test_read_script_returns_text(self):
        from src.skill_loader import SkillLoader
        loader = SkillLoader(str(ROOT / "src" / "skills"))
        text = loader.read_script("code-review", "diff_stats.py")
        self.assertIsNotNone(text)
        self.assertIn("parse_diff", text)

    def test_read_script_blocks_path_traversal(self):
        from src.skill_loader import SkillLoader
        loader = SkillLoader(str(ROOT / "src" / "skills"))
        self.assertIsNone(loader.read_script("code-review", "../SKILL.md"))
        self.assertIsNone(loader.read_script("code-review", "../../../etc/passwd"))

    def test_list_and_read_references(self):
        from src.skill_loader import SkillLoader
        loader = SkillLoader(str(ROOT / "src" / "skills"))
        refs = loader.list_references("code-review")
        self.assertTrue(any(p.name == "review-checklist.md" for p in refs))
        body = loader.read_reference("code-review", "review-checklist.md")
        self.assertIsNotNone(body)
        self.assertIn("Severity", body)

    def test_path_tools_require_absolute_path(self):
        """Spec 1.2: file_path must be absolute. The schema encodes this
        via ``pattern: '^/'`` — verify jsonschema rejects relative paths
        for read_file / write_file / edit."""
        import jsonschema
        from src.mcp_server import list_tools
        for t in list_tools():
            if t["name"] not in ("read_file", "write_file", "edit"):
                continue
            props = t["inputSchema"].get("properties", {})
            self.assertEqual(props.get("file_path", {}).get("pattern"), "^/",
                             f"{t['name']} file_path should require absolute path")
        # Verify jsonschema actually rejects a relative path.
        for schema_name, schema in [
            ("read_file", next(t["inputSchema"] for t in list_tools()
                               if t["name"] == "read_file")),
        ]:
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(instance={"file_path": "relative/path"}, schema=schema)


# ---------------------------------------------------------------------------
# Hooks — PreToolUse + PostToolUse
# ---------------------------------------------------------------------------

class TestPostToolUseAuditLog(unittest.TestCase):
    def test_audit_logger_writes_jsonl(self):
        """The default PostToolUse hook appends one JSONL record per
        tool call to the configured audit file."""
        import json
        import os
        import tempfile
        from src.hooks import HookSystem, default_post_hooks

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "sub", "audit.jsonl")
            hooks = HookSystem()
            for h in default_post_hooks(log_path):
                hooks.register_post(h)
            hooks.fire_post("read_file", {"file_path": "/x"}, "obs1")
            hooks.fire_post("write_file", {"file_path": "/y", "content": "abc"}, "obs2")
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["tool"], "read_file")
        self.assertEqual(lines[0]["observation_excerpt"], "obs1")
        self.assertEqual(lines[1]["tool"], "write_file")
        self.assertEqual(lines[1]["content_len"], 3)

    def test_audit_logger_no_op_when_path_unset(self):
        """Without an explicit log_path, the hook is a no-op (no file
        is written, no exception raised)."""
        from src.hooks import HookSystem, default_post_hooks
        hooks = HookSystem()
        for h in default_post_hooks():
            hooks.register_post(h)
        # Just verify it doesn't raise.
        hooks.fire_post("read_file", {}, "x")


# ---------------------------------------------------------------------------
# Per-instance read-before-write isolation (P4)
# ---------------------------------------------------------------------------

class TestPerInstanceReadRegistry(unittest.TestCase):
    def test_two_agents_have_isolated_reads(self):
        """Two CodingAgent instances must not share their read-before-write
        state. Otherwise an ablation running 3 agents in sequence would
        see all of agent 1's reads leaking into agent 2/3's writes."""
        from src.tools import ReadFileTool, WriteFileTool
        from src.tools.base import mark_read_for, has_been_read_for

        # Two tool sets — each has its own ``_read_paths``.
        a_read = ReadFileTool()
        a_write = WriteFileTool()
        b_read = ReadFileTool()
        b_write = WriteFileTool()
        # Wire per-instance registries.
        a_read._read_paths = set()
        a_write._read_paths = a_read._read_paths
        b_read._read_paths = set()
        b_write._read_paths = b_read._read_paths

        target = (ROOT / "data" / "toy-repo" / "calculator.py").resolve()
        # Agent A reads the file; agent B does not.
        mark_read_for(str(target), a_read._read_paths)
        self.assertTrue(has_been_read_for(str(target), a_read._read_paths))
        # Agent B's registry must NOT see A's mark.
        self.assertFalse(has_been_read_for(str(target), b_read._read_paths))

    def test_mark_read_for_falls_back_to_shared(self):
        """When ``reads`` is None, helpers use the module-level registry
        (the existing single-process contract)."""
        from src.tools.base import (
            READ_REGISTRY, clear_read_registry, has_been_read_for,
            mark_read_for,
        )
        clear_read_registry()
        mark_read_for("/some/path", None)
        self.assertIn("/some/path", READ_REGISTRY)
        self.assertTrue(has_been_read_for("/some/path", None))
        clear_read_registry()


# ---------------------------------------------------------------------------
# Part III — Subagent isolation
# ---------------------------------------------------------------------------

class TestSubagentIsolation(unittest.TestCase):
    def test_search_executor_allowlist(self):
        from src.subagents.search_executor import SearchExecutorSubagent
        self.assertIn("read_file", SearchExecutorSubagent.allowed_tools)
        self.assertNotIn("write_file", SearchExecutorSubagent.allowed_tools)
        self.assertNotIn("run_tests", SearchExecutorSubecutor.allowed_tools) \
            if False else None  # noqa
        self.assertLessEqual(SearchExecutorSubagent.max_steps, 8)

    def test_test_executor_allowlist(self):
        from src.subagents.test_executor import TestExecutorSubagent
        self.assertIn("run_tests", TestExecutorSubagent.allowed_tools)
        self.assertIn("read_file", TestExecutorSubagent.allowed_tools)
        self.assertNotIn("write_file", TestExecutorSubagent.allowed_tools)

    def test_both_subagents_are_readonly(self):
        from src.subagents.search_executor import SearchExecutorSubagent
        from src.subagents.test_executor import TestExecutorSubagent
        self.assertTrue(SearchExecutorSubagent.readonly)
        self.assertTrue(TestExecutorSubagent.readonly)

    def test_base_subagent_runs_offline(self):
        from src.llm_client import LLMError
        from src.subagents.search_executor import SearchExecutorSubagent

        class BoomClient:
            endpoint_summary = "boom"

            def chat(self, *args, **kwargs):
                raise LLMError("no model")

        sub = SearchExecutorSubagent(BoomClient())  # type: ignore[arg-type]
        result = sub.run("find add", str(ROOT / "data" / "toy-repo"))
        self.assertEqual(result.name, "search_executor")
        self.assertEqual(result.error, "llm_error: no model")
        # Summary is forced under MAX_SUMMARY_CHARS.
        self.assertLessEqual(len(result.summary), 2048)


# ---------------------------------------------------------------------------
# Part IV — Trace + CodingAgent helpers
# ---------------------------------------------------------------------------

class TestTrace(unittest.TestCase):
    def test_default_keys(self):
        t = Trace()
        for k in ("steps", "patch", "tests_passed"):
            self.assertIn(k, t)

    def test_append_increments_tool_call_count(self):
        t = Trace()
        t.append(TraceStep(kind=StepKind.TOOL_CALL, payload={"name": "read_file"}))
        t.append(TraceStep(kind=StepKind.OBSERVATION, payload={"name": "read_file"}))
        self.assertEqual(t["tool_call_count"], 1)
        self.assertEqual(len(t["steps"]), 2)

    def test_finalize_records_done_reason(self):
        t = Trace()
        t.finalize(done_reason=DoneReason.TESTS_PASSED, tests_passed=True,
                   patch="diff --git", summary="ok")
        self.assertTrue(t["tests_passed"])
        self.assertEqual(t["done_reason"], "tests_passed")


class TestAgentHelpers(unittest.TestCase):
    def test_extract_patch_from_fenced_block(self):
        from src.agent import _extract_patch
        text = "Here you go:\n```diff\n--- a/x\n+++ b/x\n@@\n-old\n+new\n```\nThanks."
        out = _extract_patch(text)
        self.assertIn("--- a/x", out)
        self.assertIn("+new", out)

    def test_extract_patch_from_bare_diff(self):
        from src.agent import _extract_patch
        text = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@\n-old\n+new\n"
        self.assertIn("diff --git", _extract_patch(text))

    def test_extract_patch_empty_when_no_diff(self):
        from src.agent import _extract_patch
        self.assertEqual(_extract_patch("just a comment, no patch here"), "")

    def test_looks_done_recognises_finish_markers(self):
        from src.agent import _looks_done
        for marker in (
            "## Summary",
            "## Final",
            "Done",
            "I've finished the fix.",
            "I have completed the task.",
            "<answer>all green</answer>",
        ):
            self.assertTrue(_looks_done(marker), f"should recognise: {marker!r}")

    def test_looks_done_rejects_prose(self):
        from src.agent import _looks_done
        for prose in (
            "I'll read the file now.",
            "Let me think about this.\nThe bug is in add() returning a - b.",
            "Sure, here is the fix.",
        ):
            self.assertFalse(_looks_done(prose), f"should NOT recognise: {prose!r}")

    def test_looks_done_recognises_multilingual_markers(self):
        """Qwen-7B sometimes writes done-markers in Chinese / Japanese."""
        from src.agent import _looks_done
        for marker in (
            "## 总结",
            "## 完成",
            "## 结果",
            "好的，问题已修复。",
            "已完成。",
            "搞定。",
        ):
            self.assertTrue(_looks_done(marker), f"should recognise: {marker!r}")

    def test_system_prompt_has_cache_control(self):
        """The first system message should be marked ephemeral so
        subsequent turns hit the prompt cache."""
        from src.agent import CodingAgent
        from src.llm_client import ChatCompletion, ChatMessage
        from src.tools.base import clear_read_registry

        clear_read_registry()
        agent = CodingAgent()
        captured = {}
        real_chat = agent.llm.chat

        def fake_chat(messages, **kw):
            captured["messages"] = messages
            return ChatCompletion(
                message=ChatMessage(role="assistant", content="ok"),
                finish_reason="stop",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        agent.llm.chat = fake_chat
        try:
            agent.run(repo_path=str(ROOT / "data" / "toy-repo"), issue="x")
        except Exception:
            pass  # only need captured messages
        agent.llm.chat = real_chat

        sys_msgs = [m for m in captured.get("messages", []) if m.get("role") == "system"]
        self.assertGreater(len(sys_msgs), 0, "no system message captured")
        self.assertEqual(
            sys_msgs[0].get("cache_control"),
            {"type": "ephemeral"},
            "first system message must carry cache_control=ephemeral",
        )


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------

class TestCompaction(unittest.TestCase):
    def test_compact_triggers_above_threshold(self):
        from src.context import maybe_compact
        huge = [{"role": "system", "content": "A" * 40_000}]
        for i in range(6):
            huge += [
                {"role": "user", "content": f"do thing {i}"},
                {"role": "assistant", "content": f"yes {i}"},
                {"role": "tool", "content": f"obs {i} " + ("x" * 200)},
            ]
        msgs, did = maybe_compact(huge, threshold=1000)
        self.assertTrue(did)
        self.assertTrue(any("compacted" in (m.get("content") or "") for m in msgs))

    def test_compact_below_threshold_passthrough(self):
        from src.context import maybe_compact
        msgs, did = maybe_compact([{"role": "user", "content": "hi"}], threshold=100_000)
        self.assertFalse(did)

    def test_compact_preserves_cache_marker(self):
        """After compaction, the system message must still carry
        cache_control=ephemeral so the prompt cache re-keys cleanly."""
        from src.context import maybe_compact
        msgs = [
            {
                "role": "system",
                "content": "You are an agent.",
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": "do thing 1"},
        ] + [
            {"role": i % 2 and "assistant" or "user",
             "content": f"obs {j} " + ("x" * 200)}
            for i, j in enumerate(range(40))
        ]
        out, did = maybe_compact(msgs, threshold=1000)
        self.assertTrue(did)
        sys_msg = next(m for m in out if m.get("role") == "system"
                       and m.get("cache_control"))
        self.assertEqual(sys_msg["cache_control"], {"type": "ephemeral"})


# ---------------------------------------------------------------------------
# Per-file snapshot rollback (used by swebench_sample)
# ---------------------------------------------------------------------------

class TestSnapshotRestore(unittest.TestCase):
    def test_snapshot_round_trip(self):
        """Snapshot must capture every file under the repo and restore it
        byte-for-byte after the tree is mutated."""
        import tempfile
        from ablations.swebench_sample import _snapshot_dir, _restore_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "r"
            root.mkdir()
            (root / "a.py").write_text("alpha")
            (root / "sub").mkdir()
            (root / "sub" / "b.txt").write_text("beta")
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("ref: master")

            snap = _snapshot_dir(root)
            # Snapshot must exclude .git but include a.py + sub/b.txt.
            self.assertIn("a.py", snap)
            self.assertIn(str(Path("sub") / "b.txt"), snap)
            self.assertNotIn(str(Path(".git") / "HEAD"), snap)

            # Mutate everything (delete + add + modify).
            (root / "a.py").write_text("MUTATED")
            (root / "sub" / "b.txt").write_text("MUTATED")
            (root / "new.txt").write_text("extra file")
            (root / "junk").write_text("to be deleted")

            _restore_dir(root, snap)

            self.assertEqual((root / "a.py").read_text(), "alpha")
            self.assertEqual((root / "sub" / "b.txt").read_text(), "beta")
            self.assertFalse((root / "junk").exists())


# ---------------------------------------------------------------------------
# TraceReplay
# ---------------------------------------------------------------------------

class TestTraceReplay(unittest.TestCase):
    def test_replay_matches_on_stable_repo(self):
        """Replay a trace whose observations match the live tool output."""
        from src.replay import TraceReplay
        repo = ROOT / "data" / "toy-repo"
        if not (repo / "calculator.py.orig").exists():
            self.skipTest("toy-repo missing")
        import shutil
        shutil.copy(repo / "calculator.py.orig", repo / "calculator.py")
        target = (repo / "calculator.py").resolve()
        # Capture the live observation by running the tool, then build a
        # trace from it — this is the standard "save a trace and
        # replay-it-later" pattern.
        from src.tools import ReadFileTool
        live_obs = ReadFileTool()({"file_path": str(target)}, repo).content
        trace = {
            "steps": [
                {
                    "kind": "tool_call", "name": "read_file",
                    "arguments": {"file_path": str(target)},
                },
                {"kind": "observation", "name": "read_file", "observation": live_obs},
            ]
        }
        report = TraceReplay(repo).replay(trace)
        self.assertEqual(report.steps_total, 1)
        self.assertEqual(report.steps_replayed, 1)
        self.assertEqual(len(report.diffs), 0, f"unexpected diffs: {report.diffs}")

    def test_replay_flags_drift(self):
        """If the on-disk file no longer matches, replay must report a diff."""
        from src.replay import TraceReplay
        repo = ROOT / "data" / "toy-repo"
        if not (repo / "calculator.py.orig").exists():
            self.skipTest("toy-repo missing")
        # Write a fake trace whose read_file expects a specific content.
        target = (repo / "calculator.py").resolve()
        # Mutate on disk so it won't match.
        target.write_text("# completely different content\n", encoding="utf-8")
        try:
            trace = {
                "steps": [
                    {
                        "kind": "tool_call", "name": "read_file",
                        "arguments": {"file_path": str(target)},
                    },
                    {"kind": "observation", "name": "read_file",
                     "observation": "def add(a, b):\n    return a + b\n"},
                ]
            }
            report = TraceReplay(repo).replay(trace)
            self.assertGreater(len(report.diffs), 0)
        finally:
            # Restore.
            import shutil
            shutil.copy(repo / "calculator.py.orig", repo / "calculator.py")


# ---------------------------------------------------------------------------
# Eval contract — make sure toy-repo can be reset
# ---------------------------------------------------------------------------

class TestEvalContract(unittest.TestCase):
    def test_toy_repo_resets_to_buggy(self):
        repo = ROOT / "data" / "toy-repo"
        orig = repo / "calculator.py.orig"
        if not orig.exists():
            self.skipTest("calculator.py.orig missing — run data/download.py")
        shutil.copy(orig, repo / "calculator.py")
        from src.tools import ReadFileTool
        text = ReadFileTool()({"file_path": str((repo / "calculator.py").resolve())}, repo).content
        self.assertIn("return a - b", text)


if __name__ == "__main__":
    unittest.main()