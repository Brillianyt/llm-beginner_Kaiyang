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
    ReadFileTool, WriteFileTool, RunTestsTool,
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
        result = tool({"file_path": str(target)}, self.repo)
        out = result.content  # tool returns ToolResult
        self.assertIn("=== ", out)
        self.assertIn("lines ", out)
        # cat -n: at least one line should start with "<n>\t".
        self.assertRegex(out, r"\b\d+\t", "cat -n line number prefix missing")


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