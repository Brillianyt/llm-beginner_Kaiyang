"""Smoke tests — run with ``python -m pytest test_smoke.py -q``.

They do **not** require a running LLM or git. Each test pins one of:
* Tool safety (path traversal blocked, dangerous git blocked)
* Skill metadata (Level-1 returns name+description)
* Skill progressive disclosure (Level-2 body only on demand)
* Subagent isolation (independent messages + step budget + allowlist)
* Agent loop termination paths (max_turns, error, completed)
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
    GitDiffTool, GitApplyTool, ListFilesTool, ALL_TOOLS,
)
from src.tools.base import BLOCKED_GIT_FRAGMENTS, check_blocked_git, safe_resolve  # noqa: E402
from src.trace import DoneReason, Trace  # noqa: E402


class TestMcpServer(unittest.TestCase):
    def test_list_tools_returns_at_least_5(self):
        tools = list_tools()
        self.assertIsInstance(tools, list)
        self.assertGreaterEqual(len(tools), 5, "must expose ≥ 5 tools")
        for t in tools:
            self.assertIn("name", t)
            self.assertIn("description", t)
            self.assertIn("inputSchema", t)

    def test_every_tool_has_schema(self):
        tools = list_tools()
        for t in tools:
            self.assertEqual(t["inputSchema"]["type"], "object")

    def test_no_duplicate_names(self):
        names = [t["name"] for t in list_tools()]
        self.assertEqual(len(names), len(set(names)))


class TestToolSafety(unittest.TestCase):
    def setUp(self):
        self.repo = ROOT / "data" / "toy-repo"
        if not self.repo.exists():
            self.skipTest("data/toy-repo missing — run data/download.py")

    def test_path_traversal_blocked(self):
        with self.assertRaises(PermissionError):
            safe_resolve("../../../etc/passwd", self.repo)

    def test_absolute_path_blocked(self):
        with self.assertRaises(PermissionError):
            safe_resolve("/etc/passwd", self.repo)

    def test_safe_resolve_normalises(self):
        p = safe_resolve("calculator.py", self.repo)
        self.assertEqual(p.name, "calculator.py")

    def test_blocked_git_fragments(self):
        for blocked in [
            "git reset --hard HEAD",
            "git clean -fd",
            "git checkout -- calculator.py",
        ]:
            self.assertIsNotNone(check_blocked_git(blocked.split()))

    def test_blocked_git_allows_safe(self):
        self.assertIsNone(check_blocked_git(["git", "--no-pager", "diff"]))

    def test_run_tests_executes_pytest(self):
        # Bring back the buggy state first so we know what we measure.
        shutil.copy(self.repo / "calculator.py.orig", self.repo / "calculator.py")
        tool = RunTestsTool()
        result = tool({}, self.repo)
        self.assertIn("exit_code", result.content)
        # When buggy, exit code should be 1.
        self.assertIn("exit_code=1", result.content)


class TestSkillLoader(unittest.TestCase):
    def setUp(self):
        self.loader = SkillLoader(str(ROOT / "src" / "skills"))

    def test_list_returns_name_and_description(self):
        skills = self.loader.list_skills()
        self.assertGreaterEqual(len(skills), 2)
        for s in skills:
            self.assertIn("name", s)
            self.assertIn("description", s)
            self.assertTrue(s["description"].strip())

    def test_load_returns_body_only(self):
        skills = self.loader.list_skills()
        self.assertGreater(len(skills), 0)
        body = self.loader.load(skills[0]["name"])
        # Body should NOT contain the YAML frontmatter delimiters.
        self.assertFalse(body.lstrip().startswith("---"))

    def test_progressive_disclosure_body_has_workflow(self):
        skills = self.loader.list_skills()
        # Find a skill whose body contains "Step"
        for s in skills:
            body = self.loader.load(s["name"])
            if "Step" in body or "## " in body:
                self.assertGreater(len(body), 100)
                return
        self.fail("no skill body had a workflow section")

    def test_load_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.loader.load("does-not-exist")

    def test_find_relevant_routes_by_keyword(self):
        names = self.loader.find_relevant("please review this PR diff")
        # Either code-review or pr-description-writer should match.
        self.assertTrue(any("review" in n or "pr-" in n for n in names))


class TestSubagentIsolation(unittest.TestCase):
    def test_code_search_subagent_isolation(self):
        from src.subagents.code_search import CodeSearchSubagent
        # allowed_tools is independent (no write_file).
        self.assertNotIn("write_file", CodeSearchSubagent.allowed_tools)
        self.assertIn("read_file", CodeSearchSubagent.allowed_tools)
        # max_steps is bounded independently of the parent (default 30).
        self.assertLessEqual(CodeSearchSubagent.max_steps, 5)

    def test_test_runner_subagent_isolation(self):
        from src.subagents.test_runner import TestRunnerSubagent
        self.assertNotIn("write_file", TestRunnerSubagent.allowed_tools)
        self.assertIn("run_tests", TestRunnerSubagent.allowed_tools)

    def test_subagent_base_runs_without_llm(self):
        from src.llm_client import LLMError
        from src.subagents.code_search import CodeSearchSubagent
        from src.llm_client import LLMClient

        class BoomClient:
            endpoint_summary = "boom"

            def chat(self, *args, **kwargs):
                raise LLMError("no model")

        sub = CodeSearchSubagent(BoomClient())  # type: ignore[arg-type]
        result = sub.run("find add", str(ROOT / "data" / "toy-repo"))
        self.assertEqual(result.name, "code_search")
        self.assertEqual(result.error, "llm_error: no model")


class TestAgentLoop(unittest.TestCase):
    def test_trace_structure(self):
        t = Trace()
        self.assertIn("steps", t)
        self.assertIn("patch", t)
        self.assertIn("tests_passed", t)

    def test_trace_finalize(self):
        t = Trace()
        t.finalize(done_reason=DoneReason.TESTS_PASSED, tests_passed=True, patch="diff --git", summary="ok")
        self.assertTrue(t["tests_passed"])
        self.assertEqual(t["done_reason"], "tests_passed")
        self.assertEqual(t["patch"], "diff --git")

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


class TestCompaction(unittest.TestCase):
    def test_compact_triggers_above_threshold(self):
        from src.context import maybe_compact
        # Provide enough messages that ``keep_head + keep_tail`` is
        # strictly less than the total — otherwise the function rightly
        # returns "no middle to compact".
        huge = [
            {"role": "system", "content": "A" * 40_000},
            {"role": "user", "content": "do thing 1"},
            {"role": "assistant", "content": "yes 1"},
            {"role": "tool", "content": "obs 1 " + ("x" * 200)},
            {"role": "user", "content": "do thing 2"},
            {"role": "assistant", "content": "yes 2"},
            {"role": "tool", "content": "obs 2 " + ("y" * 200)},
            {"role": "user", "content": "do thing 3"},
            {"role": "assistant", "content": "yes 3"},
            {"role": "tool", "content": "obs 3 " + ("z" * 200)},
            {"role": "user", "content": "do thing 4"},
            {"role": "assistant", "content": "yes 4"},
            {"role": "tool", "content": "obs 4 " + ("w" * 200)},
        ]
        msgs, did = maybe_compact(huge, threshold=1000)
        self.assertTrue(did)
        # Some "compacted" marker should be inserted.
        self.assertTrue(any("compacted" in (m.get("content") or "") for m in msgs))

    def test_compact_below_threshold_passthrough(self):
        from src.context import maybe_compact
        msgs, did = maybe_compact([{"role": "user", "content": "hi"}], threshold=100_000)
        self.assertFalse(did)


class TestEvalContract(unittest.TestCase):
    def test_toy_repo_can_be_reset(self):
        repo = ROOT / "data" / "toy-repo"
        buggy = repo / "calculator.py.orig"
        self.assertTrue(buggy.exists(), "calculator.py.orig snapshot missing")
        shutil.copy(buggy, repo / "calculator.py")
        # Original is intentionally buggy: add(2, 3) should not equal 5.
        from src.tools import ReadFileTool
        text = ReadFileTool()({"path": "calculator.py"}, repo).content
        self.assertIn("return a - b", text)


if __name__ == "__main__":
    unittest.main()
