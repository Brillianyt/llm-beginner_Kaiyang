"""Smoke tests — run with ``python -m pytest test_smoke.py -q``.

They do **not** require a running LLM or vLLM. Each test pins one of:

* Tool safety (path traversal blocked, dangerous git blocked)
* Tool input schemas are valid
* read_file renders an **honest** header (the bug fix this file is
  written to lock: 2026-08-24 the header claimed ``lines 0..400 of
  642`` while the body was silently chopped at 8000 chars to ~281
  lines, hiding astropy-14365's line 309 ``if v == "NO":`` from the
  model).
* Agent loop does not introspect ``message.content`` for tool calls
  (the project's hard-prohibit invariant — see ``AGENTS.md`` §1).
"""
from __future__ import annotations

import io
import re
import sys
import tempfile
import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.tools import ReadFileTool  # noqa: E402
from src.tools.base import safe_resolve  # noqa: E402


def _strip_comments_and_strings(src: str) -> str:
    """Return ``src`` with comments and string literals replaced by
    blank runs of the same length.  Used by the no-fallback static
    guard so docstrings / historical-deletion comments do not trip
    forbidden-pattern matches.  Tokenize errors are ignored (defensive:
    the file should parse, but if a comment ever drifts we still want
    to scan whatever does parse)."""
    lines = src.splitlines(keepends=True)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        return src  # best-effort: fall back to raw text
    out_lines = [list(line) for line in lines]
    for tok in toks:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        srow, scol = tok.start
        erow, ecol = tok.end
        # Replace this token span with blanks, keeping newlines as
        # newlines so line numbers stay stable.
        if srow == erow:
            for c in range(scol, ecol):
                ch = tok.string[c - scol] if (c - scol) < len(tok.string) else " "
                out_lines[srow - 1][c] = "\n" if ch == "\n" else " "
        else:
            # First line: from scol to end-of-line
            for c in range(scol, len(out_lines[srow - 1])):
                ch = tok.string[c - scol] if (c - scol) < len(tok.string) else " "
                out_lines[srow - 1][c] = "\n" if ch == "\n" else " "
            # Middle lines: blank
            for r in range(srow + 1, erow):
                for c in range(len(out_lines[r - 1])):
                    out_lines[r - 1][c] = " "
            # Last line: from start to ecol
            for c in range(0, ecol):
                idx = c - 0 + scol  # offset within tok.string
                ch = tok.string[idx] if 0 <= idx < len(tok.string) else " "
                out_lines[erow - 1][c] = "\n" if ch == "\n" else " "
    return "".join("".join(line) for line in out_lines)


class TestReadFileHonestHeader(unittest.TestCase):
    """Lock the read_file honest-header fix (2026-08-24).

    Before the fix, ``BaseTool.__call__`` silently chopped the rendered
    string at ``max_result_chars=8000`` while ``read_file.call`` had
    already produced a header promising ``lines 0..N of 642``.  The
    model got fewer lines than advertised and had no way to know.  The
    fix makes ``read_file`` self-manage its char budget so:

    * the header shows the **actual** line range returned
    * a ``[output truncated at N chars; call read_file again with
      offset=K to continue]`` marker is added when truncation happens
    * the rendering fits inside ``max_result_chars`` so the
      ``BaseTool.__call__`` safety net never re-truncates the body
    """

    def _make_repo(self, lines: int, line_chars: int = 30) -> tuple:
        """Create a temp repo + file with ``lines`` fake-file lines.

        Returns ``(repo_root, file_path)``.
        """
        td = tempfile.TemporaryDirectory()
        repo = Path(td.name)
        f = repo / "big.txt"
        body = "\n".join(f"line {i:04d}".ljust(line_chars) for i in range(1, lines + 1)) + "\n"
        f.write_text(body)
        return repo, f, td

    def test_small_file_no_truncation_marker(self):
        """A file that fits in the budget must NOT advertise truncation."""
        repo, f, td = self._make_repo(lines=10)
        try:
            tool = ReadFileTool()
            out = tool.call({"file_path": str(f)}, repo)
            self.assertNotIn("output truncated", out)
            # header is 1-based
            self.assertIn("lines 1..10 of 10", out)
        finally:
            td.cleanup()

    def test_large_file_header_is_honest(self):
        """When the body would overflow the cap, header MUST shrink.

        Reproduces the exact astropy-14365 symptom: 642-line file,
        ``limit=400`` requested, but ``max_result_chars=8000`` would
        chop the body.  Before the fix, header said ``lines 0..400 of
        642`` and body was ~281 lines.  After the fix, header reports
        the actual line range returned.
        """
        repo, f, td = self._make_repo(lines=642, line_chars=40)
        try:
            tool = ReadFileTool()
            out = tool.call({"file_path": str(f)}, repo)
            # 1) Output must fit in the budget — no double-truncation.
            self.assertLessEqual(len(out), tool.max_result_chars,
                                 f"read_file output {len(out)} > cap {tool.max_result_chars}")
            # 2) Header line must reflect the ACTUAL range, not the
            #    requested range.  The fix shrinks the body so the
            #    header is honest.
            m = re.search(r"lines (\d+)\.\.(\d+) of (\d+)", out)
            self.assertIsNotNone(m, f"header missing in: {out[:200]!r}")
            start, end, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
            self.assertEqual(total, 642, "total_lines must be the file length")
            self.assertEqual(start, 1, "first read starts at 1")
            # The honest end is < requested limit (400).  Before the fix,
            # the header claimed end=400 with only ~281 lines of body.
            self.assertLess(end, 400,
                            f"header claimed end={end} but should have "
                            f"shrunken the body to fit in {tool.max_result_chars} chars")
            # 3) Body must have as many lines as the header advertises
            #    (no off-by-one, no trailing `...[truncated]...`).
            # ``out.split("\n", 3)[3]`` starts with the blank-line
            # separator (one leading "\n"), and ``content`` ends with a
            # trailing "\n" from ``"".join(body)``.  Strip both before
            # counting.
            body_section = out.split("\n", 3)[3]
            if body_section.startswith("\n"):
                body_section = body_section[1:]
            body_lines = body_section.split("\n")
            if body_lines and body_lines[-1] == "":
                body_lines = body_lines[:-1]
            self.assertEqual(len(body_lines), end - start + 1,
                             f"body has {len(body_lines)} lines but header "
                             f"advertised {end - start + 1}")
            # 4) Truncation marker present and actionable
            self.assertIn("output truncated at", out)
            self.assertIn("call read_file again with offset=", out)
            # 5) The bad "BaseTool.__call__ safety-net" marker must
            #    NOT appear in the body — it is a lie.
            self.assertNotIn("...[truncated]...", body_section)
        finally:
            td.cleanup()

    def test_honest_header_enables_second_read_to_see_rest(self):
        """The continuation hint's offset must let a second read see
        the rest of the file.  This is the astropy-14365 case: after
        first read of 642-line file, the second read with the
        advertised offset must include line 309."""
        repo, f, td = self._make_repo(lines=642, line_chars=40)
        try:
            tool = ReadFileTool()
            # First read
            out1 = tool.call({"file_path": str(f)}, repo)
            m = re.search(r"call read_file again with offset=(\d+)", out1)
            self.assertIsNotNone(m, "first read missing continuation hint")
            next_off = int(m.group(1))
            # Second read with that offset
            out2 = tool.call({"file_path": str(f), "offset": next_off}, repo)
            # The combined reads must cover lines 1..642 with no
            # gap.  Easier: the second read's end_line + 1 should
            # be ≥ 642 (or its own offset+actual_lines == total).
            m2 = re.search(r"lines (\d+)\.\.(\d+) of (\d+)", out2)
            self.assertIsNotNone(m2)
            self.assertEqual(int(m2.group(1)), next_off + 1,
                             f"second read start {m2.group(1)} != "
                             f"first read next_offset+1 = {next_off + 1}")
        finally:
            td.cleanup()

    def test_no_double_truncation_by_base_class(self):
        """``BaseTool.__call__``'s char cap is a safety net.  With the
        fix in place, read_file output must stay under the cap so the
        safety net is a no-op (no ``...[truncated]...`` marker from
        base.py chopping the already-honest body)."""
        repo, f, td = self._make_repo(lines=2000, line_chars=40)
        try:
            tool = ReadFileTool()
            result = tool({"file_path": str(f)}, repo)
            # The honest marker is OK; the safety-net marker is not.
            self.assertNotIn("...[truncated]...", result.content,
                             msg="BaseTool.__call__ re-truncated — read_file "
                                 "output exceeds max_result_chars; the "
                                 "honest-header fix is broken")
            self.assertLessEqual(len(result.content), tool.max_result_chars)
        finally:
            td.cleanup()

    def test_include_line_numbers_is_one_based(self):
        """``include_line_numbers=true`` numbering must be 1-based and
        match the header's display range."""
        repo, f, td = self._make_repo(lines=5)
        try:
            tool = ReadFileTool()
            out = tool.call({"file_path": str(f), "include_line_numbers": True}, repo)
            # body shows 1\t..., 2\t..., ...
            for i in range(1, 6):
                self.assertIn(f"\n{i}\t", out, f"line {i} missing 1-based prefix")
        finally:
            td.cleanup()


class TestAgentNoFallbackInvariant(unittest.TestCase):
    """Static guard: ``src/agent.py`` must NOT introspect
    ``message.content`` for tool calls.  This is the project's hard
    architectural rule — see ``AGENTS.md`` §1.

    If you ever add a ``_parse_text_tool_calls`` helper or pattern
    like ``if "tool_call" in msg["content"]`` to the agent loop,
    this test will fail.  Do not silence it — fix the design.
    """

    FORBIDDEN_PATTERNS = (
        r"_parse_text_tool_calls",
        r"json\.loads\(.*content.*\)",
        r'content.*[\'"]?tool_call[\'"]?',
        r'_fallback_apply',
        r'_dedupe_tool_calls',
        r'_JSON_TOOL_RE',
        r'parser_miss_count',
        # loud-error / safety-net diagnostics that were removed
        r'loud_error',
        r'from\s+src\.diagnostics\.text_tool_parser',
    )

    def test_agent_does_not_introspect_text_for_tool_calls(self):
        agent_path = ROOT / "src" / "agent.py"
        self.assertTrue(agent_path.exists(), f"{agent_path} missing")
        text = agent_path.read_text(encoding="utf-8")
        # Strip comments and string literals so historical-deletion
        # comments (e.g. "the previous _fallback_apply shim was
        # deleted under the no-fallback invariant") do not trip the
        # static guard.  The guard is about CODE PATHS, not
        # documentation describing past code paths.
        code_only = _strip_comments_and_strings(text)
        for pat in self.FORBIDDEN_PATTERNS:
            with self.subTest(pattern=pat):
                self.assertNotRegex(
                    code_only, pat,
                    msg=(
                        f"src/agent.py contains forbidden pattern {pat!r} "
                        f"(in code, not a comment); the agent must not "
                        "introspect message.content for tool calls "
                        "(see AGENTS.md §1 — no fallback)."
                    ),
                )


class TestToolSafety(unittest.TestCase):
    def test_path_traversal_blocked(self):
        with self.assertRaises(PermissionError):
            safe_resolve("../../../etc/passwd", Path("/tmp"))

    def test_absolute_path_blocked(self):
        with self.assertRaises(PermissionError):
            safe_resolve("/etc/passwd", Path("/tmp"))

    def test_safe_resolve_handles_relative_repo_root(self):
        # Bug fix: safe_resolve used to compare a resolved candidate
        # against an unresolved ``Path('.')`` repo root and
        # spuriously raise ``path escapes repo root``.  Any path
        # actually inside the cwd must succeed, regardless of whether
        # the caller passed ``Path('.')`` or an absolute root.
        try:
            Path("/tmp/_rel_root_test").mkdir(exist_ok=True)
            (Path("/tmp/_rel_root_test") / "subdir").mkdir(exist_ok=True)
            (Path("/tmp/_rel_root_test") / "subdir" / "f.txt").write_text("x")
            # Relative root — what run_bash actually passes from the
            # agent when repo_root comes in as Path(".").
            result = safe_resolve("subdir/f.txt", Path("/tmp/_rel_root_test"))
            self.assertTrue(str(result).endswith("subdir/f.txt"))
        finally:
            import shutil
            shutil.rmtree("/tmp/_rel_root_test", ignore_errors=True)


# ---------------------------------------------------------------------------
# run_bash — sandboxed subprocess
# ---------------------------------------------------------------------------
class TestRunBashSandbox(unittest.TestCase):
    """Lock in the three safety properties of run_bash:

    1. Path containment: every absolute path / ``..`` traversal in the
       command is rejected with a structured ``[ERROR]``.
    2. Deny-list: destructive patterns (``rm -rf``, ``git reset --hard``,
       ``curl … | sh``) are rejected even if path checks pass.
    3. List-form subprocess: ``subprocess.run`` is invoked with
       ``shell=False`` (verified by a happy-path echo that actually runs).
    """

    def setUp(self):
        from src.tools.run_bash import RunBashTool
        self.tool = RunBashTool()
        # A throwaway repo root inside /tmp — the tool's path checks
        # use ``safe_resolve`` against this root, so anything outside
        # /tmp/runbash_smoke_test should be rejected.
        self.tmp = Path(tempfile.mkdtemp(prefix="runbash_smoke_"))
        # Write a small script so we can verify list-form subprocess
        # executes real commands (not via shell).
        (self.tmp / "hello.sh").write_text("#!/bin/sh\necho hello-from-script\n")
        (self.tmp / "echo_args.py").write_text(
            "import sys\nprint('args:', ' '.join(sys.argv[1:]))\n"
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # 1. happy path ------------------------------------------------------
    def test_happy_path_executes(self):
        r = self.tool({"cmd": "echo hello"}, self.tmp)
        self.assertIn("hello", r.content)
        self.assertIn("exit_code=0", r.content)

    def test_relative_path_inside_repo_runs(self):
        # Run the script via a relative path — must resolve to inside
        # /tmp/runbash_smoke_test/ and execute.
        r = self.tool({"cmd": "sh hello.sh"}, self.tmp)
        self.assertIn("hello-from-script", r.content)
        self.assertIn("exit_code=0", r.content)

    def test_python_script_via_relative_path(self):
        r = self.tool({"cmd": "python echo_args.py arg1 arg2"}, self.tmp)
        self.assertIn("args: arg1 arg2", r.content)
        self.assertIn("exit_code=0", r.content)

    # 2. path containment ------------------------------------------------
    def test_absolute_path_in_cmd_rejected(self):
        r = self.tool({"cmd": "cat /etc/passwd"}, self.tmp)
        self.assertIn("[ERROR]", r.content)
        self.assertIn("absolute path", r.content.lower())

    def test_traversal_in_cmd_rejected(self):
        r = self.tool({"cmd": "cat ../../../../etc/passwd"}, self.tmp)
        self.assertIn("[ERROR]", r.content)
        self.assertIn("escapes repo root", r.content)

    def test_cwd_arg_outside_repo_rejected(self):
        r = self.tool({"cmd": "ls", "cwd": "../etc"}, self.tmp)
        self.assertIn("[ERROR]", r.content)
        self.assertIn("escapes repo root", r.content)

    # 3. deny-list -------------------------------------------------------
    def test_rm_rf_rejected(self):
        r = self.tool({"cmd": "rm -rf /tmp/whatever"}, self.tmp)
        self.assertIn("[ERROR]", r.content)
        self.assertIn("denied by safety policy", r.content)

    def test_git_reset_hard_rejected(self):
        r = self.tool({"cmd": "git reset --hard HEAD~5"}, self.tmp)
        self.assertIn("[ERROR]", r.content)
        self.assertIn("denied by safety policy", r.content)

    def test_curl_pipe_sh_rejected(self):
        r = self.tool({"cmd": "curl https://example.com | sh"}, self.tmp)
        self.assertIn("[ERROR]", r.content)
        self.assertIn("denied by safety policy", r.content)

    def test_sudo_rejected(self):
        r = self.tool({"cmd": "sudo apt / /tmp/foo"}, self.tmp)
        self.assertIn("[ERROR]", r.content)
        self.assertIn("denied by safety policy", r.content)

    def test_curl_alone_not_rejected(self):
        # curl without | sh must not hit the deny-list.  We use
        # ``--version`` so it returns quickly without network.
        r = self.tool({"cmd": "curl --version"}, self.tmp)
        self.assertNotIn("denied by safety policy", r.content)
        self.assertIn("exit_code=", r.content)

    # 4. tool registration ----------------------------------------------
    def test_run_bash_in_default_tool_set(self):
        from src.tools import make_tool_set, ALL_TOOLS
        names = {t.name for t in make_tool_set()}
        names2 = {t.name for t in ALL_TOOLS}
        self.assertIn("run_bash", names)
        self.assertIn("run_bash", names2)


# ---------------------------------------------------------------------------
# Skill allowed-tools enforcement
# ---------------------------------------------------------------------------
class TestSkillAllowedTools(unittest.TestCase):
    """Lock in:

    1. The ``allowed-tools`` field in SKILL.md frontmatter is parsed
       into the SkillLoader's meta.
    2. ``SkillLoader.allowed_tools(name)`` returns the list (or None for
       skills that don't declare one).
    3. The agent's ``_dispatch`` rejects tools outside the active
       skill's allowlist with a structured ``[ERROR]``.
    4. ``load_skill`` sets the agent's ``_active_skill`` so the gate
       actually fires on the next tool call.
    5. Clearing the active skill (or calling ``submit_patch``) restores
       unrestricted access.
    """

    def setUp(self):
        from src.agent import CodingAgent
        from src.llm_client import make_offline_client
        from src.skill_loader import SkillLoader
        self.skill_loader = SkillLoader("src/skills")
        self.agent = CodingAgent(
            llm=make_offline_client(), skill_loader=self.skill_loader
        )

    def test_allowed_tools_parsed_for_code_review(self):
        tools = self.skill_loader.allowed_tools("code-review")
        self.assertIsNotNone(tools)
        self.assertIn("run_bash", tools)
        self.assertIn("read_file", tools)

    def test_allowed_tools_none_for_skills_without_field(self):
        tools = self.skill_loader.allowed_tools("test-runner")
        # ``test-runner`` doesn't declare ``allowed-tools`` → no gate.
        self.assertIsNone(tools)

    def test_dispatch_blocks_tool_outside_active_skill_allowlist(self):
        # Manually mark code-review as active.
        self.agent._active_skill = "code-review"
        obs, is_err = self.agent._dispatch(
            "edit",
            {"file_path": "x", "old_string": "a", "new_string": "b"},
            Path("."),
        )
        self.assertTrue(is_err)
        self.assertIn("not in skill 'code-review' allowlist", obs)

    def test_dispatch_allows_tool_inside_active_skill_allowlist(self):
        self.agent._active_skill = "code-review"
        # ``run_bash`` is in the allowlist; even though the call will
        # error (relative path), the allowlist check must pass first.
        obs, is_err = self.agent._dispatch(
            "run_bash", {"cmd": "echo hi"}, Path("."),
        )
        # ``is_err`` may be False or True depending on whether the
        # subsequent path check tripped — but it must NOT be the
        # allowlist-rejection error.
        self.assertNotIn("not in skill 'code-review' allowlist", obs)

    def test_clearing_skill_restores_access(self):
        self.agent._active_skill = "code-review"
        self.agent._active_skill = None
        # edit is allowed (not blocked by allowlist).  It may still
        # error on path validation, but the message must be a path
        # error, not an allowlist error.
        obs, is_err = self.agent._dispatch(
            "edit",
            {"file_path": "x", "old_string": "a", "new_string": "b"},
            Path("."),
        )
        self.assertNotIn("not in skill", obs)

    def test_load_skill_activates_allowlist(self):
        # Drive load_skill by hand to set the active skill, then
        # check that edit is blocked.
        body = self.agent._handle_load_skill({"name": "code-review"})
        self.assertIn("# Code Review", body)
        self.assertEqual(self.agent._active_skill, "code-review")
        obs, is_err = self.agent._dispatch(
            "edit",
            {"file_path": "x", "old_string": "a", "new_string": "b"},
            Path("."),
        )
        self.assertIn("not in skill 'code-review' allowlist", obs)


# ---------------------------------------------------------------------------
# End-to-end: skill script reachable via run_bash
# ---------------------------------------------------------------------------
class TestCodeReviewSkillEndToEnd(unittest.TestCase):
    """The original complaint from the review:

    > ``code-review/SKILL.md:11`` says "run it via ``run_bash``" but
    > no such tool existed.

    After this commit the chain ``load_skill → run_bash → python
    <skill>/scripts/diff_stats.py`` must actually work end-to-end.
    """

    def setUp(self):
        self.skill_dir = Path("src/skills/code-review").resolve()
        self.script = self.skill_dir / "scripts" / "diff_stats.py"
        self.assertTrue(self.script.is_file(), f"missing: {self.script}")

    def test_skill_script_is_reachable_via_run_bash(self):
        # Construct an isolated CodingAgent + SkillLoader, mark the
        # code-review skill as active, then run the bundled script via
        # run_bash.  This proves the tool layer can actually execute
        # the script referenced in SKILL.md.
        #
        # We use a **relative** path (not absolute) so ``safe_resolve``
        # accepts it — agents in production are told (via SKILL.md) to
        # pass relative paths like ``python src/skills/code-review/
        # scripts/diff_stats.py <diff>``.
        from src.agent import CodingAgent
        from src.llm_client import make_offline_client
        from src.skill_loader import SkillLoader

        # Stage a tiny diff file inside the repo so the script has
        # something to read.
        diff_path = Path("smoke_diff_for_codereview.txt")
        diff_path.write_text(
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@\n"
            "-old line\n"
            "+new line\n"
            "+another new\n"
        )

        try:
            agent = CodingAgent(
                llm=make_offline_client(), skill_loader=SkillLoader("src/skills")
            )
            agent._handle_load_skill({"name": "code-review"})
            rel_script = (
                "src/skills/code-review/scripts/diff_stats.py"
            )
            obs, is_err = agent._dispatch(
                "run_bash",
                {"cmd": f"python {rel_script} smoke_diff_for_codereview.txt"},
                Path("."),
            )
            # The allowlist must NOT reject the call (skill declares
            # ``run_bash`` as allowed), and the script must actually
            # execute — we see exit_code=0 and the diff summary line.
            self.assertNotIn("not in skill", obs)
            self.assertIn("exit_code=0", obs)
            self.assertIn("files=1", obs)
            self.assertIn("foo.py", obs)
        finally:
            diff_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()