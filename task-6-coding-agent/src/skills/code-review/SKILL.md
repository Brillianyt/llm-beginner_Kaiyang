---
name: code-review
description: "Perform a structured code review on a diff. Use when the user asks 'review my changes', before opening a PR, or after running tests."
when_to_use: "Reviewing a diff for correctness, security, or style. Pre-PR sanity check."
allowed-tools: [run_bash, read_file, grep]
---

# Code Review

## Available resources
- `scripts/diff_stats.py` — quick file-level +/−/files-touched summary.
  Invoke via the `run_bash` tool (the agent's sandboxed shell):
  ```
  run_bash(cmd="python <skill_dir>/scripts/diff_stats.py <diff_path>")
  ```
  `<skill_dir>` is the directory containing this SKILL.md
  (`src/skills/code-review/`); `<diff_path>` is a path the agent has
  written to disk (e.g. via `git_diff` followed by `write_file`) — it
  must be inside the repo.
- `references/review-checklist.md` — severity rubric + common smells.
  Read it via the SkillLoader's `read_reference` for detailed grading.

## When to use this skill
- After `git_diff` shows non-trivial changes.
- Before submitting a patch.
- When the user explicitly asks for a review.

## Inputs you receive
- `repo_path`: absolute path to repo.
- `diff`: unified diff (from `git_diff`).
- Optional `focus`: e.g. correctness | performance | security | style.

## Steps
1. (Optional) Run `scripts/diff_stats.py` via `run_bash` for a one-line overview.
2. Read each hunk with ≥ 10 lines of surrounding context (use `read_file`
   on the source files; do not pipe them through bash).
3. Classify each finding:
   - **must-fix** — bug, security issue, data loss, broken test.
   - **should-fix** — clear style/idiom violation with cited reason.
   - **nit** — subjective; mention only if asked.
4. Produce a review in this shape:

   - **Summary** (1–2 sentences)
   - **Findings** (bullets: severity, file:line, why, suggested fix)
   - **Praise** (1–3 bullets, brief)

5. If no issues: say "No issues found" + one sentence why.

## Allowed tools (enforced)

When this skill is loaded, the `allowed-tools` frontmatter above is
enforced by the agent: **only `run_bash`, `read_file`, `grep`** may be
called while the skill is active. Calling `edit`, `write_file`, or any
write tool while this skill is loaded is rejected with
`[ERROR] tool 'X' not in skill 'code-review' allowlist`.

## Output format
Markdown only, no preamble. Maximum 30 findings per call.