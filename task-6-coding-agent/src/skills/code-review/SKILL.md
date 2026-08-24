---
name: code-review
description: "Perform a structured code review on a diff. Use when the user asks 'review my changes', before opening a PR, or after running tests."
when_to_use: "Reviewing a diff for correctness, security, or style. Pre-PR sanity check."
---

# Code Review

## When to use this skill
- After `git_diff` shows non-trivial changes.
- Before submitting a patch.
- When the user explicitly asks for a review.

## Inputs you receive
- `repo_path`: absolute path to repo.
- `diff`: unified diff (from `git_diff`).
- Optional `focus`: `correctness | performance | security | style`.

## Steps
1. Read each hunk with ≥ 10 lines of surrounding context.
2. Classify each finding:
   - **must-fix** — bug, security issue, data loss, broken test.
   - **should-fix** — clear style/idiom violation with cited reason.
   - **nit** — subjective; mention only if asked.
3. Produce a review in this shape:

   - **Summary** (1–2 sentences)
   - **Findings** (bullets: severity, file:line, why, suggested fix)
   - **Praise** (1–3 bullets, brief)

4. If no issues: say "No issues found" + one sentence why.

## Output format
Markdown only, no preamble. Maximum 30 findings per call.