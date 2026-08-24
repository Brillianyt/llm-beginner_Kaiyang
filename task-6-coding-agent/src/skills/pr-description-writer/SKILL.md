---
name: pr-description-writer
description: "Generate a PR title + body from a diff and recent commits. Use when opening a PR or when the user asks for a changelog / PR description."
when_to_use: "Drafting a pull-request description or a changelog entry."
---

# PR Description Writer

## Inputs
- `repo_path`: absolute path to repo.
- `base_branch`: default `main`.
- Optional `commit_range`: e.g. `HEAD~3..HEAD`.

## Steps
1. `git log {base_branch}..HEAD --oneline` — recent commit style.
2. `git diff {base_branch}...HEAD` — full diff.
3. Write a PR title (≤ 70 chars, imperative mood, no period).
4. Write the body in this template:

```markdown
## Summary
- 1–3 bullets explaining the *what* and *why*.

## Test plan
- [ ] Unit tests added/updated
- [ ] `pytest` passes locally
- [ ] Manual verification step (if UI)

## Risk
- Rollback plan (revert commit / revert PR)
- Affected modules
```

## Hard rules
- Never claim a test exists that you did not actually run.
- Never invent commits — only describe the ones returned by `git log`.