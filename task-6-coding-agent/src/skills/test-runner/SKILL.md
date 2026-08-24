---
name: test-runner
description: "Diagnose and fix failing tests. Use when `run_tests` returns failures, or when the user reports a failing test."
when_to_use: "Test failure diagnostic loop. Read failures[], classify, fix, re-run."
---

# Test Runner & Diagnostic

## Workflow

### 1. Parse the failure
- Identify failing file/line from `failures[]`.
- Read the source under test AND the test itself (≥ 20 lines of context each).

### 2. Classify the failure
| Class           | Symptom                              | Fix path                          |
|-----------------|--------------------------------------|-----------------------------------|
| implementation  | test correct, source wrong           | edit source                       |
| test bug        | test wrong assertion / setup         | fix test (only if user permits)   |
| flaky           | first run fail, second pass          | re-run 1–2 more times             |
| environment     | missing dep / wrong cwd              | ask user; don't `pip install` blindly |

### 3. Fix loop
```
while not done:
    read source under test (read_file)
    generate patch (in your head)
    apply patch  (write_file / git_apply)
    run tests    (run_tests)
    if still failing:
        look at new failures, iterate
        if stuck after N turns → call user
```

### 4. Confirm
- Run the full test suite one more time after a green run.
- Output: `✅ <N> tests passed, 0 failed`.

## Hard rules
- Do NOT modify tests to make them pass unless the user explicitly approves.
- Stop iterating after 8 turns — summarise state and call user for help.