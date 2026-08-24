# Code Review Checklist

A short reference the agent reads while running the ``code-review``
skill. Use this to grade each hunk and decide on severity
(``must-fix`` / ``should-fix`` / ``nit``).

## Severity rubric

| Severity     | Example                                              | Required action |
|--------------|------------------------------------------------------|-----------------|
| ``must-fix`` | Security hole, off-by-one, lost test, data loss     | Block merge     |
| ``should-fix``| Style violation with cited reason                   | Request change  |
| ``nit``       | Subjective taste (``x = x + 1`` → ``x += 1``)        | Mention, don't block |

## Things to always check

1. **Public API changed** — was the breaking call site updated?
2. **Test coverage** — every new branch is exercised by at least one test.
3. **Error paths** — exceptions are caught at the right layer, not swallowed.
4. **Resource cleanup** — files / sockets / locks are closed in finally / contextlib.
5. **Naming** — does the new symbol describe its behaviour, not its implementation?
6. **Backward compat** — feature flags / deprecation warnings where needed.

## Common smells (from the 7B model that we hit during the build)

- Re-importing the same module inside a hot loop.
- Comparing floats with ``==`` instead of ``math.isclose``.
- Mutable default arguments (``def f(x=[])``).
- Bare ``except:`` clauses.
- String concatenation inside tight loops — pre-format / use ``io.StringIO``.
