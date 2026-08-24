# AGENTS.md

Mini Coding Agent — a tool-subprocess-driven Claude Code clone on
Qwen2.5-Coder-7B-Instruct, with three capability layers (Tools / Skills /
Subagents) and a single deterministic agent loop.

never owe bugs to model Qwen

 1. 请继续规划反思harness的问题完成记录线索与实验，最终通过至少一个SWE
  │  2. 你可以参考claude 反思其设计意图，进一步捕捉现有harness的问题进行改进，SWE至少一个通过是唯一证明你做对了的理由
  └  3. 按照client和vllm两端消息产出的序列，分析消息产出是否正确以及，并分析其背后原因，参考claude设计

 你应该严峻地获取每一条在server和client之间传递的信息，对于工具调用，你直接根据该信息地要求调用对应的工具查看效果
每10min最多做一次test，因此你要仔细规划test的目的是什么，一个test不能等待超过5min
每次做test之前优先明确test的意图和目的，如果没有太大意义，就不要做，喜欢做test是一个坏习惯


## Project invariant — no fallback

This codebase has a **hard architectural rule**: there is **no
fallback** for tool-call parsing.  Concretely:

1. **Tool calls arrive only via OpenAI `message.tool_calls`** from the
   vLLM `qwen_coder_json` parser plugin in `src/vllm_plugin/`.  The
   agent **never** introspects `message.content` for tool-call JSON.
2. **Adding a text-mode tool-call parser to the agent loop is a
   regression.** If you ever want one, fix the vLLM parser plugin — it
   is the canonical source.  See
   `src/diagnostics/text_tool_parser.py` for the offline-only diagnostic
   surface; it is never imported by `src/agent.py`.




## vLLM deployment (Coder 路径)

The agent assumes vLLM is started with **all four** non-default flags:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen2.5-Coder-7B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser qwen_coder_json \
  --tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py \
  --chat-template models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja \
  --generation-config vllm
```

The agent treats `message.tool_calls == []` as a deployment-level
signal that the parser plugin is misconfigured (`tool_call_native_rate`
< 1.0 in the trace).  We do **not** add a runtime rescue.

## Project layout

- `src/agent.py` — main loop (`CodingAgent.run`); consumes wire-format
  `tool_calls`; **no** text-mode fallback.
- `src/llm_client.py` — OpenAI-compatible wire layer.
- `src/mcp_server.py` — stdio MCP server (exposes `read_file`,
  `write_file`, `edit`, `list_files`, `grep`, `run_tests`, `git_diff`,
  `git_apply`).
- `src/skill_loader.py` + `src/skills/<name>/SKILL.md` — Skills.
- `src/subagents/{search,test}_executor.py` — Subagents.
- `src/vllm_plugin/qwen_coder_tool_parser.py` — Server-side tool-call
  parser plugin.  Single-path (one regex); XML-tag-split form
  (~12% of Coder output) is documented as a known-unsupported edge.
- `src/diagnostics/text_tool_parser.py` — **Offline-only** diagnostic
  helper.  Not imported by `src/agent.py`; architecturally unreachable.
- `src/hooks.py` — Pre/PostToolUse (test-write protection + audit log).
- `src/trace.py` — `Trace` dict-subclass + `DoneReason` enum.
- `test_smoke.py` — single test file; covers M1–M4 + the static guard.

## Testing instructions
Never test for no reason,before you do test,spcific a purpose,to ensure what?
For any purpose,do test <3 times,that's why you need be really careful.


## Code style

- No formatter / linter configured in this repo.  Match the surrounding
  file: type hints via `typing`, docstrings on every public symbol,
  Chinese inline comments where helpful, ASCII table-art only when it
  carries real signal.
- Module-level docstrings on every file in `src/` describing its
  responsibility and any deferred decisions (e.g. moved-out fallbacks,
  dead code, **what is intentionally NOT here**).

## PR & commit conventions

- Branch from `main`; never push to it directly.
- Commit message: imperative, ≤ 72 chars (`fix:`, `feat:`, `refactor:`,
  `docs:` etc.).
- Every PR must keep `python3 -m pytest test_smoke.py -q` green and not
  add new occurrences of `text-tool-call fallback` patterns to
  `src/agent.py`.  See the static guard test.

## Security

- All tools are sandboxed via `safe_resolve` to `repo_root`; never
  bypass `Path.resolve()` + `relative_to()` checks.
- `subprocess` calls use list form, `shell=False`, `cwd=` restricted.
- Audit log default: `<cwd>/.coding-agent-audit.jsonl`; do not commit
  audit logs to git (`.gitignore` excludes them).
- The `qwen_coder_json` parser plugin is the only path through which
  tool-call arguments reach `BaseTool.__call__`.  Never add a
  side-channel that hands user-supplied content to `BaseTool.dispatch`
  without the OpenAI protocol boundary.
