# Wire Evidence Index

本目录的 `wire_captures/` 子目录**软链**到 `eval/wire_captures/` 下全部 25 份 wire capture（HTTP request/response bodies）。每一份 capture 是一次完整 SWE 跑：包含 `summary`、`captured_http_requests[*]`（每轮 `POST /v1/chat/completions` 的 request/response 完整 body）。

按 commit 分组的索引（按时间先后）：

### 90095e4 — read_file 头部不诚实

- `audit_14365__20260824T161513Z.json` — turn 4 的 `read_file` 响应被截断但 header 仍写 `lines 0..400 of 642`
- `astropy_14365_post_fix__20260824T145554Z.json` — 修复后头部出现 `[output truncated at N chars; call read_file again]` marker

### 28754d2 — prompt 缩短 + 第一版 stuck detector

- `astropy_14365_full_trace__20260824T143334Z.json` — 修复前 system prompt 4731 chars
- `flow_analysis_14365__20260824T165254Z.json` — 9 turn 中 4 turn 是同一组 `list_files`+`read_file`（stuck 触发）

### 3b41de2 — wheel mirror + edit/read 错误信息

- `wheel_mirror_14365__20260824T162057Z.json` — 修复前 `run_tests` 返回 setuptools_scm ImportError；修复后返回真实 pass/fail
- `wheel_mirror_12907__20260824T162358Z.json` — 12907 wheel mirror 启用
- `better_edit_err_14365__20260824T162215Z.json` — edit 错误信息增强（hint 含 near-match + token match）
- `better_hint_12907__20260824T163244Z.json` — read 错误信息 hint（"EXISTING .py files in <dir>"）
- `scoped_run_12907__20260824T162642Z.json` — run_tests 缩窄到具体 test 模块
- `file_hint_12907__20260824T162522Z.json` — 模型读错误 hint 后直接调正确路径 read_file

### 127005d — edit discipline prompt

- `heuristics_12907__20260824T164021Z.json` — 模型提议 rename 函数 + 加递归变体
- `reasoning_12907__20260824T164103Z.json` — reasoning 提议 `re.IGNORECASE` 但 edit 改 ValueError
- `final_attempt_12907__20260824T164528Z.json` — 修复 prompt 后 12907 最终 turn sequence

### 662ba97 — Bug A + Bug B

- `cot_template_12907__20260824T165635Z.json` — chat template 修复后 reasoning 出现（content_len=2154）
- `patch_fix_12907__20260824T170022Z.json` — ```python``` 围栏的 fix-as-text 被 agent 提取到
- `full_cot_14365__20260825T020257Z.json` — 修复后 reasoning 出现 + edit args 含 `.upper()` 臆改

### 99ffaeb — Bug C + Bug D

- `auto_scope_14365__20260825T021519Z.json` — RECENT_EDIT_FILE 让 run_tests 缩窄到 test_qdp.py
- `stuck_detector_14365__20260825T021926Z.json` — 修复 C + D 后 8 turn 达到 PASS（含 golden fix）
- `final_12907__20260824T163151Z.json` — 12907 PASS 的最终 wire

### f7e9a1b — `run_bash` 端到端

- `run_bash_e2e__20260825T105623Z.json` — 模型 6 turn 5 次调 `run_bash`，全部被 path 沙箱正确拦截（绝对路径 + `/path/to/` 占位符）；最终 `done_reason=stuck`（同 signature lock 触发）

### Early ablations（更早的探索，未关联具体 commit）

- `astropy_14365_explore_guard__20260824T151326Z.json`
- `astropy_14365_sequential__20260824T151102Z.json`
- `astropy_14365_with_cot__20260824T150838Z.json`

## 如何使用本目录

### 1. 看某个 capture 的完整内容

```bash
python3 -c "
import json
d = json.load(open('iteration/wire_captures/stuck_detector_14365__20260825T021926Z.json'))
print('summary:', json.dumps(d['summary'], indent=2))
for i, r in enumerate(d['captured_http_requests']):
    msg = r['response_body']['choices'][0]['message']
    print(f'turn {i}: content_len={len(msg.get(\"content\") or \"\")}, tool_calls={len(msg.get(\"tool_calls\", []))}')
"
```

### 2. 找某个 commit 修复前的 evidence

```bash
# 例：找 90095e4 的 evidence（read_file 不诚实）
ls -la iteration/wire_captures/audit_14365__20260824T161513Z.json
```

### 3. 对比修复前 vs 修复后

```bash
# Bug A 修复前：
python3 -c "
import json
d = json.load(open('iteration/wire_captures/astropy_14365_full_trace__20260824T143334Z.json'))
for r in d['captured_http_requests'][:3]:
    print('content_len=', len(r['response_body']['choices'][0]['message'].get('content') or ''))
"

# Bug A 修复后：
python3 -c "
import json
d = json.load(open('iteration/wire_captures/cot_template_12907__20260824T165635Z.json'))
for r in d['captured_http_requests'][:3]:
    print('content_len=', len(r['response_body']['choices'][0]['message'].get('content') or ''))
"
```

### 4. 用 jq 高亮 edit args

```bash
# 列出某 capture 里所有 edit 调用的 args
python3 -c "
import json
d = json.load(open('iteration/wire_captures/auto_scope_14365__20260825T021519Z.json'))
for i, r in enumerate(d['captured_http_requests']):
    for tc in r['response_body']['choices'][0]['message'].get('tool_calls', []):
        if tc['function']['name'] == 'edit':
            args = json.loads(tc['function']['arguments'])
            print(f'turn {i}: {args[\"file_path\"].split(\"/\")[-1]}')
            print(f'  old: {args[\"old_string\"][:120]}')
            print(f'  new: {args[\"new_string\"][:120]}')
"
```

## Capture 数据 schema

每份 capture 是 JSON：

```typescript
{
  "summary": {
    "instance_id": "astropy__astropy-14365",
    "turn_count": int,
    "done_reason": "max_turns" | "completed" | "stuck" | "error" | "aborted",
    "post_fix_exit": int | null,    // pytest exit_code after final fix
    "post_fix_summary": str | null, // pytest summary line
    "tool_call_native_rate": float, // 1.0 if all tool_calls came via vLLM
    "fallback_markers": [],         // empty list = no text-mode fallback
    ...
  },
  "captured_http_requests": [
    {
      "url": "/v1/chat/completions",
      "method": "POST",
      "request_body_full": {...} | str,   // Full OpenAI ChatCompletionRequest
      "request_body_bytes": int,
      "response_status": 200,
      "response_body": {...}               // Full OpenAI ChatCompletionResponse
    },
    ...
  ]
}
```

## 为什么用软链而不是复制

- capture 单份 50KB ~ 420KB，25 份共 ~5.6MB
- 软链**不重复占用磁盘**
- `iteration/wire_captures/*.json` 与 `eval/wire_captures/*.json` 始终同步
- 若 harness 后续会修改 `eval/wire_captures/`（写新 capture），`iteration/` 仍指向最新版本