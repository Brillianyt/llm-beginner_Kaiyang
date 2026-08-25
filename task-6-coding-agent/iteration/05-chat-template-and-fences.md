# 05 — Chat Template Reasoning + Patch Fence Regex（commit `662ba97`）

## 主题

两个**互相耦合**的 bug，都从 client↔vLLM 的 response body 里抓到。

- **Bug A**：chat template 强制模型"只输出 JSON object, no prose, no code fences, no XML wrappers" —— 抑制了 reasoning；模型在每一轮的 `message.content` 都是空字符串，根本没机会做 chain-of-thought。
- **Bug B**：`src/agent.py:_PATCH_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)` —— 只匹配 `diff`/`patch` 围栏；模型在允许 reasoning 后把 fix-as-text 用 ``` ```python``` 围栏，agent 完全提取不到。

## 症状

### Bug A

打开任意一份 astropy-12907 wire capture，看 assistant `message.content`：

```
content_len = 0      ← 模型没说一个字
tool_calls = 1       ← 直接调工具
```

—— 50 turn 全部如此：模型直接调 `read_file`/`edit`/`run_tests`，中间**没有任何** reasoning、规划、或自言自语。Qwen2.5-Coder-7B-Instruct 不是 GPT-4，没有内置的 hidden CoT，必须**显式**在 content 里写推理才能形成思维链。

### Bug B

打开 `full_cot_14365__20260825T020257Z.json` turn 2 的 assistant content（在开启 reasoning 之后）：

```
"Looking at the issue, the problem is that `command_re = r\"READ [TS]ERR...\"` 
is case-sensitive. To make it case-insensitive, I should add `re.IGNORECASE` 
to `re.compile(_type_re)`. Here's the fix:

```python
_line_type_re = re.compile(_type_re, re.IGNORECASE)
```
"
```

—— 模型给了正确的推理（"是 case-sensitivity bug，要加 re.IGNORECASE"）和正确的 fix-as-code（围栏是 ``` ```python``` ）。

但 `src/agent.py` 的 `_PATCH_FENCE_RE` 正则是 `(?:diff|patch)?`——`python` 不在 alternative 里。`re.search` 返回 `None`。agent 退回到 `_looks_done` / `_FINALISE_RE`，把整个 content 当成"未结构化输出"对待，**最终触发 `_PATCH_FENCE_RE` 全部 miss**。

## 根因

### Bug A

`models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja` 在 system message 里硬编码：

```
{{- 'You are a coding assistant. Output exactly one JSON object for tool 
calls. Do NOT output any prose, do NOT output code fences, do NOT output 
XML wrappers.' -}}
```

—— 这条指令在 Qwen2.5-Coder-7B-Instruct 上**过强**：模型**完全**放弃了 reasoning text，content 永远是空。

对比：Qwen2.5-Instruct（工具微调版）有专门的 reasoning 训练数据，能在 tool call 前输出短推理。**Coder 版**没有这个训练数据，但 chat template 这条"no prose"指令让模型连自发尝试都放弃。

### Bug B

`src/agent.py:_PATCH_FENCE_RE`：

```python
_PATCH_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
```

—— `(?:diff|patch)?` 里的 alternative 不含 `python`/`py`。这是 commit 之前 Agent 假设备选 path：`submit_patch(diff=...)` 工具是**首选**通道（直接由 vLLM parser 解析 tool_call），`_PATCH_FENCE_RE` 是 `submit_text(text=...)` 通道的 patch 提取后备。

但当 chat template 允许 reasoning 后，模型倾向把"先解释 + 给 patch"作为整段 content 发出来，**用 ```python``` 围栏**（自然语言习惯）。`_PATCH_FENCE_RE` 漏了。

## 在 wire capture 里的发现

### Bug A — content 始终为 0

```python
import json
d = json.load(open('eval/wire_captures/cot_template_12907__20260824T165635Z.json'))
for i, r in enumerate(d['captured_http_requests']):
    msg = r['response_body']['choices'][0]['message']
    print(i, 'content_len=', len(msg.get('content') or ''), 'tool_calls=', len(msg.get('tool_calls', [])))
```

输出：

```
0  content_len=0   tool_calls=1
1  content_len=2154 tool_calls=0   ← 此处出现 reasoning（开启了 reasoning 的早期 capture）
```

—— turn 0 完全是 tool call（无可视 reasoning），turn 1 有 2154 char reasoning 但**这次是 chat template 已被 hack 修改**——这是 commit `662ba97` 之前的最后一份 baseline capture（在 chat template 替换后跑的），用于对比。

修复前真正的 capture（如 `audit_14365__20260824T161513Z.json`）：

```
0  content_len=0   tool_calls=1
1  content_len=0   tool_calls=1
2  content_len=0   tool_calls=1
... 50 轮全部 content_len=0
```

### Bug B — _PATCH_FENCE_RE miss

`full_cot_14365__20260825T020257Z.json` turn 2 model output（含 ```python``` fence 的 fix-as-text）：

```
"line = line.strip().upper().lstrip('!')"
```

—— 模型给的 fix **形式是 code**，agent 拿不到 patch，最后 `_looks_done` 触发 `_FINALISE_RE`，命中 "Done" 这种词，done_reason=completed。

## 修复

### Bug A — chat template

`models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja` 把

```
'Output exactly one JSON object for tool calls. Do NOT output any prose, 
do NOT output code fences, do NOT output XML wrappers.'
```

改成

```
'You may output a SHORT reasoning sentence (1-2 lines max) BEFORE the 
JSON object to plan your next step. After the reasoning, output exactly 
one JSON object describing the tool call. Do NOT use code fences inside 
the JSON object.'
```

—— 允许 1-2 行 reasoning（**严格限制**以避免 reasoning 占用太多 output tokens）。

### Bug B — _PATCH_FENCE_RE

`src/agent.py`：

```python
_PATCH_FENCE_RE = re.compile(r"```(?:diff|patch|python|py)?\s*\n(.*?)```", re.DOTALL)
#                              ↑                                          ↑
#                              把 'python' 和 'py' 加进 language-tag alternative
```

—— agent 现在能 extract ```python``` 围栏的 patch。

## 验证

`cot_template_12907__20260824T165635Z.json`（修复**后** capture）：

```
0  content_len=0    tool_calls=1
1  content_len=2154 tool_calls=0   ← reasoning 正常发出
2  content_len=2096 tool_calls=0   ← reasoning 累积
```

`patch_fix_12907__20260824T170022Z.json` turn 1 的 content 显示模型给出了 2096 char reasoning，**正确诊断** 12907 是 `_cstack = right` 那一行的硬编码错误（虽然模型仍没把它 patch 正确——见 14182 的失败记录）。

`full_cot_14365__20260825T020257Z.json` 显示模型在 4 turn 内：
- turn 0: tool call read_file
- turn 1: tool call edit
- turn 2: tool call edit
- turn 3: tool call submit_patch

—— reasoning 终于出现在 `message.content` 里，模型能思考了。但 astropy-14365 仍然 FAIL，因为 fix 仍不正确（仅修了一个 caller 而非两个）。SWE-bench PASS 还需要 commit `99ffaeb`（Bug C + D）。

## 副作用 / 影响

- **Token 消耗上升**。reasoning 文本占 ~2000 chars，50 turn 总 token 翻倍。
- **Output 截断风险上升**。`max_tokens=4096`，reasoning 2000 + tool_call JSON 200 + 后续回复 = 容易超。但实测没触发，Qwen2.5-Coder-7B 习惯"reasoning 简短"。
- **架构不变**。tool_call 仍然只通过 `message.tool_calls`，`tool_call_native_rate=1.0`、`fallback_markers=[]` 保持。

## Commit

```
662ba97  fix: chat template allows reasoning + agent parses python fences
```

修改文件：
- `models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja`（替换 "no prose" 段）
- `src/agent.py`（`_PATCH_FENCE_RE` +2 chars: `python|py`）
- 3 个 wire capture（`cot_template_12907__20260824T165635Z.json`、`flow_analysis_14365__20260824T165254Z.json`、`patch_fix_12907__20260824T170022Z.json`）

## 关键证据

- `eval/wire_captures/cot_template_12907__20260824T165635Z.json` — 修复**后** 12907 reasoning 出现
- `eval/wire_captures/patch_fix_12907__20260824T170022Z.json` — 2096 char reasoning 给出正确诊断
- `eval/wire_captures/full_cot_14365__20260825T020257Z.json` — ```python``` 围栏被 agent 提取到