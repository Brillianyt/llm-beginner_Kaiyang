# 13 — Chat Template Format Mismatch (verifying the rebuild on live vLLM)

> 2026-08-26（本会话）。在 vLLM 0.23.0 + ModelScope 下载的
> Qwen2.5-Coder-7B-Instruct 上端到端验证 chat_template 的"丢失后重建"——
> 包含双端消息捕获、YaRN rope_scaling 注入、ASTROPY SWE Lite 跑通。

## 1. 环境准备

| 项 | 来源 / 路径 | 备注 |
|---|---|---|
| 模型权重 | ModelScope `qwen/Qwen2.5-Coder-7B-Instruct` (15 GB safetensors) | pip mirror `-i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple`；下载到 `models/Qwen2.5-Coder-7B-Instruct/` |
| YaRN 配置 | 加到 `config.json`：`rope_scaling={factor:4.0, original_max_position_embeddings:32768, type:yarn}` | 长文本扩展到 ~131K tokens（基础 32,768 × 4） |
| astropy working tree | `data/repos/astropy`（从 bare clone @ `d16bfe05` 切出） | SWE-bench Lite 3 题都在 `data/swebench-lite-sample.parquet` |
| vLLM | `0.23.0+v0.2.0.ppu2.1.1`，`vllm serve` with `--enable-auto-tool-choice --tool-call-parser qwen_coder_json --tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py --chat-template models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja --generation-config vllm --max-model-len 16384 --gpu-memory-utilization 0.90` | 启动耗时 ~3 min（compile + CUDA graph capture）；监听 127.0.0.1:30000 |

### 1.1 完整启动方法（含 YaRN 长文本支持的端到端命令序列）

```bash
# 1) 通过指定 pip 镜像装 modelscope（已装可跳）
pip install -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple modelscope

# 2) 用 ModelScope 下载 Qwen2.5-Coder-7B-Instruct 到本地（~15 GB）
python3 -c "
from modelscope import snapshot_download
snapshot_download(
    'qwen/Qwen2.5-Coder-7B-Instruct',
    cache_dir='models',
    allow_file_pattern=['*.json', '*.txt', '*.tiktoken',
                        'tokenizer*', 'vocab.json', 'merges.txt',
                        '*.safetensors'],
)
"
# → models/Qwen2.5-Coder-7B-Instruct/{config.json, *.safetensors, ...}

# 3) 加 YaRN rope_scaling（**编辑** config.json；不加这一段 vLLM 会按 32K 截断）
#    在原 config.json 末尾加：
#       "rope_scaling": {
#           "factor": 4.0,
#           "original_max_position_embeddings": 32768,
#           "type": "yarn"
#       }
#    → 等效 max context ≈ 32K × 4 = 131,072 tokens（vLLM 实际生效以 --max-model-len 为准）

# 4) 启动 vLLM（含 chat_template + parser plugin + 自动工具调用 + YaRN-aware）
vllm serve models/Qwen2.5-Coder-7B-Instruct \
  --port 30000 --host 127.0.0.1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen_coder_json \
  --tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py \
  --chat-template models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja \
  --generation-config vllm \
  --max-model-len 16384 \            # 实际 KV 预算（YaRN 让原生支持到 131K，但本机显存允许 16K）
  --gpu-memory-utilization 0.90 \
  --served-model-name Qwen2.5-Coder-7B-Instruct

# 5) 健康检查（启动完成耗时 ~3 min：weight load 2.3s + compile 21s + CUDA graph capture ~14s）
curl http://127.0.0.1:30000/v1/models
# → {"data":[{"id":"Qwen2.5-Coder-7B-Instruct", ...}]}

# 6) 双端 wire capture + SWE 跑通
python3 eval/run_one_with_capture.py --instance-idx 0 \
    --label verify_template_12907 --max-turns 10
```

YaRN 启动要点：
- **YaRN 必须改 `config.json`**（不是启动 flag）；vLLM 0.23.0 读到 `rope_scaling.type=yarn` 后会用 YaRN rope 替换默认 RoPE，不改就退回到 32K 硬截断。
- `--max-model-len 16384` 是**实际** KV cache 预算上限，受显存约束。YaRN 在数值上把模型"能处理"的上限提到 131K，但本机 PPU 80 GB 显存允许跑 16K 上下文并发 ~79 路。
- 长 prompt（>32K）只在确实需要时才压上来——SWE-bench 实测 prompt ≈3K tokens，16K 绰绰有余。

## 2. 双端消息捕获的发现（vLLM 端 + client 端）

第一次跑 `astropy-12907` (v2 capture `verify_template_12907_v2__20260826T081509Z.json`)，
6 turns 全部命中 client↔vLLM 之间的同一个协议层 bug：

### wire 证据（model 端 output）
```
content: "After reading the file, I will identify the nested compound model and understand its structure. Then, I will manually compute the separability matrix for the nested model.\n\n```json\n{\n  \"function\": \"edit\",\n  \"parameters\": {\n    \"file_path\": \"...\",\n    \"old_string\": \"...\",\n    \"new_string\": \"...\"\n  }\n}\n```"
tool_calls: []           ← vLLM 把 message.tool_calls 设成空数组
finish_reason: "stop"    ← 不是 "tool_calls"，因为 parser 没匹配
```

### 根因

| 角色 | 看到的内容 |
|---|---|
| 模型实际输出 | `{"function": "edit", "parameters": {...}}` —— Qwen2.5-Coder **legacy** tool-call format |
| chat_template 指令（迭代 12 重建版） | "output exactly one JSON object describing the tool call" —— **没说 JSON 长什么样** |
| parser `qwen_coder_json` | 单 regex `_RE_TOOL_CALL`，匹配 `{"name": "...", "arguments": {...}}` —— Hermes/OpenAI 风格 |
| vLLM `enable-auto-tool-choice` 路径 | 调 parser；parser 返回 `tools_called=False` → `tool_calls=[]` |

→ **chat_template 指令太模糊**。Qwen2.5-Coder-7B-Instruct 的训练数据里 legacy `function/parameters` 占比很高，没有强指令就会落到这个格式上。Harness 没有"重新猜格式"这条路径（项目 invariant `AGENTS.md` 硬禁止 text-mode fallback），所以模型和 parser 完全没接上。

### 为什么 iteration/12 的 wire capture 看着正常

旧的 capture（`stuck_detector_14365__20260825T021926Z.json`）里 response 是
`{"name": "read_file", "arguments": "..."}`，证明在 **原来的环境**（不同 vLLM 版本？或
训练权重来源不同？或 YaRN 已 baked-in？）模型会落到 Hermes 格式。当前 ModelScope 下载的
权重 + vLLM 0.23.0 组合下不再保证这一点，**重建出的 chat_template 不可被"信任行为继承"**。

## 3. 修复

`models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja` line 13 把模糊的"describe the tool call"换成显式 JSON 形状：

```
修前（迭代 12 重建版、脆弱）：
"You may output a SHORT reasoning sentence (1-2 lines max) BEFORE the JSON object 
to plan your next step. After the reasoning, output exactly one JSON object 
describing the tool call. Do NOT use code fences inside the JSON object."

修后（本迭代、明示 JSON 形状）：
"You may output a SHORT reasoning sentence (1-2 lines max) BEFORE the tool call. 
After the reasoning, output exactly ONE tool call as a JSON object of the form 
{\"name\": \"<tool_name>\", \"arguments\": {<args-json-object>}}. Do NOT use code 
fences inside the JSON. Do NOT wrap the JSON in any XML tag."
```

变更要点：
1. **明确 JSON schema**（`{"name", "arguments"}`）—— 让 Hermes parser 一定能 match。
2. 明确禁 XML 包裹（与 Qwen3 风格的 `<tool_call>` 区分开）。
3. "ONE tool call" 而不是 "exactly one JSON object" —— 后者措辞可能让模型把 tool name 拼成 schema 里的 key。

→ 改完 line 13 仍在原位（"instruction lives at line 13" 的 evidence chain 保持；parser docstring 无需改）。

## 4. 验证：重新启动 vLLM 后 SWE 跑通

### 4.1 直接 API probe
```
POST /v1/chat/completions  tools=[read_file]
content: ""
tool_calls: [{"id": "chatcmpl-tool-...", "type": "function",
              "function": {"name": "read_file", "arguments": "{\"file_path\": \"...\"}"}}]
finish_reason: "tool_calls"
```
→ 模型落到 Hermes 格式、parser 一次匹配成功、vLLM 设 `finish_reason=tool_calls`。

### 4.2 SWE-bench Lite：astropy-12907 (`verify_template_12907_v3__20260826T082018Z.json`)
```
n_captures: 6
  turn 0: content=0c   tool_calls=['read_file']  status=200
  turn 1: content=0c   tool_calls=['read_file']  status=200
  turn 2: content=0c   tool_calls=['edit']        status=200
  turn 3: content=0c   tool_calls=['read_file']  status=200
  turn 4: content=0c   tool_calls=['edit']        status=200
  turn 5: content=0c   tool_calls=['submit_text'] status=200

done_reason: completed
turn_count: 6
tool_call_native_rate: 1.0
fallback_markers: []
edited_files: ['separable.py']   ← 与 golden patch 同名
verdict: PASS
```

**验收：≥1/3 PASS ✓**（astropy-12907 是 iteration/07 中 2/3 PASS 的原成员之一）。

### 4.3 与旧 capture (`stuck_detector_14365__20260825T021926Z.json`) 的对比

| 指标 | 旧 capture | 新 capture |
|---|---|---|
| system content | identical（除日期） | identical（除日期） |
| tools 数 | 12 | 13（多了 `run_bash`） |
| tool_choice | auto | auto |
| response.finish_reason | tool_calls | tool_calls |
| response.tool_calls | `name/arguments` Hermes | `name/arguments` Hermes |
| content_len (turn 0) | 0 | 0 |
| 整体行为 | 跟 harness 协议 | 跟 harness 协议 |

→ chat_template + parser 在 ModelScope 权重 + vLLM 0.23.0 + YaRN 下行为恢复到旧 capture 等价水平。

## 5. 仍待做（不阻塞当前目标）

- astropy-14182 之前 WRONG_FILE（model 选错 file `rst.py`），astropy-14365 之前 PASS（含 run_tests 验证）—— 二者本次未跑，因为验收"≥1 PASS"已达成；用户测试预算也快用完。
- agent 在 `submit_text` 路径上比 `submit_patch` 路径短一截——12907 的 PASS 是"edited the right file"级 PASS，没有跑 `run_tests` 验证真修复。这一层要用 SWE-bench harness 的 FAIL_TO_PASS test 才算"真 PASS"；当前 verify 仅证明 chat_template 工作。

## 6. 提交约束

- `models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja` 已改（line 13 措辞）。
- `models/Qwen2.5-Coder-7B-Instruct/config.json` 加了 `rope_scaling.yarn`（长文本支持）。
- `eval/run_one_with_capture.py`（新增 wire capture driver）—— 不影响 agent 行为。
- `eval/wire_captures/verify_template_12907_*` —— 三份本会话捕获，作为"重建→明示意图→vLLM 端到端"的证据链。
- `iteration/13-chat-template-format-bug.md`（本文档）。