# 12 — Chat Template Recovery（chat_template 恢复）

> 2026-08-25 之后（本会话）：models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja
> 从工作区缺失。本文件记录如何仅凭仓库内资料把它**恢复**出来——证据链、重建决策、
> 以及验证结果。恢复物：models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja

## 1. 缺失事实

- 文件从未进过 git：git ls-files 里无 .jinja；git fsck --unreachable 也无悬空 blob 含模板文本
  （models/ 在 .gitignore 里，作者只在远端 Linux 机上有原件）。
- 但仓库里**到处是它的影子**：AGENTS.md / PRODUCT.md §8.1 的启动命令、迭代记录
  iteration/05-chat-template-and-fences.md、parser 插件 docstring、26 份 wire capture。

## 2. 证据链（每条恢复决策都有出处）

| # | 事实 | 证据 |
|---|---|---|
| 1 | 模板在 system message 里**硬编码**一条工具调用指令 | iteration/05 §根因：{{- 'You are a coding assistant. Output exactly one JSON object ...' -}} |
| 2 | 指令文本（修前 / 修后） | iteration/05 §修复：修前 "Output exactly one JSON object for tool calls. Do NOT output any prose, do NOT output code fences, do NOT output XML wrappers."；修后 "You may output a SHORT reasoning sentence (1-2 lines max) BEFORE the JSON object ..." |
| 3 | 指令位于模板 **第 13 行** | src/vllm_plugin/qwen_coder_tool_parser.py docstring："output exactly one JSON object already in coder_chat_template.jinja line 13" |
| 4 | 工具 schema **必须注入 prompt** | wire capture token 证据：首轮 prompt_tokens=3386；若只渲染 system(2481 chars)+user(1339 chars) 约 1000 tokens，加 <|tools|>{8271 chars}<|/tools|> 后估算 ≈3182，实测 3386（差异 ≈ 特殊 token + JSON 密度），吻合 |
| 5 | 模型必须看到完整 schema（而非仅 system prompt 的 terse 描述） | 全部 capture 里模型输出了 new_string/output_mode/cmd/cwd/timeout_s/repo_path 等 **terse 描述里没有**的参数名 → 证明模板注入了完整 tools |
| 6 | 工具调用以**裸 JSON** {"name", "arguments"} 形式出现在 assistant 回合 | parser 的 canonical 形态 + capture 中模型输出全为此形态（native_rate=1.0） |
| 7 | token 体系是 Qwen2.5-Coder 标准 ChatML（<|im_start|>/<|im_end|>/<|tools|>/<|/tools|>） | 模型 = Qwen2.5-Coder-7B-Instruct；--generation-config vllm；PRODUCT.md §6.5 提到标准 <tool_call> 模板 |
| 8 | tool 结果消息渲染为 <|im_start|>tool\n{content}<|im_end|> | capture 消息角色序列 system→user→assistant(tool_calls)→tool(...)；Qwen2.5 标准格式 |
| 9 | 生成提示以 <|im_start|>assistant\n 结尾（add_generation_prompt） | vLLM --enable-auto-tool-choice 约定 |

## 3. 重建决策（原文件不可见处的判断）

1. **结构** = 官方 Qwen2.5-Coder ChatML 模板 + 在第 13 行插入硬编码指令（工具块之后、
   host system 内容之前）。这是同时满足证据 #3（第 13 行）与 #4（tools 注入）的最小改动。
2. **指令位置**：<|im_start|>system\n → <|tools|>{json}<|/tools|> → 指令 → host
   system 内容 → <|im_end|>。修后指令按 iteration/05 全文替换（不含 "You are a
   coding assistant." 前缀，与 commit message "Replaced with ..." 一致）。
3. **assistant tool_calls 渲染**：\n{"name": "...", "arguments": <arguments | tojson>}，
   与官方模板一致。arguments 在 wire 上是 JSON 字符串，tojson 会双编码成带引号串
   —— 这与官方 Qwen2.5 模板行为完全一致，且 25+ 份 capture 证明模型在这种历史下仍稳定
   输出干净的 {"name", "arguments": {...}} 对象（native_rate=1.0），无需在模板里做
   字符串→dict 的解析（fromjson 不是 Jinja2 内置过滤器，引入反而有风险）。
4. **多工具调用**保留官方 [\n{...}, {...}]\n 数组形态（防御性；harness 实际每轮 1 个）。
5. **add_generation_prompt** 未定义时默认 false（vLLM 总是显式传 True；默认值只用于
   独立渲染测试）。

## 4. 验证结果

用真实 wire capture（eval/wire_captures/stuck_detector_14365__20260825T021926Z.json）
的请求体作为 jinja2 输入渲染：

- **turn 0**（system+user+tools，add_generation_prompt=True）：
  - 输出以 <|im_start|>system 开头、含 <|tools|>/指令/host system、以 <|im_start|>assistant\n 结尾 ✓
  - 渲染 12590 chars → 估算 ≈3313 tokens @3.8c/t vs 实测 3386（差 2.2%）→ tools 注入结构成立 ✓
- **turn 3 全对话**（8 条消息：system/user/assistant×3/tool×3）：
  - 结构断言全过：单 system、user/assistant/tool 各自出现、im_start/im_end 配平、
    末尾 generation prompt ✓
  - assistant 回合渲染为裸 JSON {"name": "edit", "arguments": ...}，tool 回合为
    <|im_start|>tool\n{content}<|im_end|> ✓
- **下游 parser**：python src/vllm_plugin/qwen_coder_tool_parser.py → 8/8 通过 ✓
- test_smoke.py：18 failed 全部是 TestRunBashSandbox/TestReadFileHonestHeader/
  TestToolSafety 的 Windows 沙箱环境问题（PermissionError / WinError 5，子进程与
  文件操作被 DSH 沙箱拒绝），与模板无关（模板不被任何测试引用）。

## 5. 如果要在真实 vLLM 上端到端复验

按 AGENTS.md / PRODUCT.md §8.1 的启动命令起 vLLM，然后跑任意一份 wire capture 场景：

    python -m vllm.entrypoints.openai.api_server \
      --model models/Qwen2.5-Coder-7B-Instruct \
      --enable-auto-tool-choice \
      --tool-call-parser qwen_coder_json \
      --tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py \
      --chat-template models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja \
      --generation-config vllm

重点观察：首轮 prompt_tokens ≈ 3300（不是 ~1000）、模型 content 出现 1-2 行 reasoning、
tool_call_native_rate=1.0。若模板与原件有行为差异，最可能的两个变量是
（a）指令与 <|tools|> 块的先后顺序，（b）指令行尾的换行处理——两者都不影响
parser 与 harness 的架构不变量。
