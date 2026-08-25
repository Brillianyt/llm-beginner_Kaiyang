# TODO

本文件记录已知的技术债、待验证的想法和未来改进方向。格式：一行摘要 + 一句上下文 + （可选）验证方式。

## 已修的 Harness bug（2026-08-24 ~ 08-25）

完整记录（症状 / 根因 / wire 证据 / 修复 / 验证）见
[`iteration/`](./iteration/README.md)。下面是单行摘要 + commit：

- **read_file 头部不诚实**（`90095e4`）— `lines 0..400 of 642` 实际只回 360 行；改成 self-managed `max_result_chars` + 显式 marker `[output truncated at N chars; call read_file again with offset=K]`
- **prompt 太长 + 无 stuck 检测**（`28754d2`）— 4731 → 1462 chars；引入 3 连续相同 `(tool_name, args)` 签名的 stuck 检测 + `DoneReason.STUCK`
- **astropy run_tests ImportError 淹没**（`3b41de2`）— 引入 wheel mirror 模式（`WHEEL_MIRROR_ROOT` + `SKIP_SYNC_FILES` 跳过 `version.py`/`_version.py`/`_dev/scm_version.py`）；edit "old_string not found" 错误加入 near-match + token-match hint；read_file "file not found" 错误列出目录内 `.py` 文件
- **edit discipline 写入 prompt**（`127005d`）— system prompt 加入 "minimum change / never rename / never recursive variant" + "bug-location heuristics: case-sensitivity bugs prefer `re.IGNORECASE`"
- **chat template 抑制 reasoning**（`662ba97` Bug A）— "no prose, no code fences" 改成 "you may output a SHORT reasoning sentence (1-2 lines max) BEFORE the JSON object"；修复后模型 reasoning 2000+ char 出现
- **`_PATCH_FENCE_RE` 漏 `python` 围栏**（`662ba97` Bug B）— regex alternative 从 `(diff|patch)?` 改成 `(diff|patch|python|py)?`；agent 现在能提取 ```python``` 围栏的 fix-as-text
- **`run_tests` 默认 scope 太宽**（`99ffaeb` Bug C）— 引入 `RECENT_EDIT_FILE` 模块全局；agent loop 在 edit/write_file 成功后调 `set_recent_edit`；run_tests 在 `extra_args` 空时自动缩窄到 `<mirror>/<pkg>/<sub>/tests/test_<name>.py`
- **stuck 检测器抓不到 cosmetic edit**（`99ffaeb` Bug D）— 增加 test-summary lock：3 连续相同 `exit_code=N passed=N failed=M errors=K` summary → `done_reason=stuck` + hint "edits are cosmetic, revisit bug location"

## 技术债

- **[已删除 · 列入项目 invariant `AGENTS.md`] 文本模式 tool-call 解析路径**
  - 2026-08-24 A-2 + hard-prohibit 路线落地后,**agent 端彻底不再做工具调用的文本解析**。`_parse_text_tool_calls` / `_JSON_TOOL_RE` / `_dedupe_tool_calls` / `_fallback_apply` 已删除。`text_tool_call_fallback` / `parser_miss_count` / `fallback_*` 健康指标也已删除。
  - 真正的工具调用解析由 vLLM 自定义 parser plugin `qwen_coder_json`(`src/vllm_plugin/qwen_coder_tool_parser.py`,**单路径**—— strip fence + 一个 regex 找 `{name, arguments}` JSON)完成。gate 实测覆盖 7/8 形态(88%);唯一未支持的是 XML-tag-split 形态(`<tool_call><name>X</name><arguments>{...}</arguments></tool_call>`,gate 12%)—— 该形态落到 msg.tool_calls=[]、agent 当文本处理,部署层修复路径是改 chat template / system prompt。
  - 启动命令固化:`--tool-call-parser qwen_coder_json --tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py`(`PRODUCT.md` §8.1)。
  - 静态守卫:`test_smoke.py::test_agent_never_introspects_text_for_tool_calls` 强制 `src/agent.py` 不导入任何文本解析路径 —— 一旦有人想加 fallback,该测试立刻 fail。
  - **别重新引入 fallback**:后续 agent 接手时,如果模型输出解析失败,正确做法是修 `src/vllm_plugin/qwen_coder_tool_parser.py`(单路径,易审计),不是给 agent 主循环加补救路径。AGENTS.md 把这条作为项目 invariant 钉死。

- **[已记录] vLLM 原生 tool_calls 解析完整支持**
  - `Qwen2.5-7B-Instruct`(工具微调版)+ vLLM `--enable-auto-tool-choice --tool-call-parser hermes`,裸 API `finish_reason=tool_calls` + `message.tool_calls` 非空;agent 端到端 toy_repo_patch 11 步 `tests_passed`。
  - **别用 `qwen3_xml` parser**:这个 PPU 定制版的 `Qwen3XMLToolParser` 有 bug,本地单测都无法解析 `<tool_call>` 包裹的 JSON;`hermes` 对 Qwen2.5 格式完美。

- **[已修] harness `arguments` 回传序列化 bug**
  - `LLMClient.chat` 把 vLLM 返回的 `arguments`(字符串) `json.loads` 成 dict 直接存进 message;agent 回传 assistant message 时 vLLM OpenAI 协议校验失败(要求 `arguments` 是字符串)。
  - 已做:加 `_to_wire_tool_calls` 在回传边界统一序列化为 JSON 字符串,内部保持 dict 方便消费(`src/agent.py`)。

- **[已修] `_DONE_MARKER_RE` 把"好的"误判为完成信号**
  - 中文 LLM 最常用的开场白("好的,我来看看...")被中文完成标记正则命中,导致 Coder 模型第一轮就停在 1 步。
  - 已做:从正则中移除 `好的`,保留"已修复/已完成/搞定"等真正的完成信号(`src/agent.py`)。

## 待验证

- ~~**SGLang 换 Qwen function-calling template 能否消除 fallback**~~(已转 vLLM + 自定义模板路径解决,见上)

## 未来改进（低优先级）

- 多 agent 并行时 `CODING_AGENT_AUDIT_LOG` 默认路径会争抢，建议 per-instance 显式传路径（已在 PRODUCT.md FAQ 提及）。
- trace 完全 1.0 replay 需要自带时间戳 monkeypatch（状态链 diff 是正确行为，非 bug）。
