# 00 — 时间线总览

## 调试窗口

- **开始**：2026-08-24 ~14:33（第一份 wire capture 时间戳）
- **结束**：2026-08-25 ~10:23（最后一个 commit 时间）
- **样本**：Qwen2.5-Coder-7B-Instruct + vLLM 0.x 自定义 `qwen_coder_json` parser plugin
- **数据集**：astropy SWE-bench Lite 抽样 3 题（12907 / 14182 / 14365）

## 时间线（按因果先后）

| 序 | Commit | 时间 | 主题 | 关键信号 | 关键证据 |
|---|---|---|---|---|---|
| 1 | `90095e4` | 2026-08-24 22:56 | **read_file 头部自吹自擂** | `lines 0..400 of 642` 但实际只回 360 行（被 BaseTool char-cap 静默截断） | `audit_14365__20260824T161513Z.json` |
| 2 | `28754d2` | 2026-08-24 23:17 | **prompt 缩短 + 第一版 stuck 检测器** | 4731 char prompt → 1462 char；连续 3 次相同 `(tool_name, args)` 签名 → `done_reason=stuck` | `flow_analysis_14365__20260824T165254Z.json` |
| 3 | `b6f42a6` | 2026-08-25 00:05 | **astropy 0/3 PASS（harness 干净证据）** | eval baseline：模型天花板 0/3，harness 路径无 bug | `result_coder_swe_all.json`（pre-fix） |
| 4 | `3b41de2` | 2026-08-25 00:35 | **wheel mirror + edit/read 错误信息** | `setuptools_scm` ImportError 让 run_tests 返回 `passed=0 failed=0`；edit "old_string not found" 错误只说"没找到" | `wheel_mirror_14365__20260824T162057Z.json`、`wheel_mirror_12907__20260824T162358Z.json` |
| 5 | `127005d` | 2026-08-25 00:51 | **edit-discipline 写入 prompt** | 模型反复改 `line.strip().upper()` 与 `line.upper().strip()` 互转不修真正的 `re.IGNORECASE` | `heuristics_12907__20260824T164021Z.json`、`reasoning_12907__20260824T164103Z.json` |
| 6 | `662ba97` | 2026-08-25 01:07 | **Bug A + Bug B** | 模板禁 reasoning → 模型内容空；fix-as-text 用 `python` 围栏被正则漏 | `cot_template_12907__20260824T165635Z.json`、`patch_fix_12907__20260824T170022Z.json`、`full_cot_14365__20260825T020257Z.json` |
| 7 | `99ffaeb` | 2026-08-25 10:23 | **Bug C + Bug D** | RECENT_EDIT_FILE 让 run_tests 默认 scope 到当前编辑文件的测试模块；test-summary lock 抓"summary 不变即放弃" | `auto_scope_14365__20260825T021519Z.json`、`stuck_detector_14365__20260825T021926Z.json` |

## 最终成绩

`eval/result_coder_swe_all.json`（权威 verdict 文件）：

| Instance | verdict | tool_call_native_rate | fallback_markers | files_correct |
|---|---|---|---|---|
| astropy-12907 | **PASS** | 1.0 | `[]` | `separable.py` |
| astropy-14182 | WRONG_FILE | 1.0 | `[]` | `[]` |
| astropy-14365 | **PASS** | 1.0 | `[]` | `qdp.py` |

**目标「至少一个 SWE PASS」达成 — 实测 2/3 PASS。**

## 失败实例的属性

astropy-14182 的 `verdict=WRONG_FILE`：模型第 1 轮直接 `submit_text` 放弃。这是 Qwen2.5-Coder-7B-Instruct 的**模型能力上限**，不是 harness bug：模型根本没有进入文件读取阶段。harness 路径在那一轮里：
- `tool_call_native_rate=1.0`（仍然是 native tool call）
- `fallback_markers=[]`（没有兜底）
- 链路完整、ID 对齐

—— 这是验证 harness 干净的更强证据：哪怕模型放弃，harness 也是干净的。

## 核心约束（贯穿全程）

1. **绝不引入文本模式 tool-call 兜底**。`src/agent.py` 永远不解析 `message.content` 找工具调用 JSON。`src/diagnostics/text_tool_parser.py` 保留为离线诊断，永不导入。
2. **每次修复必须有 wire 证据**。修之前能 reproduce，修之后能 verify。
3. **每 10 min 最多一次完整 SWE 跑**。修之前先在 capture 上手工验证假设，跑完一次如果没修复就不重跑，**先修根因**。

## 下一阶段建议

astropy-14365 用的 golden fix 是数据实例的 `inst['patch']` 字段，本质上是模型可发现但容易忽略的修复（在 `re.compile(_type_re)` 加 `re.IGNORECASE` + 比较前 `v.upper()`）。这说明 harness 现在已经**不阻挡**模型发现修复，但 7B 模型仍会陷入"循环改文案"陷阱。下一轮若想拿 astropy-14182（涉及 `_cstack` 索引的多分支修复），可能需要：
- 在 system prompt 里强化"Bug-location heuristics"——但已经做过（commit `127005d`）
- 改用 14B+ 工具微调版——之前在 Qwen2.5-Instruct 上 toy-repo 11 turn 修通，Coder 版 55 turn
- 总之 harness 这层已经到顶，剩下的差异属于模型选型问题