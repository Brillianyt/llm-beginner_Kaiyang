# `iteration/` — Harness 调试全程记录

本目录按**时间先后**完整记录 2026-08-24 ~ 2026-08-25 这一轮 harness 调试：每个 bug 的**症状**、**根因**、**如何在 client↔vLLM 双向消息里发现**、**修复代码**、**验证证据**。最终目标：通过至少一个 SWE-bench Lite 实例。

## 阅读顺序

| # | 文件 | 对应 commit | 主题 |
|---|---|---|---|
| 1 | [`00-chronological-summary.md`](./00-chronological-summary.md) | — | 一页纸总览：时间线 + 最终状态 |
| 2 | [`01-read-file-honest-header.md`](./01-read-file-honest-header.md) | `90095e4` | read_file 头部"自吹自擂"（声称返回 0..400 行实际被悄悄截断） |
| 3 | [`02-prompt-shorter-and-stuck-detector.md`](./02-prompt-shorter-and-stuck-detector.md) | `28754d2` | 系统提示词 4731 → 1462 chars + 第一版签名 stuck 检测器 |
| 4 | [`03-wheel-mirror-and-tool-errors.md`](./03-wheel-mirror-and-tool-errors.md) | `3b41de2` | wheel mirror 让 astropy 可运行；edit/read 错误信息增强 |
| 5 | [`04-edit-discipline-prompt.md`](./04-edit-discipline-prompt.md) | `127005d` | "minimum change / don't rename" 等编辑纪律写入 system prompt |
| 6 | [`05-chat-template-and-fences.md`](./05-chat-template-and-fences.md) | `662ba97` | **Bug A**：chat template 抑制 reasoning；**Bug B**：`_PATCH_FENCE_RE` 漏 `python` 围栏 |
| 7 | [`06-recent-edit-scope-and-summary-lock.md`](./06-recent-edit-scope-and-summary-lock.md) | `99ffaeb` | **Bug C**：`run_tests` 默认范围淹没模型；**Bug D**：stuck 检测器抓不到"修改文案而非 bug" |
| 8 | [`07-final-state-and-swe-pass.md`](./07-final-state-and-swe-pass.md) | — | 最终 harness 状态：2/3 SWE PASS（astropy-12907 + 14365） |
| 9 | [`08-run-bash-and-allowed-tools.md`](./08-run-bash-and-allowed-tools.md) | `f7e9a1b` | `run_bash` 沙箱 + skill `allowed-tools` 强制执行 + `safe_resolve` 预存 bug 修复 |
| 10 | [`wire-evidence-index.md`](./wire-evidence-index.md) | — | 全部 wire capture 索引（26 份 HTTP body） |

## 调试方法学

整个会话用一句话总结：

> **每个 bug 都从 client↔vLLM 双向消息序列里抓到。** 让 vLLM 每次响应落入磁盘上的 wire capture（`eval/wire_captures/<instance>__<label>__<ISO8601>.json`），然后用 Python 脚本读取每条请求/响应，统计 `message.content` 是否空、`message.tool_calls` 是否非空、tool_call_id 链路是否对齐、assistant 的 args 与 tool 的 response 是否匹配。一旦某条数字异常就定位原因。

## 调过的 4 类 wire 异常信号

1. **`message.content` 始终为空** —— chat template 抑制 reasoning（Bug A）
2. **`tool_calls` 在第 N 轮后突然消失，content 突现大段散文** —— patch 提取正则漏形态（Bug B）
3. **`run_tests` 响应里出现大量 `pytest_remotedata ImportError`、与编辑的文件无关** —— 默认 scope 太宽（Bug C）
4. **连续多轮 `run_tests` summary 完全相同（passed=N failed=M errors=K）** —— 模型在改文案而非修 bug（Bug D）

## 6 个 commit 的因果链

```
bed72b2 → 90095e4 → 28754d2 → b6f42a6 → 3b41de2 → 127005d → 662ba97 → 99ffaeb
                                          │
                                          └─ astropy 0/3 PASS（harness 干净证据）
                                                                       │
                                                                       └─ 2/3 PASS ✓
```

每一 commit 对应一个或多个 wire 异常信号；每一 commit 都通过新的 wire capture 验证修复有效；最终全部 25 份 capture 都在 `wire_captures/`（链接）。