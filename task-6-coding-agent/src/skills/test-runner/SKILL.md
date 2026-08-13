---
name: test-runner
description: 当需要运行测试、解读 pytest 失败、或修复测试相关错误时加载。涵盖常见 pytest 错误诊断与修复。
when_to_use: 用户要求"跑测试 / pytest 失败 / 解析 test 输出 / 测试挂了"
allowed-tools:
  - read_file
  - run_tests
  - write_file
  - git_diff
---

# Test Runner 工作流

## 何时使用
- 修复完 bug 想验证
- pytest 报失败，要解读输出
- CI 挂了，要本地复现

## 工作流

### Step 1：跑全量测试
- 调用 `run_tests`（默认 `python -m pytest -q`，timeout 60s）
- 如果是 SWE-bench 这种重任务，`extra_args=["-x"]` 先停第一个失败

### Step 2：解读 pytest 输出
常见模式：
| 模式 | 含义 |
|---|---|
| `AssertionError` | 期望值不匹配 — 看 `assert` 那行 |
| `ImportError` | 模块没装 / 路径错 / 循环 import |
| `IndentationError` | 缩进错（混了 tab 和空格） |
| `ModuleNotFoundError` | 包没装 |
| `TimeoutExpired` | 测试卡住；可能无限循环 |
| `E   ... expected X but got Y` | 输出对比失败 |

### Step 3：常见修复
- `AssertionError on expected == X`: 用 `read_file` 看实际代码；多半是边界 case 没覆盖
- `ImportError`: 检查 `sys.path` / 是否漏装依赖
- 多个 test 一起挂：通常是底层函数坏了；先修底层
- 单个 test 挂：可能是测试本身写错；先看测试代码（但**不要改 tests/**）

### Step 4：再跑一次
- 修复后**必须**再调一次 `run_tests` 验证
- 如果 timeout 调到 60 还不够，说明有性能问题或死循环

## 注意事项
- 严格不修改 `test_*.py` 文件（任务约定）
- 如果测试预期本身就错，先跑通了再和用户讨论
- 失败的 traceback 要截取关键栈帧；不要全部塞给 LLM（token 浪费）
