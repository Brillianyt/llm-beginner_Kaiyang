# 10 — Toy-repo Skill 端到端：3 skill × 2 触发模式

## 目标

在 toy-repo（`data/toy-repo`）上端到端验证 **3 个 skill 都能被调用**，且调用后模型确实使用了 skill 规定的工具链。

## 实验设计

两个触发模式 × 3 个 skill = 6 次跑：

### 模式 1：隐式触发（issue 字面匹配 skill "Use when ..."）

| Skill | Issue |
|---|---|
| test-runner | "My pytest test suite is failing. Please diagnose and fix the bug." |
| code-review | "I just edited calculator.py. Please review my changes." |
| pr-description-writer | "I just fixed a bug. Please write a PR description summarizing the fix." |

### 模式 2：显式触发（issue 直接说"先用 load_skill"）

| Skill | Issue |
|---|---|
| test-runner | "Use the test-runner skill to diagnose my failing tests, then fix the calculator.add bug. First call `load_skill('test-runner')`." |
| code-review | "Use the code-review skill to review calculator.py. First call `load_skill('code-review')`, then run diff_stats.py via `run_bash`." |
| pr-description-writer | "Use the pr-description-writer skill to write a PR description. First call `load_skill('pr-description-writer')`." |

每个 scenario 跑前都 `cp calculator.py.orig calculator.py` 把 toy-repo 重置到 broken 状态（`return a - b`）。`max_turns=10`。`bootstrap_explore=False`。

## 结果

### 隐式触发（3/3 失败 — 但 toy-repo 仍然修通）

| Skill | load_skill | done_reason | turns | tests_passed | tool 序列 |
|---|---|---|---|---|---|
| test-runner | **0** | tests_passed | 8 | ✅ | run_tests → read_file → edit → run_tests |
| code-review | **0** | tests_passed | 2 | ❌ | submit_patch（直接交，没 review） |
| pr-description-writer | **0** | completed | 4 | ❌ | read_file → list_files → submit_text |

**3/3 模型都没调 load_skill**——印证 `iteration/09-experiments.md §D` 的发现：Qwen 7B 自发调 load_skill 的概率是 0。模型有强烈的 "fix the bug" 先验，**忽略**了 description 里写的 "Use when ..."。

但**有意思的副作用**：test-runner 隐式触发**反而 PASS 了**——因为模型无视 skill 直接用了"测 → 改 → 测"的硬编码 ReAct 循环。这是 prompt 里写好的 "Workflow: explore → read → edit → test → submit_patch" 在起作用。

### 显式触发（3/3 成功 — 真的触发了 skill 规定的工具链）

| Skill | load_skill | done_reason | turns | tests_passed | tool 序列 |
|---|---|---|---|---|---|
| **test-runner** | **1** | max_turns | 10 | ❌ | `load_skill → dispatch_subagent → run_tests → read_file → edit → read_file → edit → ...` |
| **code-review** | **1** | stuck | 5 | ❌ | `load_skill → run_bash × 4` |
| **pr-description-writer** | **1** | completed | 2 | ❌ | `load_skill → submit_text` |

**3/3 都成功 load_skill**，而且每个 skill **触发后调用了 skill 规定的工具**：

#### test-runner: 触发了 `dispatch_subagent`

```
turn 0: load_skill(name='test-runner')
turn 1: dispatch_subagent(name='test_executor', task='diagnose and fix the failing tests')
turn 2: run_tests
turn 4-11: read_file / edit 循环
```

模型把"fix 循环"翻译成"派给 test_executor subagent"——这是它自发结合 Skill + Subagent 两层能力栈的行为。**subagent 在这里**第一次被实际调用**（在所有 capture 里首次出现）**。

#### code-review: 触发了 `run_bash` × 4

```
turn 1: run_bash(cmd='python /mnt/.../src/skills/code-review/scripts/diff_stats.py /mnt/.../data/toy-repo/.git/index')
        ↑ 绝对路径 → 被 safe_resolve 拒绝
turn 2: run_bash(cmd='python src/skills/code-review/scripts/diff_stats.py .git/index')
        ↑ 相对路径 OK，但 .git/index 不是 diff 文件
turn 3-4: 重复 turn 2 命令 → stuck detector (3 同 signature lock) 触发
```

完整展示了 **skill → run_bash → 沙箱 → stuck detector** 的链条：
- `allowed-tools: [run_bash, read_file, grep]` 强制执行 ✅ — run_bash 被允许，edit/write_file 被拒绝
- `safe_resolve` 在 absolute path 时拒绝 ✅
- stuck detector 在 repeated signature 时触发 ✅

#### pr-description-writer: 触发了 `submit_text`

```
turn 0: load_skill(name='pr-description-writer')
turn 1: submit_text
```

模型读完 skill body 后直接 generate PR description 文本 → 调 submit_text（skill 内部说"不要 claim 不存在的测试 / 不要 invent commits"），没有 edit 行为，符合 skill 语义。

## 关键发现

### 1. **Skill 链端到端可用**

skill body 内容到达模型后，模型确实按 skill 的 workflow 行动：
- test-runner → dispatch_subagent（间接 skill workflow）
- code-review → run_bash 调用 scripts/diff_stats.py（直接 skill workflow）
- pr-description-writer → 只调 submit_text 不改文件（符合 skill "never claim tests that don't exist" 规则）

**smoke test `TestCodeReviewSkillEndToEnd` 之前已经验证了 dispatch 链路**，这次是**模型真正调起来**——更接近生产场景。

### 2. **隐式 vs 显式触发**

| 触发 | 隐式 | 显式 |
|---|---|---|
| load_skill call rate | 0/3 | **3/3** |
| 模型有 bug-fix 先验 | ✅ 强烈 | 仍存在但被 issue 引导覆盖 |
| Skill 规定工具被调 | 否 | ✅ |

`code-review` 这个 skill 特别有意思：模型**两次**用上了skill 描述的工具（run_bash），第一次被沙箱挡（绝对路径），第二次用相对路径（成功），第三次重复第二次触发 stuck detector。**整个 5 轮的交互就是沙箱 + stuck detector 的端到端演练**。

### 3. **test-runner 第一次触发 subagent**

之前 26 次 capture 都没见过 `dispatch_subagent` tool call。这次是**第一次**。说明：

- 模型有 dispatch_subagent 的 schema（system prompt 里写了），但**之前没有 issue 触发它**
- test-runner skill 的"循环"语义暗示模型可以派子任务 → 模型真的派了
- 这是 **skill + subagent 两层栈联合工作**的第一个证据

### 4. **prompt 加了"scan Level-1 first"指令 — 0/3 没救**

本 commit 同时改了 `src/prompt.py`，加了一段：

```
**Before deciding an approach, scan the Level-1 list above.** 
If a skill's "Use when ..." description matches your task, 
call `load_skill(name)` to load its workflow (Level-2 body). 
The skill may prescribe specific tools, sequences, or scripts. 
After loading, follow its steps; do not skip directly to read/edit.
```

但隐式触发的 3 次跑**仍然 0 次**调 load_skill。说明：
- **prompt 提示不够强**——Qwen 7B 的 "fix the bug" 先验压过了 skill-list 提示
- **需要的是 issue 措辞里直接出现 skill 描述关键词**，或者更激进的方式（system prompt 里给 skill 优先级 > bug fix 先验）
- **架构层无解**——3 个 skill 的存在对 toy-repo 修复是冗余信息

## 启示：skill 系统要真正工作，需要什么？

1. **Issue 触发词和 skill description 强匹配**——本轮 6 次跑只有显式触发能让 skill 加载，证实了这一点。
2. **System prompt 把 skill 优先级提得比 "Workflow: explore → read → edit → test" 还高**——目前 Workflow 在 system prompt 里是独立段，Level-1 在另一段。位置/优先级可能影响。
3. **或者在 Workflow 里嵌入："If a Level-1 skill matches, load_skill(name) BEFORE explore"**——这是最直接的指令。
4. **或者测试时让 issue 字面触发**（现实里用户可能就是这么用的："review my code" 直接是 code-review skill 的 trigger phrase）。

## 文件

新增 6 份 wire capture（3 implicit + 3 explicit）：

```
eval/wire_captures/skill_trigger_test-runner_*.json
eval/wire_captures/skill_trigger_code-review_*.json
eval/wire_captures/skill_trigger_pr-description_*.json
```

软链同步到 `iteration/wire_captures/`。

## Commit

```
<pending>  feat: toy-repo skill e2e + system prompt scan-Level-1 nudge
```

修改：
- `src/prompt.py` — 加 "Before deciding an approach, scan the Level-1 list above" 段（**已被实验证伪**——prompt cue 不足以让模型自发调 skill）
- `iteration/10-toyrepo-skill-e2e.md`（本文件）
- 6 份新 wire capture

## 关键 wire evidence

| Capture | 关键观察 |
|---|---|
| `skill_trigger_test-runner_20260825T111945Z.json` | 隐式触发：tests_passed=True 但 0 次 load_skill（模型用 prompt 的 "explore → read → edit → test" workflow 而非 skill workflow） |
| `skill_trigger_code-review_20260825T111947Z.json` | 隐式触发：2 turn 直接 submit_patch，0 次 load_skill |
| `skill_trigger_pr-description_20260825T111949Z.json` | 隐式触发：4 turn 调 list_files 后 submit_text，0 次 load_skill |
| `skill_trigger_test-runner-explicit_20260825T112129Z.json` | **显式触发**：load_skill → dispatch_subagent → run_tests → edit 循环（**subagent 第一次被实际调用**） |
| `skill_trigger_code-review-explicit_20260825T112131Z.json` | **显式触发**：load_skill → run_bash × 4（**run_bash + 沙箱 + stuck detector 端到端**） |
| `skill_trigger_pr-description-explicit_20260825T112133Z.json` | **显式触发**：load_skill → submit_text（仅生成文本不修改代码，符合 skill 语义） |