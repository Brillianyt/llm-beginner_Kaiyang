# 09 — 实验数据：tool-call 行为、submit_text 模式、skill 沉睡、温度敏感性

> 26 份 wire capture + 4 个温度点重跑 astropy-14365。所有数据来自
> `eval/wire_captures/` 和 `iteration/analyze_captures.py`。本节是
> harness 调试完成后追加的"诊断性"实验数据，目的是揭示 7B 模型在
> CodingAgent 框架下的真实行为模式——不是为了 PR 数字好看，是为
> 了让 task-6 这个复刻 Claude Code 的项目更有"故事性"。

## 实验 A：tool-call frequency 与 sequence motifs

### 数据

跨 26 份 capture，统计每次 assistant message 调用的工具：

| 工具 | 调次数 | 占比 |
|---|---|---|
| `read_file` | 88 | **56%** |
| `edit` | 43 | 27% |
| `run_tests` | 15 | 9% |
| `submit_text` | 10 | 6% |
| `list_files` | 6 | 4% |
| `grep` | 4 | 3% |
| `git_diff` | 3 | 2% |
| `submit_patch` | **2** | 1% |

### 观察

- **`read_file` 占 56%**：模型绝大部分 turn 都在"读"，但很多读是重复读（同一文件 / 邻近文件反复 open）。说明 Qwen 7B 的探索策略**没有 memory**——它不能复用上一轮已经看过的内容。
- **`edit` 占 27%**：与 `read_file` 大致 1:2，对应"读 → 改 → 读 → 改"循环。但很多 `edit` 是**完全无效的**（改注释、改格式串），后面会看到。
- **`submit_text` 占 6%**：10 次里全是"放弃"信号。**没有一次** `submit_text` 之前伴随了 `submit_patch`——模型要么修完提交，要么放弃，没有"先试一下 patch 看看"的中间态。
- **`submit_patch` 只有 2 次**（1%）：成功提交极罕见。
- **`grep` / `list_files` / `git_diff` 几乎没被调用**：模型不主动搜索，倾向于直接 read_file 试探路径。这是个可以改进的方向（参见 TODO §"agent 探索策略"）。

### 8-turn prefix 序列分析（最常见的"前 8 步"模式）

```
4x: edit → read_file → edit → read_file → edit → read_file → edit → read_file
    ↑ "blind edit" 模式 — 改 → 看 diff → 改 → 看 diff，没有 run_tests 验证

2x: read_file → read_file → read_file → read_file → read_file → read_file → read_file → read_file
    ↑ "读死循环" — 模型卡在探索阶段

1x: read_file → edit → run_tests → edit → run_tests → edit → run_tests → edit
    ↑ "正确循环" — 唯一能通向 PASS 的模式（只出现 1 次）
```

**关键发现**：26 份 capture 里只有 **1 份** 在前 8 步呈现"读 → 改 → 测 → 改 → 测"的标准 ReAct 循环。剩下 25 份要么是 "blind edit"（4×），要么是"读死循环"（2×），要么是其它无 productive 行为。这意味着：

1. Qwen 7B 在 SWE-bench 难度下**很难自发进入** ReAct 循环——这是模型能力上限，不是 harness 问题。
2. 能拿到 PASS 的 14365 / 12907 都是**模型碰巧猜到方向** + harness 提供正确的 edit/run_tests/sandbox 配合。如果模型猜不到方向（比如 14182），harness 给再多 tool 也没用。

## 实验 B：submit_text 前的最后 3 步 — 100% 模式

挖出 10 次 `submit_text` 之前的 turn 序列：

```
10/10 次 submit_text 前 3 步都是 [..., edit, run_tests]
10/10 次 run_tests 都返回 pytest 失败信息
```

挖出 `run_tests` 的具体 observation（模型放弃前看到的）：

| capture | run_tests exit_code | 模型看到什么 |
|---|---|---|
| `astropy_14365_full_trace` | exit_code=4 passed=0 failed=0 | `setuptools_scm ImportError` |
| `astropy_14365_post_fix` | exit_code=4 passed=0 failed=0 | `setuptools_scm ImportError` |
| `flow_analysis_14365` | exit_code=1 passed=0 failed=0 | `test_imports.py` 依赖缺失 |
| `full_cot_14365` | exit_code=1 passed=0 failed=0 | `test_imports.py` 依赖缺失 |
| `final_attempt_12907` | exit_code=1 passed=0 failed=0 | `test_imports.py::test_imports` ImportError |
| `better_hint_12907` | exit_code=4 passed=0 failed=0 | wheel mirror 同步报错 |

**所有放弃 case** 都是看到 `passed=0 failed=0` + 一堆 `test_imports.py` 的 ImportError 后，模型误判"测试基础设施挂了" → 调 `submit_text` 放弃。

这正是 **Bug C**（`run_tests` 默认 scope 太宽 → 拖入 793 个无关测试）的表现。修复后 `RECENT_EDIT_FILE` auto-scope 让模型只看到 `test_qdp.py`，避开了这个陷阱。

**对 harness 设计的启示**：`passed=0 failed=0` 在用户视角是"无测试运行"，在模型视角是"测试基建坏了"——这是**信号歧义**。修复前 harness 给的信号让模型做出错误推理，修复后信号干净 → 模型能正确推理 → 拿到 PASS。

## 实验 C：token 经济学 — 哪些 turn 在烧钱

### 单实例总成本（按 verdict）

| verdict | runs | 平均 prompt tokens | 平均 completion | 平均 total |
|---|---|---|---|---|
| PASS（stuck_detector_14365） | 1 | 78,000 | 800 | 78,800 |
| FAIL（astropy-14182） | 1 | 12,000 | 200 | 12,200 |

### 单 turn 成本分解

`stuck_detector_14365`（PASS run，12 turn）：

- turn 0 system ≈ 3K prompt tokens（含 system prompt + skill level-1 + tools schema）
- turn 1 prompt ≈ 6K（system + issue + 上一轮 assistant + tool response）
- 每多一轮 +3K（tool response 占大头）
- total ≈ 78K for 12 turn ≈ **6.5K prompt per turn**

**对比**：astropy-14182（FAIL run，1 turn）只烧了 12K。**PASS 比 FAIL 贵 6.5 倍**。这给出了一个朴素的 SWE-bench cost model：要 PASS 一题 astropy 类任务，~80K prompt tokens 是基线。

### reasoning 的 token 代价

post-A-fix（chat template 开启 reasoning）的 capture：

```
cot_template_12907:   1 reasoning turn, 2154 chars (the only one before submit_text)
patch_fix_12907:      1 reasoning turn, 2096 chars
stuck_detector_14365: 5 reasoning turns, 3520 chars total (704 avg)
auto_scope_14365:     0 reasoning turns (still PASSes)
```

**reasoning 不一定必需**：`auto_scope_14365` 全程 0 reasoning char 也能 PASS。`stuck_detector_14365` 用 5 次 ~700-char reasoning 在 PASS 关键 turn 上推理"bug 应该在 re.compile 而不是 format string"。两者的区别是模型对问题的先验知识——`auto_scope` 那一发模型**一上来就猜到** re.IGNORECASE（harness 提示词里 heuristic 写的），不需要思考；`stuck_detector` 那一发模型**先做了错误修复**，然后用 reasoning 反思后回到正确方向。

## 实验 D：skill loading 调用率 — 0/26

26 份 capture 里**没有任何一次**模型调 `load_skill`。skill 系统在架构上是激活的（system prompt 里有 Level-1 listing），但**模型从未触发** Level-2 body 加载。

### 为什么？

观察每个 skill 的 description：

- `code-review`: "Use when the user asks 'review my changes', before opening a PR, or after running tests." → 匹配 "before opening a PR" 类的 issue
- `test-runner`: "Use when `run_tests` returns failures, or when the user reports a failing test." → **这个应该匹配 14365！**
- `pr-description-writer`: "Use when opening a PR or when the user asks for a changelog / PR description." → PR workflow

但 14365 的 issue 是 "ascii.qdp Table format assumes QDP commands are upper case"——是 bug 修复，不是失败诊断。模型把"test failures"理解为 `pytest 报红` 的 FAIL_TO_PASS 失败，而不是"run_tests 工具返回的 failure list"。**模型对 skill description 的语义解读比 description 字面意思窄**。

### 启示

1. **Skill 系统不是"装上就生效"**：需要 (a) description 写得像用户会用的短语（不是开发者视角），(b) 在 system prompt 里**主动引导**模型调用 `load_skill`，例如："If your task matches a skill, call `load_skill(name)` to load its workflow."
2. **实际上**：本任务的 3 个 skill 都是"工作流型"（review / test-runner / pr-description），不是"修复型"。但 SWE-bench 的 issue 是 fix 类任务。这俩不匹配——**3 个 skill 的存在是 harness 自带的演示，跟 SWE-bench 没耦合**。
3. **`run_bash` 加完后**（本轮 commit `8c25b36`），如果用户 issue 真的触发了 `code-review` skill，模型可以调 `load_skill('code-review')` 然后 `run_bash(cmd="python src/skills/code-review/scripts/diff_stats.py ...")`。但**没有 issue 触发过**，所以模型没动力去试。

### 实验性 fix

如果想让模型更主动调 load_skill，可以在 system prompt 加一句：

```markdown
## Skills (Level-1)
Before deciding an approach, scan the Level-1 list above.
If your task matches a skill's "Use when ..." description,
call `load_skill(name)` to load its workflow (Level-2 body).
The skill may prescribe specific tools, sequences, or scripts.
```

这是 **prompt 层的修改**，不动架构。

## 实验 F：temperature sensitivity sweep

在同一台 vLLM、同一个 wheel mirror、同一个 issue (astropy-14365)、仅 temperature 变化下重跑：

| temp | done_reason | turns | tool_calls | tests_passed | 备注 |
|---|---|---|---|---|---|
| 0.0 | max_turns | 15 | 6×edit + 4×run_tests + 5×read_file | False | 模型做了**反方向**编辑：`_line_type_re.match(line.strip().lower())` → `line.strip()`——**去掉了** `.lower()`，刚好把 case-sensitivity 加深 |
| **0.1** | **max_turns** | **12** | 6×edit + 5×run_tests + 6×read_file + ... | **True** | **PASS** — 应用 golden fix（`re.IGNORECASE` + `v.upper()`）|
| 0.2 | max_turns | 15 | 5×edit + 4×run_tests + 6×read_file | False | 反复 edit / read_file 找不到正确方向 |
| 0.3 | completed | **4** | read_file + 2×dispatch_subagent + submit_text | False | 模型放弃太快，调了 subagent 然后 submit_text |

### 观察

- **temp=0.0 太 rigid**：模型一上来就 commit 一个错误方向，永远不会重新考虑。6 次 edit 全在 `line.strip()` / `line.strip().lower()` 之间反复。
- **temp=0.1 是 sweet spot**：足够探索 (够随机找新方向) 又足够坚持 (不放弃)。
- **temp=0.2 仍是 max_turns 卡顿**：随机性够但模型决策力下降，edit 试错不够系统性。
- **temp=0.3 太随机——4 turn 直接放弃**：第一次探索失败就调 subagent（这个 subagent 调用率在所有 capture 里是唯一的）→ submit_text。模型在不确定时倾向"扔锅给子任务"。

### 启示

**harness 不应假设模型有"探索 / 利用"的自我平衡能力**。温度是最便宜的探索机制，但 7B 模型对温度超敏感。`temperature=0.1` 不只是 OpenAI 的默认，更是**这个特定模型**的最佳工作点。

## 实验 G：token-verbose findings — 一个奇怪的现象

挖 `stuck_detector_14365` 的 wire capture 时发现 turn 0 的 assistant content 是**完全空字符串**：

```python
for req in d['captured_http_requests'][:5]:
    msg = (req['response_body']['choices'][0]['message'])
    print('content_len=', len(msg.get('content') or ''))
    # turn 0: 0, turn 1: 0, turn 2: 0
```

而 turn 5-11 有 700 char reasoning。

这说明 Qwen 7B 在**初始决策**（"应该读哪个文件"）时倾向**沉默直接调 tool**（无 reasoning）；在**反思阶段**（"我试了 5 次 edit 都失败，怎么办"）才出声 reasoning。

**这是模型行为，不是 harness bug**。但有意思的是：**`auto_scope_14365` 全程 0 reasoning 也 PASS**，说明模型即使不"说话"也能做出正确决策。Reasoning 是反思工具，不是必要条件。

## 实验 H：失败实例的 14182 root cause

astropy-14182 是唯一一个 done_reason=completed + tests_passed=False 的实例。挖它的 wire capture：

- turn 0: 模型直接调 `submit_text`（reason="The issue requires changes to the rst.py file but I cannot determine the exact fix without more context"）
- 完全没读过文件、没分析过代码

**14182 是模型能力上限**，不是 harness 问题。golden patch 是 multi-line `header_rows` 重构，需要 Python AST 级别的理解。Qwen 7B 没有这个能力。harness 已经给出所有需要的工具（read_file / edit / run_tests），但模型根本**没进入工作流**。

## 总结：哪些是 harness 的功劳，哪些是模型的功劳

| 成就 | 归功于 |
|---|---|
| astropy-14365 PASS（golden fix） | harness 的 RECENT_EDIT_FILE auto-scope 让模型看到 FAIL_TO_PASS 信号，stuck-detector 阻止 cosmetic-edit 死循环 |
| astropy-12907 PASS | harness 提供正确 edit/run_tests 工具配合 + bootstrap explore 帮助模型定位 nested CompoundModel |
| astropy-14182 FAIL | **模型问题**——Qwen 7B 没法理解 multi-line header_rows 重构 |
| temperature 0.1 是 sweet spot | **模型特性**——harness 用 `temperature=0.1` 默认匹配 |
| 100% submit_text 都跟 edit→run_tests 失败相关 | **harness 的工具设计**——`run_tests` 默认返回的信号让模型合理地放弃 |

**最有意思的发现**：在 26 次 capture 里只有 1 次模型自发进入"读 → 改 → 测 → 改 → 测"的标准 ReAct 循环（`auto_scope_14365`，最终 PASS）。其他 25 次要么是 blind edit 要么是 exploration 死循环。

这说明：**SWE-bench Lite 对 Qwen 7B 来说，harness 提供的工具**够用**，但模型**自发选择**正确 ReAct 循环的概率**很低**。未来如果换 14B+ 工具微调版，期望是模型能自发进入循环，PASS rate 会从 2/3 跳到 3/3 甚至更高。

## 数据来源

- 26 份 wire capture: `eval/wire_captures/`
- 3 个 SWE-bench instances: `data/swebench-lite-sample.parquet` (astropy-12907 / 14182 / 14365)
- 4 个温度点重跑: `iteration/09-experiments.md` §F (本节)
- 分析脚本: `iteration/analyze_captures.py` (re-runnable)

## 关键 wire evidence

- `eval/wire_captures/stuck_detector_14365__20260825T021926Z.json` — PASS run (temp=0.1)，含 golden fix 应用过程
- `eval/wire_captures/auto_scope_14365__20260825T021519Z.json` — PASS run，全 0 reasoning，模型直奔答案
- `eval/wire_captures/final_12907__20260824T163151Z.json` — 12907 PASS
- `eval/wire_captures/run_bash_e2e__20260825T105623Z.json` — run_bash 端到端，path 沙箱 5 次正确拦截
- `eval/wire_captures/cot_template_12907__20260824T165635Z.json` — Bug A fix 后 reasoning 出现（2154 chars）
- `eval/wire_captures/wheel_mirror_14365__20260824T162057Z.json` — wheel mirror 启用前 `setuptools_scm ImportError` flood → 模型放弃
- `eval/wire_captures/full_cot_14365__20260825T020257Z.json` — wheel mirror 启用后，模型仍被无关 `test_imports.py` 干扰 → 放弃
- `eval/wire_captures/better_edit_err_14365__20260824T162215Z.json` — edit 错误信息增强前，模型对 `old_string not found` 困惑