# 14 — grep / list_files prompt promotion + Claude Code 参考

> 2026-08-26（本会话）。三个迭代：
  - 14a: 系统 prompt 把 grep 推上去（不动 schema）
  - 14b: 升级 grep tool 到 Claude Code parity（schema + 参数 + 应用限位）
  - 14c: 新增 "Locate-before-test" 段 + 改 workflow 为 `locate → read → edit`

## 1. 改动总览

| 文件 | 改动 | 来源 |
|---|---|---|
| `src/prompt.py:_TERSE_TOOL_DESC["grep"]` | 11 → 165 字（Claude Code 风格：ALWAYS、NEVER、反斜杠转义、3 mode 说明、-i 用途、head_limit） | 14a |
| `src/prompt.py:_TERSE_TOOL_DESC["list_files"]` | 40 → 100 字（加场景） | 14a |
| `src/prompt.py` workflow 行 | `explore → read → edit → test → submit_patch` → **`locate → read → edit → test → submit_patch`** | 14c |
| `src/prompt.py:TERMINATION_PROTOCOL` | 末尾加 **Locate-before-test** 段：禁止先 run_tests、明确 traceback 顶层通常是 dispatch | 14c |
| `src/tools/grep.py` schema | 加 `glob`、`type`、`-i`、`multiline`、`head_limit`（默认 250）、`offset`；maxResultChars 30K → 20K；输出加 `applied_limit` 反馈 | 14b |

### 1.1 Claude Code 对照（参考 `claude-code/packages/builtin-tools/src/tools/GrepTool/prompt.ts` + `GrepTool.ts`）

claude-code 做法（直接搬运的 4 点）：
- **"ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command"**——我的描述里也有
- **literal brace escape**——举 `interface\\{\\}` 作为示例
- **head_limit + applied_limit 反馈**——我加了 `head_limit=250`、并在 truncation 时返回 footer `[truncated at ...; N total entries; pass offset/head_limit to paginate]`
- **Output modes 配示例**——我列出三种

claude-code 独有的（**没抄**，因为本项目是 Python + 7B 模型够用即可）：
- `type` 参数（file type filter）：抄了
- `multiline`：抄了
- `head_limit` 软限制 + `offset` 分页：抄了
- `-A/-B/-C` 单独参数：未抄（保留合并的 `context=N`）

## 2. 实验结果（astropy-14182 跟踪）

```
阶段                  turn   tool_calls                                     done      verdict
─────────────────────────────────────────────────────────────────────────────────────────────
v0 (原 baseline)      3      run_tests × 3                                 stuck     WRONG_FILE
v1 (14a: prompt)      12     read_file × 6 + edit × 6                       max      WRONG_FILE (connect.py)
v2 (14b: tool only)    3      run_tests × 3                                 stuck     WRONG_FILE
v3 (14c: locate)      12     list_files + grep + read_file × 10             max      WRONG_FILE (table.py)
```

→ **v3 是质变**：模型终于**用了** `list_files` + `grep`。但它挑的关键词是 `QTable`（issue 里反复出现的类名），不是 `header_rows`（真正描述特性的 token）——这是 7B 模型 instruction-following 的天花板，目前无解。

## 3. 三个 instance 在 v3 下的实测

```
instance_id                verdict (eval)   verdict (real)   turns  tool mix
─────────────────────────────────────────────────────────────────────────────────────
astropy__astropy-12907      PASS             PASS             8    read_file × 4 + grep × 3 + edit
astropy__astropy-14182      WRONG_FILE       WRONG_FILE       12    list_files + grep + read_file × 10
astropy__astropy-14365      PARTIAL*         PASS             11    read_file + list_files + grep × 1 + edit × 3
```

\* `eval/run_one_with_capture.py` 的 `edited_files` heuristic 把 read_file 的 file_path 也算进去，导致 14365 误判为 PARTIAL（`ascii.py` 来自 turn 0 read_file）。**实际 edit 只动 qdp.py，verdict 是 PASS**。

### 3.1 astropy-12907 (verify_template_12907_grep_v3)
- 模型用 grep 3 次，仍然编辑 `separable.py`（golden 文件）→ **PASS**。
- 多用 grep 没导致回归。

### 3.2 astropy-14365 (verify_template_14365_grep_v3)
- 模型 turn 0 误读了 `ascii.py`（错误的），turn 2 用 grep 重新定位到 `qdp.py`
- turn 6-8 在 `qdp.py` 上做了 3 个 edit：`READ SERR → read serr`、`READ TERR → read terr`、再 `READ SERR → read serr` 加 `replace_all=true`
- run_tests + submit_text
- 实际是**正确的 fix 模式**（case-insensitivity），但 model 仍 submit_text 认输

## 4. 总结

| 维度 | 修前 | 修后 |
|---|---|---|
| astropy-12907 | PASS | PASS（无回归） |
| astropy-14182 | WRONG_FILE（卡 run_tests） | WRONG_FILE（用 grep 但挑错关键词） |
| astropy-14365 | PASS | PASS（多走一步 list_files+grep，无回归） |
| 2/3 PASS | 保持 | 保持 |

→ **SWE 验收不退化**；模型行为从"卡住"变成"主动探索"是质变；14182 的胜败留给 keyword-selection 问题的下一次尝试。

## 5. 提交

```
- src/prompt.py
- src/tools/grep.py
- eval/wire_captures/verify_template_14182_grep__20260826T084200Z.json      (14a)
- eval/wire_captures/verify_template_12907_grep_v3__20260826T085957Z.json  (14c)
- eval/wire_captures/verify_template_14182_grep_v3__20260826T085749Z.json  (14c)
- eval/wire_captures/verify_template_14365_grep_v3__20260826T090021Z.json  (14c)
- iteration/14-grep-prompt-promotion.md（本文件，覆盖 14a/14b/14c 三轮）