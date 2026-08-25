# 03 — Wheel Mirror + Edit/Read 错误信息增强（commit `3b41de2`）

## 主题

两件事：(a) `run_tests` 在 astropy 这种"源码不可直接构建"的场景下用 wheel mirror 而不是 `repo_root`；(b) `edit` 的 "old_string not found" 错误和 `read_file` 的 "file not found" 错误从"裸字符串"升级为"含 hint 的结构化错误"。

## 症状

### (a) astropy run_tests 全 ImportError

`repo_root = /mnt/.../data/repos/astropy` 是 `git clone` 的源码树，运行 `python -m pytest` 时：

```
ERROR astropy/_version.py - ImportError: cannot import name 'version' from 'astropy._dev.scm_version'
```

—— astropy 用 `setuptools_scm` 生成 version，clone 的源码里 `_dev/scm_version.py` 缺失（这是 setuptools_scm 故意 gitignore 的）。结果：pytest 一行测试都没收集到（`passed=0 failed=0`），但 stderr 全部 ImportError。

模型看到 `passed=0 failed=0` 的总结 → 误判为"测试基础设施挂了" → `submit_text` 放弃。

### (b) 错误信息无用

- `edit` 报 `[ERROR] old_string not found in <file>`。但模型根本没看到**原文件里长什么样**——Qwen2.5-Coder-7B 倾向于：
  - 臆造不存在的代码行（`if line.strip().upper().startswith('READ SERR')`，实际文件里没有）
  - 丢掉前导 whitespace（`command_re` 行有 4 空格缩进，模型写成 0 空格 → `text.count` miss）
  - 丢换行符（diff 输出少一行）

  结果：模型反复发同一个失败的 edit，进入"retry loop"。

- `read_file` 报 `[ERROR] not a regular file: <abs_path>`。模型根本没看到**目录里有什么文件**——Qwen2.5-Coder-7B 在 7B-class 上倾向于猜测路径：
  - `astropy/io/ascii/qdp.py`（正确）vs `astropy/io/qdp/qdp.py`（臆造）

  结果：模型反复调 `read_file` 不同变体直到 50 turn 用完。

## 在 wire capture 里的发现

### (a)

`wheel_mirror_14365__20260824T162057Z.json` turn 5 的 `run_tests` response：

```
exit_code=4
passed=0 failed=0 errors=0
duration_s=12.34
stderr_tail: ImportError: cannot import name 'version' from 'astropy._dev.scm_version'
...
```

模型第 6 轮调 `submit_text` —— **就因为 `passed=0`**。

### (b)

`wheel_mirror_14365__20260824T162057Z.json` turn 1 的 `read_file` response：

```
[ERROR] not a regular file: /mnt/.../astropy/io/qdp/qdp.py
[HINT] if you are guessing a path, use `list_files`
```

模型第 2 轮调 `list_files` —— **浪费一整轮只是为了看一眼目录**。修复后，模型读完错误信息**直接调** `read_file("astropy/io/ascii/qdp.py")`（正确路径）。

`wheel_mirror_12907__20260824T162358Z.json` 的 edit turn 4 显示连续 3 个失败的 edit，args 里的 `old_string` 字段差异只在尾随空格：

```
attempt 1: old="def _separable(transform):\n    matrix = np.zeros"
attempt 2: old="def _separable(transform):\n    matrix = np.zeros((noutp"  ← 多了 1 char
attempt 3: old="def _separable(transform):\n    matrix = np.zeros((noutp"  ← 又少了
```

—— `text.count` 始终为 0，模型不知道文件里到底是什么样。

## 修复

### (a) Wheel Mirror

`src/tools/run_tests.py`：

```python
# Wheel-mirror mode: when WHEEL_MIRROR_ROOT is set, sync every *.py
# under <repo_root>/<package> to <WHEEL_MIRROR_ROOT>/<package> before
# pytest, then run pytest with --rootdir=<WHEEL_MIRROR_ROOT>.
WHEEL_MIRROR_ROOT = os.environ.get("WHEEL_MIRROR_ROOT")
WHEEL_MIRROR_PKG  = os.environ.get("WHEEL_MIRROR_PKG", "astropy")
SKIP_SYNC_FILES = {"version.py", "_version.py", "_dev/scm_version.py"}
```

新建 `iteration` flow：
1. agent 把 model 的 edit 应用到 `repo_root`（裸源码）
2. `run_tests` 启动时把 `repo_root/astropy/*.py` 同步到 `wheel_mirror/astropy/*.py`（**`SKIP_SYNC_FILES` 除外**，避免 wheel 的正确 `_version.py` 被 source 的 broken fallback 覆盖）
3. `python -m pytest --rootdir=<WHEEL_MIRROR_ROOT> <mirror>/astropy/io/ascii/tests/test_qdp.py::test_roundtrip`
4. astropy 的 wheel 安装版有 `*.so`、有正确的 `version.py`、有 `iers/eopc04_1962-now` 数据文件
5. 模型看到真实的 `passed=N failed=M` 总结

注意 SKIP_SYNC 是关键：源代码的 `version.py` 是 `setuptools_scm` 的 broken fallback（`try: ... except: version="0.0.0"`），如果同步会覆盖 wheel 的正确 `version.py`，让所有 `astropy.<func>` import 都返回 `0.0.0` 而不是真实版本号 → 引入新 bug。

### (b) Edit 错误信息

`src/tools/edit.py`：

```python
# When edit fails because old_string isn't found, compute:
# 1. "lines containing <first_token>" — substring match (catches whitespace-diff)
# 2. "near matches (token-stripped)" — difflib ratio ≥ 0.5 (catches typos/extra chars)
# Then dump those lines with their original indentation preserved.
import difflib
old_first_token = old.splitlines()[0].split()[0] if old.splitlines() else ""
hint_lines = []
for ln, line in enumerate(text.splitlines(), start=1):
    if old_first_token in line:
        hint_lines.append(f"  L{ln}: {line.rstrip()[:160]}")
        if len(hint_lines) >= 10: break
near_lines = []
for ln, line in enumerate(text.splitlines(), start=1):
    ratio = difflib.SequenceMatcher(
        a=old_first_line.strip(), b=line.strip(), autojunk=False
    ).ratio()
    if ratio >= 0.5:
        near_lines.append((ratio, ln, line.rstrip()[:160]))
near_lines.sort(reverse=True)
near_hint = "\n--- near matches (token-stripped) ---\n" + "\n".join(
    f"  {ratio:.2f} L{ln}: {content}" for ratio, ln, content in near_lines[:5]
)
hint = "\n--- lines containing " + old_first_token + " ---\n" + "\n".join(hint_lines)
return f"[ERROR] old_string not found in {path}\n{hint}{near_hint}"
```

### (b) Read 错误信息

`src/tools/read_file.py`：

```python
# When read_file fails because the file doesn't exist:
existing = []
parent = target.parent if target.parent.is_dir() else repo_root
target_stem = target.stem.lower()
for entry in sorted(parent.iterdir()):
    if entry.is_file() and entry.suffix == ".py":
        stem = entry.stem.lower()
        score = 0 if stem == target_stem else (
            1 if stem.startswith(target_stem[:4]) or target_stem.startswith(stem[:4]) else 2
        )
        existing.append((score, entry.name))
existing.sort()
existing_block = "[EXISTING .py files in " + str(parent) + "]\n" + "\n".join(
    f"  {n}" for _, n in existing[:30]
)
return f"[ERROR] file not found: {target}\n{existing_block}"
```

—— 输出示例：

```
[ERROR] file not found: /mnt/.../astropy/io/qdp/qdp.py
[EXISTING .py files in /mnt/.../astropy/io/ascii]
  cds.py
  fixedwidth.py
  qdp.py    ← correct path
  ...
```

## 验证

- `wheel_mirror_14365__20260824T162057Z.json` 显示 wheel mirror 启用后，`run_tests` response 出现 `passed=N failed=M` 真实数字。
- `wheel_mirror_12907__20260824T162358Z.json` 显示模型读错误信息后**直接调**正确路径的 `read_file`，不再触发 `list_files` 兜底。
- `better_edit_err_14365__20260824T162215Z.json` 显示修复后第一个 `edit` 失败的 response 包含完整 hint，模型立刻调 `read_file` 重新读文件并修复 `old_string`。
- 8/8 smoke tests 仍通过。

## Commit

```
3b41de2  fix: wheel-mirror run_tests + better tool errors
```

修改文件：
- `src/tools/run_tests.py`（+ WHEEL_MIRROR_ROOT/LOG/SKIP_SYNC 机制，168 行 diff）
- `src/tools/edit.py`（+ hint 计算逻辑）
- `src/tools/read_file.py`（+ EXISTING .py files 列出）
- `src/prompt.py`（小修）
- 3 个 wire capture（wheel_mirror_14365 / wheel_mirror_12907 / scoped_run_12907）

## 关键证据

- `eval/wire_captures/wheel_mirror_14365__20260824T162057Z.json`
- `eval/wire_captures/wheel_mirror_12907__20260824T162358Z.json`
- `eval/wire_captures/better_edit_err_14365__20260824T162215Z.json`
- `eval/wire_captures/better_hint_12907__20260824T163244Z.json`
- `eval/wire_captures/scoped_run_12907__20260824T162642Z.json`
- `eval/wire_captures/file_hint_12907__20260824T162522Z.json`