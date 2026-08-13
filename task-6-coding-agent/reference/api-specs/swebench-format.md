# SWE-bench 数据格式（精简版）

## 来源
- 官方仓库：https://github.com/SWE-bench/SWE-bench
- 数据集：https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite
- 论文：https://arxiv.org/abs/2310.06770

## 关键要点
1. **数据集结构**：每个 instance = (issue 文本 + repo + 修复 patch + 测试 patch)
2. **核心字段**（Lite parquet 包含）：
   - `instance_id` —— 唯一标识（`repo__pr-number` 格式）
   - `repo` —— GitHub 仓库（`owner/name`）
   - `base_commit` —— 修改前的 commit SHA
   - `problem_statement` —— issue 描述文本（Markdown）
   - `patch` —— 修复 patch（unified diff）
   - `test_patch` —— 测试 patch（unified diff，**禁止修改**）
   - `FAIL_TO_PASS` —— list[str]，修复前 FAIL、修复后 PASS 的测试 ID
   - `PASS_TO_PASS` —— list[str]，修复前后都 PASS 的测试 ID（防回归）
3. **评测规则**：
   - Agent 拿到 `repo@base_commit` + `problem_statement`
   - Agent 产出 patch
   - 应用 patch → 跑 `FAIL_TO_PASS` 测试（应全 PASS） → 跑 `PASS_TO_PASS` 测试（应全 PASS）
   - 两类都通过才算 resolved
4. **变体**：
   - SWE-bench（2,294 题）—— 全量
   - SWE-bench Lite（534 题）—— 精选，迭代快
   - SWE-bench Verified（500 题）—— OpenAI 人工校验版
5. **评测 harness**：Docker 容器内运行（避免环境差异）；超时通常 5-30 分钟/题
6. **加载方式**：
   ```python
   from datasets import load_dataset
   ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
   row = ds[0]
   ```

## 与我们任务的关联
- **S4（可选加分）**：抽样 3 题至少 1 题 tests_passed
- **自检脚本**（`eval/run.py`）：
  - 没下元数据 → skip
  - 没 clone repo → skip
  - 跑了 → 看 `trace["tests_passed"]` 是否 True
- **评测必须严格**：
  - 不能改 `test_patch`（题目禁止）
  - 不能改 base_commit 之前的代码（除非 patch 涉及）
  - 必须跑**全量**测试（不能只跑 issue 提到的某个）

## 文字版数据流

```
SWE-bench Lite (534 题)
  │
  ├── instance["django__django-11099"]
  │     repo: "django/django"
  │     base_commit: "abc123..."
  │     problem_statement: "Bug: ..."
  │     gold_patch: "diff --git a/..."
  │     FAIL_TO_PASS: ["test_x.py::test_y"]
  │     PASS_TO_PASS: [...]
  │
  ▼
我们的 CodingAgent
  │
  ├── 输入：repo@base_commit + problem_statement
  ├── 输出：predicted_patch
  │
  ▼
SWE-bench harness（Docker 内）
  │
  ├── git checkout base_commit
  ├── git apply predicted_patch
  ├── pytest FAIL_TO_PASS → 应全 PASS
  ├── pytest PASS_TO_PASS → 应全 PASS
  │
  ▼
resolved: True / False
```

## 代码片段（用 SWE-bench 抽样跑通一道）

```python
import pandas as pd
from pathlib import Path

df = pd.read_parquet("data/swebench-lite-sample.parquet")
row = df.iloc[0]
print(row["instance_id"], row["repo"], row["base_commit"])
print(row["problem_statement"][:300])

# 准备 repo（学生需自行 clone）
repo_path = Path("data/repos") / row["repo"].split("/")[-1]
if (repo_path / ".git").exists():
    import subprocess
    subprocess.run(["git", "checkout", row["base_commit"]], cwd=repo_path, check=True)

# 跑 agent
from src.agent import CodingAgent
agent = CodingAgent()
trace = agent.run(repo_path=str(repo_path), issue=row["problem_statement"])
print(f"tests_passed: {trace.get('tests_passed')}")
print(f"patch:\n{trace.get('patch', '')[:500]}")
```

## 我们应该怎么借鉴
1. **抽样先打印 gold patch 看格式**：我们的 CodingAgent 输出的 patch 格式应与 gold 一致（unified diff）
2. **不要改 test_patch**：prompt 要明示「禁止修改 tests/ 目录」
3. **必须 git checkout base_commit**：题面是 base_commit 上的代码；不在该 commit 上跑可能环境错位
4. **跑全量测试**：`python -m pytest -q`（不限单个测试）；`FAIL_TO_PASS` 通过 + `PASS_TO_PASS` 通过才算成功
5. **超时设置**：SWE-bench 一题可能 5-10 分钟；我们 run_tests 工具要支持 timeout=300+ 秒
6. **失败恢复**：如果 patch 应用失败（`git apply` 报错），agent 应该回滚重试，不要让目录污染状态
7. **评测沙箱**：生产用 Docker；我们 toy repo 直接跑就行；SWE-bench 真跑时考虑用 `subprocess + venv` 隔离，避免污染主机 Python 环境

## 主要参考来源
- SWE-bench 官方：https://www.swebench.com/
- 仓库：https://github.com/SWE-bench/SWE-bench
- 论文：https://arxiv.org/abs/2310.06770
- 数据集：https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite
- 中文评测指南：https://blog.csdn.net/gitblog_01170/article/details/159339650