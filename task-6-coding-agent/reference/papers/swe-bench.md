# SWE-bench: Real-World GitHub Issue Resolution Benchmark

## 来源
- 论文链接：https://arxiv.org/abs/2310.06770
- 官方站点：https://www.swebench.com/
- 评测仓库：https://github.com/SWE-bench/SWE-bench
- 作者：Carlos E. Jimenez et al.（Princeton NLP 团队）
- 发布时间：2023-10
- 核心定位：**衡量 LLM 解决真实 GitHub issue 能力的基准**，含 2,294 个 Issue-PR 对（12 个 Python 仓库）

## 关键要点
1. **数据集结构**：每个 instance 是 (issue 文本, 修复 PR 的代码 patch, 测试 patch)。Agent 拿到 issue + 仓库初始状态，必须产出能通过隐藏测试的 patch。
2. **测试分类（核心概念）**：
   - `FAIL_TO_PASS`：issue 修复前 FAIL、修复后 PASS 的测试；用来判断 issue 是否真的被解决。
   - `PASS_TO_PASS`：修复前后都 PASS 的测试；用来检测回归。
   - 只有两类全过才算 instance 解决。
3. **变体版本**：
   - **SWE-bench**（完整版，2,294 题）：跑一次要几十小时
   - **SWE-bench Lite**（534 题精选）：迭代更快
   - **SWE-bench Verified**（500 题 + 人工校验，OpenAI 2024-08）：修了「测试过严/描述不清/环境难配」三大问题
   - **SWE-bench Multimodal / Multilingual**：扩展版
4. **评测方式**：Docker 容器跑测试 harness，应用 patch → 跑 `FAIL_TO_PASS` 和 `PASS_TO_PASS` → 全过则记为 resolved。
5. **题面字段**（Lite parquet 包含）：`instance_id`, `repo`, `problem_statement`, `patch`（gold patch）, `test_patch`, `base_commit`, `FAIL_TO_PASS`, `PASS_TO_PASS` 等。

## 与我们任务的关联
- **S4（可选进阶 SWE-bench Lite）**：本任务的 S4 是「抽样 3 题至少 1 题 tests_passed」。我们的 CodingAgent 必须能：拿到 `problem_statement` + 本地 clone 的仓库，产出 patch 并让 harness 跑通。
- **自检脚本逻辑**（`eval/run.py`）：
  - 没下 `--with-swebench` 时直接 `skip`（不要报失败）
  - 下了 parquet 但没 clone 对应 repo 也 skip
  - 跑了至少 1 题才算 attempted
  - `tests_passed` 通过 `trace.get("tests_passed")` 读取
- **本任务评测可比性**：SWE-bench 标准 harness 在 Docker 里跑；本任务给的 60 秒 timeout 远不够 SWE-bench 全量测试，但抽样的简单 issue 大概率够用。**实现时记得把 run_tests 的 timeout 设可调**（比如 SWE-bench 跑 300 秒、toy repo 跑 60 秒）。

## 代码片段（Lite parquet 字段示例）

```python
import pandas as pd
df = pd.read_parquet("data/swebench-lite-sample.parquet")
row = df.iloc[0]
print(row["instance_id"], row["repo"], row["base_commit"])
print(row["problem_statement"][:200])
# 'FAIL_TO_PASS' 和 'PASS_TO_PASS' 是 list[str]（测试函数全名）
```

## 我们应该怎么借鉴
1. **gold patch 当 ground truth**：抽样完先打印一份 gold patch 看「正确答案长什么样」，反推 CodingAgent 应该输出的格式。我们建议统一成 unified diff（git diff 格式），这样可以直接灌给 `git apply`。
2. **不能改测试**：SWE-bench 规则是 `test_patch` 不让改，只改 `patch`（源码）。CodingAgent 的 prompt 要明示「禁止修改 `tests/` 目录下的文件」。
3. **base_commit 必读**：抽样的 SWE-bench 实例对应 `data/repos/<repo>` 的某个 commit，agent 必须在那个 commit 上跑（不要 reset 到 HEAD，否则代码不一样）。我们的简单实现里就先 `git checkout <base_commit>` 再让 agent 改。
4. **回归检测**：`PASS_TO_PASS` 测试相当于「你的修改不能搞坏别的功能」。CodingAgent 每次改完代码必须跑**全量 pytest**（而不是只跑 issue 提到的某个测试），否则可能修了一个测、破了十个。
5. **小步快跑**：SWE-bench Lite 单题平均 LLM 调用 50-100 次。我们给 Qwen2.5-Coder-7B 设计的最大步数建议 30-50，再大就要做 context compaction。

## 主要参考来源
- arXiv：https://arxiv.org/abs/2310.06770
- 官方：https://www.swebench.com/
- 数据集：https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite
- 评测详解：https://blog.csdn.net/gitblog_01170/article/details/159339650
- OpenAI Verified：https://blog.csdn.net/weixin_41429382/article/details/144055292