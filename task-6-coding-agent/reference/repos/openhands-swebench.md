# OpenHands (OpenDevin) - SWE-bench Coding Agent 学习笔记

## 来源
- 仓库：https://github.com/All-Hands-AI/OpenHands
- 论文：OpenHands: An Open Platform for AI Software Developers as Generalist Agents
- 核心定位：**目前在 SWE-bench 上效果领先的开源 coding agent**，42.3% 任务通过率，14.2 分钟平均修复时间

## 关键要点
1. **架构组件**：
   - `CodeActAgent`：核心 agent，每步生成代码动作（受 CodeAct 论文启发）
   - `agent_controller.py`：实现有限状态机（FSM）控制 agent 生命周期
   - `DelegatorAgent`：任务分发
   - 后端服务：事件流管理、状态存储、安全沙箱
   - 前端：Web UI + CLI
2. **CodeActAgent 内置工具**（我们要对照精简）：
   - `execute_bash`：在沙箱里跑命令
   - `execute_ipython_cell`：跑 Python
   - `web_read`：读网页
   - `browser`：浏览器自动化
   - `str_replace_editor` / `edit_file`：字符串替换式编辑（比 diff 更适合 LLM）
3. **SWE-bench 评测管线**：`evaluation/benchmarks/swe_bench/` 目录下，含 Docker 镜像生成、patch 应用、测试执行全套。
4. **状态机设计**：agent 行为受 FSM 约束——`init → plan → step → finish → exit`，每步可以被打断、回滚、分叉。
5. **九大设计原则**（从 OpenHands 论文提取）：
   - 统一执行/业务状态（不是双状态机）
   - 无状态 reducer（每步是纯函数）
   - 有限状态机控制 agent flow
   - 事件流驱动（agent 与 controller 解耦）
   - 安全沙箱（Docker / 容器化）
   - 多 agent 协作
   - 可观测性（事件流 replay）
   - 可扩展性（plugin / hook）
   - 失败恢复与重试
6. **str_replace_editor 模式**：与 `diff` 相比，**「找唯一字符串 → 替换为新字符串」**的编辑方式对 LLM 更友好。OpenHands 默认用它而不是整文件覆写。

## 与我们任务的关联
- **M1（5+ 工具）**：我们暴露 `read_file / write_file / run_tests / git_diff / git_apply`——与 OpenHands 工具集有交集；可以参考 `str_replace_editor` 加一个 `edit_file(old_str, new_str)` 工具。
- **M3（agent loop）**：OpenHands 的 FSM 思路非常清晰；我们至少要有显式的状态转移（READ/UNDERSTAND/PLAN/ACT/VERIFY/DONE），即使不用 enum 写，也要能在 Trace 里看出来每步处于哪个状态。
- **M4（Trace）**：OpenHands 的事件流是完整 replay-able 的——我们 Trace 设计可以参考其字段（`step / state / action / observation / timestamp`）。
- **S4（SWE-bench Lite）**：直接复用 OpenHands 的评测管线思路，但用更小的 7B 模型和简化工具集。

## 代码片段（OpenHands 风格的状态机伪代码）

```python
class AgentState(Enum):
    INIT = "init"
    PLAN = "plan"
    STEP = "step"
    FINISH = "finish"

class CodeActAgent:
    def step(self, state: AgentState):
        if state == AgentState.INIT:
            plan = self.llm.generate_plan(self.task)
            return AgentState.PLAN
        elif state == AgentState.PLAN:
            return AgentState.STEP
        elif state == AgentState.STEP:
            action = self.llm.generate_action(self.history)
            obs = self.execute(action)
            self.history.append((action, obs))
            if self.should_finish():
                return AgentState.FINISH
            return AgentState.STEP
```

## str_replace_editor 示例

```python
# 比整文件覆写更安全的编辑模式
@server.tool()
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """将文件中第一次出现的 old_text 替换为 new_text。"""
    p = (REPO_ROOT / path).resolve()
    assert p.is_relative_to(REPO_ROOT)
    content = p.read_text(encoding="utf-8")
    if content.count(old_text) != 1:
        raise ValueError(f"old_text 出现 {content.count(old_text)} 次，必须恰好 1 次")
    new_content = content.replace(old_text, new_text)
    p.write_text(new_content, encoding="utf-8")
    return "ok"
```

## 我们应该怎么借鉴
1. **状态机优于 while 循环**：用 `current_state` 变量显式记录，每步转移，Trace 里可读性更强（而不是塞进 thought）。
2. **edit_file 工具**：除了 `write_file`（整文件覆写），再加一个 `edit_file(old, new)` 工具——降低 LLM 出错的概率（整文件覆写容易漏行）。
3. **事件流驱动**：每步产出一个 dict `{state, thought, action, observation}`，append 到 trace；不要在 step 之间共享可变状态（除了 trace list）。
4. **失败重试**：OpenHands 默认每个 action 重试 2-3 次。我们也给工具失败加 retry 逻辑，但**整体步数上限内**——比如 step 30 步上限里最多重试 5 次。
5. **沙箱**：OpenHands 走 Docker；我们没那么严格，**subprocess + cwd 限定**就够，但 SWE-bench 真跑时要小心——SWE-bench 标准 harness 用 Docker，我们直接 subprocess 容易污染主机。建议 SWE-bench 跑时也启一个临时 Docker / venv 隔离。
6. **不要照抄复杂度**：OpenHands 是工业级产品（含 Web UI、20+ 工具、Docker 管理）；我们是 Mini 版。**原则：能复用的就复用其思想，不复制的就放弃**。

## 主要参考来源
- 仓库：https://github.com/All-Hands-AI/OpenHands
- 评测目录：https://github.com/All-Hands-AI/OpenHands/tree/main/evaluation/benchmarks/swe_bench
- 中文架构解析：https://blog.csdn.net/gitblog_00371/article/details/151462614
- CodeActAgent 解读：https://www.cnblogs.com/rossiXYZ/p/19620422