"""S1 · Qwen-Agent 对照实验。

实现思路（来自 SYNTHESIS §7.1）：
- 用 qwen_agent.agents.ReActAgent + 同样的 4 个工具（wrapping 成 QwenAgent 工具格式）
- 跑同样的 10 题，对比自写 ReActAgent 与 Qwen-Agent 的成功率
- 如 qwen-agent 未安装，优雅降级：打印提示，不让脚本崩溃
- 不依赖网络（用假 LLM 或 stub）

跑：`python ablations/qwen_agent_baseline.py [--smoke]`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools import default_registry
from src.tools.base import Tool


def _qwen_tool_wrapper(tool: Tool):
    """把 Tool 包装成 qwen_agent.tools.BaseTool 子类。

    qwen-agent 工具要求继承 BaseTool 并实现 `call(...)`。
    """
    try:
        from qwen_agent.tools.base import BaseTool  # type: ignore
    except ImportError:
        return None

    class _WrappedTool(BaseTool):
        def __init__(self):
            super().__init__()
            self.name = tool.name
            self.description = tool.description
            self.parameters = tool.parameters

        def call(self, params: str, **kwargs) -> str:
            import json as _json
            try:
                args = _json.loads(params) if isinstance(params, str) else params
            except Exception:
                args = {"_raw": params}
            return tool.run(args)

    return _WrappedTool()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="仅跑 1 题做接口校验")
    ap.add_argument("--compare-with-self", action="store_true",
                    help="同时跑自写 ReActAgent 做对比")
    args = ap.parse_args()

    reg = default_registry()
    qwen_tools = []
    for t in reg._tools.values():
        wrapped = _qwen_tool_wrapper(t)
        if wrapped is not None:
            qwen_tools.append(wrapped)

    if not qwen_tools:
        print("[SKIP] qwen-agent 未安装；本消融实验需先 `pip install qwen-agent`")
        print("       （自写 ReActAgent 不依赖 qwen-agent，照常可用）")
        return 0

    try:
        from qwen_agent.agents import ReActAgent as QwenAgent  # type: ignore
    except ImportError:
        print("[SKIP] 无法 import qwen_agent.agents")
        return 0

    # 数据
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        print(f"[SKIP] {tasks_path} 不存在")
        return 0
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if args.smoke:
        tasks = tasks[:1]

    # 跑 QwenAgent
    llm_cfg = {
        "model": "qwen2.5:7b-instruct",
        "model_server": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
    print("=== Qwen-Agent baseline ===")
    qwen_agent = QwenAgent(llm=llm_cfg, tool_list=qwen_tools)
    qwen_success = 0
    qwen_results = []
    for t in tasks:
        try:
            response = []
            for chunk in qwen_agent.run(t["task"]):
                response.append(chunk)
            # 提取 final answer
            ans = str(response[-1]) if response else ""
            ok = any(kw in ans for kw in t.get("expected_answer_contains", []))
            qwen_success += int(ok)
            qwen_results.append({"id": t["id"], "success": ok, "answer": ans[:120]})
        except Exception as e:
            qwen_results.append({"id": t["id"], "success": False,
                                 "error": str(e)})
    qwen_rate = qwen_success / max(1, len(tasks))
    print(f"Qwen-Agent 成功率：{qwen_rate:.1%} ({qwen_success}/{len(tasks)})")

    # 对比自写（可选）
    if args.compare_with_self:
        print("\n=== 自写 ReActAgent ===")
        from src.agent import ReActAgent

        # 用一个会失败的假 LLM，避免依赖真实模型（仅接口对比）
        class _StubLLM:
            def chat(self, messages, model=None, **kw):
                # 直接返回期望格式（用 task 自身的关键词当答案）
                return ("Thought: stub\nAction: Final Answer\n"
                        "Action Input: stub 答案")

        # 这里只对比 trace 结构，不对比真实成功率
        agent = ReActAgent(llm_client=_StubLLM(), max_steps=3)
        for t in tasks[:1]:
            try:
                trace = agent.run(t["task"])
                print(f"自写 trace 结构 OK: "
                      f"steps={len(trace['steps'])}, "
                      f"final={trace['final_answer'][:60]!r}")
            except Exception as e:
                print(f"自写跑失败：{e}")

    # 写结果
    out = ROOT / "eval" / "s1_qwen_agent_result.json"
    out.write_text(json.dumps({
        "qwen_agent_rate": qwen_rate,
        "qwen_results": qwen_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())