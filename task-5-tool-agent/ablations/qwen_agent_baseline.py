"""S1 · Qwen-Agent 对照实验。

实现思路：
- 用 qwen_agent.agents.ReActChat + 同样的 4 个工具（wrapping 成 QwenAgent 工具格式）
- 跑同样的 10 题，对比自写 ReActAgent 与 Qwen-Agent 的成功率
- 优雅降级：qwen-agent 未安装时打印 SKIP

跑：
    # 启动 SGLang (Qwen2.5-7B-Instruct) 后:
    OPENAI_BASE_URL=http://localhost:30000/v1 \
    OPENAI_API_KEY=EMPTY \
    OPENAI_MODEL=/root/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master \
    python ablations/qwen_agent_baseline.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------- helpers
def _normalize(text: str) -> str:
    text = str(text).lower()
    text = text.replace(",", "").replace("，", "")
    return re.sub(r"\s+", "", text)


def _answer_matches(answer: str, expected_keywords) -> bool:
    norm = _normalize(answer)
    for expected in expected_keywords:
        if isinstance(expected, list):
            if not any(_normalize(k) in norm for k in expected):
                return False
        elif _normalize(expected) not in norm:
            return False
    return True


def _qwen_tool_wrapper(tool):
    """把自写 Tool 包装成 qwen_agent.tools.BaseTool 子类。"""
    try:
        from qwen_agent.tools.base import BaseTool  # type: ignore
    except ImportError:
        return None

    # 类属性必须 class-level 设置 (BaseTool 校验)
    class _WrappedTool(BaseTool):
        name = tool.name
        description = tool.description
        parameters = tool.parameters

        def call(self, params, **kwargs) -> str:
            import json as _json
            if isinstance(params, str):
                try:
                    args = _json.loads(params)
                except Exception:
                    args = {"_raw": params}
            else:
                args = params
            return tool.run(args)

    return _WrappedTool()


def _qwen_final_answer(response) -> str:
    """从 Qwen-Agent 的响应中提取 final answer。

    Qwen-Agent 的 `run()` 返回 generator,产出 list[Message]（最后一次 yield 是完整历史）。
    """
    # response 可能是一个 list[Message]（最后一次 yield 的结果）
    msgs = response if isinstance(response, list) else []
    if not msgs and isinstance(response, str):
        # 直接是字符串（异常或简短回复）
        return response
    for m in reversed(msgs):
        if isinstance(m, dict):
            if m.get("role") == "assistant":
                return str(m.get("content", ""))
            # 兼容 message 对象有 .role/.content 属性
        elif hasattr(m, "role") and getattr(m, "role", None) == "assistant":
            return str(getattr(m, "content", ""))
    return str(response) if not isinstance(response, list) else ""


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="仅跑 1 题做接口校验")
    ap.add_argument("--compare-with-self", action="store_true",
                    help="同时跑自写 ReActAgent 做对比")
    args = ap.parse_args()

    # 数据
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        print(f"[SKIP] {tasks_path} 不存在；先跑 python data/download.py")
        return 0
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if args.smoke:
        tasks = tasks[:1]

    # 包工具
    from src.tools import default_registry
    reg = default_registry()
    qwen_tools = []
    for t in reg._tools.values():
        wrapped = _qwen_tool_wrapper(t)
        if wrapped is not None:
            qwen_tools.append(wrapped)

    if not qwen_tools:
        print("[SKIP] qwen-agent 未安装；本消融实验需先 `pip install qwen-agent`")
        return 0

    try:
        from qwen_agent.agents import ReActChat  # type: ignore
    except ImportError:
        print("[SKIP] 无法 import qwen_agent.agents")
        return 0

    # LLM 配置（OpenAI 兼容 → 指向 SGLang）
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:30000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    model = os.environ.get("OPENAI_MODEL",
                            "/root/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master")
    llm_cfg = {
        "model": model,
        "model_server": base_url,
        "api_key": api_key,
        "generate_cfg": {"max_tokens": 1024, "temperature": 0.0},
    }
    print(f"=== Qwen-Agent baseline ({len(tasks)} 题) ===")
    print(f"  endpoint: {base_url}, model: {model}")

    # 跑 Qwen-Agent
    qwen_success = 0
    qwen_results = []
    qwen_agent = ReActChat(llm=llm_cfg, function_list=qwen_tools,
                             system_message="你是一个工具调用助手。")
    for t in tasks:
        try:
            # Qwen-Agent 的 run() 需要 messages 列表（不是裸字符串）
            messages = [{"role": "user", "content": t["task"]}]
            response = list(qwen_agent.run(messages))
            # response[-1] 是最终历史 messages（list of dict）
            ans = _qwen_final_answer(response[-1] if response else "")
            ok = _answer_matches(ans, t.get("expected_answer_contains", []))
            qwen_success += int(ok)
            qwen_results.append({"id": t["id"], "success": ok,
                                  "answer_preview": ans[:120]})
            print(f"  task {t['id']}: {'✅' if ok else '❌'} | {ans[:80]}")
        except Exception as e:
            qwen_results.append({"id": t["id"], "success": False,
                                  "error": str(e)[:120]})
            print(f"  task {t['id']}: ❌ {str(e)[:80]}")
    qwen_rate = qwen_success / max(1, len(tasks))
    print(f"\nQwen-Agent 成功率：{qwen_rate:.1%} ({qwen_success}/{len(tasks)})")

    # 对比自写（可选）
    self_rate = None
    if args.compare_with_self:
        print("\n=== 自写 ReActAgent ===")
        from src.agent import ReActAgent
        agent = ReActAgent()
        self_success = 0
        self_results = []
        for t in tasks:
            try:
                trace = agent.run(t["task"])
                ans = trace.get("final_answer", "")
                ok = _answer_matches(ans, t.get("expected_answer_contains", []))
                self_success += int(ok)
                self_results.append({"id": t["id"], "success": ok,
                                      "answer_preview": ans[:120]})
            except Exception as e:
                self_results.append({"id": t["id"], "success": False,
                                      "error": str(e)[:120]})
        self_rate = self_success / max(1, len(tasks))
        print(f"自写 ReActAgent 成功率：{self_rate:.1%} ({self_success}/{len(tasks)})")
        delta = (self_rate - qwen_rate) * 100
        sign = "+" if delta >= 0 else ""
        print(f"差值：{sign}{delta:.1f} pp (自写 - qwen-agent)")

    # 写结果
    out = ROOT / "eval" / "s1_qwen_agent_result.json"
    out.write_text(json.dumps({
        "endpoint": base_url,
        "model": model,
        "qwen_agent_rate": round(qwen_rate, 3),
        "self_rate": round(self_rate, 3) if self_rate is not None else None,
        "n_tasks": len(tasks),
        "qwen_results": qwen_results,
        "self_results": self_results if args.compare_with_self else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())