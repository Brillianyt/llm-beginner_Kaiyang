"""S4 · 错误注入消融。

实现思路（来自 SYNTHESIS §7.4 + README §M3）：
- 用 ToolRegistry 的 `set_error_rate(name, rate)` 按概率注入 [ERROR: ...]
- 跑 10 题，分别测 error_rate ∈ {0.0, 0.2, 0.5, 0.8} 下的命中率
- 验证 M3 的"失败即字符串"路径在不同错误率下是否还能保持合理命中率
- 无 LLM 时优雅降级：用 stub 假 LLM 验证注入逻辑（不影响主流程）

跑：`python ablations/error_injection.py [--smoke]`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ReActAgent
from src.llm_client import LLMClient, LLMConfig

ERROR_RATES = [0.0, 0.2, 0.5, 0.8]
TOOLS_TO_INJECT = ["calculator", "python_sandbox"]


def _normalize(text: str) -> str:
    import re
    t = str(text).lower().replace(",", "").replace("，", "")
    return re.sub(r"\s+", "", t)


def _answer_matches(answer, expected) -> bool:
    norm = _normalize(answer)
    for kw in expected:
        if isinstance(kw, list):
            if not any(_normalize(k) in norm for k in kw):
                return False
        else:
            if _normalize(kw) not in norm:
                return False
    return True


def _probe_endpoint(cfg: LLMConfig, timeout: float = 5.0) -> bool:
    try:
        c = LLMClient(cfg)
        c.chat([{"role": "user", "content": "ok"}],
               max_tokens=4, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001
        return False


class _RepeatFakeLLM:
    """知道 task 答案的假 LLM：先调 calculator，再 Final Answer。

    设计：让 agent 实际调用一次 calculator，这样错误注入能触发。
    如果 calculator 注入错误，第二次改为用 python_sandbox。
    """

    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    def chat(self, messages, model=None, **kw):
        self.calls += 1
        # 检查历史里有没有 [ERROR: ...] Observation（注入的错误）
        last_obs = ""
        for m in reversed(messages):
            if m.get("role") == "user" and "Observation:" in m["content"]:
                last_obs = m["content"]
                break
        if "[ERROR" in last_obs:
            # 上一步报错，换工具
            return (
                "Thought: 换工具\n"
                "Action: python_sandbox\n"
                "Action Input: {\"code\": \"print(42)\"}"
            )
        if self.calls == 1:
            return (
                "Thought: 用 calculator\n"
                "Action: calculator\n"
                "Action Input: {\"expression\": \"1+1\"}"
            )
        return (
            "Thought: 给答案\n"
            "Action: Final Answer\n"
            f"Action Input: {self.answer}"
        )


def _run_with_stub(task: dict, error_rate: float, tools_to_inject: list[str]) -> dict:
    """用假 LLM 验证错误注入逻辑本身（不验证真实命中率）。"""
    # 取期望答案（简化版：取第一个关键词当答案）
    expected = task.get("expected_answer_contains", [])
    if isinstance(expected[0], list):
        answer = expected[0][0]
    else:
        answer = expected[0]
    fake = _RepeatFakeLLM(answer)
    agent = ReActAgent(llm_client=fake, max_steps=8)
    for name in tools_to_inject:
        agent.set_error_rate(name, error_rate, msg="[Injected]")
    trace = agent.run(task["task"])
    steps_with_error = sum(1 for s in trace["steps"] if s.get("is_error"))
    return {
        "id": task["id"],
        "steps": len(trace["steps"]),
        "error_steps": steps_with_error,
        "success": trace["success"],
        "final": trace["final_answer"][:80],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="仅跑 1 题 + 2 个 rate")
    ap.add_argument("--no-llm-check", action="store_true",
                    help="跳过 LLM 探测（强制 stub 模式）")
    args = ap.parse_args()

    # 数据
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        print(f"[SKIP] {tasks_path} 不存在")
        return 0
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))

    # 1) 注入逻辑 stub 验证（不依赖 LLM）
    print("=== Stub 验证：注入逻辑生效 ===")
    rates_to_test = [0.0, 0.5, 0.8] if args.smoke else ERROR_RATES
    stub_results = []
    for rate in rates_to_test:
        print(f"\n--- error_rate = {rate} ---")
        for t in tasks[:3]:  # 仅前 3 题做注入验证
            r = _run_with_stub(t, rate, TOOLS_TO_INJECT)
            r["error_rate"] = rate
            print(f"  task {r['id']:>2}: "
                  f"steps={r['steps']}, "
                  f"error_steps={r['error_steps']}, "
                  f"success={r['success']}")
            stub_results.append(r)

    # 2) 真模型消融（可选）
    base_url = os.environ.get("OPENAI_BASE_URL",
                              "http://localhost:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "ollama")
    model = os.environ.get("OPENAI_MODEL", "qwen2.5:7b-instruct")
    cfg_obj = LLMConfig(base_url=base_url, api_key=api_key, model=model,
                        timeout=60.0)

    model_results: list[dict] = []
    if not args.no_llm_check and _probe_endpoint(cfg_obj):
        print(f"\n=== 真模型错误注入消融（{model}）===")
        for rate in ERROR_RATES:
            print(f"\n--- error_rate = {rate} ---")
            agent = ReActAgent(llm_client=LLMClient(cfg_obj), max_steps=10)
            success = 0
            for t in tasks:
                agent.clear_errors()
                for name in TOOLS_TO_INJECT:
                    agent.set_error_rate(name, rate, msg="[Injected]")
                try:
                    trace = agent.run(t["task"])
                    ans = trace.get("final_answer", "")
                    ok = _answer_matches(
                        ans, t.get("expected_answer_contains", [])
                    )
                    success += int(ok)
                except Exception:
                    pass
            rate_val = success / max(1, len(tasks))
            model_results.append({
                "error_rate": rate,
                "rate_val": rate_val,
                "success": success,
                "n": len(tasks),
            })
            print(f"  命中率：{rate_val:.1%} ({success}/{len(tasks)})")
    else:
        if not args.no_llm_check:
            print(f"\n[SKIP] {base_url} 不可达；真模型消融需本地 LLM 运行")

    # 写结果
    out = ROOT / "eval" / "s4_error_injection_result.json"
    out.write_text(json.dumps({
        "stub_verification": stub_results,
        "model_runs": model_results,
        "skipped_model_runs": not bool(model_results),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())