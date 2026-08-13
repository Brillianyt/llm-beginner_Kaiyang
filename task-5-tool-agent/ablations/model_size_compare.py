"""S2 · 不同模型尺寸对比：1.5B / 7B / 14B。

实现思路（来自 SYNTHESIS §7.2）：
- OpenAI 兼容 API 切换模型名即可（同 base_url）
- 跑同样的 10 题，看每个模型的命中率
- 1.5B 通常 < 50%（格式遵从差），7B 60%+，14B 70%+
- 无 Ollama 时优雅降级：打印提示，不让脚本崩溃

跑：`python ablations/model_size_compare.py [--smoke]`
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

# Ollama 模型命名约定
MODELS = [
    ("qwen2.5:1.5b-instruct", "1.5B"),
    ("qwen2.5:7b-instruct", "7B"),
    ("qwen2.5:14b-instruct", "14B"),
]


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
    """探测 endpoint 是否可达。"""
    try:
        c = LLMClient(cfg)
        # 用极小请求验证
        c.chat([{"role": "user", "content": "ok"}],
               max_tokens=4, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="仅跑 1 题做接口校验")
    ap.add_argument("--models", nargs="*", default=None,
                    help="指定模型列表，覆盖默认")
    args = ap.parse_args()

    # 数据
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        print(f"[SKIP] {tasks_path} 不存在")
        return 0
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if args.smoke:
        tasks = tasks[:1]

    # 默认 base_url = Ollama
    base_url = os.environ.get("OPENAI_BASE_URL",
                              "http://localhost:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "ollama")

    # 探测
    probe_cfg = LLMConfig(base_url=base_url, api_key=api_key,
                          model="qwen2.5:7b-instruct", timeout=5.0)
    if not _probe_endpoint(probe_cfg):
        print(f"[SKIP] {base_url} 不可达；S2 需要本地 Ollama / vLLM 运行")
        print("       启动后重跑：`ollama serve` + `ollama pull qwen2.5:7b-instruct`")
        # 写 stub 结果（标记未跑）
        out = ROOT / "eval" / "s2_model_size_result.json"
        out.write_text(json.dumps({
            "skipped": True,
            "reason": f"{base_url} 不可达",
            "models_tested": [],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"占位结果写入 {out.relative_to(ROOT)}")
        return 0

    model_list = args.models or [m for m, _ in MODELS]
    results: dict[str, dict] = {}

    for model_name in model_list:
        print(f"\n=== Model: {model_name} ===")
        cfg = LLMConfig(base_url=base_url, api_key=api_key, model=model_name,
                        timeout=60.0)
        agent = ReActAgent(llm_client=LLMClient(cfg), max_steps=10)
        success = 0
        details = []
        for t in tasks:
            try:
                trace = agent.run(t["task"])
                ans = trace.get("final_answer", "")
                ok = _answer_matches(ans, t.get("expected_answer_contains", []))
                success += int(ok)
                details.append({"id": t["id"], "success": ok,
                                "answer": ans[:80]})
            except Exception as e:
                details.append({"id": t["id"], "success": False, "error": str(e)})
        rate = success / max(1, len(tasks))
        results[model_name] = {"rate": rate, "n": len(tasks),
                               "details": details}
        print(f"  命中率：{rate:.1%} ({success}/{len(tasks)})")

    # 写结果
    out = ROOT / "eval" / "s2_model_size_result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())