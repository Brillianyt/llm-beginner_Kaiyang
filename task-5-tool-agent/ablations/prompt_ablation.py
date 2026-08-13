"""S3 · Prompt 模板消融：few-shot 数量 + 错误提示开关。

实现思路（来自 SYNTHESIS §7.3）：
- 参数化 PromptBuilder：few_shot_count ∈ {0, 1, 3} × include_error_hint ∈ {True, False}
- 共 6 组组合（3×2），跑同样的 10 题
- 观察哪个参数对命中率影响最大
- 无 LLM 时优雅降级：仅做"prompt 长度 / token 估算"消融，不真跑模型

跑：`python ablations/prompt_ablation.py [--smoke]`
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
from src.prompt import PromptBuilder
from src.tools import default_registry

CONFIGS = [
    {"few_shot_count": 0, "include_error_hint": False, "label": "0-shot, no hint"},
    {"few_shot_count": 0, "include_error_hint": True,  "label": "0-shot, +hint"},
    {"few_shot_count": 1, "include_error_hint": False, "label": "1-shot, no hint"},
    {"few_shot_count": 1, "include_error_hint": True,  "label": "1-shot, +hint"},
    {"few_shot_count": 3, "include_error_hint": False, "label": "3-shot, no hint"},
    {"few_shot_count": 3, "include_error_hint": True,  "label": "3-shot, +hint"},
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
    try:
        c = LLMClient(cfg)
        c.chat([{"role": "user", "content": "ok"}],
               max_tokens=4, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001
        return False


def _prompt_lengths() -> list[dict]:
    """统计每组 config 的 prompt 长度（仅消息条数 + 字符数）。"""
    reg = default_registry()
    out = []
    for cfg in CONFIGS:
        pb = PromptBuilder(
            tool_schemas=reg.schema_list(),
            few_shot_count=cfg["few_shot_count"],
            include_error_hint=cfg["include_error_hint"],
        )
        msgs = pb.initial_messages("样例任务")
        char_count = sum(len(m["content"]) for m in msgs)
        out.append({
            "label": cfg["label"],
            "few_shot_count": cfg["few_shot_count"],
            "include_error_hint": cfg["include_error_hint"],
            "messages": len(msgs),
            "chars": char_count,
            "approx_tokens": char_count // 4,  # 粗估
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="仅做 prompt 长度消融")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="指定 config label 列表")
    args = ap.parse_args()

    # 数据
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        print(f"[SKIP] {tasks_path} 不存在")
        return 0
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if args.smoke:
        tasks = tasks[:1]

    # 1) 必跑：prompt 长度消融（不依赖 LLM）
    print("=== Prompt 长度消融（不依赖 LLM）===")
    lengths = _prompt_lengths()
    for row in lengths:
        print(f"  [{row['label']:>20}] "
              f"messages={row['messages']:>2}, "
              f"chars={row['chars']:>5}, "
              f"approx_tokens={row['approx_tokens']:>4}")

    # 2) 跑模型消融（可选）
    base_url = os.environ.get("OPENAI_BASE_URL",
                              "http://localhost:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "ollama")
    probe_cfg = LLMConfig(base_url=base_url, api_key=api_key,
                          model="qwen2.5:7b-instruct", timeout=5.0)
    if not _probe_endpoint(probe_cfg):
        print(f"\n[SKIP] {base_url} 不可达；S3 真模型消融需要 Ollama / vLLM 运行")
        # 写 stub 结果（含 prompt 长度数据）
        out = ROOT / "eval" / "s3_prompt_ablation_result.json"
        out.write_text(json.dumps({
            "skipped_model_runs": True,
            "reason": f"{base_url} 不可达",
            "prompt_lengths": lengths,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"占位结果写入 {out.relative_to(ROOT)}")
        return 0

    print("\n=== 真模型消融（依赖 LLM）===")
    model = os.environ.get("OPENAI_MODEL", "qwen2.5:7b-instruct")
    cfg_obj = LLMConfig(base_url=base_url, api_key=api_key, model=model,
                        timeout=60.0)
    config_results: list[dict] = []
    for cfg in CONFIGS:
        if args.configs and cfg["label"] not in args.configs:
            continue
        print(f"\n--- {cfg['label']} ---")
        agent = ReActAgent(
            llm_client=LLMClient(cfg_obj),
            max_steps=10,
            few_shot_count=cfg["few_shot_count"],
            include_error_hint=cfg["include_error_hint"],
        )
        success = 0
        for t in tasks:
            try:
                trace = agent.run(t["task"])
                ans = trace.get("final_answer", "")
                ok = _answer_matches(ans,
                                     t.get("expected_answer_contains", []))
                success += int(ok)
            except Exception:
                pass
        rate = success / max(1, len(tasks))
        config_results.append({**cfg, "rate": rate,
                               "n": len(tasks), "success": success})
        print(f"  命中率：{rate:.1%} ({success}/{len(tasks)})")

    out = ROOT / "eval" / "s3_prompt_ablation_result.json"
    out.write_text(json.dumps({
        "prompt_lengths": lengths,
        "model_runs": config_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())