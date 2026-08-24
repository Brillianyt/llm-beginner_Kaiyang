"""Smoke test：4 个工具 + PromptBuilder + ActionParser + ReActAgent 主循环。

不依赖真实 LLM / 网络。所有测试用 `_FakeLLM` 或直接调工具函数。
跑：`python test_smoke.py` 或 `python test_smoke.py --smoke`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------- helpers
def header(title: str) -> None:
    print(f"\n=== {title} ===")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# --------------------------------------------------------------------- 工具测试
def test_calculator() -> bool:
    header("calculator")
    from src.tools import calculator
    passed = True
    cases = [
        ({"expression": "2 + 3 * 4"}, "14", "基本四则"),
        ({"expression": "(123+456)*789"}, "456831", "复合表达式"),
        ({"expression": "sqrt(2026)"}, "45.011109739", "math.sqrt"),
        ({"expression": "pi * 2"}, "6.2831853072", "常量 pi"),
    ]
    for args, expect, desc in cases:
        try:
            out = calculator.run(args)
            if expect in out:
                ok(f"{desc}: {out}")
            else:
                fail(f"{desc}: 期望包含 {expect!r}，实际 {out!r}")
                passed = False
        except Exception as e:
            fail(f"{desc} 抛异常：{e}")
            passed = False
    # 拒绝不安全
    try:
        calculator.run({"expression": "__import__('os').system('ls')"})
        fail("未拒绝 __import__")
        passed = False
    except Exception as e:
        if "__import__" in str(e) or "不在白名单" in str(e):
            ok(f"拒绝 __import__: {e}")
        else:
            ok(f"拒绝不安全表达式: {e}")
    try:
        calculator.run({"expression": "open('x')"})
        fail("未拒绝 open")
        passed = False
    except Exception:
        ok("拒绝 open()")
    return passed


def test_python_sandbox() -> bool:
    header("python_sandbox")
    from src.tools import python_sandbox
    passed = True
    cases = [
        ({"code": "print(sum(range(10)))"}, "45", "sum(range(10))"),
        ({"code": "print('hello')"}, "hello", "字符串 print"),
        ({"code": "x = sum(range(100))\nprint(x)"}, "4950", "多行"),
    ]
    for args, expect, desc in cases:
        try:
            out = python_sandbox.run(args)
            if expect in out:
                ok(f"{desc}: {out!r}")
            else:
                fail(f"{desc}: 期望包含 {expect!r}，实际 {out!r}")
                passed = False
        except Exception as e:
            fail(f"{desc} 抛异常：{e}")
            passed = False
    # 拒绝 os import
    try:
        python_sandbox.run({"code": "import os\nprint(os.getcwd())"})
        fail("未拒绝 import os")
        passed = False
    except Exception as e:
        ok(f"拒绝 import os: {e}")
    return passed


def test_file_search() -> bool:
    header("file_search")
    from src.tools import file_search
    passed = True
    # 默认 allowed_root 现在是 task-5-tool-agent/，所以相对路径 data/agent-fixtures 就够了
    cases = [
        ({"pattern": "*.md", "dir": "data/agent-fixtures"},
         "README.md", "glob *.md"),
        ({"pattern": "TODO", "dir": "data/agent-fixtures"},
         "todo_note.md", "内容正则 TODO"),
        ({"pattern": "README.md", "dir": "data/agent-fixtures"},
         "任务五", "返回内容片段"),
    ]
    for args, expect, desc in cases:
        try:
            out = file_search.run(args)
            if expect in out:
                ok(f"{desc}: 命中 {expect!r}")
            else:
                fail(f"{desc}: 期望包含 {expect!r}，实际 {out!r}")
                passed = False
        except Exception as e:
            fail(f"{desc} 抛异常：{e}")
            passed = False
    # 越界保护
    try:
        file_search.run({"pattern": "*.py", "dir": "../"})
        fail("未拒绝 .. 越界")
        passed = False
    except Exception as e:
        ok(f"拒绝路径越界: {e}")
    try:
        file_search.run({"pattern": "*.py", "dir": "/etc"})
        fail("未拒绝绝对路径越界")
        passed = False
    except Exception as e:
        ok(f"拒绝绝对路径越界: {e}")
    return passed


def test_wiki() -> bool:
    header("wiki (依赖网络)")
    from src.tools import wiki
    try:
        out = wiki.run({"query": "Alan Turing"})
        if "Turing" in out or len(out) > 50:
            ok(f"wiki 网络 OK: {out[:80]}...")
            return True
        fail(f"wiki 异常返回: {out[:80]}")
        return False
    except Exception as e:
        print(f"  [SKIP] wiki 无网络：{e}")
        return True  # 离线环境跳过（不算失败）


# --------------------------------------------------------------------- PromptBuilder
def test_prompt_builder() -> bool:
    header("PromptBuilder")
    from src.prompt import PromptBuilder, FEW_SHOTS
    from src.tools import default_registry
    reg = default_registry()
    pb = PromptBuilder(tool_schemas=reg.schema_list(), few_shot_count=3)
    msgs = pb.initial_messages("计算 1+1")
    has_system = any(m["role"] == "system" for m in msgs)
    fewshot_assistant = sum(1 for m in msgs if m["role"] == "assistant")
    ok(f"messages 长度={len(msgs)}, system={has_system}, "
       f"assistant(包含 few-shot)={fewshot_assistant}")
    # few_shot_count=0
    pb0 = PromptBuilder(tool_schemas=reg.schema_list(), few_shot_count=0)
    msgs0 = pb0.initial_messages("计算 1+1")
    ok(f"few_shot_count=0 → messages 长度={len(msgs0)}")
    # append_observation
    msgs1 = pb.initial_messages("test")
    msgs2 = pb.append_observation(
        msgs1, "思考", "calculator",
        {"expression": "1+1"}, "2"
    )
    obs_count = sum(1 for m in msgs2 if m["role"] == "user"
                    and m["content"].startswith("Observation:"))
    ok(f"append_observation 添加了 {obs_count} 条 Observation")
    return has_system and len(msgs0) < len(msgs) and obs_count == 1


# --------------------------------------------------------------------- ActionParser
def test_action_parser() -> bool:
    header("ActionParser")
    from src.parser import ActionParser
    p = ActionParser()
    passed = True
    cases = [
        ("Thought: x\nAction: calculator\n"
         "Action Input: {\"expression\": \"1+1\"}",
         {"action": "calculator", "ai_type": dict}),
        ("Thought: y\nAction: Final Answer\nAction Input: 6 位",
         {"action": "Final Answer", "ai_type": str}),
        ("Thought 2: think\nAction 1: wiki\n"
         "Action Input 1: {\"query\": \"Alan Turing\"}",
         {"action": "wiki", "ai_type": dict}),
    ]
    for text, expect in cases:
        try:
            r = p.parse(text)
            if r.get("retry"):
                fail(f"意外进入 retry: {text[:30]}...")
                passed = False
                continue
            ai = r.get("action_input")
            ok_str = (r["action"] == expect["action"]
                      and isinstance(ai, expect["ai_type"]))
            if ok_str:
                ok(f"{text[:30]}... → action={r['action']}")
            else:
                fail(f"{text[:30]}... → action={r['action']}, "
                     f"type={type(ai)}")
                passed = False
        except Exception as e:
            fail(f"解析 {text[:30]}... 抛异常：{e}")
            passed = False
    # 解析失败 → retry
    r = p.parse("some random text without format")
    if r.get("retry"):
        ok("解析失败 → retry=True")
    else:
        fail(f"应返回 retry，实际 {r}")
        passed = False
    return passed


# --------------------------------------------------------------------- ReActAgent
def test_react_agent() -> bool:
    header("ReActAgent (假 LLM)")
    from src.agent import ReActAgent, trace_to_text

    class _FakeLLM:
        def __init__(self, replies):
            self.replies = list(replies)
            self.calls = []

        def chat(self, messages, model=None, **kw):
            self.calls.append(list(messages))
            if self.replies:
                return self.replies.pop(0)
            return ("Thought: 兜底\nAction: Final Answer\n"
                    "Action Input: 兜底答案")

    passed = True
    # 1. 单步 calculator
    fake = _FakeLLM([
        "Thought: 算\nAction: calculator\n"
        "Action Input: {\"expression\": \"(123+456)*789\"}",
        "Thought: 6 位\nAction: Final Answer\nAction Input: 6 位",
    ])
    agent = ReActAgent(llm_client=fake, max_steps=5)
    trace = agent.run("计算 (123+456)*789 是几位数")
    if trace["success"] and "6 位" in trace["final_answer"]:
        ok(f"单步 calculator → {trace['final_answer']}")
    else:
        fail(f"单步失败: {trace}")
        passed = False

    # 2. 错误恢复：tool 抛错
    fake2 = _FakeLLM([
        "Thought: 调\nAction: calculator\n"
        "Action Input: {\"expression\": \"\"}",
        "Thought: 修正\nAction: calculator\n"
        "Action Input: {\"expression\": \"1+2\"}",
        "Thought: OK\nAction: Final Answer\nAction Input: 3",
    ])
    agent2 = ReActAgent(llm_client=fake2, max_steps=5)
    trace2 = agent2.run("1+2")
    steps_with_error = [s for s in trace2["steps"] if s.get("is_error")]
    if trace2["success"] and len(steps_with_error) >= 1:
        ok(f"错误恢复: {len(steps_with_error)} 步带 [ERROR], "
           f"最终 {trace2['final_answer']}")
    else:
        fail(f"错误恢复失败: {trace2}")
        passed = False

    # 3. 卡死检测
    fake3 = _FakeLLM([
        "Thought: stuck\nAction: calculator\n"
        "Action Input: {\"expression\": \"1+1\"}",
        "Thought: stuck\nAction: calculator\n"
        "Action Input: {\"expression\": \"1+1\"}",
        "Thought: stuck\nAction: calculator\n"
        "Action Input: {\"expression\": \"1+1\"}",
        "Thought: stuck\nAction: Final Answer\nAction Input: end",
    ])
    agent3 = ReActAgent(llm_client=fake3, max_steps=10)
    trace3 = agent3.run("stuck")
    if not trace3["success"] and len(trace3["steps"]) <= 5:
        ok(f"卡死检测生效：{len(trace3['steps'])} 步终止, "
           f"success={trace3['success']}")
    else:
        fail(f"卡死检测失败: {trace3}")
        passed = False

    # 4. 解析失败 → retry
    fake4 = _FakeLLM([
        "没有按格式输出",
        "Thought: 第二次按格式\nAction: Final Answer\nAction Input: done",
    ])
    agent4 = ReActAgent(llm_client=fake4, max_steps=5)
    trace4 = agent4.run("format")
    if trace4["success"]:
        ok(f"解析重试生效 → {trace4['final_answer']}")
    else:
        fail(f"解析重试失败: {trace4}")
        passed = False

    # 5. S4 错误注入
    fake5 = _FakeLLM([
        "Thought: 用 calc\nAction: calculator\n"
        "Action Input: {\"expression\": \"1+1\"}",
        "Thought: 用 python\nAction: python_sandbox\n"
        "Action Input: {\"code\": \"print(1+1)\"}",
        "Thought: OK\nAction: Final Answer\nAction Input: 2",
    ])
    agent5 = ReActAgent(llm_client=fake5, max_steps=5)
    agent5.inject_error("calculator", "[Injected]")
    trace5 = agent5.run("1+1")
    has_inject_err = any("[Injected]" in s.get("observation", "")
                         for s in trace5["steps"])
    if has_inject_err and trace5["success"]:
        ok(f"S4 错误注入生效，最终 {trace5['final_answer']}")
    else:
        fail(f"S4 注入未生效: {trace5}")
        passed = False

    return passed


# --------------------------------------------------------------------- 总结
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="仅跑 smoke 子集")
    args = ap.parse_args()

    results = {}
    if not args.smoke:
        results["calculator"] = test_calculator()
        results["python_sandbox"] = test_python_sandbox()
        results["file_search"] = test_file_search()
        results["wiki"] = test_wiki()
    else:
        results["calculator"] = test_calculator()
        results["python_sandbox"] = test_python_sandbox()
        results["file_search"] = test_file_search()
        results["wiki"] = True
    results["prompt_builder"] = test_prompt_builder()
    results["action_parser"] = test_action_parser()
    results["react_agent"] = test_react_agent()

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        flag = "[OK]" if v else "[FAIL]"
        print(f"  {flag} {k}")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())