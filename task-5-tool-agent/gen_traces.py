#!/usr/bin/env python3
"""Generate full ReAct traces for a few representative tasks."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:30000/v1")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_MODEL",
                       "/root/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master")

from src.agent import ReActAgent
from src.trace import trace_to_text
from src.tools import default_registry
from src.prompt import PromptBuilder

# Pick 3 representative tasks: simple calc, file search, multi-tool wiki+calc
tasks_path = ROOT / "data" / "tasks.json"
tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
selected = [tasks[1], tasks[6], tasks[4]]  # python_sandbox, file_search, multi-tool

agent = ReActAgent()
out_lines = ["# Sample ReAct Traces (Qwen2.5-7B-Instruct via SGLang)\n"]
for t in selected:
    out_lines.append(f"\n## Task {t['id']}: {t['task']}\n")
    out_lines.append(f"**Expected:** {t['expected_answer_contains']}\n")
    trace = agent.run(t["task"])
    out_lines.append("```\n" + trace_to_text(trace) + "\n```")
    out_lines.append(f"\n**Final Answer:** {trace['final_answer']}")
    out_lines.append(f"\n**Success:** {trace['success']}\n")
    out_lines.append("")

out_path = ROOT / "eval" / "sample_traces.md"
out_path.write_text("\n".join(out_lines), encoding="utf-8")
print(f"wrote {out_path}")