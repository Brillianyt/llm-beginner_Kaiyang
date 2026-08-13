# 渐进式披露（Progressive Disclosure）—— Skill 加载策略

## 来源
- Anthropic Skills 设计文档：https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview
- Anthropic Skills 仓库：https://github.com/anthropics/skills
- Claude Code 源码：`src/skills/loadSkillsDir.ts`
- 概念出处：UX 设计原则（progressive disclosure of information），应用到 prompt 工程

## 关键要点
1. **核心思想**：不要一次把所有信息塞进 context；分层、按需暴露
2. **三层暴露**：
   - **Level 1 — Skill 列表**：只暴露 `name + description`（约 20-50 token/skill），用于 LLM 判断「是否相关」
   - **Level 2 — Skill 正文**：LLM 决定用某个 skill 后，才把 SKILL.md 完整 markdown 读进 context（可达 5K-20K token）
   - **Level 3 — Skill 内部资源**：Skill 内的 `scripts/` `references/` 文件，被 skill 正文里 `Read references/foo.md` 之类的指令触发加载
3. **为什么有效**：
   - 减少 prompt 长度 → 节省 token、降成本
   - 减少 LLM 干扰（信号噪声比提升）→ 提高准确率
   - 让 Skill 库可以无限扩展而不爆 context
4. **路由策略**：基于 description 的关键词匹配；或让 LLM 自评「是否相关」（更准确但更贵）
5. **状态**：被加载的 skill 进 context 后「粘住」——直到 conversation 结束或被显式丢弃
6. **估算 token**（来自 Claude Code 源码）：`estimateSkillFrontmatterTokens = roughTokenCountEstimation(name + description + whenToUse)`，只算 frontmatter 的开销

## 与我们任务的关联
- **M2（Skill 加载器）**：直接对应 progressive disclosure
- **description 是关键设计**：决定 LLM 何时加载——太宽（"代码相关"）=永远命中=浪费 token；太窄（"特定 API 错误处理"）=永不命中
- **list_skills 与 load 分离**：list 只返回 frontmatter 元数据；load(name) 才读完整正文

## 文字版数据流

```
                    ┌────────────────────────────────────┐
                    │   CodingAgent 启动                 │
                    └────────────┬───────────────────────┘
                                 │ scan src/skills/*/SKILL.md
                                 ▼
                    ┌────────────────────────────────────┐
                    │   SkillLoader.list_skills()        │
                    │   → 返回 [{name, description}]     │   ← Level 1
                    └────────────┬───────────────────────┘
                                 │ 注入 system prompt
                                 │ "以下 skill 可用：..."
                                 ▼
                    ┌────────────────────────────────────┐
                    │   LLM 判断当前任务是否需要某个      │
                    │   skill（基于 description 匹配）    │
                    └────────────┬───────────────────────┘
                                 │ LLM 在 thought 里说
                                 │ "要加载 code-review skill"
                                 ▼
                    ┌────────────────────────────────────┐
                    │   SkillLoader.load("code-review")  │
                    │   → 读 SKILL.md 完整正文           │   ← Level 2
                    │   → 塞进 messages 作为 system 消息 │
                    └────────────┬───────────────────────┘
                                 │ skill 正文里可能写
                                 │ "参考 templates/pr.md"
                                 ▼
                    ┌────────────────────────────────────┐
                    │   按需加载 templates/pr.md          │   ← Level 3
                    └────────────────────────────────────┘
```

## 代码片段（Python 实现，约 50 行）

```python
import yaml
from pathlib import Path
from typing import Dict, List

class SkillLoader:
    """扫描 skills_dir 下所有 SKILL.md，支持渐进式披露。"""

    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)
        self._meta: Dict[str, dict] = {}      # name -> {description, path}
        if not self.skills_dir.exists():
            return
        for md in self.skills_dir.glob("*/SKILL.md"):
            text = md.read_text(encoding="utf-8")
            meta, _body = self._parse_frontmatter(text)
            if "name" not in meta or "description" not in meta:
                continue  # 跳过无效 skill
            self._meta[meta["name"]] = {
                "description": meta["description"],
                "path": md,
                "when_to_use": meta.get("when_to_use"),
            }

    def list_skills(self) -> List[dict]:
        """Level 1：只返回 name + description（轻量）"""
        return [{"name": n, "description": m["description"]} for n, m in self._meta.items()]

    def load(self, name: str) -> str:
        """Level 2：返回 SKILL.md 完整正文（按需）"""
        if name not in self._meta:
            raise KeyError(f"skill not found: {name}")
        text = self._meta[name]["path"].read_text(encoding="utf-8")
        _, body = self._parse_frontmatter(text)
        return body

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end != -1:
                meta = yaml.safe_load(text[4:end])
                body = text[end + 5:]
                return meta or {}, body
        return {}, text


# CodingAgent 中使用：
class CodingAgent:
    def __init__(self, ..., skill_loader: SkillLoader):
        self.skill_loader = skill_loader

    def build_system_prompt(self) -> str:
        skills = self.skill_loader.list_skills()
        skill_block = "\n".join(f"- {s['name']}: {s['description']}" for s in skills)
        return f"""你是 coding agent。可用 skill：
{skill_block}

需要时调用 `load_skill(name)` 加载完整正文。
"""

    def handle_load_skill_call(self, name: str) -> str:
        return self.skill_loader.load(name)   # Level 2 暴露
```

## 我们应该怎么借鉴
1. **description 写法**：写成「**何时加载**」而非「**是什么**」。正例：「当需要 review PR diff、检查命名规范、识别潜在 bug 时加载」。反例：「代码 review 工具」（太宽）
2. **list_skills 在 system prompt 里只占几百 token**：description 写 1-2 句就够，不要塞完整工作流到 description
3. **正文里写 workflow 步骤**：「Step 1: read_file → Step 2: git_diff → Step 3: 输出格式...」
4. **Level 3 可选**：v1 不实现 references/ 子目录，v2 再加
5. **不要预加载**：agent 启动时**不要**把所有 SKILL.md 读进 context；只在 `list_skills()` 时读 frontmatter，正文按需加载
6. **缓存**：同一个 skill 在同一 conversation 里多次 load 时可以缓存；但 v1 不用做
7. **allowed-tools**（来自 Claude Code）：Skill 可以声明自己只能用某些 tool，避免 skill 内部的 prompt 误导 LLM 调不该用的 tool

## 主要参考来源
- Anthropic Skills 设计：https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview
- Claude Code `src/skills/loadSkillsDir.ts`：完整的渐进式披露实现
- Claude Code `src/skills/bundledSkills.ts`：内置 skill 集合