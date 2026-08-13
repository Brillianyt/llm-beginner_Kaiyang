# Anthropic Skills 仓库学习笔记

## 来源
- 仓库：https://github.com/anthropics/skills
- 文档：https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview
- 维护者：Anthropic
- 核心定位：**Claude 可调用的「技能包」目录**，每个 Skill 是一个含 SKILL.md 的文件夹，可被 Agent 按需加载（progressive disclosure）

## 关键要点
1. **目录结构约定**：
   ```
   skills/
   └── <skill-name>/
       ├── SKILL.md          # 必含：frontmatter + 正文
       ├── scripts/          # 可选：可执行脚本
       └── references/       # 可选：参考文档（API 文档、模板等）
   ```
2. **SKILL.md frontmatter 格式**：
   ```yaml
   ---
   name: code-review
   description: 何时加载此 skill（具体触发条件，如「review 某段 PR diff」「检查函数命名规范」）
   ---
   ```
   - `name`：skill 唯一标识，文件夹名应一致
   - `description`：**关键字段**——LLM 用它决定何时加载该 skill。要写清楚「什么时候用」，不能太泛（否则一直命中）或太窄（否则永不命中）
3. **正文**：Markdown 格式的工作流描述，类似 prompt 模板。比如「先 X 后 Y 再 Z」「按以下格式输出」。
4. **Progressive Disclosure（渐进式披露）**：
   - 阶段 1：只暴露 skill 列表（含 name + description），不进 prompt
   - 阶段 2：LLM 决定加载某个 skill 时才把完整正文塞进 context
   - 阶段 3：可进一步让 skill 引用 `references/` 文件，按需加载
5. **Skills 与 Tools 的区别**：Tool 是「能调用的函数」；Skill 是「组织化的最佳实践+模板+脚本」，可能在内部用多个 tool 完成一组操作。

## 与我们任务的关联
- **M2（Skill 加载器 + SKILL.md）**：直接对齐 Anthropic 的 SKILL.md frontmatter 格式，让自检通过 `list_skills()` 返回的 dict 含 `name` + `description`。
- **加载器设计**（约 50 行）：扫描 `src/skills/*/SKILL.md` → 解析 YAML frontmatter → 缓存元数据；`load(name)` 时才读完整正文。
- **至少 2-3 个 Skill**：建议覆盖三类典型场景——「代码审查」「PR 描述生成」「测试运行与失败诊断」。

## 代码片段（SKILL.md 示例）

```markdown
---
name: code-review
description: 当需要审查代码质量、检查命名规范或识别潜在 bug 时加载。适用于 review PR diff、review 单文件改动、给出重构建议。
---

# Code Review 工作流

## Step 1：理解改动范围
先用 `read_file` 读取待 review 的文件；如涉及 PR，用 `git_diff` 看完整 diff。

## Step 2：分类检查
按以下清单逐项检查：
- 命名是否清晰（变量、函数、类）
- 错误处理是否完整
- 是否有明显 bug（边界条件、空值）
- 是否有性能问题

## Step 3：输出格式
按以下 markdown 结构输出：
### 发现的问题
- [严重性] 描述：位置 + 原因
### 改进建议
- 描述
```

## 我们应该怎么借鉴
1. **description 是触发器**：写成「何时加载 X」而非「X 是什么」。反例："A code review skill"；正例："当用户请求代码 review、检查代码质量、识别潜在 bug 时加载"。
2. **正文写工作流而非事实**：Skill 的正文是给 LLM 看的 prompt 模板，应含「按 X 步走」「输出格式为 Y」之类的指令。
3. **scripts/ 子目录放可复用代码**：比如 test-runner skill 可以带一个 `run_pytest.sh` 脚本，LLM 通过 tool 调它而不是自己拼命令。
4. **references/ 子目录放参考文档**：比如 pr-description-writer skill 可以带 `pr_template.md` 作为输出模板，按需加载避免污染主 context。
5. **加载器实现核心**（伪代码）：
   ```python
   class SkillLoader:
       def __init__(self, skills_dir: str):
           self.skills_dir = Path(skills_dir)
           self._meta = {}  # name -> {"description": ..., "path": ...}
           for md in self.skills_dir.glob("*/SKILL.md"):
               meta, _ = parse_frontmatter(md.read_text())
               self._meta[meta["name"]] = meta
       def list_skills(self):
           return [{"name": n, "description": m["description"]} for n, m in self._meta.items()]
       def load(self, name):
           return self.skills_dir / name / "SKILL.md").read_text()
   ```
6. **description 匹配策略**：v1 简单做关键词包含匹配；v2 让 LLM 自评「是否相关」（成本高）。我们先用 v1。

## 主要参考来源
- Skills 仓库：https://github.com/anthropics/skills
- 官方文档：https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview
- 综述博客：https://blog.csdn.net/qq_44810930/article/details/156146071