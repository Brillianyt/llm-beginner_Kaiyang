---
name: code-review
description: 当需要审查改动 diff、检查命名规范、识别潜在 bug 或给出重构建议时加载。适用于 review 单文件改动或小型 PR。
when_to_use: 用户要求"review / 审查 / 检查代码 / 看 diff"等场景
allowed-tools:
  - read_file
  - git_diff
  - list_files
---

# Code Review 工作流

## 何时使用本 Skill
- 用户提交了一个 PR 或一段 diff，让你"review"
- 用户问"这段代码有什么问题？"、"看看命名/风格"
- 修复完 bug 后想自检质量

## 工作流（按顺序执行）

### Step 1：理解改动范围
- 调用 `git_diff` 看全部改动
- 如 diff 超过 200 行，调用 `list_files` 列出 repo 结构先定位关键文件

### Step 2：逐文件阅读改动
- 用 `read_file` 读被修改的文件完整内容（不仅看 diff）
- 检查上下文（函数前后、调用方）

### Step 3：按以下维度评审
1. **正确性**：逻辑是否覆盖所有边界？异常路径？空值？
2. **命名**：变量/函数名是否清晰？是否与已有命名风格一致？
3. **风格**：是否符合 PEP 8 / 项目约定？
4. **性能**：是否有 N² 循环？是否多次 IO？
5. **安全**：是否有注入风险？路径穿越？硬编码 secrets？
6. **测试**：是否补了测试？是否覆盖新分支？

### Step 4：输出格式（直接给用户）
```
## 总体评价
[一两句概括]

## 必须修改 (blocker)
- [ ] issue 1：位置 + 描述 + 建议
- [ ] issue 2：...

## 建议优化 (nice-to-have)
- suggestion 1
- suggestion 2

## 亮点
- what went well
```

## 注意事项
- 不要修改任何代码！review 只读
- 如果用户要求"自动改"，用 `write_file` 改；review skill 不改
- 中文 review 也行；如果用户用英文，输出英文
