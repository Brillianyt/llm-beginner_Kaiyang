---
name: pr-description-writer
description: 当需要为已完成的改动写 PR 描述、commit message 或 release note 时加载。基于 git diff 自动生成结构化描述。
when_to_use: 用户说"写 PR / 写 changelog / 写 commit message / 总结这次改动"
allowed-tools:
  - read_file
  - git_diff
  - list_files
---

# PR Description Writer

## 何时使用
- 提交 PR 前需要 description
- 写 changelog / release note
- 给团队解释"这次改了啥"

## 工作流

### Step 1：拿到 diff
- 调用 `git_diff` 看本次改动
- 如果改动很大（> 500 行），按文件分组总结而不是逐行

### Step 2：识别改动类型
按以下分类定位每个文件：
- **feat**: 新增功能
- **fix**: bug 修复
- **refactor**: 重构（无功能变化）
- **test**: 测试相关
- **docs**: 文档
- **chore**: 杂项

### Step 3：套模板输出

```markdown
## Summary
[1-3 句话概括：做什么、为什么]

## Changes
- **feat**: <file> — <what it does>
- **fix**: <file> — <bug fixed>

## Testing
- [ ] 跑了 `python -m pytest`
- [ ] 手工验证了 X

## Risks
- <known risk 1>
- <known risk 2>
```

### Step 4：自检
- Summary 是否能让 reviewer 30 秒内看懂？
- 是否漏了 breaking change 警告？
- 是否引用了相关 issue / ticket？

## 风格
- 英文 PR：imperative mood ("Add feature", not "Added")
- 中文 PR：正常陈述即可
- 不超过 200 行（超长拆 bullet）
