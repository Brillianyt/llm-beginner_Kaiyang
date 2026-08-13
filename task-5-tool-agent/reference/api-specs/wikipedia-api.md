# Wikipedia API 与 python-wikipedia-api

## 来源
- 链接：https://github.com/martin-majlis/python-wikipedia-api + https://pypi.org/project/wikipedia-api/
- 作者/组织：Martin Majlis（社区维护）
- 发布时间：持续维护
- 核心定位：Python 客户端封装，简化访问 MediaWiki API（en/zh wiki 通用）

## 关键要点

### 1. 安装

```bash
pip install wikipedia-api
```

依赖 `requests`，已在 `requirements.txt`。

### 2. 基本用法

```python
import wikipediaapi

# 初始化客户端（指定语言）
wiki = wikipediaapi.Wikipedia(
    user_agent="MyToolAgent/1.0 (contact@example.com)",  # 必须填，Wikipedia 现在强制 UA
    language="zh",  # 'en'、'zh'、'zh-cn' 都行
)

# 按标题获取页面
page = wiki.page("图灵机")

# 关键属性
print(page.exists())           # bool，是否存在
print(page.title)              # str，标准化后的标题
print(page.summary[:500])      # str，前 500 字摘要
print(page.text[:2000])        # str，完整正文（可能很长）
print(page.fullurl)            # str，维基百科链接
```

### 3. 关键约定

- **必须设置 `user_agent`**：Wikipedia 现在强制要求 User-Agent，否则返回 403。格式建议 `AppName/version (contact)`。
- **语言代码**：`zh` 是简体中文，`en` 是英文。Wikipedia 还有 `zh-tw`、`zh-hk` 等变体。
- **页面不存在**：`page.exists()` 返回 False，不要直接访问 `page.summary`，会变成空字符串。
- **网络依赖**：本任务 wiki 工具需要联网，离线 / 被墙时按 README 规则"跳过"。

### 4. 检索 vs 直接查

python-wikipedia-api 提供两种方式：

| 方式 | API | 适用场景 |
|---|---|---|
| 精确标题查询 | `wiki.page("图灵机")` | 已知标题（任务集给的查询词大多是这种） |
| 模糊搜索 | 需要用 MediaWiki `action=query&list=search` | 不知道精确标题 |

任务集的 wiki 题（"查维基百科里图灵机的发明者"）通常标题明确，**精确查询够用**。但 `wiki.run` 内部建议加一层模糊搜索兜底——如果 `wiki.page(query).exists()` 为 False，就用 `action=query&list=search` 找最相似的标题。

### 5. MediaWiki API 直接调用（兜底用）

如果 python-wikipedia-api 不够灵活，可以直接调 MediaWiki REST API：

```python
import requests

def wiki_search(query, lang="zh", limit=3):
    """模糊搜索，返回最相似标题列表。"""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return [hit["title"] for hit in data["query"]["search"]]
```

### 6. 任务集中的 wiki 题分析

| 任务 | 标题候选 | 关键信息位置 |
|---|---|---|
| 4. 图灵机发明者 | "图灵机" / "Turing machine" | summary 第一句 |
| 5. Geoffrey Hinton 出生年份 | "Geoffrey Hinton" / "杰弗里·辛顿" | infobox / summary |
| 9. Transformer (机器学习) 发表年份 | "Transformer (机器学习模型)" / "Transformer (deep learning architecture)" | summary |

这些题的关键事实都在 summary 里（前 200 字），所以 wiki 工具实现可以**只返回 summary**，避免 LLM 上下文爆掉。

### 7. 常见坑

- **`user_agent` 为空或默认**：403 错误。务必填项目标识。
- **中文编码**：MediaWiki 返回 UTF-8，Python 3 默认 OK，但要注意终端打印时的 console 编码。
- **网络超时**：默认 requests 超时很长，建议显式 `timeout=10`。
- **页面存在但 summary 为空**：极少数页面（重定向 / 消歧义页），要兜底返回完整 text 或提示。
- **断网 / 被墙**：catch 所有 requests 异常，返回 `[ERROR: wiki 网络不可用: ...]`。

## 与我们任务的关联

- **M1 wiki 工具实现**：基于 python-wikipedia-api，固定参数 `{"query": str}`，返回 summary + 链接。
- **M3 错误恢复**：网络异常 → 错误字符串 → Observation，模型能感知并改用其他工具。
- **M4 命中率**：第 4 / 5 / 9 题完全依赖 wiki 工具——wiki 工具返回的字符串必须包含"图灵"、"1947"、"2017"这些关键词。
- **README 评测**：wiki 工具若抛异常，eval 脚本判定为"跳过"而非"失败"，不拖累 M4 总分。

## 代码片段（wiki 工具的最小可用实现）

```python
# src/tools/wiki.py
import wikipediaapi

WIKI = wikipediaapi.Wikipedia(
    user_agent="Task5ToolAgent/1.0 (educational use)",
    language="zh",
)

def run(args: dict) -> str:
    query = args["query"]
    page = WIKI.page(query)
    if not page.exists():
        # 兜底：尝试英文
        en_wiki = wikipediaapi.Wikipedia(
            user_agent="Task5ToolAgent/1.0 (educational use)",
            language="en",
        )
        page = en_wiki.page(query)
    if not page.exists():
        return f"[NOT FOUND: 维基百科无 '{query}' 的页面]"
    
    summary = page.summary[:500]
    return f"标题: {page.title}\n摘要: {summary}\n链接: {page.fullurl}"
```

```python
# src/tools/wiki.py 的 schema
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": "查询维基百科页面的摘要。支持中英文，"
                       "输入关键词（如 '图灵机' 或 'Geoffrey Hinton'），"
                       "返回标题、摘要和链接。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "维基百科条目标题或关键词，例如 '图灵机' 或 'Transformer (machine learning model)'",
                },
            },
            "required": ["query"],
        },
    },
}
```

## 不确定 / 需验证的点
- Wikipedia API 端点是否被国内网络封禁：若被封，wiki 工具只能靠缓存 / 替代源（DBpedia、百度百科）。
- `user_agent` 校验严格程度：偶尔会因格式不对返回 403。
- 中文版和英文版 Wikipedia 标题不一致："图灵机" 中文，"Turing machine" 英文——工具内部需要尝试两个。
