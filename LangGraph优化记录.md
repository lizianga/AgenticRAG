# LangGraph 优化记录

## 一、AgentNode（路由决策节点）

### 优化前问题

1. **判断解析脆弱** — 用 `== "需要检索"` 精确匹配 LLM 输出，带标点或空格就会失败
2. **忽略对话历史** — state 有 `chat_history` 但完全没使用，多轮追问如"那充电呢？"会误判
3. **全量走 LLM** — 问候/闲聊/数学题也要调一次模型，浪费 token 和延迟
4. **分类规则粗糙** — prompt 只有简单的二分法，缺少边界 case 处理

### 优化方案

**两层分类策略：**

```
用户查询 → 第一层：正则快速分类（0ms，零 token）
              ├─ 问候/闲聊/数学 → 直接 skip
              ├─ 产品关键词命中 → 直接 retrieve
              └─ 无法确定 → 第二层：LLM 分类（结合对话历史）
```

- **规则层**：`_SKIP_PATTERNS`（问候/数学/结束语）+ `_FORCE_RETRIEVE_PATTERNS`（故障/维护/报错等关键词）
- **LLM 层**：prompt 融入 `chat_history`（最近 3 轮），支持多轮追问；返回 JSON `{"decision": "retrieve"}` 格式
- **响应解析**：`_parse_llm_decision` 先匹配 JSON，兜底关键词模糊匹配

### 测试结果：12/12 通过

| 测试组 | 用例数 | 耗时 |
|--------|--------|------|
| 规则快速分类 | 6 | ~0s |
| LLM 分类 | 2 | 6-9s |
| 多轮对话上下文 | 2 | 0-15s |
| 边界情况 | 2 | ~0s |

---

## 二、RelevanceNode（相关性评估节点）

### 优化前问题

1. **用 Embedding 余弦相似度做评估，精度差** — 项目有 bge-reranker 但没用上
2. **取平均相似度** — 一个高度相关的 doc 被其他不相关 doc 拉低分数
3. **逐个 embed 调用** — k=3 就是 4 次 API 调用（1 query + 3 docs），未批量化
4. **无异常处理和日志**

### 优化方案

**三层评分架构：**

```
检索结果 → 第一选择：外部 Reranker API（bge-reranker-v2-m3）
              └─ 不可用 → 回退：Embedding 余弦相似度(30%) + 关键词重叠率(70%)
```

- **主路径**：调用 `http://1.15.95.222:6405/v1/rerank`，取最高 relevance_score
- **回退路径**：`embed_documents` 批量编码 + 中文 2-gram 关键词重叠率加权
  - 关键词重叠率是区分相关/不相关的核心信号（"制作蛋糕"与"扫地机器人"零重叠 → 分数被压低）
- **双阈值**：Reranker 在线用 0.5，回退用 0.45（适配混合评分分布）
- **懒探测**：`_api_ok` 状态缓存，首次失败后不再重试

### 测试结果：6/6 通过

| 用例 | 分数 | 阈值 | 结果 |
|------|------|------|------|
| 充电故障-多文档 | 0.6826 | 0.45 | PASS |
| 重写查询-相关 | 0.7916 | 0.45 | PASS |
| 滤网清洗-强相关 | 0.4656 | 0.45 | PASS |
| 制作蛋糕-不相关 | 0.2017 | 0.45 | PASS |
| Python文件-跨领域 | 0.0831 | 0.45 | PASS |
| 无文档-边界 | 0.0000 | 0.45 | PASS |

---

## 三、State（节点间状态设计）

### 优化前问题

| 问题 | 说明 |
|------|------|
| `relevance_score` 死字段 | RelevanceNode 写入后无人读取 |
| `chat_history` 与记忆系统断裂 | AgentNode 读 state 的 chat_history，GenerateNode 完全不用，只走 SessionManager |
| `max_rewrite_count` 误入 state | 运行时不变的配置常量，每个调用方都要手动写 |
| `state.copy()` 浅拷贝隐患 | `retrieved_docs` 等 list 浅拷贝后新旧 state 共享引用 |
| graph.py `messages` bug | `generate_message` 函数读 `messages` 字段，但 state 定义的是 `response` |
| 无错误状态 | 任何节点异常直接崩溃，无法优雅降级 |
| 类型定义松散 | `chat_history: List[Dict[str, str]]` 无 key 约束 |

### 优化方案

#### 1. State 结构重设计

```python
class ChatMessage(TypedDict):
    role: str
    content: str

class RagState(TypedDict):
    # --- 输入 ---
    query: str
    session_id: str
    chat_history: List[ChatMessage]       # 替代 List[Dict[str, str]]

    # --- 流程控制 ---
    needs_retrieval: bool
    is_relevant: bool
    rewrite_count: int

    # --- 检索管线 ---
    rewritten_query: str
    retrieved_docs: List[Document]
    relevance_score: float

    # --- 输出 ---
    response: str
    error: str                            # 新增：错误传递
```

变更：
- 新增 `ChatMessage` 类型，约束 `role` + `content`
- 新增 `error` 字段
- 移除 `max_rewrite_count`（提升为 `RagGraph.MAX_REWRITE_COUNT` 类常量）

#### 2. 消除 `state.copy()` 反模式

**之前（每个节点）：**
```python
updated_state = state.copy()           # 浅拷贝，list 引用共享
updated_state["needs_retrieval"] = True
return updated_state                    # 返回完整 state
```

**之后：**
```python
return {"needs_retrieval": True}        # 只返回修改的字段，LangGraph 自动合并
```

#### 3. GenerateNode 融合 chat_history

**之前：** GenerateNode 完全不读 `state["chat_history"]`，只用 SessionManager。

**之后：**
- RAG 回答：`_build_rag_context()` 将检索文档 + 对话历史一起作为上下文
- 直接回答：`_direct_answer()` 将对话历史拼入 prompt
- 异常捕获：生成失败写入 `error` 字段而非崩溃

#### 4. graph.py 修复

- `generate_message()` 函数：`value.get("messages")` → `value.get("response")`
- `max_rewrite_count` → `RagGraph.MAX_REWRITE_COUNT = 2`
- 条件路由用 `state.get()` 防御空值

---

## 四、文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `agent/langgraph/state.py` | 重写 | 新增 ChatMessage、error 字段，移除 max_rewrite_count |
| `agent/langgraph/graph.py` | 重写 | 修复 messages bug，max_rewrite_count 提升为类常量 |
| `agent/langgraph/nodes/agent_node.py` | 优化 | 两层分类、chat_history 融入、消除 state.copy() |
| `agent/langgraph/nodes/relevance_node.py` | 优化 | 外部 Reranker API + embedding 回退、双阈值、消除 state.copy() |
| `agent/langgraph/nodes/retrieve_node.py` | 重构 | 消除 state.copy()，返回 partial dict |
| `agent/langgraph/nodes/rewrite_node.py` | 重构 | 消除 state.copy()，返回 partial dict |
| `agent/langgraph/nodes/generate_node.py` | 重构 | 融合 chat_history、error 处理、消除 state.copy() |
