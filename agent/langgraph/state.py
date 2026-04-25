from typing import List, TypedDict
from langchain_core.documents import Document


class ChatMessage(TypedDict):
    role: str
    content: str


class RagState(TypedDict):
    """RAG 系统节点间传递的状态

    字段分类：
    - 输入：query, session_id, chat_history（由调用方设置）
    - 流程控制：needs_retrieval, is_relevant, rewrite_count（由节点写入，路由逻辑读取）
    - 检索管线：rewritten_query, retrieved_docs, relevance_score
    - 输出：response, error
    """
    # --- 输入 ---
    query: str
    session_id: str
    chat_history: List[ChatMessage]

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
    error: str
