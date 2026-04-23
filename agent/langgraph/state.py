from typing import List, Dict, Any, TypedDict, Sequence
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RagState(TypedDict):
    """RAG系统状态定义"""
    # 用户原始查询
    query: str
    # 重写后的查询
    rewritten_query: str
    # 检索到的文档
    retrieved_docs: List[Document]
    # 相关性评分
    relevance_score: float
    # 生成的回答
    response: str
    # 对话历史
    chat_history: List[Dict[str, str]]
    # 是否需要检索的标志
    needs_retrieval: bool
    # 检索结果是否相关的标志
    is_relevant: bool
    # 重写次数
    rewrite_count: int
    # 最大重写次数
    max_rewrite_count: int
    # 会话ID
    session_id: str