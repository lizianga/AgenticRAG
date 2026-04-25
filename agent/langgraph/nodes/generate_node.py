import os
import sys
import logging

# 计算并添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(project_root)
sys.path.append(project_root)

from agent.langgraph.state import RagState
from rag.rag_service import RagSummariceService
from model.factory import chat_model
from langchain_core.prompts import PromptTemplate
from agent.memory.session_manager import SessionManager
from agent.memory.short_term_memory import ShortTermMemory
from agent.memory.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)


class GenerateNode:
    """Generate节点，用于生成最终回答"""

    def __init__(self):
        self.rag_service = RagSummariceService()
        self.model = chat_model
        self.session_manager = SessionManager()
        self.short_term_memory = ShortTermMemory(self.session_manager)
        self.long_term_memory = LongTermMemory()

    def generate_response(self, state: RagState) -> dict:
        """生成最终回答"""
        logger.info("GenerateNode 开始生成回答")

        try:
            response = self._do_generate(state)
            self._update_memory(state, response)
            return {"response": response, "session_id": state.get("session_id", ""), "error": ""}
        except Exception as e:
            logger.error("GenerateNode 生成失败: %s", e)
            return {"response": "", "error": str(e)}

    def _do_generate(self, state: RagState) -> str:
        """核心生成逻辑"""
        query = state["query"]
        chat_history = state.get("chat_history", [])

        # 需要检索且有文档 → RAG 回答
        if state.get("needs_retrieval") and state.get("retrieved_docs"):
            retrieve_query = state.get("rewritten_query") or query
            context = self._build_rag_context(state["retrieved_docs"], chat_history)
            return self.rag_service.chain.invoke({"input": retrieve_query, "context": context})

        # 不需要检索 → 直接回答（结合对话历史）
        return self._direct_answer(query, chat_history)

    def _build_rag_context(self, docs: list, chat_history: list) -> str:
        """构建 RAG 上下文：检索文档 + 对话历史"""
        context = ""
        for i, doc in enumerate(docs, 1):
            context += f"【参考资料{i}】：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        history_text = self._format_chat_history(chat_history)
        if history_text:
            context += f"\n对话历史:\n{history_text}"

        return context

    def _direct_answer(self, query: str, chat_history: list) -> str:
        """直接回答（不检索），结合对话历史"""
        history_text = self._format_chat_history(chat_history)
        if history_text:
            prompt = (
                "你是一个智能助手，请结合对话历史直接回答用户的问题，"
                "不需要从知识库中检索信息，也不需要提及知识库或检索。\n\n"
                f"对话历史:\n{history_text}\n\n"
                f"当前用户查询: {query}\n\n"
                "请直接回答："
            )
        else:
            prompt = (
                "你是一个智能助手，请直接回答用户的问题，"
                "不需要从知识库中检索信息，也不需要提及知识库或检索。\n\n"
                f"用户查询: {query}\n\n"
                "请直接回答："
            )
        return self.model.invoke(prompt).content

    def _format_chat_history(self, chat_history: list) -> str:
        """将对话历史格式化为文本"""
        if not chat_history:
            return ""
        lines = []
        for msg in chat_history[-6:]:  # 最近3轮
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _update_memory(self, state: RagState, response: str):
        """更新短期和长期记忆"""
        session_id = state.get("session_id", "")
        if not session_id:
            session_id = self.session_manager.create_session()
        state["session_id"] = session_id

        query = state["query"]
        self.short_term_memory.add_memory(session_id, query, response)

        chat_history = self.short_term_memory.get_memory(session_id)
        if chat_history and len(chat_history) % 3 == 0:
            self.long_term_memory.add_memory(session_id, query, response)
