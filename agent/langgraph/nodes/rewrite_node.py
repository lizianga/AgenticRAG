import logging
from agent.langgraph.state import RagState
from model.factory import chat_model
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class RewriteNode:
    """Rewrite节点，用于重写查询"""

    def __init__(self):
        self.prompt_template = PromptTemplate(
            template="""你是一个查询重写助手，需要根据原始查询和检索结果，生成一个更精确的查询，以便检索到更相关的文档。

原始查询: {query}

检索结果:
{retrieved_docs}

请分析原始查询和检索结果，然后生成一个更精确的查询，使得新的查询能够检索到更相关的文档。

要求：
1. 新查询应该更具体，包含更多关键词
2. 新查询应该明确表达用户的意图
3. 新查询应该避免模糊不清的表述
4. 只返回重写后的查询，不要返回其他内容
""",
            input_variables=["query", "retrieved_docs"]
        )
        self.model = chat_model

    def rewrite_query(self, state: RagState) -> dict:
        """重写查询"""
        # 构建检索结果文本
        retrieved_docs_text = ""
        for i, doc in enumerate(state.get("retrieved_docs", [])[:3]):
            retrieved_docs_text += f"[{i+1}] {doc.page_content[:200]}...\n"

        query = state["query"]
        prompt = self.prompt_template.format(
            query=query,
            retrieved_docs=retrieved_docs_text
        )

        response = self.model.invoke(prompt)
        rewritten_query = response.content.strip()
        new_count = state.get("rewrite_count", 0) + 1

        logger.info("RewriteNode 原始查询: %s → 重写: %s (第%d次)", query, rewritten_query, new_count)

        return {"rewritten_query": rewritten_query, "rewrite_count": new_count}
