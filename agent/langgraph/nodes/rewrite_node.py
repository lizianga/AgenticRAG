from typing import Dict, Any
from agent.langgraph.state import RagState
from model.factory import chat_model
from langchain_core.prompts import PromptTemplate


class RewriteNode:
    """Rewrite节点，用于重写查询"""
    
    def __init__(self):
        # 定义查询重写的提示词
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
    
    def rewrite_query(self, state: RagState) -> RagState:
        """重写查询
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        # 构建检索结果文本
        retrieved_docs_text = ""
        for i, doc in enumerate(state["retrieved_docs"][:3]):  # 只使用前3个文档
            retrieved_docs_text += f"[{i+1}] {doc.page_content[:200]}...\n"
        
        # 构建提示词
        prompt = self.prompt_template.format(
            query=state["query"],
            retrieved_docs=retrieved_docs_text
        )
        
        # 调用模型重写查询
        response = self.model.invoke(prompt)
        
        # 解析模型响应
        rewritten_query = response.content.strip()
        
        # 更新状态
        updated_state = state.copy()
        updated_state["rewritten_query"] = rewritten_query
        updated_state["rewrite_count"] = state["rewrite_count"] + 1
        
        return updated_state