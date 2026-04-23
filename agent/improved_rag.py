import sys
import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model.factory import chat_model, embed_model
from rag.rag_service import RagSummariceService
import numpy as np


class ImprovedRAG:
    """改进的RAG系统"""
    
    def __init__(self):
        # 初始化RAG服务
        self.rag_service = RagSummariceService()
        # 初始化模型
        self.chat_model = chat_model
        self.embed_model = embed_model
        # 相关性阈值
        self.relevance_threshold = 0.7
        # 最大重写次数
        self.max_rewrite_count = 2
        
        # 定义判断是否需要检索的提示词
        self.retrieve_prompt = PromptTemplate(
            template="""你是一个决策助手，需要判断用户的查询是否需要从知识库中检索信息。

用户查询: {query}

请分析上述查询，判断是否需要从知识库中检索信息：
- 如果查询是关于扫地机器人的专业问题（如故障排除、维护保养、选购指南等），需要检索
- 如果查询是常识性问题、简单指令或与扫地机器人无关的问题，不需要检索

请只返回"需要检索"或"不需要检索"，不要返回其他内容。""",
            input_variables=["query"]
        )
        
        # 定义查询重写的提示词
        self.rewrite_prompt = PromptTemplate(
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
        
        # 定义直接回答的提示词
        self.direct_prompt = PromptTemplate(
            template="""你是一个智能助手，请直接回答用户的问题，不需要从知识库中检索信息。

用户查询: {query}

请直接回答用户的问题，不需要提及知识库或检索相关的内容。""",
            input_variables=["query"]
        )
    
    def _should_retrieve(self, query: str) -> bool:
        """判断是否需要检索
        
        Args:
            query: 用户查询
            
        Returns:
            是否需要检索
        """
        # 构建提示词
        prompt = self.retrieve_prompt.format(query=query)
        
        # 调用模型判断是否需要检索
        response = self.chat_model.invoke(prompt)
        
        # 解析模型响应
        return "需要检索" in response.content
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """检索文档
        
        Args:
            query: 查询
            
        Returns:
            检索到的文档
        """
        return self.rag_service.retriver_docs(query)
    
    def _evaluate_relevance(self, query: str, retrieved_docs: List[Document]) -> float:
        """评估检索结果与查询的相关性
        
        Args:
            query: 查询
            retrieved_docs: 检索到的文档
            
        Returns:
            相关性评分
        """
        # 如果没有检索到文档，直接返回0
        if not retrieved_docs:
            return 0.0
        
        # 计算查询的嵌入向量
        query_embedding = self.embed_model.embed_query(query)
        
        # 计算文档与查询的相似度
        similarities = []
        for doc in retrieved_docs:
            # 计算文档内容的嵌入向量
            doc_embedding = self.embed_model.embed_query(doc.page_content)
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # 计算平均相似度作为相关性评分
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0
    
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """计算两个向量的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            余弦相似度
        """
        # 计算点积
        dot_product = np.dot(vec1, vec2)
        
        # 计算向量长度
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # 计算余弦相似度
        if norm_vec1 > 0 and norm_vec2 > 0:
            return dot_product / (norm_vec1 * norm_vec2)
        else:
            return 0.0
    
    def _rewrite_query(self, query: str, retrieved_docs: List[Document]) -> str:
        """重写查询
        
        Args:
            query: 原始查询
            retrieved_docs: 检索到的文档
            
        Returns:
            重写后的查询
        """
        # 构建检索结果文本
        retrieved_docs_text = ""
        for i, doc in enumerate(retrieved_docs[:3]):  # 只使用前3个文档
            retrieved_docs_text += f"[{i+1}] {doc.page_content[:200]}...\n"
        
        # 构建提示词
        prompt = self.rewrite_prompt.format(
            query=query,
            retrieved_docs=retrieved_docs_text
        )
        
        # 调用模型重写查询
        response = self.chat_model.invoke(prompt)
        
        # 解析模型响应
        return response.content.strip()
    
    def _generate_response(self, query: str, retrieved_docs: List[Document], needs_retrieval: bool) -> str:
        """生成最终回答
        
        Args:
            query: 查询
            retrieved_docs: 检索到的文档
            needs_retrieval: 是否需要检索
            
        Returns:
            生成的回答
        """
        # 如果需要检索且有相关文档，使用RAG服务生成回答
        if needs_retrieval and retrieved_docs:
            # 使用RAG服务生成回答
            return self.rag_service.rag_summarize(query)
        else:
            # 不需要检索，直接使用模型回答
            prompt = self.direct_prompt.format(query=query)
            return self.chat_model.invoke(prompt).content
    
    def run(self, query: str) -> Dict[str, Any]:
        """运行RAG系统
        
        Args:
            query: 用户查询
            
        Returns:
            包含结果的字典
        """
        # 初始化结果
        result = {
            "query": query,
            "rewritten_query": "",
            "retrieved_docs": [],
            "relevance_score": 0.0,
            "response": "",
            "needs_retrieval": False,
            "is_relevant": False,
            "rewrite_count": 0
        }
        
        # 判断是否需要检索
        needs_retrieval = self._should_retrieve(query)
        result["needs_retrieval"] = needs_retrieval
        
        if needs_retrieval:
            # 初始化重写次数
            rewrite_count = 0
            current_query = query
            
            while rewrite_count <= self.max_rewrite_count:
                # 检索文档
                retrieved_docs = self._retrieve_documents(current_query)
                
                # 评估相关性
                relevance_score = self._evaluate_relevance(current_query, retrieved_docs)
                is_relevant = relevance_score >= self.relevance_threshold
                
                # 更新结果
                result["retrieved_docs"] = retrieved_docs
                result["relevance_score"] = relevance_score
                result["is_relevant"] = is_relevant
                result["rewrite_count"] = rewrite_count
                
                if is_relevant or rewrite_count >= self.max_rewrite_count:
                    # 生成回答
                    response = self._generate_response(current_query, retrieved_docs, needs_retrieval)
                    result["response"] = response
                    break
                else:
                    # 重写查询
                    rewritten_query = self._rewrite_query(current_query, retrieved_docs)
                    result["rewritten_query"] = rewritten_query
                    current_query = rewritten_query
                    rewrite_count += 1
        else:
            # 不需要检索，直接生成回答
            response = self._generate_response(query, [], needs_retrieval)
            result["response"] = response
        
        return result