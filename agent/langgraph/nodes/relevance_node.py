import os
import sys

# 计算并添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 修正项目根目录，应该是E:\code\AgenticRAG而不是E:\code\AgenticRAG\agent
project_root = os.path.dirname(project_root)
sys.path.append(project_root)

from typing import Dict, Any
from agent.langgraph.state import RagState
from model.factory import embed_model
import numpy as np


class RelevanceNode:
    """Relevance节点，用于评估检索结果与查询的相关性"""
    
    def __init__(self):
        # 初始化嵌入模型
        self.embed_model = embed_model
        # 相关性阈值
        self.relevance_threshold = 0.7
    
    def evaluate_relevance(self, state: RagState) -> RagState:
        """评估检索结果与查询的相关性
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        print("正在判断是否相关。。。。。。。。。")
        # 如果没有检索到文档，直接返回不相关
        if not state["retrieved_docs"]:
            updated_state = state.copy()
            updated_state["is_relevant"] = False
            updated_state["relevance_score"] = 0.0
            return updated_state
        
        # 确定使用哪个查询（重写后的查询或原始查询）
        query = state["rewritten_query"] if state["rewritten_query"] else state["query"]
        
        # 计算查询的嵌入向量
        query_embedding = self.embed_model.embed_query(query)
        
        # 计算文档与查询的相似度
        similarities = []
        for doc in state["retrieved_docs"]:
            # 计算文档内容的嵌入向量
            doc_embedding = self.embed_model.embed_query(doc.page_content)
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # 计算平均相似度作为相关性评分
        updated_state = state.copy()
        if similarities:
            updated_state["relevance_score"] = np.mean(similarities)
        else:
            updated_state["relevance_score"] = 0.0
        
        # 判断是否相关
        updated_state["is_relevant"] = updated_state["relevance_score"] >= self.relevance_threshold
        
        return updated_state
    
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


if __name__ == "__main__":
    """测试RelevanceNode类"""
    from agent.langgraph.state import RagState
    from langchain_core.documents import Document
    
    # 创建RelevanceNode实例
    relevance_node = RelevanceNode()
    
    # 测试用例1：有检索文档且相关
    print("测试用例1：有检索文档且相关")
    state1: RagState = {
        "query": "扫地机器人无法充电怎么办？",
        "rewritten_query": "",
        "retrieved_docs": [
            Document(
                page_content="扫地机器人无法充电可能是因为电池问题、充电器故障或接触不良。建议检查电池是否损坏，充电器是否正常工作，以及充电接口是否干净。",
                metadata={"source": "故障排除.txt"}
            ),
            Document(
                page_content="扫地机器人的电池寿命一般为1-2年，使用不当会缩短电池寿命。建议每次使用后及时充电，避免过度放电。",
                metadata={"source": "维护保养.txt"}
            )
        ],
        "relevance_score": 0.0,
        "response": "",
        "chat_history": [],
        "needs_retrieval": True,
        "is_relevant": False,
        "rewrite_count": 0,
        "max_rewrite_count": 2
    }
    result1 = relevance_node.evaluate_relevance(state1)
    print(f"查询: {result1['query']}")
    print(f"检索到的文档数量: {len(result1['retrieved_docs'])}")
    print(f"相关性评分: {result1['relevance_score']:.4f}")
    print(f"是否相关: {result1['is_relevant']}")
    print()
    
    # 测试用例2：有检索文档但不相关
    print("测试用例2：有检索文档但不相关")
    state2: RagState = {
        "query": "如何制作蛋糕？",
        "rewritten_query": "",
        "retrieved_docs": [
            Document(
                page_content="扫地机器人无法充电可能是因为电池问题、充电器故障或接触不良。",
                metadata={"source": "故障排除.txt"}
            )
        ],
        "relevance_score": 0.0,
        "response": "",
        "chat_history": [],
        "needs_retrieval": True,
        "is_relevant": False,
        "rewrite_count": 0,
        "max_rewrite_count": 2
    }
    result2 = relevance_node.evaluate_relevance(state2)
    print(f"查询: {result2['query']}")
    print(f"检索到的文档数量: {len(result2['retrieved_docs'])}")
    print(f"相关性评分: {result2['relevance_score']:.4f}")
    print(f"是否相关: {result2['is_relevant']}")
    print()
    
    # 测试用例3：无检索文档
    print("测试用例3：无检索文档")
    state3: RagState = {
        "query": "扫地机器人如何维护？",
        "rewritten_query": "",
        "retrieved_docs": [],
        "relevance_score": 0.0,
        "response": "",
        "chat_history": [],
        "needs_retrieval": True,
        "is_relevant": False,
        "rewrite_count": 0,
        "max_rewrite_count": 2
    }
    result3 = relevance_node.evaluate_relevance(state3)
    print(f"查询: {result3['query']}")
    print(f"检索到的文档数量: {len(result3['retrieved_docs'])}")
    print(f"相关性评分: {result3['relevance_score']:.4f}")
    print(f"是否相关: {result3['is_relevant']}")
    print()
    
    # 测试用例4：使用重写后的查询
    print("测试用例4：使用重写后的查询")
    state4: RagState = {
        "query": "扫地机器人问题",
        "rewritten_query": "扫地机器人常见故障及解决方法",
        "retrieved_docs": [
            Document(
                page_content="扫地机器人常见故障包括无法充电、无法启动、清扫效果差等。针对不同故障，有相应的解决方法。",
                metadata={"source": "故障排除.txt"}
            )
        ],
        "relevance_score": 0.0,
        "response": "",
        "chat_history": [],
        "needs_retrieval": True,
        "is_relevant": False,
        "rewrite_count": 0,
        "max_rewrite_count": 2
    }
    result4 = relevance_node.evaluate_relevance(state4)
    print(f"原始查询: {result4['query']}")
    print(f"重写后的查询: {result4['rewritten_query']}")
    print(f"检索到的文档数量: {len(result4['retrieved_docs'])}")
    print(f"相关性评分: {result4['relevance_score']:.4f}")
    print(f"是否相关: {result4['is_relevant']}")