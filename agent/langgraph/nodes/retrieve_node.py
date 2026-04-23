import os
import sys

# 计算并添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 修正项目根目录，应该是E:\code\AgenticRAG而不是E:\code\AgenticRAG\agent
project_root = os.path.dirname(project_root)
sys.path.append(project_root)

from typing import Dict, Any
from agent.langgraph.state import RagState
from rag.rag_service import RagSummariceService


class RetrieveNode:
    """Retrieve节点，用于从向量存储中检索文档"""
    
    def __init__(self):
        # 初始化RAG服务
        self.rag_service = RagSummariceService()
    
    def retrieve_documents(self, state: RagState) -> RagState:
        """检索文档
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        # 确定使用哪个查询（重写后的查询或原始查询）
        query = state["rewritten_query"] if state["rewritten_query"] else state["query"]
        print("使用的query",query)
        # 检索文档
        retrieved_docs = self.rag_service.retriver_docs(query)
        print("检索到的文档数量", len(retrieved_docs))
        # 更新状态
        updated_state = state.copy()
        updated_state["retrieved_docs"] = retrieved_docs
        
        return updated_state


if __name__ == "__main__":
    """测试RetrieveNode类"""
    from agent.langgraph.state import RagState
    
    # 创建RetrieveNode实例
    retrieve_node = RetrieveNode()
    
    # 测试用例1：使用原始查询检索
    print("测试用例1：使用原始查询检索")
    state1: RagState = {
        "query": "扫地机器人无法充电怎么办？",
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
    result1 = retrieve_node.retrieve_documents(state1)
    print(f"查询: {result1['query']}")
    print(f"检索到的文档数量: {len(result1['retrieved_docs'])}")
    if result1['retrieved_docs']:
        print(f"第一个文档内容: {result1['retrieved_docs'][0]['page_content'][:100]}...")
    print()
    
    # 测试用例2：使用重写后的查询检索
    print("测试用例2：使用重写后的查询检索")
    state2: RagState = {
        "query": "扫地机器人问题",
        "rewritten_query": "扫地机器人常见故障及解决方法",
        "retrieved_docs": [],
        "relevance_score": 0.0,
        "response": "",
        "chat_history": [],
        "needs_retrieval": True,
        "is_relevant": False,
        "rewrite_count": 0,
        "max_rewrite_count": 2
    }
    result2 = retrieve_node.retrieve_documents(state2)
    print(f"原始查询: {result2['query']}")
    print(f"重写后的查询: {result2['rewritten_query']}")
    print(f"检索到的文档数量: {len(result2['retrieved_docs'])}")
    if result2['retrieved_docs']:
        print(f"第一个文档内容: {result2['retrieved_docs'][0]['page_content'][:100]}...")
    print()
    
    # # 测试用例3：空查询
    # print("测试用例3：空查询")
    # state3: RagState = {
    #     "query": "",
    #     "rewritten_query": "",
    #     "retrieved_docs": [],
    #     "relevance_score": 0.0,
    #     "response": "",
    #     "chat_history": [],
    #     "needs_retrieval": True,
    #     "is_relevant": False,
    #     "rewrite_count": 0,
    #     "max_rewrite_count": 2
    # }
    # result3 = retrieve_node.retrieve_documents(state3)
    # print(f"查询: '{result3['query']}'")
    # print(f"检索到的文档数量: {len(result3['retrieved_docs'])}")
    # print()