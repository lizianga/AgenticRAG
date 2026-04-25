import os
import sys
import logging

# 计算并添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(project_root)
sys.path.append(project_root)

from agent.langgraph.state import RagState
from rag.rag_service import RagSummariceService

logger = logging.getLogger(__name__)


class RetrieveNode:
    """Retrieve节点，用于从向量存储中检索文档"""
    
    def __init__(self):
        # 初始化RAG服务
        self.rag_service = RagSummariceService()
    
    def retrieve_documents(self, state: RagState) -> dict:
        """检索文档"""
        query = state.get("rewritten_query") or state["query"]
        logger.info("RetrieveNode 使用查询: %s", query)
        retrieved_docs = self.rag_service.retriver_docs(query)
        logger.info("RetrieveNode 检索到 %d 个文档", len(retrieved_docs))
        return {"retrieved_docs": retrieved_docs}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    retrieve_node = RetrieveNode()

    print("测试用例1：使用原始查询检索")
    state1 = {
        "query": "扫地机器人无法充电怎么办？",
        "session_id": "", "chat_history": [], "needs_retrieval": True,
        "is_relevant": False, "rewrite_count": 0, "rewritten_query": "",
        "retrieved_docs": [], "relevance_score": 0.0, "response": "", "error": "",
    }
    result1 = retrieve_node.retrieve_documents(state1)
    print(f"检索到 {len(result1['retrieved_docs'])} 个文档\n")

    print("测试用例2：使用重写后的查询检索")
    state2 = {
        "query": "扫地机器人问题",
        "session_id": "", "chat_history": [], "needs_retrieval": True,
        "is_relevant": False, "rewrite_count": 0,
        "rewritten_query": "扫地机器人常见故障及解决方法",
        "retrieved_docs": [], "relevance_score": 0.0, "response": "", "error": "",
    }
    result2 = retrieve_node.retrieve_documents(state2)
    print(f"检索到 {len(result2['retrieved_docs'])} 个文档")