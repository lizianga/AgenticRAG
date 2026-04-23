import os
import sys

# 计算并添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 修正项目根目录，应该是E:\code\AgenticRAG而不是E:\code\AgenticRAG\agent
project_root = os.path.dirname(project_root)
sys.path.append(project_root)

from typing import Dict, Any
from agent.langgraph.state import RagState
from model.factory import chat_model
from langchain_core.prompts import PromptTemplate


class AgentNode:
    """Agent节点，用于判断是否需要检索文档"""
    
    def __init__(self):
        # 定义判断是否需要检索的提示词
        self.prompt_template = PromptTemplate(
            template="""你是一个决策助手，需要判断用户的查询是否需要从知识库中检索信息。

用户查询: {query}

请分析上述查询，判断是否需要从知识库中检索信息：
- 如果查询是关于扫地机器人的专业问题（如故障排除、维护保养、选购指南等），需要检索
- 如果查询是常识性问题、简单指令或与扫地机器人无关的问题，不需要检索

请只返回"需要检索"或"不需要检索"，不要返回其他内容。""",
            input_variables=["query"]
        )
        self.model = chat_model
    
    def should_retrieve(self, state: RagState) -> RagState:
        """判断是否需要检索
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        # 构建提示词
        prompt = self.prompt_template.format(query=state["query"])
        print("query", state["query"])
        # 调用模型判断是否需要检索
        response = self.model.invoke(prompt)
        print("Agent_node:",response.content)
        # 解析模型响应
        # 确保只有当响应完全等于"需要检索"时才返回True
        needs_retrieval = response.content.strip() == "需要检索"
        print("-----------已判断是否需要检索：", needs_retrieval)
        # 更新状态
        updated_state = state.copy()
        updated_state["needs_retrieval"] = needs_retrieval
        
        return updated_state


if __name__ == "__main__":
    """测试AgentNode类"""
    from agent.langgraph.state import RagState
    
    # 创建AgentNode实例
    agent_node = AgentNode()
    
    # 测试用例1：关于扫地机器人的专业问题
    print("测试用例1：关于扫地机器人的专业问题")
    state1: RagState = {
        "query": "扫地机器人无法连接wifi了，怎么办？",
        "rewritten_query": "",
        "retrieved_docs": [],
        "relevance_score": 0.0,
        "response": "",
        "chat_history": [],
        "needs_retrieval": False,
        "is_relevant": False,
        "rewrite_count": 0,
        "max_rewrite_count": 2
    }
    result1 = agent_node.should_retrieve(state1)
    print(f"查询: {result1['query']}")
    print(result1)
    print(f"是否需要检索: {result1['needs_retrieval']}")
    print()
    
    # # 测试用例2：与扫地机器人无关的问题
    # print("测试用例2：与扫地机器人无关的问题")
    # state2: RagState = {
    #     "query": "今天天气怎么样？",
    #     "rewritten_query": "",
    #     "retrieved_docs": [],
    #     "relevance_score": 0.0,
    #     "response": "",
    #     "chat_history": [],
    #     "needs_retrieval": False,
    #     "is_relevant": False,
    #     "rewrite_count": 0,
    #     "max_rewrite_count": 2
    # }
    # result2 = agent_node.should_retrieve(state2)
    # print(f"查询: {result2['query']}")
    # print(f"是否需要检索: {result2['needs_retrieval']}")
    # print()
    
    # # 测试用例3：扫地机器人的维护保养问题
    # print("测试用例3：扫地机器人的维护保养问题")
    # state3: RagState = {
    #     "query": "如何清洁扫地机器人的滤网？",
    #     "rewritten_query": "",
    #     "retrieved_docs": [],
    #     "relevance_score": 0.0,
    #     "response": "",
    #     "chat_history": [],
    #     "needs_retrieval": False,
    #     "is_relevant": False,
    #     "rewrite_count": 0,
    #     "max_rewrite_count": 2
    # }
    # result3 = agent_node.should_retrieve(state3)
    # print(f"查询: {result3['query']}")
    # print(f"是否需要检索: {result3['needs_retrieval']}")
    # print()
    
    # # 测试用例4：常识性问题
    # print("测试用例4：常识性问题")
    # state4: RagState = {
    #     "query": "1+1等于多少？",
    #     "rewritten_query": "",
    #     "retrieved_docs": [],
    #     "relevance_score": 0.0,
    #     "response": "",
    #     "chat_history": [],
    #     "needs_retrieval": False,
    #     "is_relevant": False,
    #     "rewrite_count": 0,
    #     "max_rewrite_count": 2
    # }
    # result4 = agent_node.should_retrieve(state4)
    # print(f"查询: {result4['query']}")
    # print(f"是否需要检索: {result4['needs_retrieval']}")

