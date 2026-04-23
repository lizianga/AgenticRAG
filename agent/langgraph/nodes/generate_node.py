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
from model.factory import chat_model
from langchain_core.prompts import PromptTemplate
from agent.memory.session_manager import SessionManager
from agent.memory.short_term_memory import ShortTermMemory
from agent.memory.long_term_memory import LongTermMemory


class GenerateNode:
    """Generate节点，用于生成最终回答"""
    
    def __init__(self):
        # 初始化RAG服务
        self.rag_service = RagSummariceService()
        # 定义直接回答的提示词（不需要检索时使用）
        self.direct_prompt_template = PromptTemplate(
            template="""你是一个智能助手，请直接回答用户的问题，不需要从知识库中检索信息。

用户查询: {query}

请直接回答用户的问题，不需要提及知识库或检索相关的内容。""",
            input_variables=["query"]
        )
        # 初始化记忆管理
        self.session_manager = SessionManager()
        self.short_term_memory = ShortTermMemory(self.session_manager)
        self.long_term_memory = LongTermMemory()
        self.model = chat_model
    
    def generate_response(self, state: RagState) -> RagState:
        """生成最终回答
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        print("正在生成--------")
        
        # 获取会话ID
        session_id = state.get("session_id", "")
        if not session_id:
            # 如果没有会话ID，创建一个新的
            session_id = self.session_manager.create_session()
        
        # 获取短期记忆作为上下文
        short_term_context = self.short_term_memory.format_memory_for_context(session_id)
        
        # 如果需要检索且有相关文档，使用状态中的检索结果生成回答
        if state["needs_retrieval"] and state["retrieved_docs"]:
            # 确定使用哪个查询（重写后的查询或原始查询）
            query = state["rewritten_query"] if state["rewritten_query"] else state["query"]
            
            # 构建上下文
            context = ""
            counter = 0
            for doc in state["retrieved_docs"]:
                counter += 1
                context += f"【参考资料{counter}】：{doc.page_content} | 参考元数据：{doc.metadata}\n"
            
            # 添加短期记忆到上下文
            if short_term_context:
                context += "\n" + short_term_context
            
            # 使用RAG服务的chain生成回答
            response = self.rag_service.chain.invoke(
                {
                    "input": query,
                    "context": context
                }
            )
        else:
            # 不需要检索，直接使用模型回答
            if short_term_context:
                # 如果有短期记忆，将其添加到提示中
                prompt = f"""你是一个智能助手，请直接回答用户的问题，不需要从知识库中检索信息。

{short_term_context}
用户查询: {state['query']}

请直接回答用户的问题，不需要提及知识库或检索相关的内容。"""
            else:
                prompt = self.direct_prompt_template.format(query=state["query"])
            response = self.model.invoke(prompt).content
        
        # 更新短期记忆
        self.short_term_memory.add_memory(session_id, state["query"], response)
        
        # 更新长期记忆（每3轮对话保存一次）
        chat_history = self.short_term_memory.get_memory(session_id)
        if chat_history and len(chat_history) % 3 == 0:
            self.long_term_memory.add_memory(session_id, state["query"], response)
        
        # 更新状态
        updated_state = state.copy()
        updated_state["response"] = response
        updated_state["session_id"] = session_id
        
        return updated_state


if __name__ == "__main__":
    """测试GenerateNode类"""
    from agent.langgraph.state import RagState
    from langchain_core.documents import Document
    
    # 创建GenerateNode实例
    generate_node = GenerateNode()
    
    # 测试用例1：需要检索且有相关文档
    print("测试用例1：需要检索且有相关文档")
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
        "is_relevant": True,
        "rewrite_count": 0,
        "max_rewrite_count": 2,
        "session_id": ""
    }
    result1 = generate_node.generate_response(state1)
    print(f"查询: {result1['query']}")
    print(f"是否需要检索: {result1['needs_retrieval']}")
    print(f"检索到的文档数量: {len(result1['retrieved_docs'])}")
    print(f"生成的回答: {result1['response']}")
    print(f"会话ID: {result1['session_id']}")
    print()
    
    # 测试用例2：不需要检索
    print("测试用例2：不需要检索")
    state2: RagState = {
        "query": "今天天气怎么样？",
        "rewritten_query": "",
        "retrieved_docs": [],
        "relevance_score": 0.0,
        "response": "",
        "chat_history": [],
        "needs_retrieval": False,
        "is_relevant": False,
        "rewrite_count": 0,
        "max_rewrite_count": 2,
        "session_id": ""
    }
    result2 = generate_node.generate_response(state2)
    print(f"查询: {result2['query']}")
    print(f"是否需要检索: {result2['needs_retrieval']}")
    print(f"生成的回答: {result2['response']}")
    print(f"会话ID: {result2['session_id']}")
    print()
    
    # 测试用例3：需要检索但无相关文档
    print("测试用例3：需要检索但无相关文档")
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
        "max_rewrite_count": 2,
        "session_id": ""
    }
    result3 = generate_node.generate_response(state3)
    print(f"查询: {result3['query']}")
    print(f"是否需要检索: {result3['needs_retrieval']}")
    print(f"检索到的文档数量: {len(result3['retrieved_docs'])}")
    print(f"生成的回答: {result3['response']}")
    print(f"会话ID: {result3['session_id']}")