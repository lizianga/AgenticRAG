import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from langgraph.graph import StateGraph, END, START
from agent.langgraph.state import RagState
from agent.langgraph.nodes.agent_node import AgentNode
from agent.langgraph.nodes.retrieve_node import RetrieveNode
from agent.langgraph.nodes.relevance_node import RelevanceNode
from agent.langgraph.nodes.rewrite_node import RewriteNode
from agent.langgraph.nodes.generate_node import GenerateNode

# 添加当前目录到Python路径
class RagGraph:
    """RAG系统状态图"""
    
    def __init__(self):
        # 初始化各个节点
        self.agent_node = AgentNode()
        self.retrieve_node = RetrieveNode()
        self.relevance_node = RelevanceNode()
        self.rewrite_node = RewriteNode()
        self.generate_node = GenerateNode()
        
        # 构建状态图
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建状态图
        
        Returns:
            构建好的状态图
        """
        # 创建状态图，使用RagState作为状态类型
        graph = StateGraph(RagState)
        
        # 添加节点
        graph.add_node("agent", self.agent_node.should_retrieve)
        graph.add_node("retrieve", self.retrieve_node.retrieve_documents)
        graph.add_node("relevance", self.relevance_node.evaluate_relevance)
        graph.add_node("rewrite", self.rewrite_node.rewrite_query)
        graph.add_node("generate", self.generate_node.generate_response)
        
        # 设置入口节点
        graph.add_edge(START, "agent")
        # 定义节点间的流转逻辑
        
        # Agent节点的输出
        def agent_conditional(state):
            if state["needs_retrieval"]:
                return "retrieve"
            else:
                return "generate"
        
        graph.add_conditional_edges(
            "agent",
            agent_conditional,
            {
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )
        
        # Retrieve节点的输出
        graph.add_edge("retrieve", "relevance")
        
        # Relevance节点的输出
        def relevance_conditional(state):
            if state["is_relevant"] or state["rewrite_count"] >= state["max_rewrite_count"]:
                return "generate"
            else:
                return "rewrite"
        
        graph.add_conditional_edges(
            "relevance",
            relevance_conditional,
            {
                "generate": "generate",
                "rewrite": "rewrite"
            }
        )
        
        # Rewrite节点的输出
        graph.add_edge("rewrite", "retrieve")
        
        # Generate节点的输出
        graph.add_edge("generate", END)
        
        return graph
    
    def compile(self):
        """编译状态图
        
        Returns:
            编译后的状态图
        """
        return self.graph.compile()


def generate_message(graph, inputs):
    """
    从图的流式输出中提取生成的回答
    
    参数:
        graph: 编译后的LangGraph图
        inputs: 输入数据
        
    返回:
        str: 生成的回答文本
    """
    generated_message = ""

    for output in graph.stream(inputs):
        for key, value in output.items():
            if key == "generate" and isinstance(value, dict):
                generated_message = value.get("messages", [""])[0]
    
    return generated_message

if __name__ == "__main__":
    rag_graph = RagGraph()
    compiled_graph = rag_graph.compile()
    print(compiled_graph)
    # 测试案例1：需要检索的查询
    print("测试案例1：需要检索的查询")
    print("查询：扫地机器人无法连接wifi了，怎么办？")
    
    # 创建初始状态
    state1: RagState = {
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
    print(state1)
    # 执行图
    # result1 = compiled_graph.stream(state1)
    generated_message = ''
    for output in compiled_graph.stream(state1):
        for key,value in output.items():
            if key == "generate" and isinstance(value, dict):
                generated_message = value.get("response", [""])
    print(generated_message)
    # print(result1)