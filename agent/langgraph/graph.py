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


class RagGraph:
    """RAG系统状态图"""

    # 配置常量（不属于运行时状态）
    MAX_REWRITE_COUNT = 2

    def __init__(self):
        self.agent_node = AgentNode()
        self.retrieve_node = RetrieveNode()
        self.relevance_node = RelevanceNode()
        self.rewrite_node = RewriteNode()
        self.generate_node = GenerateNode()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(RagState)

        graph.add_node("agent", self.agent_node.should_retrieve)
        graph.add_node("retrieve", self.retrieve_node.retrieve_documents)
        graph.add_node("relevance", self.relevance_node.evaluate_relevance)
        graph.add_node("rewrite", self.rewrite_node.rewrite_query)
        graph.add_node("generate", self.generate_node.generate_response)

        graph.add_edge(START, "agent")

        def agent_conditional(state):
            if state.get("needs_retrieval"):
                return "retrieve"
            return "generate"

        graph.add_conditional_edges("agent", agent_conditional, {
            "retrieve": "retrieve",
            "generate": "generate",
        })

        graph.add_edge("retrieve", "relevance")

        def relevance_conditional(state):
            if state.get("is_relevant") or state.get("rewrite_count", 0) >= self.MAX_REWRITE_COUNT:
                return "generate"
            return "rewrite"

        graph.add_conditional_edges("relevance", relevance_conditional, {
            "generate": "generate",
            "rewrite": "rewrite",
        })

        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", END)

        return graph

    def compile(self):
        return self.graph.compile()


def generate_message(graph, inputs):
    """从图的流式输出中提取生成的回答"""
    generated_message = ""
    for output in graph.stream(inputs):
        for key, value in output.items():
            if key == "generate" and isinstance(value, dict):
                generated_message = value.get("response", "")
    return generated_message


if __name__ == "__main__":
    rag_graph = RagGraph()
    compiled_graph = rag_graph.compile()

    print("=== 测试：需要检索的查询 ===")
    print("查询：扫地机器人无法连接wifi了，怎么办？")

    state1 = {
        "query": "扫地机器人无法连接wifi了，怎么办？",
        "session_id": "",
        "chat_history": [],
        "needs_retrieval": False,
        "is_relevant": False,
        "rewrite_count": 0,
        "rewritten_query": "",
        "retrieved_docs": [],
        "relevance_score": 0.0,
        "response": "",
        "error": "",
    }

    for output in compiled_graph.stream(state1):
        for key, value in output.items():
            if key == "generate" and isinstance(value, dict):
                print(f"回答: {value.get('response', '')}")
            if key == "generate" and isinstance(value, dict) and value.get("error"):
                print(f"错误: {value['error']}")
