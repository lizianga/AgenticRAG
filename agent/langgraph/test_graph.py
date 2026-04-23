import sys
import os

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.langgraph.graph import RagGraph
from agent.langgraph.state import RagState



def test_rag_system():
    """测试RAG系统"""
    # 初始化RAG图
    rag_graph = RagGraph()
    compiled_graph = rag_graph.compile()
    
    # 测试案例1：需要检索的查询
    print("测试案例1：需要检索的查询")
    print("查询：扫地机器人无法连接wifi了，怎么办？")
    
    # 创建初始状态
    state1 = RagState()
    state1.query = "扫地机器人无法连接wifi了，怎么办？"
    
    # 执行图
    result1 = compiled_graph.invoke(state1.to_dict())
    print(result1)
    # 打印结果
    print(f"是否需要检索: {result1['needs_retrieval']}")
    print(f"检索结果数量: {len(result1['retrieved_docs'])}")
    print(f"相关性评分: {result1['relevance_score']}")
    print(f"是否相关: {result1['is_relevant']}")
    print(f"重写次数: {result1['rewrite_count']}")
    print(f"重写后的查询: {result1['rewritten_query']}")
    print(f"回答: {result1['response']}")
    print("-" * 50)
    
    # 测试案例2：不需要检索的查询
    print("测试案例2：不需要检索的查询")
    print("查询：今天天气怎么样？")
    
    # 创建初始状态
    state2 = RagState()
    state2.query = "今天天气怎么样？"
    
    # 执行图
    result2 = compiled_graph.invoke(state2.to_dict())
    
    # 打印结果
    print(f"是否需要检索: {result2['needs_retrieval']}")
    print(f"回答: {result2['response']}")
    print("-" * 50)
    
    # 测试案例3：模糊查询（需要重写）
    print("测试案例3：模糊查询（需要重写）")
    print("查询：机器人问题")
    
    # 创建初始状态
    state3 = RagState()
    state3.query = "机器人问题"
    
    # 执行图
    result3 = compiled_graph.invoke(state3.to_dict())
    
    # 打印结果
    print(f"是否需要检索: {result3['needs_retrieval']}")
    print(f"检索结果数量: {len(result3['retrieved_docs'])}")
    print(f"相关性评分: {result3['relevance_score']}")
    print(f"是否相关: {result3['is_relevant']}")
    print(f"重写次数: {result3['rewrite_count']}")
    print(f"重写后的查询: {result3['rewritten_query']}")
    print(f"回答: {result3['response']}")
    print("-" * 50)


if __name__ == "__main__":
    test_rag_system()