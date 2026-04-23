import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.improved_rag import ImprovedRAG


def test_improved_rag():
    """测试改进的RAG系统"""
    # 初始化ImprovedRAG
    rag = ImprovedRAG()
    
    # 测试案例1：需要检索的查询
    print("测试案例1：需要检索的查询")
    print("查询：扫地机器人无法连接wifi了，怎么办？")
    
    # 运行RAG系统
    result1 = rag.run("扫地机器人无法连接wifi了，怎么办？")
    
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
    
    # 运行RAG系统
    result2 = rag.run("今天天气怎么样？")
    
    # 打印结果
    print(f"是否需要检索: {result2['needs_retrieval']}")
    print(f"回答: {result2['response']}")
    print("-" * 50)
    
    # 测试案例3：模糊查询（需要重写）
    print("测试案例3：模糊查询（需要重写）")
    print("查询：机器人问题")
    
    # 运行RAG系统
    result3 = rag.run("机器人问题")
    
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
    test_improved_rag()