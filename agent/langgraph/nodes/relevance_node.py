import os
import re
import sys
import logging

# 计算并添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(project_root)
sys.path.append(project_root)

from typing import Dict, Any
from agent.langgraph.state import RagState
from model.factory import embed_model
import numpy as np
import httpx

logger = logging.getLogger(__name__)

# 中文分词：按 2-gram + 单字提取关键词
_CHAR_PATTERN = re.compile(r'[一-鿿\w]+')


def _extract_tokens(text: str) -> set:
    """提取文本中的关键 token（中文按 2-gram，其他按词）"""
    tokens = set()
    for word in _CHAR_PATTERN.findall(text.lower()):
        if len(word) >= 2:
            # 中文 2-gram
            for i in range(len(word) - 1):
                tokens.add(word[i:i + 2])
        tokens.add(word)
    return tokens


class RelevanceNode:
    """Relevance节点，评估检索结果与查询的相关性

    优先使用外部 Reranker API（bge-reranker-v2-m3），
    服务不可用时回退到 embedding 余弦相似度 + 关键词重叠加权。
    """

    def __init__(self):
        self.embed_model = embed_model
        self.reranker_threshold = 0.5
        self.fallback_threshold = 0.45
        self.rerank_api_url = "http://1.15.95.222:6405/v1/rerank"
        self.rerank_model = "bge-reranker-v2-m3"
        self._api_ok = None  # None=未检测, True=可用, False=不可用

    def evaluate_relevance(self, state: RagState) -> dict:
        """评估检索结果与查询的相关性"""
        logger.info("RelevanceNode 开始评估相关性")

        if not state.get("retrieved_docs"):
            logger.info("无检索文档，判定不相关")
            return {"is_relevant": False, "relevance_score": 0.0}

        query = state.get("rewritten_query") or state["query"]

        if self._api_ok is not False:
            try:
                score = self._score_with_reranker(query, state["retrieved_docs"])
                self._api_ok = True
            except Exception as e:
                self._api_ok = False
                logger.warning("Reranker API 调用失败，回退到 embedding: %s", e)
                score = self._score_with_embedding(query, state["retrieved_docs"])
        else:
            score = self._score_with_embedding(query, state["retrieved_docs"])

        threshold = self.reranker_threshold if self._api_ok else self.fallback_threshold
        is_relevant = score >= threshold
        logger.info(
            "query=%s score=%.4f threshold=%.2f(%s) is_relevant=%s",
            query, score, threshold, "reranker" if self._api_ok else "fallback", is_relevant,
        )

        return {"relevance_score": score, "is_relevant": is_relevant}

    def _score_with_reranker(self, query: str, docs: list) -> float:
        """调用外部 Reranker API，返回最高相关性分数 (0-1)"""
        documents = [doc.page_content for doc in docs]
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
        }
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(self.rerank_api_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            logger.warning("Reranker API 返回空结果")
            return 0.0

        scores = [item.get("relevance_score", 0) for item in results]
        best = max(scores)
        logger.debug("Reranker 各文档分数: %s, 最高分: %.4f", scores, best)
        return best

    def _score_with_embedding(self, query: str, docs: list) -> float:
        """回退方案：embedding 余弦相似度(30%) + 关键词重叠率(70%)，取最高分"""
        query_embedding = self.embed_model.embed_query(query)
        query_tokens = _extract_tokens(query)
        doc_texts = [doc.page_content for doc in docs]
        doc_embeddings = self.embed_model.embed_documents(doc_texts)

        scores = []
        for doc_text, doc_emb in zip(doc_texts, doc_embeddings):
            cos_sim = self._cosine_similarity(query_embedding, doc_emb)
            doc_tokens = _extract_tokens(doc_text)
            overlap = len(query_tokens & doc_tokens) / len(query_tokens) if query_tokens else 0.0
            combined = 0.3 * cos_sim + 0.7 * overlap
            scores.append(combined)
            logger.debug(
                "doc=%.40s... cos=%.4f overlap=%.4f combined=%.4f",
                doc_text, cos_sim, overlap, combined,
            )

        return max(scores) if scores else 0.0

    @staticmethod
    def _cosine_similarity(vec1: list, vec2: list) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            return float(dot_product / (norm1 * norm2))
        return 0.0


def _new_state(query: str, docs: list = None, rewritten_query: str = "") -> dict:
    return {
        "query": query,
        "session_id": "",
        "chat_history": [],
        "needs_retrieval": True,
        "is_relevant": False,
        "rewrite_count": 0,
        "rewritten_query": rewritten_query,
        "retrieved_docs": docs or [],
        "relevance_score": 0.0,
        "response": "",
        "error": "",
    }


if __name__ == "__main__":
    import time
    from langchain_core.documents import Document

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    node = RelevanceNode()
    passed = 0
    failed = 0

    def run_test(name: str, expected: bool, state: dict):
        global passed, failed
        print(f"\n{'='*50}")
        print(f"测试: {name}")
        print(f"查询: {state['query']}")
        if state['rewritten_query']:
            print(f"重写查询: {state['rewritten_query']}")
        print(f"文档数: {len(state['retrieved_docs'])}")
        t0 = time.time()
        result = node.evaluate_relevance(state)
        elapsed = time.time() - t0
        actual = result["is_relevant"]
        ok = actual == expected
        if ok:
            passed += 1
        else:
            failed += 1
        status = "PASS" if ok else "FAIL"
        threshold = node.reranker_threshold if node._api_ok else node.fallback_threshold
        print(f"分数: {result['relevance_score']:.4f}  阈值: {threshold:.2f}")
        print(f"期望: {expected}  实际: {actual}  耗时: {elapsed:.3f}s  [{status}]")

    # ---- 相关文档 ----
    print("=" * 50)
    print("第一组：相关文档")

    run_test(
        "充电故障-多文档",
        True,
        _new_state("扫地机器人无法充电怎么办？", [
            Document(page_content="扫地机器人无法充电可能是因为电池问题、充电器故障或接触不良。建议检查电池是否损坏，充电器是否正常工作，以及充电接口是否干净。", metadata={"source": "故障排除.txt"}),
            Document(page_content="扫地机器人的电池寿命一般为1-2年，使用不当会缩短电池寿命。建议每次使用后及时充电，避免过度放电。", metadata={"source": "维护保养.txt"}),
        ]),
    )
    run_test(
        "重写查询-相关",
        True,
        _new_state(
            "扫地机器人问题",
            [Document(page_content="扫地机器人常见故障包括无法充电、无法启动、清扫效果差等。针对不同故障，有相应的解决方法。", metadata={"source": "故障排除.txt"})],
            rewritten_query="扫地机器人常见故障及解决方法",
        ),
    )
    run_test(
        "单文档-强相关",
        True,
        _new_state("扫地机器人的滤网怎么清洗？", [
            Document(page_content="清洗扫地机器人滤网的步骤：1. 取出尘盒；2. 拿出滤网；3. 用清水冲洗；4. 晾干后装回。建议每两周清洗一次滤网。", metadata={"source": "维护保养.txt"}),
        ]),
    )

    # ---- 不相关文档 ----
    print("\n" + "=" * 50)
    print("第二组：不相关文档")

    run_test(
        "完全不相关",
        False,
        _new_state("如何制作蛋糕？", [
            Document(page_content="扫地机器人无法充电可能是因为电池问题、充电器故障或接触不良。", metadata={"source": "故障排除.txt"}),
        ]),
    )
    run_test(
        "跨领域文档",
        False,
        _new_state("Python如何读取文件？", [
            Document(page_content="扫地机器人的电池寿命一般为1-2年，使用不当会缩短电池寿命。", metadata={"source": "维护保养.txt"}),
        ]),
    )

    # ---- 边界情况 ----
    print("\n" + "=" * 50)
    print("第三组：边界情况")

    run_test("无文档", False, _new_state("扫地机器人如何维护？", []))

    # ---- 汇总 ----
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"测试完成: {passed}/{total} 通过, {failed} 失败")