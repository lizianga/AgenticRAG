import sys
import os
# 获取当前文件（vector_store.py）的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录（rag目录）
rag_dir = os.path.dirname(current_file_path)
# 获取根目录（heima_agent目录，即rag的父目录）
root_dir = os.path.dirname(rag_dir)
# 把根目录加入sys.path
sys.path.append(root_dir)
from utils.prompt_loader import load_rag_prompts
from utils.config_handler import rag_conf
from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.file_handler import get_file_md5_hex, pdf_loader, txt_loader, listdir_with_allowed_type
from model.factory import chat_model, embed_model
from rag.vector_store import VectorStoreService
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank

from langchain_openai import OpenAI
# from langchain_classic.retrievers import EnsembleRetriever,ContextualCompressionRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class RagSummariceService():
    def __init__(self):
        self.vector_store = VectorStoreService().vector_store
        self.retriver = VectorStoreService().get_hybrid_retriever()  # 使用混合检索
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
        self.rerank_retriever = self._init_rerank_retriever()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain
    
    def _init_rerank_retriever(self):
        """初始化带Rerank的检索器"""
        # 使用BGE模型进行重排
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retrievertriever=self.retriver
        )
        
        return rerank_retriever
    
    def retriver_docs(self, query:str)-> list[Document]:
        # return self.retriver.invoke(query)
        # 使用带Rerank的检索器
        return self.rerank_retriever.invoke(query)
    
    def rag_summarize(self, query:str)->str:
        context_docs = self.retriver_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )

if __name__ == '__main__':
    rag = RagSummariceService()
    
    # 测试混合检索
    print("=== 测试混合检索 ===")
    query = "扫地机器人无法连接wifi了"
    
    # 测试1：使用混合检索+重排
    print("\n1. 使用混合检索+重排：")
    reranked_docs = rag.retriver_docs(query)
    print(f"检索到 {len(reranked_docs)} 个文档")
    for i, doc in enumerate(reranked_docs):
        print(f"\n文档 {i+1}:")
        print(f"内容: {doc.page_content[:200]}...")
        print(f"元数据: {doc.metadata}")
    
    # 测试2：使用原始向量检索（用于对比）
    print("\n2. 使用原始向量检索（对比）：")
    vector_retriever = VectorStoreService().get_retriever()
    vector_docs = vector_retriever.invoke(query)
    print(f"检索到 {len(vector_docs)} 个文档")
    for i, doc in enumerate(vector_docs):
        print(f"\n文档 {i+1}:")
        print(f"内容: {doc.page_content[:200]}...")
        print(f"元数据: {doc.metadata}")
    
    # 测试3：生成最终回答
    print("\n3. 生成最终回答：")
    answer = rag.rag_summarize(query)
    print(f"回答: {answer}")