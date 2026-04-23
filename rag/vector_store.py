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

from utils.config_handler import rag_conf
from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.file_handler import get_file_md5_hex, pdf_loader, txt_loader, listdir_with_allowed_type
from model.factory import chat_model, embed_model
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever



class VectorStoreService():

    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf['name'],
            embedding_function=embed_model,
            persist_directory=chroma_conf['persist_directory']
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = chroma_conf['chunk_size'],
            chunk_overlap = chroma_conf['chunk_overlap'],
            separators = chroma_conf['separators']
        )

    def get_retriever(self):
            return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf['k']})
    
    def get_hybrid_retriever(self):
        """获取混合检索器（向量检索 + 关键词检索）"""
        # 向量检索
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": chroma_conf['k']})
        
        # 关键词检索（BM25）
        # 首先获取所有文档
        all_docs = self.vector_store.get()['documents']
        bm25_retriever = BM25Retriever.from_texts(all_docs)
        bm25_retriever.k = chroma_conf['k']
        
        # 创建混合检索器
        hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # 向量检索权重70%，关键词检索权重30%
        )
        
        return hybrid_retriever

    def load_document(self):
            def check_md5_hex(md5_hex_for_check):
                if not os.path.exists(get_abs_path(chroma_conf['md5_hex_store'])):
                    open(get_abs_path(chroma_conf['md5_hex_store']), 'w', encoding='utf-8').close()
                    return False
                
                with open(get_abs_path(chroma_conf['md5_hex_store']), 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line == md5_hex_for_check:
                            return True
                return False
            
            def save_md5_hex(md5_for_save):
                with open(get_abs_path(chroma_conf['md5_hex_store']), 'a', encoding='utf-8') as f:
                    f.write(md5_for_save + "\n")
            
            def get_file_documents(read_path:str):
                if read_path.endswith("txt"):
                    return txt_loader(read_path)
                if read_path.endswith("pdf"):
                    return pdf_loader(read_path)
                return []
            
            allowed_files_path: list[str] = listdir_with_allowed_type(
                get_abs_path(chroma_conf['data_path']),
                tuple(chroma_conf["allowed_files_type"])
            )

            for path in allowed_files_path:
                file_md5 = get_file_md5_hex(path)
                if check_md5_hex(file_md5):
                    logger.info(f"[加载知识库]{path}内容以存在知识库中，跳过")
                    continue

                try:
                    documents:list[Document] = get_file_documents(path)
                    if not documents:
                        logger.warning(f"[加载知识库]{path}无有效内容，跳过")
                    split_documents = self.spliter.split_documents(documents)
                    if not split_documents:
                        logger.warning(f"[加载知识库]{path}分片后无有效内容，跳过")
                    if len(split_documents) > 64:
                        batch_size = 64
                        # 把split_documents拆分成多个子列表，每个子列表最多64条
                        batches = [
                        split_documents[i:i+batch_size] 
                        for i in range(0, len(split_documents), batch_size)
                        ]
                        # 3. 分批添加到向量库
                        for batch in batches:
                            if batch:  # 跳过空批次
                                self.vector_store.add_documents(batch)
                    else:
                        self.vector_store.add_documents(split_documents)
                    save_md5_hex(file_md5)
                    logger.info(f"[加载知识库]{path}成功！")
                except Exception as e:
                    # 记录详细报错堆栈
                    logger.error(f"[加载知识库]{path}失败！", exc_info=True)
                    continue

if __name__ == '__main__':
    vector_store = VectorStoreService()
    retriver = vector_store.get_retriever()
    vector_store.load_document()
    res = retriver.invoke("迷路")
    for r in res:
        print(r.page_content)
        print('-'*20)