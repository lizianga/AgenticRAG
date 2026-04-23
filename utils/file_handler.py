import os, hashlib
from typing import List, Tuple
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader

# 获取 文件 MD5的十六进制字符串
def get_file_md5_hex(filepath):
    if not os.path.exists(filepath):
        logger.error(f"【md5计算】文件{filepath}不存在")
        return

    if not os.path.isfile(filepath):
        logger.error(f"【md5计算】路径{filepath}不是文件")
        return
    
    md5_obj = hashlib.md5()
    chunk_size = 4096         # 4kb
    try:
        with open(filepath, "rb") as f:     # 必须二进制读取
            while chunk := f.read(chunk_size):   # chunk = f.read( )  while chunk:
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}MD5失败")
        return None

# 返回文件夹内的文件列表（允许的文件后缀）
def listdir_with_allowed_type(path: str, allowed_types: Tuple[str]):
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types
    
    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path,f))

    return tuple(files)

def pdf_loader(file, password = None)->List[Document]:
    return PyPDFLoader(file, password).load()

def txt_loader(file) -> List[Document]:
    return TextLoader(file, encoding='utf-8').load()