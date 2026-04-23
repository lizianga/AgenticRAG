from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model
from abc import ABC, abstractmethod
from typing import Optional, Union
from utils.config_handler import agent_conf, rag_conf
from langchain_openai import OpenAIEmbeddings

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self):
        return init_chat_model(
                rag_conf['chat_model_name'],
                model_provider="openai",
                base_url = "https://api.siliconflow.cn/v1",
                api_key = agent_conf['api_key'],
            )
    
class EmbeddingsFactory(BaseModelFactory):
    def generator(self):
        return OpenAIEmbeddings(
                base_url="https://api.siliconflow.cn/v1",
                api_key = agent_conf['api_key'],
                model=rag_conf['embeddings_model_name'],
            )

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()