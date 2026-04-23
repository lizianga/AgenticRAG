import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any


class LongTermMemory:
    """长期记忆管理类（基于文件系统JSON）"""
    
    def __init__(self, memory_dir: str = "data/memory"):
        """初始化长期记忆管理器
        
        Args:
            memory_dir: 记忆存储目录
        """
        self.memory_dir = memory_dir
        # 确保目录存在
        os.makedirs(self.memory_dir, exist_ok=True)
    
    def _get_memory_file_path(self, session_id: str) -> str:
        """获取记忆文件路径
        
        Args:
            session_id: 会话ID
            
        Returns:
            记忆文件路径
        """
        return os.path.join(self.memory_dir, f"{session_id}.json")
    
    def get_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取长期记忆
        
        Args:
            session_id: 会话ID
            
        Returns:
            长期记忆数据，如果不存在则返回None
        """
        file_path = self._get_memory_file_path(session_id)
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def add_memory(self, session_id: str, user_query: str, system_response: str, summary: str = "") -> bool:
        """添加长期记忆
        
        Args:
            session_id: 会话ID
            user_query: 用户查询
            system_response: 系统回答
            summary: 对话摘要
            
        Returns:
            是否添加成功
        """
        try:
            # 获取现有记忆
            memory = self.get_memory(session_id)
            if not memory:
                # 创建新的记忆文件
                memory = {
                    "session_id": session_id,
                    "conversations": [],
                    "last_updated": datetime.now().isoformat()
                }
            
            # 添加新的对话记录
            memory["conversations"].append({
                "user_query": user_query,
                "system_response": system_response,
                "timestamp": datetime.now().isoformat(),
                "summary": summary
            })
            
            # 更新最后更新时间
            memory["last_updated"] = datetime.now().isoformat()
            
            # 保存到文件
            file_path = self._get_memory_file_path(session_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception:
            return False
    
    def update_summary(self, session_id: str, conversation_index: int, summary: str) -> bool:
        """更新对话摘要
        
        Args:
            session_id: 会话ID
            conversation_index: 对话索引
            summary: 新的摘要
            
        Returns:
            是否更新成功
        """
        memory = self.get_memory(session_id)
        if not memory:
            return False
        
        if conversation_index < 0 or conversation_index >= len(memory["conversations"]):
            return False
        
        try:
            memory["conversations"][conversation_index]["summary"] = summary
            memory["last_updated"] = datetime.now().isoformat()
            
            file_path = self._get_memory_file_path(session_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception:
            return False
    
    def get_conversations(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取会话的所有对话
        
        Args:
            session_id: 会话ID
            
        Returns:
            对话列表，如果不存在则返回None
        """
        memory = self.get_memory(session_id)
        if not memory:
            return None
        return memory.get("conversations", [])
    
    def delete_memory(self, session_id: str) -> bool:
        """删除长期记忆
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
        """
        file_path = self._get_memory_file_path(session_id)
        if not os.path.exists(file_path):
            return True
        
        try:
            os.remove(file_path)
            return True
        except Exception:
            return False
    
    def list_sessions(self) -> List[str]:
        """列出所有存储的会话
        
        Returns:
            会话ID列表
        """
        sessions = []
        try:
            for filename in os.listdir(self.memory_dir):
                if filename.endswith('.json'):
                    session_id = filename[:-5]  # 移除.json后缀
                    sessions.append(session_id)
        except Exception:
            pass
        return sessions