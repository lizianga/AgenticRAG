from typing import Dict, List, Optional, Any
from .session_manager import SessionManager


class ShortTermMemory:
    """短期记忆管理类"""
    
    def __init__(self, session_manager: SessionManager):
        """初始化短期记忆管理器
        
        Args:
            session_manager: 会话管理器实例
        """
        self.session_manager = session_manager
    
    def get_memory(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取短期记忆
        
        Args:
            session_id: 会话ID
            
        Returns:
            短期记忆列表，如果会话不存在或已超时则返回None
        """
        return self.session_manager.get_chat_history(session_id)
    
    def add_memory(self, session_id: str, user_query: str, system_response: str) -> bool:
        """添加短期记忆
        
        Args:
            session_id: 会话ID
            user_query: 用户查询
            system_response: 系统回答
            
        Returns:
            是否添加成功
        """
        return self.session_manager.update_chat_history(session_id, user_query, system_response)
    
    def clear_memory(self, session_id: str) -> bool:
        """清除短期记忆
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否清除成功
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return False
        
        session["chat_history"] = []
        return True
    
    def get_recent_memory(self, session_id: str, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        """获取最近的短期记忆
        
        Args:
            session_id: 会话ID
            limit: 返回的记忆数量限制
            
        Returns:
            最近的短期记忆列表，如果会话不存在或已超时则返回None
        """
        memory = self.get_memory(session_id)
        if not memory:
            return None
        
        return memory[-limit:]
    
    def format_memory_for_context(self, session_id: str) -> Optional[str]:
        """格式化短期记忆为上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            格式化后的上下文字符串，如果会话不存在或已超时则返回None
        """
        memory = self.get_memory(session_id)
        if not memory:
            return None
        
        context = """最近的对话历史：
"""
        
        for i, item in enumerate(reversed(memory)):
            context += f"用户: {item['user_query']}\n"
            context += f"助手: {item['system_response']}\n\n"
        
        return context