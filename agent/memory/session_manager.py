import uuid
import time
from datetime import datetime
from typing import Dict, Optional, Any


class SessionManager:
    """会话管理类"""
    
    def __init__(self, session_timeout: int = 1800):
        """初始化会话管理器
        
        Args:
            session_timeout: 会话超时时间（秒），默认30分钟
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = session_timeout
    
    def create_session(self) -> str:
        """创建新会话
        
        Returns:
            会话ID
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": time.time(),
            "last_accessed": time.time(),
            "chat_history": []
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话信息，如果会话不存在或已超时则返回None
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        current_time = time.time()
        
        # 检查会话是否超时
        if current_time - session["last_accessed"] > self.session_timeout:
            del self.sessions[session_id]
            return None
        
        # 更新最后访问时间
        session["last_accessed"] = current_time
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def update_chat_history(self, session_id: str, user_query: str, system_response: str) -> bool:
        """更新会话的聊天历史
        
        Args:
            session_id: 会话ID
            user_query: 用户查询
            system_response: 系统回答
            
        Returns:
            是否更新成功
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        # 添加新的对话记录
        session["chat_history"].append({
            "user_query": user_query,
            "system_response": system_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # 限制聊天历史长度（最多10条）
        if len(session["chat_history"]) > 10:
            session["chat_history"] = session["chat_history"][-10:]
        
        return True
    
    def get_chat_history(self, session_id: str) -> Optional[list]:
        """获取会话的聊天历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            聊天历史，如果会话不存在或已超时则返回None
        """
        session = self.get_session(session_id)
        if not session:
            return None
        return session["chat_history"]
    
    def clean_expired_sessions(self) -> int:
        """清理过期会话
        
        Returns:
            清理的会话数量
        """
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_accessed"] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)