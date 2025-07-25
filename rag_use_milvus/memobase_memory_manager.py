#!/usr/bin/env python3
"""
Memobase集成记忆管理器

基于Memobase的用户画像和长期记忆管理，提供：
1. 用户画像管理：自动构建和更新用户档案
2. 时间感知记忆：支持时间相关的记忆查询
3. 事件记录：记录用户的重要事件和互动
4. 智能记忆检索：基于用户画像的个性化记忆
5. 与现有系统的兼容性：可以与混合记忆管理器协同工作
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from config import settings
from models import Memory, MemoryType, Message, MessageRole, ConversationSession
from database import conversation_db

# 注意：需要安装 memobase: pip install memobase
try:
    from memobase import MemoBaseClient, ChatBlob
    MEMOBASE_AVAILABLE = True
except ImportError:
    MEMOBASE_AVAILABLE = False
    logging.warning("Memobase not installed. Install with: pip install memobase")


logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """用户画像数据结构"""
    user_id: str
    basic_info: Dict[str, Any] = None
    demographics: Dict[str, Any] = None
    interests: Dict[str, Any] = None
    preferences: Dict[str, Any] = None
    psychological: Dict[str, Any] = None
    relationships: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.basic_info is None:
            self.basic_info = {}
        if self.demographics is None:
            self.demographics = {}
        if self.interests is None:
            self.interests = {}
        if self.preferences is None:
            self.preferences = {}
        if self.psychological is None:
            self.psychological = {}
        if self.relationships is None:
            self.relationships = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class MemobaseMemoryManager:
    """基于Memobase的记忆管理器"""
    
    def __init__(self, project_url: str = None, api_key: str = None, fallback_manager=None):
        """
        初始化Memobase记忆管理器
        
        Args:
            project_url: Memobase项目URL (默认使用本地服务)
            api_key: Memobase API密钥
            fallback_manager: 备用记忆管理器（当Memobase不可用时使用）
        """
        self.project_url = project_url or "http://localhost:8019"
        self.api_key = api_key or "secret"
        self.fallback_manager = fallback_manager
        self.client = None
        self.user_cache: Dict[str, Any] = {}  # 用户对象缓存
        
        # 配置参数
        self.batch_size = 10  # 批处理大小
        self.flush_interval = 300  # 自动刷新间隔（秒）
        self.profile_update_threshold = 5  # 触发画像更新的消息数量
        
        if MEMOBASE_AVAILABLE:
            try:
                self.client = MemoBaseClient(
                    project_url=self.project_url,
                    api_key=self.api_key
                )
                # 测试连接
                if self.client.ping():
                    logger.info("Memobase client initialized successfully")
                else:
                    logger.warning("Memobase connection test failed")
                    self.client = None
            except Exception as e:
                logger.error(f"Failed to initialize Memobase client: {e}")
                self.client = None
        else:
            logger.warning("Memobase not available, using fallback manager")
    
    async def initialize(self):
        """初始化记忆管理器"""
        try:
            if self.fallback_manager:
                await self.fallback_manager.initialize()
            logger.info("Memobase memory manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Memobase memory manager: {e}")
            raise
    
    def _get_or_create_user(self, user_id: str, character_id: str = "") -> Any:
        """获取或创建Memobase用户"""
        if not self.client:
            return None
            
        cache_key = f"{user_id}_{character_id}"
        
        if cache_key in self.user_cache:
            return self.user_cache[cache_key]
        
        try:
            # 尝试获取现有用户
            user_data = {
                "user_id": user_id,
                "character_id": character_id,
                "created_at": datetime.now().isoformat()
            }
            
            uid = self.client.add_user(user_data)
            user = self.client.get_user(uid)
            
            self.user_cache[cache_key] = user
            logger.debug(f"Created/retrieved Memobase user for {user_id}")
            
            return user
            
        except Exception as e:
            logger.error(f"Failed to get/create Memobase user: {e}")
            return None
    
    async def add_message_to_conversation(
        self, user_id: str, session_id: str, message: Message
    ) -> ConversationSession:
        """添加消息到Memobase记忆系统"""
        try:
            character_id = message.metadata.get("character_id", "")
            
            # 如果Memobase可用，使用Memobase
            if self.client:
                await self._add_to_memobase(user_id, character_id, session_id, message)
            
            # 同时使用备用管理器（如果有）
            if self.fallback_manager:
                return await self.fallback_manager.add_message_to_conversation(
                    user_id, session_id, message
                )
            
            # 如果没有备用管理器，创建基本的会话对象
            conversation = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                character_id=character_id,
                messages=[message],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            
            logger.info(f"Added message to Memobase memory system for session {session_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"Failed to add message to Memobase memory: {e}")
            # 如果有备用管理器，尝试使用它
            if self.fallback_manager:
                return await self.fallback_manager.add_message_to_conversation(
                    user_id, session_id, message
                )
            raise
    
    async def _add_to_memobase(self, user_id: str, character_id: str, session_id: str, message: Message):
        """添加消息到Memobase"""
        try:
            user = self._get_or_create_user(user_id, character_id)
            if not user:
                return
            
            # 构建聊天消息
            chat_messages = [{
                "role": message.role.value,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "session_id": session_id,
                **message.metadata
            }]
            
            # 创建ChatBlob并插入
            chat_blob = ChatBlob(messages=chat_messages)
            blob_id = user.insert(chat_blob)
            
            logger.debug(f"Inserted message to Memobase with blob_id: {blob_id}")
            
        except Exception as e:
            logger.error(f"Failed to add message to Memobase: {e}")
    
    async def retrieve_relevant_memories(
        self,
        query: str,
        user_id: str,
        character_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """检索相关记忆"""
        try:
            memories = []
            
            # 如果Memobase可用，从Memobase检索
            if self.client:
                memobase_memories = await self._retrieve_from_memobase(
                    query, user_id, character_id, limit
                )
                memories.extend(memobase_memories)
            
            # 如果有备用管理器，也从备用管理器检索
            if self.fallback_manager:
                fallback_memories = await self.fallback_manager.retrieve_relevant_memories(
                    query, user_id, character_id, session_id, memory_type, limit
                )
                memories.extend(fallback_memories)
            
            # 去重和排序
            unique_memories = self._deduplicate_memories(memories)
            sorted_memories = sorted(
                unique_memories,
                key=lambda x: (x.importance_score, x.created_at),
                reverse=True
            )
            
            result = sorted_memories[:limit]
            logger.info(f"Retrieved {len(result)} relevant memories from Memobase system")
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant memories: {e}")
            # 如果有备用管理器，尝试使用它
            if self.fallback_manager:
                return await self.fallback_manager.retrieve_relevant_memories(
                    query, user_id, character_id, session_id, memory_type, limit
                )
            return []
    
    async def _retrieve_from_memobase(
        self, query: str, user_id: str, character_id: str, limit: int
    ) -> List[Memory]:
        """从Memobase检索记忆"""
        try:
            user = self._get_or_create_user(user_id, character_id)
            if not user:
                return []
            
            # 刷新用户记忆以获取最新的画像和事件
            user.flush(sync=True)
            
            # 获取用户画像
            profile = await self._get_user_profile(user_id, character_id)
            
            # 基于画像和查询构建记忆对象
            memories = []
            
            if profile:
                # 将用户画像转换为记忆对象
                profile_memory = Memory(
                    user_id=user_id,
                    character_id=character_id,
                    content=f"用户画像: {json.dumps(asdict(profile), ensure_ascii=False, indent=2)}",
                    memory_type=MemoryType.LONG_TERM,
                    importance_score=0.9,
                    created_at=profile.updated_at,
                    metadata={
                        "source": "memobase_profile",
                        "profile_type": "user_profile"
                    }
                )
                memories.append(profile_memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve from Memobase: {e}")
            return []
    
    async def _get_user_profile(self, user_id: str, character_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        try:
            user = self._get_or_create_user(user_id, character_id)
            if not user:
                return None
            
            # 这里需要根据Memobase的实际API来获取用户画像
            # 由于Memobase的画像API可能需要特定的调用方式，这里提供一个基础实现
            
            # 创建基础用户画像
            profile = UserProfile(
                user_id=user_id,
                basic_info={"character_id": character_id},
                updated_at=datetime.now()
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return None
    
    async def get_conversation_history(self, session_id: str) -> List[Message]:
        """获取对话历史"""
        try:
            # 优先使用备用管理器
            if self.fallback_manager:
                return await self.fallback_manager.get_conversation_history(session_id)
            
            # 如果没有备用管理器，从数据库获取
            return await conversation_db.get_conversation_messages(session_id)
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """获取对话上下文"""
        try:
            if self.fallback_manager:
                return await self.fallback_manager.get_conversation_context(session_id)
            
            # 基础实现
            messages = await self.get_conversation_history(session_id)
            return {
                "session_id": session_id,
                "conversation_history": [
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat()
                    }
                    for msg in messages
                ],
                "total_messages": len(messages)
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return {"session_id": session_id, "error": str(e)}
    
    async def generate_conversation_summary(
        self, session_id: str, user_id: str, character_id: str, location: str = ""
    ) -> str:
        """生成对话总结"""
        try:
            if self.fallback_manager:
                return await self.fallback_manager.generate_conversation_summary(
                    session_id, user_id, character_id, location
                )
            
            # 基础实现
            messages = await self.get_conversation_history(session_id)
            if not messages:
                return "对话为空，无需总结"
            
            # 简单的总结生成
            summary = f"对话包含{len(messages)}条消息，时间范围从{messages[0].timestamp}到{messages[-1].timestamp}"
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return f"总结生成失败: {str(e)}"
    
    def _deduplicate_memories(self, memories: List[Memory]) -> List[Memory]:
        """去重记忆"""
        seen_contents = set()
        unique_memories = []
        
        for memory in memories:
            content_hash = hash(memory.content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_memories.append(memory)
        
        return unique_memories
    
    async def flush_user_memories(self, user_id: str, character_id: str = ""):
        """刷新用户记忆（触发Memobase处理）"""
        try:
            if not self.client:
                return
            
            user = self._get_or_create_user(user_id, character_id)
            if user:
                user.flush(sync=True)
                logger.info(f"Flushed memories for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to flush user memories: {e}")
    
    async def cleanup_old_sessions(self, days_threshold: int = 30):
        """清理旧会话"""
        try:
            if self.fallback_manager:
                await self.fallback_manager.cleanup_old_sessions(days_threshold)
            logger.info(f"Cleaned up sessions older than {days_threshold} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")


# 全局实例（将在main.py中初始化）
memobase_memory_manager = None