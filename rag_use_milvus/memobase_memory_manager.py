#!/usr/bin/env python3
"""
Memobase记忆管理器 - 完全基于Memobase的记忆管理系统

彻底重构的记忆管理架构，完全基于Memobase提供：
1. 用户画像管理：自动构建和更新结构化用户档案
2. 时间感知记忆：支持时间相关的记忆查询和事件记录
3. 智能记忆检索：基于用户画像的个性化记忆检索
4. 对话上下文管理：利用Memobase的上下文API
5. 批处理优化：高效的记忆处理和存储
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from config import settings
from models import Memory, MemoryType, Message, MessageRole, ConversationSession
from database import conversation_db


try:
    from memobase import MemoBaseClient, ChatBlob
    MEMOBASE_AVAILABLE = True
except ImportError:
    MEMOBASE_AVAILABLE = False
    logging.warning("Memobase not installed. Using mock implementation.")


logger = logging.getLogger(__name__)


@dataclass
class MemobaseConfig:
    """Memobase配置"""
    project_url: str = "http://localhost:8019"

    api_key: str = "secret"
    batch_size: int = 2  # 降低批处理大小，确保及时刷新
    flush_interval: int = 300  # 秒
    profile_update_threshold: int = 5
    context_limit: int = 20
    memory_retention_days: int = 365


class MemobaseMemoryManager:
    """基于Memobase的完整记忆管理器"""
    
    def __init__(self, config: MemobaseConfig = None):
        """
        初始化Memobase记忆管理器
        
        Args:
            config: Memobase配置对象
        """
        self.config = config or MemobaseConfig()
        self.client = None
        self.user_cache: Dict[str, Any] = {}  # 用户对象缓存
        self.session_cache: Dict[str, List[Message]] = {}  # 会话消息缓存
        self.pending_flushes: Dict[str, List[Dict]] = {}  # 待处理的刷新队列
        self.user_id_mapping: Dict[str, str] = {}  # 字符串用户ID到UUID的映射
        
        # 初始化Memobase客户端
        self._initialize_client()
        
        # 启动后台任务
        self._background_tasks = []
        
        logger.info("Memobase memory manager initialized")
    
    async def _get_or_create_user_uuid(self, user_id: str, character_id: str = "") -> str:
        """获取或创建用户的UUID映射"""
        # 标准化character_id，确保一致性
        normalized_character_id = character_id or "default"
        cache_key = f"{user_id}_{normalized_character_id}"
        
        # 检查缓存
        if cache_key in self.user_id_mapping:
            return self.user_id_mapping[cache_key]
        
        try:
            import httpx
            
            # 尝试创建用户并获取UUID
            create_url = f"{self.config.project_url}/api/v1/users"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "user_id": user_id,
                "name": user_id,
                "character_id": normalized_character_id
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(create_url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    uuid = result.get("data", {}).get("id")
                    if uuid:
                        self.user_id_mapping[cache_key] = uuid
                        logger.debug(f"Created user UUID mapping: {user_id} -> {uuid}")
                        return uuid
                else:
                    logger.warning(f"Failed to create user: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"Failed to get/create user UUID: {e}")
        
        # Fallback: 使用原始用户ID
        self.user_id_mapping[cache_key] = user_id
        return user_id
    
    def _initialize_client(self):
        """初始化Memobase客户端"""
        if not MEMOBASE_AVAILABLE:
            logger.warning("Memobase library not installed. Using fallback mode.")
            self.client = None
            return
        
        try:
            self.client = MemoBaseClient(
                project_url=self.config.project_url,
                api_key=self.config.api_key
            )
            
            # 测试连接 - 使用正确的健康检查端点
            try:
                import httpx
                response = httpx.get(f"{self.config.project_url}/api/v1/healthcheck", timeout=5.0)
                if response.status_code == 200:
                    logger.info(f"Connected to Memobase at {self.config.project_url}")
                else:
                    logger.warning(f"Memobase healthcheck failed with status {response.status_code}, using fallback mode")
                    self.client = None
            except Exception as e:
                logger.warning(f"Failed to connect to Memobase: {e}, using fallback mode")
                self.client = None
                
        except Exception as e:
            logger.warning(f"Failed to initialize Memobase client: {e}, using fallback mode")
            self.client = None
    
    async def initialize(self):
        """异步初始化"""
        try:
            # 初始化数据库（用于本地缓存和备份）
            await conversation_db.initialize()
            
            # 启动后台任务
            self._start_background_tasks()
            
            logger.info("Memobase memory manager fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memobase memory manager: {e}")
            raise
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 定期刷新任务
        flush_task = asyncio.create_task(self._periodic_flush())
        self._background_tasks.append(flush_task)
        
        # 清理任务
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._background_tasks.append(cleanup_task)
    
    def _get_or_create_user(self, user_id: str, character_id: str = "") -> Any:
        """获取或创建Memobase用户"""
        cache_key = f"{user_id}_{character_id}"
        
        if cache_key in self.user_cache:
            return self.user_cache[cache_key]
        
        if self.client is None:
            logger.info(f"Memobase not available, using fallback for user {user_id}")
            self.user_cache[cache_key] = None
            return None
        
        try:
            # 尝试简化的用户创建方式
            # 根据Memobase文档，add_user应该接受简单的字典
            user_data = {
                "user_id": user_id,
                "character_id": character_id or "default"
            }
            
            try:
                # 尝试添加用户到Memobase
                uid = self.client.add_user(user_data)
                user = self.client.get_user(uid)
            except Exception as api_error:
                logger.warning(f"Memobase add_user failed: {api_error}")
                # 如果add_user失败，尝试直接使用客户端作为用户对象
                # 这是一个fallback方案
                user = self.client
                uid = f"{user_id}_{character_id}"
            
            # 缓存用户对象
            self.user_cache[cache_key] = user
            
            logger.debug(f"Created/retrieved Memobase user: {user_id}")
            return user
            
        except Exception as e:
            logger.error(f"Failed to get/create Memobase user: {e}")
            # 返回客户端对象作为fallback
            self.user_cache[cache_key] = self.client
            return self.client
    
    async def add_message_to_conversation(
        self, user_id: str, session_id: str, message: Message
    ) -> ConversationSession:
        """添加消息到对话"""
        try:
            character_id = message.metadata.get("character_id", "")
            
            # 1. 添加到本地缓存
            if session_id not in self.session_cache:
                self.session_cache[session_id] = []
            
            self.session_cache[session_id].append(message)
            
            # 2. 保存到本地数据库（备份）
            await conversation_db.save_message(session_id, message)
            
            # 3. 准备Memobase数据
            await self._prepare_memobase_data(user_id, character_id, session_id, message)
            
            # 4. 检查是否需要立即刷新
            await self._check_immediate_flush(user_id, character_id)
            
            # 5. 构建返回对象
            conversation = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                character_id=character_id,
                messages=self.session_cache[session_id][-self.config.context_limit:],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            
            logger.debug(f"Added message to conversation {session_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"Failed to add message to conversation: {e}")
            raise
    
    async def _prepare_memobase_data(
        self, user_id: str, character_id: str, session_id: str, message: Message
    ):
        """准备Memobase数据"""
        try:
            # 构建聊天消息格式
            chat_message = {
                "role": message.role.value,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "session_id": session_id,
                "message_id": message.id,
                **message.metadata
            }
            
            # 添加到待处理队列
            cache_key = f"{user_id}_{character_id}"
            if cache_key not in self.pending_flushes:
                self.pending_flushes[cache_key] = []
            
            self.pending_flushes[cache_key].append(chat_message)
            
            logger.debug(f"Prepared Memobase data for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to prepare Memobase data: {e}")
    
    async def _check_immediate_flush(self, user_id: str, character_id: str):
        """检查是否需要立即刷新"""
        try:
            cache_key = f"{user_id}_{character_id}"
            
            if cache_key in self.pending_flushes:
                pending_count = len(self.pending_flushes[cache_key])
                logger.info(f"Checking flush for {user_id}: {pending_count}/{self.config.batch_size} messages pending")
                
                # 如果达到批处理大小，立即刷新
                if pending_count >= self.config.batch_size:
                    logger.info(f"Triggering immediate flush for {user_id} (reached batch size {self.config.batch_size})")
                    await self._flush_user_data(user_id, character_id)
                else:
                    logger.debug(f"Not flushing yet for {user_id}: {pending_count} < {self.config.batch_size}")
            else:
                logger.debug(f"No pending messages for {user_id}")
                    
        except Exception as e:
            logger.error(f"Failed to check immediate flush: {e}")
    
    async def _flush_user_data(self, user_id: str, character_id: str):
        """刷新用户数据到Memobase"""
        try:
            cache_key = f"{user_id}_{character_id}"
            
            if cache_key not in self.pending_flushes or not self.pending_flushes[cache_key]:
                return
            
            # 获取用户对象
            user = self._get_or_create_user(user_id, character_id)
            
            # 批量处理消息
            messages = self.pending_flushes[cache_key]
            
            if user is None:
                logger.info(f"Memobase not available, skipping flush for user {user_id}")
                # 清空待处理队列
                self.pending_flushes[cache_key] = []
                return
            
            # 检查用户对象类型并相应处理
            try:
                if hasattr(user, 'insert') and hasattr(user, 'flush'):
                    # 真实的Memobase用户对象
                    chat_blob = ChatBlob(messages=messages)
                    blob_id = user.insert(chat_blob)
                    user.flush(sync=True)
                    logger.info(f"Flushed {len(messages)} messages for user {user_id}, blob_id: {blob_id}")
                else:
                    # Fallback用户对象（MemoBaseClient），通过HTTP API发送数据
                    await self._flush_via_http_api(user_id, character_id, messages)
                    logger.info(f"Flushed {len(messages)} messages for user {user_id} via HTTP API")
            except Exception as flush_error:
                logger.warning(f"Primary flush failed, trying HTTP API fallback: {flush_error}")
                await self._flush_via_http_api(user_id, character_id, messages)
            
            # 清空待处理队列
            self.pending_flushes[cache_key] = []
            
        except Exception as e:
            logger.error(f"Failed to flush user data: {e}")
    
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
            
            # 1. 确保用户数据已刷新
            await self._flush_user_data(user_id, character_id)
            
            # 2. 获取用户对象
            user = self._get_or_create_user(user_id, character_id)
            
            # 3. 使用Memobase的上下文API获取相关记忆
            context_data = await self._get_memobase_context(user, query, limit, user_id, character_id)
            
            # 4. 转换为Memory对象
            if context_data:
                memories.extend(self._convert_context_to_memories(
                    context_data, user_id, character_id
                ))
            
            # 5. 添加最近的会话上下文
            if session_id and session_id in self.session_cache:
                recent_memories = self._get_recent_session_memories(
                    session_id, user_id, character_id, query
                )
                memories.extend(recent_memories)
            
            # 6. 排序和去重
            unique_memories = self._deduplicate_memories(memories)
            sorted_memories = sorted(
                unique_memories,
                key=lambda x: (x.importance_score, x.created_at),
                reverse=True
            )
            
            result = sorted_memories[:limit]
            logger.info(f"Retrieved {len(result)} relevant memories for query: {query[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant memories: {e}")
            return []
    
    async def _get_memobase_context(self, user: Any, query: str, limit: int, user_id: str = None, character_id: str = "") -> Optional[Dict]:
        """获取Memobase上下文"""
        try:
            if user is None:
                logger.info("Memobase not available, returning empty context")
                return {
                    "query": query,
                    "user_profile": None,
                    "relevant_events": [],
                    "memories": []
                }
            
            # 刷新用户记忆以获取最新状态
            try:
                if hasattr(user, 'flush'):
                    user.flush(sync=True)
                    logger.debug("User data flushed successfully")
            except Exception as e:
                logger.warning(f"Failed to flush user in context retrieval: {e}")
            
            # 调用Memobase的上下文API
            try:
                # 根据Memobase文档，使用HTTP API获取用户上下文
                import httpx
                
                # 构建上下文请求 - 使用正确的API端点
                if not user_id:
                    logger.warning("No user_id provided for context API")
                    raise ValueError("user_id is required for context API")
                
                # 获取或创建用户UUID
                user_uuid = await self._get_or_create_user_uuid(user_id, character_id)
                logger.debug(f"Using user UUID for context API: {user_id} -> {user_uuid}")
                    
                context_url = f"{self.config.project_url}/api/v1/users/context/{user_uuid}"
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
                
                # 请求参数 - 根据API文档使用chats_str
                params = {
                    "chats_str": query,
                    "limit": limit
                }
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(context_url, headers=headers, params=params)
                    
                    if response.status_code == 200:
                        context_data = response.json()
                        logger.debug(f"Retrieved context from Memobase: {len(str(context_data))} chars")
                        return context_data
                    else:
                        logger.warning(f"Memobase context API returned status {response.status_code}: {response.text}")
                        
            except Exception as api_error:
                logger.warning(f"Failed to call Memobase context API: {api_error}")
            
            # Fallback: 尝试获取用户画像和事件
            try:
                profile_data = await self._get_user_profile_fallback(user)
                events_data = await self._get_user_events_fallback(user, query, limit)
                
                context = {
                    "query": query,
                    "user_profile": profile_data,
                    "relevant_events": events_data,
                    "memories": []
                }
                
                logger.debug("Using fallback context data")
                return context
                
            except Exception as fallback_error:
                logger.error(f"Fallback context retrieval failed: {fallback_error}")
            
            # 最后的fallback：返回空上下文
            return {
                "query": query,
                "user_profile": None,
                "relevant_events": [],
                "memories": []
            }
            
        except Exception as e:
            logger.error(f"Failed to get Memobase context: {e}")
            return None
    
    async def _flush_via_http_api(self, user_id: str, character_id: str, messages: List[Dict]):
        """通过HTTP API刷新数据到Memobase"""
        try:
            import httpx
            
            # 获取或创建用户UUID
            user_uuid = await self._get_or_create_user_uuid(user_id, character_id)
            logger.debug(f"Using user UUID for flush API: {user_id} -> {user_uuid}")
            
            # 构建数据刷新请求
            flush_url = f"{self.config.project_url}/api/v1/users/{user_uuid}/messages"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建请求数据
            data = {
                "user_id": user_uuid,
                "character_id": character_id,
                "messages": messages
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(flush_url, headers=headers, json=data)
                
                if response.status_code in [200, 201]:
                    logger.debug(f"Successfully flushed {len(messages)} messages via HTTP API")
                else:
                    logger.warning(f"HTTP API flush returned status {response.status_code}: {response.text}")
                    
        except Exception as e:
            logger.error(f"Failed to flush via HTTP API: {e}")
    
    async def _get_user_profile_fallback(self, user: Any) -> Optional[Dict]:
        """获取用户画像的fallback方法"""
        try:
            # 尝试通过HTTP API获取用户画像
            import httpx
            
            profile_url = f"{self.config.project_url}/api/v1/users/profile"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(profile_url, headers=headers)
                
                if response.status_code == 200:
                    profile_data = response.json()
                    logger.debug("Retrieved user profile from Memobase API")
                    return profile_data
                else:
                    logger.debug(f"Profile API returned status {response.status_code}")
                    
        except Exception as e:
            logger.debug(f"Failed to get user profile via API: {e}")
        
        return None
    
    async def _get_user_events_fallback(self, user: Any, query: str, limit: int) -> List[Dict]:
        """获取用户事件的fallback方法"""
        try:
            # 尝试通过HTTP API搜索用户事件
            import httpx
            
            events_url = f"{self.config.project_url}/api/v1/users/events/search"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "query": query,
                "limit": limit
            }
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(events_url, headers=headers, json=data)
                
                if response.status_code == 200:
                    events_data = response.json()
                    logger.debug(f"Retrieved {len(events_data)} events from Memobase API")
                    return events_data.get('events', [])
                else:
                    logger.debug(f"Events API returned status {response.status_code}")
                    
        except Exception as e:
            logger.debug(f"Failed to get user events via API: {e}")
        
        return []
    
    def _convert_context_to_memories(
        self, context_data: Dict, user_id: str, character_id: str
    ) -> List[Memory]:
        """将Memobase上下文转换为Memory对象"""
        memories = []
        
        try:
            # 转换用户画像
            if "user_profile" in context_data and context_data["user_profile"]:
                profile_data = context_data["user_profile"]
                
                # 处理结构化的用户画像数据
                if isinstance(profile_data, dict):
                    for category, details in profile_data.items():
                        if details:  # 只处理非空数据
                            content = f"用户{category}: {json.dumps(details, ensure_ascii=False) if isinstance(details, dict) else str(details)}"
                            profile_memory = Memory(
                                user_id=user_id,
                                character_id=character_id,
                                content=content,
                                memory_type=MemoryType.LONG_TERM,
                                importance_score=0.9,
                                created_at=datetime.now(),
                                metadata={
                                    "source": "memobase_profile",
                                    "type": "user_profile",
                                    "category": category
                                }
                            )
                            memories.append(profile_memory)
                else:
                    # 处理简单字符串格式的画像
                    profile_memory = Memory(
                        user_id=user_id,
                        character_id=character_id,
                        content=f"用户画像: {str(profile_data)}",
                        memory_type=MemoryType.LONG_TERM,
                        importance_score=0.9,
                        created_at=datetime.now(),
                        metadata={
                            "source": "memobase_profile",
                            "type": "user_profile"
                        }
                    )
                    memories.append(profile_memory)
            
            # 转换相关事件
            if "relevant_events" in context_data and context_data["relevant_events"]:
                for event in context_data["relevant_events"]:
                    # 处理事件数据结构
                    if isinstance(event, dict):
                        event_content = event.get('content', str(event))
                        event_time = event.get('timestamp', datetime.now().isoformat())
                        event_tags = event.get('tags', [])
                    else:
                        event_content = str(event)
                        event_time = datetime.now().isoformat()
                        event_tags = []
                    
                    event_memory = Memory(
                        user_id=user_id,
                        character_id=character_id,
                        content=f"相关事件: {event_content}",
                        memory_type=MemoryType.LONG_TERM,
                        importance_score=0.8,
                        created_at=datetime.now(),
                        metadata={
                            "source": "memobase_event",
                            "type": "event",
                            "event_time": event_time,
                            "tags": event_tags
                        }
                    )
                    memories.append(event_memory)
            
            # 转换记忆数据
            if "memories" in context_data and context_data["memories"]:
                for memory_data in context_data["memories"]:
                    # 处理记忆数据结构
                    if isinstance(memory_data, dict):
                        memory_content = memory_data.get('content', str(memory_data))
                        memory_score = memory_data.get('score', 0.7)
                        memory_timestamp = memory_data.get('timestamp')
                    else:
                        memory_content = str(memory_data)
                        memory_score = 0.7
                        memory_timestamp = None
                    
                    memory = Memory(
                        user_id=user_id,
                        character_id=character_id,
                        content=memory_content,
                        memory_type=MemoryType.LONG_TERM,
                        importance_score=float(memory_score),
                        created_at=datetime.fromisoformat(memory_timestamp) if memory_timestamp else datetime.now(),
                        metadata={
                            "source": "memobase_memory",
                            "type": "memory"
                        }
                    )
                    memories.append(memory)
            
            logger.debug(f"Converted {len(memories)} context items to Memory objects")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to convert context to memories: {e}")
            return []
    
    def _get_recent_session_memories(
        self, session_id: str, user_id: str, character_id: str, query: str
    ) -> List[Memory]:
        """获取最近会话的相关记忆"""
        memories = []
        
        try:
            if session_id not in self.session_cache:
                return memories
            
            messages = self.session_cache[session_id]
            query_lower = query.lower()
            
            # 简单的关键词匹配
            for message in messages[-self.config.context_limit:]:
                if query_lower in message.content.lower():
                    memory = Memory(
                        user_id=user_id,
                        character_id=character_id,
                        content=f"{message.role.value}: {message.content}",
                        memory_type=MemoryType.SHORT_TERM,
                        importance_score=0.6,
                        created_at=message.timestamp,
                        metadata={
                            "source": "recent_session",
                            "session_id": session_id,
                            "message_id": message.id
                        }
                    )
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get recent session memories: {e}")
            return []
    
    def _deduplicate_memories(self, memories: List[Memory]) -> List[Memory]:
        """去重记忆"""
        seen_contents = set()
        unique_memories = []
        
        for memory in memories:
            # 使用内容的哈希值进行去重
            content_hash = hash(memory.content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_memories.append(memory)
        
        return unique_memories
    
    async def get_conversation_history(self, session_id: str) -> List[Message]:
        """获取对话历史"""
        try:
            # 优先从缓存获取
            if session_id in self.session_cache:
                return self.session_cache[session_id]
            
            # 从数据库获取
            messages = await conversation_db.load_messages(session_id)
            
            # 更新缓存
            self.session_cache[session_id] = messages
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """获取对话上下文"""
        try:
            messages = await self.get_conversation_history(session_id)
            
            # 获取用户信息（从第一条消息中提取）
            user_id = ""
            character_id = ""
            
            if messages:
                first_message = messages[0]
                user_id = first_message.metadata.get("user_id", "")
                character_id = first_message.metadata.get("character_id", "")
            
            # 构建上下文
            context = {
                "session_id": session_id,
                "user_id": user_id,
                "character_id": character_id,
                "conversation_history": [
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat()
                    }
                    for msg in messages
                ],
                "total_messages": len(messages),
                "recent_messages": len(messages[-self.config.context_limit:]),
            }
            
            # 如果有用户信息，尝试获取用户画像
            if user_id and character_id:
                try:
                    user = self._get_or_create_user(user_id, character_id)
                    if user is not None and hasattr(user, 'flush'):
                        user.flush(sync=True)
                        context["user_profile_available"] = True
                    else:
                        context["user_profile_available"] = False
                except Exception:
                    context["user_profile_available"] = False
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "conversation_history": [],
                "total_messages": 0
            }
    
    async def add_message(
        self, user_id: str, content: str, role: str, session_id: str = None, metadata: Dict = None
    ) -> bool:
        """添加消息到Memobase"""
        try:
            from models import Message, MessageRole
            
            # 创建消息对象
            message = Message(
                role=MessageRole(role),
                content=content,
                metadata=metadata or {}
            )
            
            # 添加到对话
            await self.add_message_to_conversation(user_id, session_id or "default", message)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            return False
    
    async def search_memories(
        self, user_id: str, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索记忆"""
        try:
            # 获取用户对象
            user = self._get_or_create_user(user_id)
            
            if user is None:
                logger.info(f"Memobase not available, using local search for user {user_id}")
            
            # 直接使用本地搜索（Memobase客户端没有search_memories方法）
            results = await self._local_memory_search(user_id, query, limit)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    async def _local_memory_search(
        self, user_id: str, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """本地记忆搜索（备用方案）"""
        try:
            results = []
            query_lower = query.lower()
            
            # 搜索会话缓存
            for session_id, messages in self.session_cache.items():
                for message in messages:
                    if (message.metadata.get("user_id") == user_id and 
                        query_lower in message.content.lower()):
                        results.append({
                            'content': message.content,
                            'score': 0.7,
                            'timestamp': message.timestamp.isoformat(),
                            'metadata': {
                                'session_id': session_id,
                                'role': message.role.value,
                                **message.metadata
                            }
                        })
            
            # 按时间排序，返回最新的
            results.sort(key=lambda x: x['timestamp'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to perform local memory search: {e}")
            return []
    
    async def cleanup_old_sessions(self, days: int = 30) -> Dict[str, Any]:
        """清理旧会话"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cleaned_sessions = 0
            
            # 清理会话缓存
            sessions_to_remove = []
            for session_id, messages in self.session_cache.items():
                if messages and messages[-1].timestamp < cutoff_date:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.session_cache[session_id]
                cleaned_sessions += 1
            
            # 清理数据库中的旧数据
            try:
                await conversation_db.cleanup_old_data(days)
            except AttributeError:
                logger.warning("Database cleanup method not available")
            
            # 注意：Memobase客户端没有cleanup_old_data方法
            # 数据清理主要通过本地缓存和数据库清理完成
            logger.debug("Memobase data cleanup not available in current client version")
            
            result = {
                "cleaned_sessions": cleaned_sessions,
                "cutoff_date": cutoff_date.isoformat(),
                "status": "success"
            }
            
            logger.info(f"Cleaned up {cleaned_sessions} old sessions")
            return result
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    async def generate_conversation_summary(
        self, session_id: str, user_id: str, character_id: str = "", location: str = ""
    ) -> str:
        """生成对话总结"""
        try:
            # 确保数据已刷新到Memobase
            if character_id:
                await self._flush_user_data(user_id, character_id)
            
            # 获取对话历史
            messages = await self.get_conversation_history(session_id)
            
            if not messages:
                return "对话为空，无需总结"
            
            # 使用Memobase的能力生成总结
            try:
                user = self._get_or_create_user(user_id, character_id)
                # 刷新用户数据
                if user is not None and hasattr(user, 'flush'):
                    user.flush(sync=True)
            except Exception as e:
                logger.warning(f"Failed to flush user data: {e}")
            
            # 构建总结
            summary_parts = [
                f"对话会话: {session_id}",
                f"消息数量: {len(messages)}",
                f"时间范围: {messages[0].timestamp} 到 {messages[-1].timestamp}",
                f"地点: {location}" if location else ""
            ]
            
            # 添加关键内容摘要
            key_messages = [msg for msg in messages if len(msg.content) > 20]
            if key_messages:
                summary_parts.append(f"主要讨论了 {len(key_messages)} 个重要话题")
            
            summary = "\n".join(filter(None, summary_parts))
            
            # 保存总结到数据库
            try:
                await conversation_db.save_summary(
                    session_id=session_id,
                    user_id=user_id,
                    summary=summary
                )
            except (AttributeError, Exception) as e:
                logger.warning(f"Failed to save summary to database: {e}")
            
            logger.info(f"Generated conversation summary for session {session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return f"总结生成失败: {str(e)}"
    
    async def _periodic_flush(self):
        """定期刷新任务"""
        while True:
            try:
                await asyncio.sleep(self.config.flush_interval)
                
                # 刷新所有待处理的用户数据
                for cache_key in list(self.pending_flushes.keys()):
                    if self.pending_flushes[cache_key]:
                        user_id, character_id = cache_key.split("_", 1)
                        await self._flush_user_data(user_id, character_id)
                
                logger.debug("Completed periodic flush")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
    
    async def _periodic_cleanup(self):
        """定期清理任务"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时执行一次
                
                # 清理过期的会话缓存
                await self._cleanup_session_cache()
                
                # 清理用户缓存
                await self._cleanup_user_cache()
                
                logger.debug("Completed periodic cleanup")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def _cleanup_session_cache(self):
        """清理会话缓存"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            sessions_to_remove = []
            
            for session_id, messages in self.session_cache.items():
                if messages and messages[-1].timestamp < cutoff_time:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.session_cache[session_id]
                logger.debug(f"Cleaned up session cache for {session_id}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup session cache: {e}")
    
    async def _cleanup_user_cache(self):
        """清理用户缓存"""
        try:
            # 保持用户缓存，因为用户对象相对稳定
            # 只在内存压力大时才清理
            if len(self.user_cache) > 1000:
                # 清理一半最旧的缓存
                items_to_remove = len(self.user_cache) // 2
                cache_keys = list(self.user_cache.keys())
                for key in cache_keys[:items_to_remove]:
                    del self.user_cache[key]
                logger.debug(f"Cleaned up {items_to_remove} user cache entries")
                
        except Exception as e:
            logger.error(f"Failed to cleanup user cache: {e}")
    

    
    async def shutdown(self):
        """关闭记忆管理器"""
        try:
            # 刷新所有待处理的数据
            for cache_key in list(self.pending_flushes.keys()):
                if self.pending_flushes[cache_key]:
                    user_id, character_id = cache_key.split("_", 1)
                    await self._flush_user_data(user_id, character_id)
            
            # 取消后台任务
            for task in self._background_tasks:
                task.cancel()
            
            # 等待任务完成
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            logger.info("Memobase memory manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# 全局实例（将在main.py中初始化）
memobase_memory_manager = None