import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from config import settings
from models import Memory, MemoryType, Message, MessageRole, ConversationSession
from vector_store import vector_store
from database import conversation_db


logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器 - 统一向量化存储所有对话历史"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            temperature=settings.temperature,
        )
        # 保留conversations用于兼容性
        self.conversations: Dict[str, ConversationSession] = {}

    async def initialize(self):
        """初始化记忆管理器"""
        try:
            await conversation_db.initialize()
            logger.info("Memory manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise

    async def add_message_to_conversation(
        self, user_id: str, session_id: str, message: Message
    ) -> ConversationSession:
        """添加消息到对话会话并保存到向量库"""
        try:
            # 保存到数据库（持久化）
            await conversation_db.save_message(session_id, message)

            # 将对话消息保存到向量库
            await self._save_message_to_vector_store(user_id, session_id, message)

            # 更新内存中的对话会话
            if session_id not in self.conversations:
                self.conversations[session_id] = ConversationSession(
                    session_id=session_id,
                    user_id=user_id,
                    character_id=message.metadata.get("character_id", ""),
                    messages=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

            conversation = self.conversations[session_id]
            conversation.messages.append(message)
            conversation.updated_at = datetime.now()

            # 保持内存中的消息数量限制
            if len(conversation.messages) > settings.max_conversation_history:
                conversation.messages = conversation.messages[
                    -settings.max_conversation_history :
                ]

            logger.info(f"Added message to conversation {session_id} and vector store")
            return conversation

        except Exception as e:
            logger.error(f"Failed to add message to conversation: {e}")
            raise

    async def _save_message_to_vector_store(
        self, user_id: str, session_id: str, message: Message
    ):
        """将消息保存到向量库"""
        try:
            # 创建记忆对象
            memory = Memory(
                user_id=user_id,
                character_id=message.metadata.get("character_id", ""),
                content=f"{message.role.value}: {message.content}",
                memory_type=MemoryType.SHORT_TERM,  # 所有对话都作为短期记忆
                importance_score=0.5,  # 统一重要性评分
                created_at=message.timestamp,
                metadata={
                    "session_id": session_id,
                    "message_id": message.id,
                    "role": message.role.value,
                    **message.metadata,
                },
            )

            # 保存到向量存储
            await vector_store.add_memory(memory)

            logger.debug(f"Saved message to vector store: {message.role.value}")

        except Exception as e:
            logger.error(f"Failed to save message to vector store: {e}")
            # 不抛出异常，避免影响主流程

    async def extract_important_memories(
        self, user_id: str, character_id: str = None, days: int = 7
    ) -> List[Memory]:
        """提取重要记忆（已废弃，所有记忆都保存在向量库中）"""
        try:
            # 直接从向量库搜索记忆
            memories = await vector_store.search_memories(
                user_id=user_id, character_id=character_id, query="", limit=100
            )

            return memories

        except Exception as e:
            logger.error(f"Failed to extract memories: {e}")
            return []

    async def save_memories(self, memories: List[Memory]) -> List[str]:
        """保存记忆到向量存储"""
        try:
            memory_ids = []
            for memory in memories:
                memory_id = await vector_store.add_memory(memory)
                memory_ids.append(memory_id)

            logger.info(f"Saved {len(memory_ids)} memories to vector store")
            return memory_ids

        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
            raise

    async def retrieve_relevant_memories(
        self,
        query: str,
        user_id: str,
        character_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 5,
    ) -> List[Memory]:
        """检索相关记忆"""
        try:
            results = await vector_store.search_memories(
                query=query,
                user_id=user_id,
                character_id=character_id,
                memory_type=memory_type,
                limit=limit,
            )

            memories = [memory for memory, score in results]

            # 更新访问信息
            for memory in memories:
                if memory.id:
                    await vector_store.update_memory_access(memory.id)

            logger.info(f"Retrieved {len(memories)} relevant memories")
            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve relevant memories: {e}")
            return []

    async def get_conversation_context(
        self,
        user_id: str,
        current_message: str = "",
        character_id: str = None,
        limit: int = 10,
    ) -> str:
        """获取对话上下文，基于当前消息查询相关历史对话"""
        try:
            # 如果有当前消息，使用它作为查询来搜索相关对话历史
            query = current_message if current_message else ""

            # 从向量库搜索相关对话历史
            memories = await vector_store.search_memories(
                user_id=user_id, character_id=character_id, query=query, limit=limit
            )

            if not memories:
                return "No relevant conversation context found."

            # 构建上下文，按时间排序
            memories.sort(key=lambda x: x.created_at)
            context_parts = []

            for memory in memories:
                # 提取角色和内容
                content = memory.content
                timestamp = memory.created_at.strftime("%Y-%m-%d %H:%M")
                context_parts.append(f"[{timestamp}] {content}")

            context = "\n".join(context_parts)
            logger.debug(f"Retrieved {len(memories)} relevant conversation memories")
            return context

        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return "Error retrieving conversation context."

    async def get_conversation_history(
        self, session_id: str, limit: int = 50
    ) -> List[Message]:
        """获取对话历史"""
        try:
            # 优先从内存中获取
            if session_id in self.conversations:
                conversation = self.conversations[session_id]
                if conversation.messages:
                    return conversation.messages[-limit:]

            # 从数据库加载
            messages = await conversation_db.load_messages(session_id, limit)

            # 更新内存缓存
            if messages and session_id not in self.conversations:
                # 获取第一条消息的元数据来确定user_id和character_id
                first_message = messages[0]
                self.conversations[session_id] = ConversationSession(
                    session_id=session_id,
                    user_id=first_message.metadata.get("user_id", ""),
                    character_id=first_message.metadata.get("character_id", ""),
                    messages=messages,
                    created_at=messages[0].timestamp if messages else datetime.now(),
                    updated_at=messages[-1].timestamp if messages else datetime.now(),
                )

            logger.debug(f"Retrieved {len(messages)} messages from database")
            return messages

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    async def clear_old_conversations(self, days: int = 30):
        """清理旧对话"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            sessions_to_remove = []

            for session_id, conversation in self.conversations.items():
                if conversation.updated_at < cutoff_date:
                    sessions_to_remove.append(session_id)

            for session_id in sessions_to_remove:
                del self.conversations[session_id]

            logger.info(f"Cleared {len(sessions_to_remove)} old conversations")

        except Exception as e:
            logger.error(f"Failed to clear old conversations: {e}")

    async def generate_conversation_summary(
        self, 
        session_id: str, 
        user_id: str, 
        character_id: str,
        location: str = "网络空间"
    ) -> Optional[Memory]:
        """生成对话总结记忆"""
        try:
            # 获取当前对话的所有消息
            messages = await self.get_conversation_history(session_id)
            
            if not messages or len(messages) < 2:
                logger.info(f"Session {session_id} has insufficient messages for summary")
                return None
            
            # 构建对话内容用于总结
            conversation_text = ""
            for msg in messages:
                role = "用户" if msg.role == MessageRole.USER else "AI角色"
                conversation_text += f"{role}: {msg.content}\n"
            
            # 创建总结提示
            summary_prompt = ChatPromptTemplate.from_template(
                """请根据以下对话内容，生成一个简洁的记忆总结，格式如下：
                时间 + 地点 + 发生的事 + 心情
                
                例如：2025.7.23 20:17:30 - 三夏向我介绍了他，他是一名程序员，出生于1996年3月10日，我很开心
                
                对话内容：
                {conversation}
                
                地点：{location}
                当前时间：{current_time}
                
                请生成一个自然、简洁的记忆总结（不超过100字）："""
            )
            
            # 调用LLM生成总结
            current_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
            prompt_messages = summary_prompt.format_messages(
                conversation=conversation_text,
                location=location,
                current_time=current_time
            )
            
            response = await self.llm.ainvoke(prompt_messages)
            summary_content = response.content.strip()
            
            # 创建总结记忆
            summary_memory = Memory(
                user_id=user_id,
                character_id=character_id,
                content=summary_content,
                memory_type=MemoryType.LONG_TERM,  # 总结作为长期记忆
                importance_score=0.8,  # 总结具有较高重要性
                created_at=datetime.now(),
                metadata={
                    "session_id": session_id,
                    "summary_type": "conversation_summary",
                    "message_count": len(messages),
                    "location": location,
                    "generated_at": current_time
                }
            )
            
            # 保存总结记忆到向量存储
            memory_id = await vector_store.add_memory(summary_memory)
            summary_memory.id = memory_id
            
            logger.info(f"Generated conversation summary for session {session_id}: {summary_content[:50]}...")
            return summary_memory
            
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return None

    async def end_conversation_with_summary(
        self, 
        session_id: str, 
        user_id: str, 
        character_id: str,
        location: str = "未知地点"
    ) -> Optional[str]:
        """结束对话并生成总结"""
        try:
            # 生成对话总结
            summary_memory = await self.generate_conversation_summary(
                session_id, user_id, character_id, location
            )
            
            if summary_memory:
                # 清理内存中的对话会话（可选）
                if session_id in self.conversations:
                    del self.conversations[session_id]
                
                logger.info(f"Conversation {session_id} ended with summary generated")
                return summary_memory.content
            else:
                logger.warning(f"Failed to generate summary for conversation {session_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to end conversation with summary: {e}")
            return None


# 全局记忆管理器实例
memory_manager = MemoryManager()
