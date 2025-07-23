#!/usr/bin/env python3
"""
混合记忆管理器

结合向量记忆和持久化摘要记忆，解决记忆丢失和错乱问题：
1. 短期记忆：使用向量存储保存最近的对话
2. 中期记忆：使用ConversationSummaryMemory进行滚动摘要
3. 长期记忆：重要事件和总结的持久化存储
4. 智能检索：根据查询类型选择合适的记忆源
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from config import settings
from models import Memory, MemoryType, Message, MessageRole, ConversationSession
from vector_store import vector_store
from database import conversation_db


logger = logging.getLogger(__name__)


class HybridMemoryManager:
    """混合记忆管理器 - 结合向量记忆、摘要记忆和长期记忆"""

    def __init__(self, vector_store, db_manager):
        self.vector_store = vector_store
        self.db_manager = db_manager
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            temperature=settings.temperature,
        )
        
        # 会话级别的摘要记忆管理器
        self.session_summaries: Dict[str, ConversationSummaryMemory] = {}
        
        # 短期记忆缓存（最近的对话）
        self.short_term_cache: Dict[str, List[Message]] = {}
        
        # 配置参数
        self.short_term_limit = 20  # 短期记忆保留的消息数量
        self.summary_trigger_count = 10  # 触发摘要的消息数量
        self.max_summary_age_hours = 24  # 摘要的最大有效期（小时）
        
        logger.info("Hybrid memory manager initialized")

    async def initialize(self):
        """初始化混合记忆管理器"""
        try:
            await conversation_db.initialize()
            logger.info("Hybrid memory manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid memory manager: {e}")
            raise

    def _get_session_summary_memory(self, session_id: str) -> ConversationSummaryMemory:
        """获取或创建会话的摘要记忆管理器"""
        if session_id not in self.session_summaries:
            # 创建新的摘要记忆管理器
            self.session_summaries[session_id] = ConversationSummaryMemory(
                llm=self.llm,
                return_messages=True,
                ai_prefix="AI角色",
                human_prefix="用户"
            )
            logger.debug(f"Created new summary memory for session {session_id}")
        
        return self.session_summaries[session_id]

    async def add_message_to_conversation(
        self, user_id: str, session_id: str, message: Message
    ) -> ConversationSession:
        """添加消息到混合记忆系统"""
        try:
            # 1. 保存到数据库（持久化）
            await conversation_db.save_message(session_id, message)
            
            # 2. 更新短期记忆缓存
            if session_id not in self.short_term_cache:
                self.short_term_cache[session_id] = []
            
            self.short_term_cache[session_id].append(message)
            
            # 保持短期记忆限制
            if len(self.short_term_cache[session_id]) > self.short_term_limit:
                # 将超出的消息移到摘要记忆中
                overflow_messages = self.short_term_cache[session_id][:-self.short_term_limit]
                self.short_term_cache[session_id] = self.short_term_cache[session_id][-self.short_term_limit:]
                
                # 更新摘要记忆
                await self._update_summary_memory(session_id, overflow_messages)
            
            # 3. 保存重要消息到向量存储
            await self._save_important_message_to_vector(user_id, session_id, message)
            
            # 4. 检查是否需要生成中期摘要
            await self._check_and_generate_summary(session_id, user_id, message.metadata.get("character_id", ""))
            
            # 5. 构建返回的对话会话对象
            conversation = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                character_id=message.metadata.get("character_id", ""),
                messages=self.short_term_cache[session_id],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            
            logger.info(f"Added message to hybrid memory system for session {session_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"Failed to add message to hybrid memory: {e}")
            raise

    async def _update_summary_memory(self, session_id: str, messages: List[Message]):
        """更新摘要记忆"""
        try:
            summary_memory = self._get_session_summary_memory(session_id)
            
            # 将消息转换为LangChain格式并添加到摘要记忆
            for message in messages:
                if message.role == MessageRole.USER:
                    langchain_message = HumanMessage(content=message.content)
                else:
                    langchain_message = AIMessage(content=message.content)
                
                # 添加到摘要记忆的聊天历史
                summary_memory.chat_memory.add_message(langchain_message)
            
            logger.debug(f"Updated summary memory for session {session_id} with {len(messages)} messages")
            
        except Exception as e:
            logger.error(f"Failed to update summary memory: {e}")

    async def _save_important_message_to_vector(
        self, user_id: str, session_id: str, message: Message
    ):
        """保存重要消息到向量存储"""
        try:
            # 判断消息重要性
            importance_score = await self._calculate_message_importance(message)
            
            # 只保存重要性较高的消息到向量存储
            if importance_score >= 0.6:
                memory = Memory(
                    user_id=user_id,
                    character_id=message.metadata.get("character_id", ""),
                    content=f"{message.role.value}: {message.content}",
                    memory_type=MemoryType.SHORT_TERM,
                    importance_score=importance_score,
                    created_at=message.timestamp,
                    metadata={
                        "session_id": session_id,
                        "message_id": message.id,
                        "role": message.role.value,
                        **message.metadata,
                    },
                )
                
                await vector_store.add_memory(memory)
                logger.debug(f"Saved important message to vector store (importance: {importance_score:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to save important message to vector store: {e}")

    async def _calculate_message_importance(self, message: Message) -> float:
        """计算消息重要性评分"""
        try:
            # 基于内容长度、关键词等简单规则计算重要性
            content = message.content.lower()
            importance = 0.3  # 基础分数
            
            # 长度因子
            if len(content) > 50:
                importance += 0.2
            
            # 关键词因子
            important_keywords = [
                "记住", "重要", "名字", "生日", "喜欢", "不喜欢", "工作", "家庭",
                "remember", "important", "name", "birthday", "like", "dislike", "work", "family"
            ]
            
            for keyword in important_keywords:
                if keyword in content:
                    importance += 0.1
            
            # 问题因子
            if "?" in content or "？" in content:
                importance += 0.1
            
            return min(importance, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate message importance: {e}")
            return 0.5

    async def _check_and_generate_summary(
        self, session_id: str, user_id: str, character_id: str
    ):
        """检查是否需要生成中期摘要"""
        try:
            # 获取当前会话的消息数量
            message_count = len(self.short_term_cache.get(session_id, []))
            
            # 每达到触发数量就生成一次摘要
            if message_count > 0 and message_count % self.summary_trigger_count == 0:
                await self._generate_periodic_summary(session_id, user_id, character_id)
                
        except Exception as e:
            logger.error(f"Failed to check and generate summary: {e}")

    async def _generate_periodic_summary(
        self, session_id: str, user_id: str, character_id: str
    ):
        """生成周期性摘要"""
        try:
            summary_memory = self._get_session_summary_memory(session_id)
            
            # 获取当前摘要
            memory_variables = summary_memory.load_memory_variables({})
            current_summary = ""
            
            if "history" in memory_variables:
                history = memory_variables["history"]
                if isinstance(history, list) and history:
                    # 如果是消息列表，提取系统消息的内容
                    for msg in history:
                        if isinstance(msg, SystemMessage):
                            current_summary = msg.content
                            break
                elif isinstance(history, str):
                    current_summary = history
            
            if current_summary and current_summary.strip():
                # 保存摘要到向量存储作为长期记忆
                summary_memory_obj = Memory(
                    user_id=user_id,
                    character_id=character_id,
                    content=f"对话摘要: {current_summary}",
                    memory_type=MemoryType.LONG_TERM,
                    importance_score=0.8,
                    created_at=datetime.now(),
                    metadata={
                        "session_id": session_id,
                        "summary_type": "periodic_summary",
                        "generated_at": datetime.now().isoformat()
                    }
                )
                
                await vector_store.add_memory(summary_memory_obj)
                logger.info(f"Generated periodic summary for session {session_id}: {current_summary[:50]}...")
                
        except Exception as e:
            logger.error(f"Failed to generate periodic summary: {e}")

    async def retrieve_relevant_memories(
        self,
        query: str,
        user_id: str,
        character_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """智能检索相关记忆"""
        try:
            all_memories = []
            
            # 1. 从短期记忆缓存中搜索
            if session_id and session_id in self.short_term_cache:
                short_term_memories = await self._search_short_term_memories(
                    query, session_id, user_id, character_id
                )
                all_memories.extend(short_term_memories)
            
            # 2. 从摘要记忆中获取上下文
            if session_id and session_id in self.session_summaries:
                summary_context = await self._get_summary_context(session_id)
                if summary_context:
                    # 将摘要作为记忆对象
                    summary_memory = Memory(
                        user_id=user_id,
                        character_id=character_id,
                        content=f"会话摘要: {summary_context}",
                        memory_type=MemoryType.LONG_TERM,
                        importance_score=0.9,
                        created_at=datetime.now(),
                        metadata={"source": "summary_memory"}
                    )
                    all_memories.append(summary_memory)
            
            # 3. 从向量存储中搜索长期记忆
            vector_memories = await vector_store.search_memories(
                query=query,
                user_id=user_id,
                character_id=character_id,
                memory_type=memory_type,
                limit=limit
            )
            
            # 提取记忆对象（忽略分数）
            for item in vector_memories:
                if isinstance(item, tuple):
                    memory, score = item
                    all_memories.append(memory)
                else:
                    all_memories.append(item)
            
            # 4. 去重和排序
            unique_memories = self._deduplicate_memories(all_memories)
            sorted_memories = sorted(
                unique_memories, 
                key=lambda x: (x.importance_score, x.created_at), 
                reverse=True
            )
            
            # 5. 限制返回数量
            result = sorted_memories[:limit]
            
            logger.info(f"Retrieved {len(result)} relevant memories from hybrid system")
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant memories: {e}")
            return []

    async def _search_short_term_memories(
        self, query: str, session_id: str, user_id: str, character_id: str
    ) -> List[Memory]:
        """搜索短期记忆缓存"""
        try:
            if session_id not in self.short_term_cache:
                return []
            
            messages = self.short_term_cache[session_id]
            relevant_memories = []
            
            # 简单的关键词匹配
            query_lower = query.lower()
            
            for message in messages:
                if query_lower in message.content.lower():
                    memory = Memory(
                        user_id=user_id,
                        character_id=character_id,
                        content=f"{message.role.value}: {message.content}",
                        memory_type=MemoryType.SHORT_TERM,
                        importance_score=0.7,
                        created_at=message.timestamp,
                        metadata={
                            "source": "short_term_cache",
                            "session_id": session_id
                        }
                    )
                    relevant_memories.append(memory)
            
            return relevant_memories
            
        except Exception as e:
            logger.error(f"Failed to search short term memories: {e}")
            return []

    async def _get_summary_context(self, session_id: str) -> Optional[str]:
        """获取摘要上下文"""
        try:
            if session_id not in self.session_summaries:
                return None
            
            summary_memory = self.session_summaries[session_id]
            memory_variables = summary_memory.load_memory_variables({})
            
            if "history" in memory_variables:
                history = memory_variables["history"]
                if isinstance(history, list) and history:
                    # 提取系统消息的内容
                    for msg in history:
                        if isinstance(msg, SystemMessage):
                            return msg.content
                elif isinstance(history, str):
                    return history
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get summary context: {e}")
            return None

    def _deduplicate_memories(self, memories: List[Memory]) -> List[Memory]:
        """去重记忆"""
        seen_contents = set()
        unique_memories = []
        
        for memory in memories:
            # 使用内容的前50个字符作为去重标识
            content_key = memory.content[:50].strip()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_memories.append(memory)
        
        return unique_memories

    async def get_conversation_history(self, session_id: str) -> List[Message]:
        """获取对话历史记录"""
        try:
            # 首先从短期记忆缓存获取
            if session_id in self.short_term_cache:
                return self.short_term_cache[session_id]
            
            # 如果缓存中没有，从数据库获取
            db_messages = await conversation_db.load_messages(session_id)
            
            # 更新缓存
            self.short_term_cache[session_id] = db_messages[-self.short_term_limit:]
            
            return db_messages
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    async def get_conversation_context(
        self,
        user_id: str,
        session_id: str,
        current_message: str = "",
        character_id: str = None,
        limit: int = 10,
    ) -> str:
        """获取混合对话上下文"""
        try:
            # 使用智能检索获取相关记忆
            memories = await self.retrieve_relevant_memories(
                query=current_message,
                user_id=user_id,
                character_id=character_id,
                session_id=session_id,
                limit=limit
            )
            
            if not memories:
                return "No relevant conversation context found."
            
            # 构建上下文
            context_parts = []
            
            for memory in memories:
                timestamp = memory.created_at.strftime("%Y-%m-%d %H:%M")
                source = memory.metadata.get("source", "vector_store")
                context_parts.append(f"[{timestamp}|{source}] {memory.content}")
            
            context = "\n".join(context_parts)
            logger.debug(f"Retrieved hybrid context with {len(memories)} memories")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get hybrid conversation context: {e}")
            return "Error retrieving conversation context."

    async def cleanup_old_sessions(self, hours: int = 24):
        """清理旧会话的内存数据"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            sessions_to_remove = []
            
            # 清理短期记忆缓存
            for session_id in list(self.short_term_cache.keys()):
                # 这里简化处理，实际应该检查最后更新时间
                if len(self.short_term_cache[session_id]) == 0:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                if session_id in self.short_term_cache:
                    del self.short_term_cache[session_id]
                if session_id in self.session_summaries:
                    del self.session_summaries[session_id]
            
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")

    async def generate_conversation_summary(self, session_id: str, user_id: str, character_id: str, location: str = "") -> str:
        """生成对话总结"""
        try:
            # 获取对话历史
            messages = await self.get_conversation_history(session_id)
            
            if not messages:
                return "对话为空，无需总结"
            
            # 生成摘要
            summary_memory = self._get_session_summary_memory(session_id)
            
            # 将所有消息添加到摘要记忆中
            for msg in messages:
                if msg.role == MessageRole.USER:
                    summary_memory.chat_memory.add_user_message(msg.content)
                else:
                    summary_memory.chat_memory.add_ai_message(msg.content)
            
            # 获取摘要
            summary = summary_memory.buffer
            
            logger.info(f"Generated summary for conversation {session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return f"总结生成失败: {str(e)}"
    
    async def end_conversation_and_summarize(self, session_id: str, user_id: str, character_id: str, location: str = "") -> str:
        """结束对话并生成总结"""
        try:
            # 生成最终摘要
            final_summary = await self.generate_conversation_summary(session_id, user_id, character_id, location)
            
            # 清理缓存
            if session_id in self.short_term_cache:
                del self.short_term_cache[session_id]
            if session_id in self.session_summaries:
                del self.session_summaries[session_id]
            
            logger.info(f"Conversation {session_id} ended and summarized")
            return final_summary
            
        except Exception as e:
            logger.error(f"Failed to end conversation and summarize: {e}")
            return f"总结生成失败: {str(e)}"

    async def generate_final_summary(
        self, 
        session_id: str, 
        user_id: str, 
        character_id: str,
        location: str = "未知地点"
    ) -> Optional[Memory]:
        """生成最终对话总结"""
        try:
            # 获取所有相关记忆
            all_messages = []
            
            # 从短期缓存获取
            if session_id in self.short_term_cache:
                all_messages.extend(self.short_term_cache[session_id])
            
            # 从数据库获取完整历史
            db_messages = await conversation_db.load_messages(session_id, limit=1000)
            all_messages.extend(db_messages)
            
            # 去重
            seen_ids = set()
            unique_messages = []
            for msg in all_messages:
                if msg.id not in seen_ids:
                    seen_ids.add(msg.id)
                    unique_messages.append(msg)
            
            if len(unique_messages) < 2:
                return None
            
            # 获取摘要上下文
            summary_context = ""
            if session_id in self.session_summaries:
                summary_context = await self._get_summary_context(session_id) or ""
            
            # 构建完整对话内容
            conversation_parts = []
            if summary_context:
                conversation_parts.append(f"之前的对话摘要: {summary_context}")
            
            for msg in unique_messages[-20:]:  # 只取最近20条消息
                role = "用户" if msg.role == MessageRole.USER else "AI角色"
                conversation_parts.append(f"{role}: {msg.content}")
            
            conversation_text = "\n".join(conversation_parts)
            
            # 生成最终总结
            summary_prompt = ChatPromptTemplate.from_template(
                """请根据以下完整对话内容，生成一个全面的记忆总结，格式如下：
                时间 + 地点 + 发生的事 + 心情
                
                要求：
                1. 总结要包含对话中的关键信息和重要事件
                2. 体现用户的情感状态和AI角色的互动
                3. 简洁但信息丰富（不超过150字）
                
                对话内容：
                {conversation}
                
                地点：{location}
                当前时间：{current_time}
                
                请生成最终总结："""
            )
            
            current_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
            prompt_messages = summary_prompt.format_messages(
                conversation=conversation_text,
                location=location,
                current_time=current_time
            )
            
            response = await self.llm.ainvoke(prompt_messages)
            summary_content = response.content.strip()
            
            # 创建最终总结记忆
            final_summary = Memory(
                user_id=user_id,
                character_id=character_id,
                content=summary_content,
                memory_type=MemoryType.LONG_TERM,
                importance_score=0.95,  # 最终总结具有最高重要性
                created_at=datetime.now(),
                metadata={
                    "session_id": session_id,
                    "summary_type": "final_summary",
                    "message_count": len(unique_messages),
                    "location": location,
                    "generated_at": current_time,
                    "has_previous_summary": bool(summary_context)
                }
            )
            
            # 保存到向量存储
            memory_id = await vector_store.add_memory(final_summary)
            final_summary.id = memory_id
            
            # 清理会话数据
            if session_id in self.short_term_cache:
                del self.short_term_cache[session_id]
            if session_id in self.session_summaries:
                del self.session_summaries[session_id]
            
            logger.info(f"Generated final summary for session {session_id}: {summary_content[:50]}...")
            return final_summary
            
        except Exception as e:
            logger.error(f"Failed to generate final summary: {e}")
            return None


# 全局混合记忆管理器实例将在main.py中初始化
hybrid_memory_manager = None