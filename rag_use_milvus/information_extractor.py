import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from config import settings
from models import (
    Message, MessageRole, ConversationBufferMemory, 
    CharacterUpdate, KeyInformationExtraction, Memory, MemoryType,
    CharacterBackground
)

logger = logging.getLogger(__name__)


class InformationExtractor:
    """关键信息提取器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            temperature=0.3  # 降低温度以获得更一致的提取结果
        )
    
    async def extract_key_information(
        self, 
        buffer_memory: ConversationBufferMemory,
        character_background: Optional[CharacterBackground] = None
    ) -> KeyInformationExtraction:
        """从对话缓冲区提取关键信息"""
        try:
            # 获取对话文本
            conversation_text = self._format_conversation(buffer_memory.messages)
            
            # 提取角色更新信息
            character_updates = []
            if character_background and settings.extract_character_changes:
                character_updates = await self._extract_character_updates(
                    conversation_text, character_background, buffer_memory.session_id
                )
            
            # 提取重要记忆
            important_memories = await self._extract_important_memories(
                conversation_text, buffer_memory.session_id
            )
            
            # 生成提取摘要
            extraction_summary = await self._generate_extraction_summary(
                character_updates, important_memories
            )
            
            return KeyInformationExtraction(
                session_id=buffer_memory.session_id,
                character_updates=character_updates,
                important_memories=important_memories,
                extraction_summary=extraction_summary,
                metadata={
                    "message_count": len(buffer_memory.messages),
                    "extraction_time": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to extract key information: {e}")
            return KeyInformationExtraction(
                session_id=buffer_memory.session_id,
                character_updates=[],
                important_memories=[],
                extraction_summary="提取失败"
            )
    
    def _format_conversation(self, messages: List[Message]) -> str:
        """格式化对话文本"""
        formatted_messages = []
        for msg in messages:
            if msg.role != MessageRole.SYSTEM:  # 跳过系统消息
                role_name = "用户" if msg.role == MessageRole.USER else "AI助手"
                formatted_messages.append(f"{role_name}: {msg.content}")
        return "\n".join(formatted_messages)
    
    async def _extract_character_updates(
        self, 
        conversation_text: str, 
        character_background: CharacterBackground,
        session_id: str
    ) -> List[CharacterUpdate]:
        """提取角色信息更新"""
        try:
            # 构建角色信息提取提示
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """
你是一个角色信息分析专家。请分析以下对话，识别是否包含对角色信息的更新或修改。

当前角色信息：
- 姓名：{character_name}
- 年龄：{character_age}
- 职业：{character_occupation}
- 性格特征：{personality_traits}
- 说话风格：{speaking_style}
- 背景故事：{background_story}
- 人际关系：{relationships}
- 目标：{goals}
- 恐惧：{fears}
- 技能：{skills}

请识别对话中是否包含以下类型的角色信息更新：
1. 性格特征的变化或补充
2. 人际关系的新增、修改或删除
3. 背景信息的更新
4. 目标、恐惧、技能的变化
5. 说话风格的调整

请以JSON格式返回，包含以下字段：
- updates: 更新列表，每个更新包含：
  - update_type: 更新类型（personality/relationship/background/goals/fears/skills/speaking_style）
  - field_name: 具体字段名
  - old_value: 旧值（如果有）
  - new_value: 新值
  - confidence_score: 置信度（0-1）
  - description: 更新描述

如果没有发现角色信息更新，返回空的updates列表。
                """),
                ("user", "对话内容：\n{conversation}")
            ])
            
            # 格式化角色信息
            character_info = {
                "character_name": character_background.name,
                "character_age": character_background.age or "未知",
                "character_occupation": character_background.occupation or "未知",
                "personality_traits": ", ".join(character_background.personality.traits),
                "speaking_style": character_background.personality.speaking_style,
                "background_story": character_background.background_story,
                "relationships": ", ".join([f"{r.name}({r.relationship_type})" for r in character_background.relationships]),
                "goals": ", ".join(character_background.goals),
                "fears": ", ".join(character_background.fears),
                "skills": ", ".join(character_background.skills),
                "conversation": conversation_text
            }
            
            response = await self.llm.ainvoke(
                extraction_prompt.format_messages(**character_info)
            )
            
            # 解析响应
            try:
                result = json.loads(response.content)
                character_updates = []
                
                for update_data in result.get("updates", []):
                    if update_data.get("confidence_score", 0) >= settings.character_update_threshold:
                        update = CharacterUpdate(
                            character_id=character_background.name,  # 使用角色名作为ID
                            update_type=update_data["update_type"],
                            field_name=update_data["field_name"],
                            old_value=update_data.get("old_value"),
                            new_value=update_data["new_value"],
                            confidence_score=update_data["confidence_score"],
                            source_session=session_id,
                            metadata={
                                "description": update_data.get("description", ""),
                                "extraction_method": "llm_analysis"
                            }
                        )
                        character_updates.append(update)
                
                logger.info(f"Extracted {len(character_updates)} character updates")
                return character_updates
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse character update extraction response")
                return []
            
        except Exception as e:
            logger.error(f"Failed to extract character updates: {e}")
            return []
    
    async def _extract_important_memories(
        self, 
        conversation_text: str,
        session_id: str
    ) -> List[Memory]:
        """提取重要记忆"""
        try:
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """
你是一个记忆提取专家。请从以下对话中提取重要的信息，这些信息应该被保存为长期记忆。

提取标准：
1. 用户的个人信息、喜好、经历
2. 重要的事件、决定或计划
3. 情感状态的重要变化
4. 有价值的知识、见解或学习内容
5. 人际关系信息
6. 特殊的对话主题或兴趣点

请以JSON格式返回，包含以下字段：
- memories: 记忆列表，每个记忆包含：
  - content: 记忆内容（简洁明确）
  - memory_type: 记忆类型（episodic/semantic）
  - importance_score: 重要性评分（0-1）
  - summary: 简短总结
  - keywords: 关键词列表

如果没有重要信息，返回空的memories列表。
                """),
                ("user", "对话内容：\n{conversation}")
            ])
            
            response = await self.llm.ainvoke(
                extraction_prompt.format_messages(conversation=conversation_text)
            )
            
            # 解析响应
            try:
                result = json.loads(response.content)
                memories = []
                
                for memory_data in result.get("memories", []):
                    if memory_data.get("importance_score", 0) >= settings.memory_importance_threshold:
                        memory = Memory(
                            user_id="",  # 将在调用时设置
                            character_id="",  # 将在调用时设置
                            content=memory_data["content"],
                            memory_type=MemoryType(memory_data.get("memory_type", "semantic")),
                            importance_score=memory_data["importance_score"],
                            metadata={
                                "summary": memory_data.get("summary", ""),
                                "keywords": memory_data.get("keywords", []),
                                "extracted_from_session": session_id,
                                "extraction_time": datetime.now().isoformat(),
                                "extraction_method": "llm_analysis"
                            }
                        )
                        memories.append(memory)
                
                logger.info(f"Extracted {len(memories)} important memories")
                return memories
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse memory extraction response")
                return []
            
        except Exception as e:
            logger.error(f"Failed to extract important memories: {e}")
            return []
    
    async def _generate_extraction_summary(
        self, 
        character_updates: List[CharacterUpdate],
        important_memories: List[Memory]
    ) -> str:
        """生成提取摘要"""
        try:
            summary_parts = []
            
            if character_updates:
                update_types = set(update.update_type for update in character_updates)
                summary_parts.append(f"检测到{len(character_updates)}个角色信息更新（{', '.join(update_types)}）")
            
            if important_memories:
                memory_types = set(memory.memory_type.value for memory in important_memories)
                summary_parts.append(f"提取了{len(important_memories)}条重要记忆（{', '.join(memory_types)}）")
            
            if not summary_parts:
                return "本轮对话未发现需要提取的关键信息"
            
            return "；".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate extraction summary: {e}")
            return "摘要生成失败"


# 全局信息提取器实例
information_extractor = InformationExtractor()