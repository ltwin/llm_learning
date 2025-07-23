from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """消息角色枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MemoryType(str, Enum):
    """记忆类型枚举"""
    SHORT_TERM = "short_term"  # 短期记忆（对话历史）
    LONG_TERM = "long_term"    # 长期记忆（重要信息）
    EPISODIC = "episodic"      # 情节记忆（特定事件）
    SEMANTIC = "semantic"      # 语义记忆（知识概念）


class CharacterPersonality(BaseModel):
    """角色性格模型"""
    traits: List[str] = Field(description="性格特征列表")
    speaking_style: str = Field(description="说话风格")
    emotional_tendency: str = Field(description="情感倾向")
    humor_level: float = Field(default=0.5, ge=0, le=1, description="幽默程度(0-1)")
    formality_level: float = Field(default=0.5, ge=0, le=1, description="正式程度(0-1)")


class CharacterRelationship(BaseModel):
    """角色关系模型"""
    name: str = Field(description="关系对象名称")
    relationship_type: str = Field(description="关系类型")
    closeness_level: float = Field(ge=0, le=1, description="亲密程度(0-1)")
    description: str = Field(description="关系描述")


class CharacterBackground(BaseModel):
    """角色背景模型"""
    name: str = Field(description="角色名称")
    age: Optional[int] = Field(default=None, description="年龄")
    occupation: Optional[str] = Field(default=None, description="职业")
    background_story: str = Field(description="背景故事")
    world_setting: str = Field(description="世界观设定")
    personality: CharacterPersonality = Field(description="性格设定")
    relationships: List[CharacterRelationship] = Field(default=[], description="人物关系网")
    goals: List[str] = Field(default=[], description="角色目标")
    fears: List[str] = Field(default=[], description="角色恐惧")
    skills: List[str] = Field(default=[], description="技能特长")


class Message(BaseModel):
    """消息模型"""
    id: Optional[str] = Field(default=None, description="消息ID")
    role: MessageRole = Field(description="消息角色")
    content: str = Field(description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")


class ConversationSession(BaseModel):
    """对话会话模型"""
    session_id: str = Field(description="会话ID")
    user_id: str = Field(description="用户ID")
    character_id: str = Field(description="角色ID")
    messages: List[Message] = Field(default=[], description="消息列表")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    metadata: Dict[str, Any] = Field(default={}, description="会话元数据")


class Memory(BaseModel):
    """记忆模型"""
    id: Optional[str] = Field(default=None, description="记忆ID")
    user_id: str = Field(description="用户ID")
    character_id: str = Field(description="角色ID")
    content: str = Field(description="记忆内容")
    memory_type: MemoryType = Field(description="记忆类型")
    importance_score: float = Field(ge=0, le=1, description="重要性评分(0-1)")
    embedding: Optional[List[float]] = Field(default=None, description="向量嵌入")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    accessed_at: datetime = Field(default_factory=datetime.now, description="最后访问时间")
    access_count: int = Field(default=0, description="访问次数")
    metadata: Dict[str, Any] = Field(default={}, description="记忆元数据")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    user_id: str = Field(description="用户ID")
    character_id: str = Field(description="角色ID")
    message: str = Field(description="用户消息")
    session_id: Optional[str] = Field(default=None, description="会话ID")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    session_id: str = Field(description="会话ID")
    message: str = Field(description="AI回复")
    character_name: str = Field(description="角色名称")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    metadata: Dict[str, Any] = Field(default={}, description="响应元数据")


class CreateCharacterRequest(BaseModel):
    """创建角色请求模型"""
    background: CharacterBackground = Field(description="角色背景")


class UpdateCharacterRequest(BaseModel):
    """更新角色请求模型"""
    background: CharacterBackground = Field(description="角色背景")


class MemorySearchRequest(BaseModel):
    """记忆搜索请求模型"""
    user_id: str = Field(description="用户ID")
    character_id: str = Field(description="角色ID")
    query: str = Field(description="搜索查询")
    memory_type: Optional[MemoryType] = Field(default=None, description="记忆类型过滤")
    limit: int = Field(default=10, ge=1, le=50, description="返回数量限制")


class ConversationBufferMemory(BaseModel):
    """对话缓冲记忆模型"""
    session_id: str = Field(description="会话ID")
    messages: List[Message] = Field(default=[], description="短时记忆消息列表")
    max_size: int = Field(default=10, description="最大缓存大小")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    def add_message(self, message: Message):
        """添加消息到缓冲区"""
        self.messages.append(message)
        self.updated_at = datetime.now()
        
        # 保持缓冲区大小限制
        if len(self.messages) > self.max_size:
            # 保留系统消息和最近的消息
            system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
            recent_messages = [msg for msg in self.messages if msg.role != MessageRole.SYSTEM][-self.max_size:]
            self.messages = system_messages + recent_messages
    
    def get_recent_messages(self, count: int = None) -> List[Message]:
        """获取最近的消息"""
        if count is None:
            return self.messages
        return self.messages[-count:]
    
    def clear(self):
        """清空缓冲区"""
        self.messages = []
        self.updated_at = datetime.now()


class CharacterUpdate(BaseModel):
    """角色信息更新模型"""
    character_id: str = Field(description="角色ID")
    update_type: str = Field(description="更新类型：personality/relationship/background/goals")
    field_name: str = Field(description="更新字段名")
    old_value: Any = Field(description="旧值")
    new_value: Any = Field(description="新值")
    confidence_score: float = Field(ge=0, le=1, description="置信度评分")
    source_session: str = Field(description="来源会话ID")
    extracted_at: datetime = Field(default_factory=datetime.now, description="提取时间")
    metadata: Dict[str, Any] = Field(default={}, description="更新元数据")


class KeyInformationExtraction(BaseModel):
    """关键信息提取结果模型"""
    session_id: str = Field(description="会话ID")
    character_updates: List[CharacterUpdate] = Field(default=[], description="角色更新列表")
    important_memories: List[Memory] = Field(default=[], description="重要记忆列表")
    extraction_summary: str = Field(description="提取摘要")
    extracted_at: datetime = Field(default_factory=datetime.now, description="提取时间")
    metadata: Dict[str, Any] = Field(default={}, description="提取元数据")