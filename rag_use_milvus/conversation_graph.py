import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

from config import settings
from models import Message, MessageRole, ChatRequest, ChatResponse
# memory_manager将在main.py中初始化后导入
from character_manager import character_manager

logger = logging.getLogger(__name__)


class ConversationState(BaseModel):
    """对话状态模型"""
    session_id: str
    user_id: str
    character_id: str
    messages: List[BaseMessage] = []
    user_input: str = ""
    ai_response: str = ""
    relevant_memories: List[Dict[str, Any]] = []
    character_prompt: str = ""
    context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class ConversationGraph:
    """基于LangGraph的对话流程管理器"""
    
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            temperature=settings.temperature
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建对话流程图"""
        # 创建状态图
        workflow = StateGraph(ConversationState)
        
        # 添加节点
        workflow.add_node("initialize", self._initialize_conversation)
        workflow.add_node("retrieve_memories", self._retrieve_memories)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("process_memories", self._process_memories)
        workflow.add_node("finalize", self._finalize_response)
        
        # 设置入口点
        workflow.set_entry_point("initialize")
        
        # 添加边
        workflow.add_edge("initialize", "retrieve_memories")
        workflow.add_edge("retrieve_memories", "prepare_context")
        workflow.add_edge("prepare_context", "generate_response")
        workflow.add_edge("generate_response", "process_memories")
        workflow.add_edge("process_memories", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _initialize_conversation(self, state: ConversationState) -> ConversationState:
        """初始化对话"""
        try:
            logger.info(f"Initializing conversation for session {state.session_id}")
            
            # 获取角色信息
            character = character_manager.get_character(state.character_id)
            if character:
                state.character_prompt = character_manager.get_character_prompt(state.character_id)
                state.metadata["character_name"] = character.name
            else:
                state.character_prompt = "你是一个友善的AI助手。"
                state.metadata["character_name"] = "AI助手"
            
            # 获取对话历史
            conversation_history = await self.memory_manager.get_conversation_history(state.session_id)
            state.messages = [
                HumanMessage(content=msg.content) if msg.role == MessageRole.USER 
                else AIMessage(content=msg.content) if msg.role == MessageRole.ASSISTANT
                else SystemMessage(content=msg.content)
                for msg in conversation_history[-10:]  # 最近10条消息
            ]
            
            logger.info(f"Initialized conversation with {len(state.messages)} historical messages")
            return state
            
        except Exception as e:
            logger.error(f"Failed to initialize conversation: {e}, traceback: {traceback.format_exc()}")
            state.metadata["error"] = str(e)
            return state
    
    async def _retrieve_memories(self, state: ConversationState) -> ConversationState:
        """检索相关记忆"""
        try:
            logger.info(f"Retrieving memories for query: {state.user_input[:50]}...")
            
            # 检索相关记忆
            memories = await self.memory_manager.retrieve_relevant_memories(
                query=state.user_input,
                user_id=state.user_id,
                character_id=state.character_id,
                limit=5
            )
            
            # 转换为字典格式
            state.relevant_memories = [
                {
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "importance_score": memory.importance_score,
                    "created_at": memory.created_at.isoformat()
                }
                for memory in memories
            ]
            
            logger.info(f"Retrieved {len(state.relevant_memories)} relevant memories")
            return state
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            state.metadata["memory_error"] = str(e)
            return state
    
    async def _prepare_context(self, state: ConversationState) -> ConversationState:
        """准备对话上下文"""
        try:
            logger.info("Preparing conversation context")
            
            # 构建系统消息
            system_parts = [state.character_prompt]
            
            # 添加相关记忆
            if state.relevant_memories:
                memory_context = "\n\n相关记忆：\n"
                for i, memory in enumerate(state.relevant_memories, 1):
                    memory_context += f"{i}. {memory['content']} (重要性: {memory['importance_score']:.2f})\n"
                system_parts.append(memory_context)
            
            # 添加对话指导
            system_parts.append(
                "\n请基于以上角色设定和相关记忆，以自然、连贯的方式回应用户。"
                "如果相关记忆中有重要信息，请适当地融入到回复中。"
                "保持角色的一致性和个性特点。"
            )
            
            system_message = SystemMessage(content="\n".join(system_parts))
            
            # 构建完整的消息列表
            full_messages = [system_message] + state.messages + [HumanMessage(content=state.user_input)]
            state.messages = full_messages
            
            logger.info("Context prepared successfully")
            return state
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            state.metadata["context_error"] = str(e)
            return state
    
    async def _generate_response(self, state: ConversationState) -> ConversationState:
        """生成AI回复"""
        try:
            logger.info("Generating AI response")
            
            # 调用LLM生成回复
            response = await self.llm.ainvoke(state.messages)
            state.ai_response = response.content
            
            logger.info(f"Generated response: {state.ai_response[:100]}...")
            return state
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            state.ai_response = "抱歉，我现在无法回复。请稍后再试。"
            state.metadata["generation_error"] = str(e)
            return state
    
    async def _process_memories(self, state: ConversationState) -> ConversationState:
        """处理记忆存储"""
        try:
            logger.info("Processing conversation memories")
            
            # 添加用户消息到对话
            from models import Message, MessageRole
            user_msg = Message(
                role=MessageRole.USER,
                content=state.user_input,
                metadata={"user_id": state.user_id, "character_id": state.character_id}
            )
            await self.memory_manager.add_message_to_conversation(
                user_id=state.user_id,
                session_id=state.session_id,
                message=user_msg
            )
            
            # 添加AI回复到对话
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=state.ai_response,
                metadata={"user_id": state.user_id, "character_id": state.character_id}
            )
            await self.memory_manager.add_message_to_conversation(
                user_id=state.user_id,
                session_id=state.session_id,
                message=assistant_msg
            )
            
            state.metadata["messages_saved"] = 2
            logger.info("Processed memories, saved user and assistant messages")
            return state
            
        except Exception as e:
            logger.error(f"Failed to process memories: {e}")
            state.metadata["memory_processing_error"] = str(e)
            return state
    
    async def _finalize_response(self, state: ConversationState) -> ConversationState:
        """最终化回复"""
        try:
            logger.info("Finalizing response")
            
            # 添加响应元数据
            state.metadata.update({
                "response_generated_at": datetime.now().isoformat(),
                "memories_used": len(state.relevant_memories),
                "session_id": state.session_id
            })
            
            # 检查是否需要生成对话总结
            await self._check_and_generate_summary(state)
            
            logger.info("Response finalized successfully")
            return state
            
        except Exception as e:
            logger.error(f"Failed to finalize response: {e}")
            state.metadata["finalization_error"] = str(e)
            return state
    
    async def _check_and_generate_summary(self, state: ConversationState):
        """检查并生成对话总结"""
        try:
            # 获取当前对话的消息数量
            conversation_history = await self.memory_manager.get_conversation_history(state.session_id)
            message_count = len(conversation_history)
            
            # 设置自动总结的触发条件
            AUTO_SUMMARY_THRESHOLD = 20  # 每20轮对话生成一次总结
            
            # 如果消息数量达到阈值且是阈值的倍数，则生成总结
            if message_count >= AUTO_SUMMARY_THRESHOLD and message_count % AUTO_SUMMARY_THRESHOLD == 0:
                logger.info(f"Auto-generating summary for session {state.session_id} at {message_count} messages")
                
                summary_memory = await self.memory_manager.generate_conversation_summary(
                    session_id=state.session_id,
                    user_id=state.user_id,
                    character_id=state.character_id,
                    location="对话中"  # 默认地点
                )
                
                if summary_memory:
                    state.metadata["auto_summary_generated"] = True
                    state.metadata["summary_content"] = summary_memory.content[:50] + "..."
                    logger.info(f"Auto-generated summary for session {state.session_id}")
                else:
                    logger.warning(f"Failed to auto-generate summary for session {state.session_id}")
                    
        except Exception as e:
            logger.error(f"Failed to check and generate summary: {e}")
            # 不抛出异常，避免影响主流程
    
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求"""
        try:
            # 生成会话ID（如果没有提供）
            session_id = request.session_id or str(uuid.uuid4())
            
            # 创建初始状态
            initial_state = ConversationState(
                session_id=session_id,
                user_id=request.user_id,
                character_id=request.character_id,
                user_input=request.message
            )
            
            # 运行对话流程
            final_state = await self.graph.ainvoke(initial_state)
            
            # 构建响应
            character = character_manager.get_character(request.character_id)
            character_name = character.name if character else "AI助手"
            
            response = ChatResponse(
                session_id=session_id,
                message=final_state["ai_response"],
                character_name=character_name,
                metadata=final_state["metadata"]
            )
            
            logger.info(f"Chat processed successfully for session {session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process chat: {e}")
            # 返回错误响应
            return ChatResponse(
                session_id=request.session_id or "error",
                message="抱歉，处理您的消息时出现了错误。请稍后再试。",
                character_name="系统",
                metadata={"error": str(e)}
            )
    
    async def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """获取对话摘要"""
        try:
            context = await self.memory_manager.get_conversation_context(session_id)
            
            # 添加统计信息
            summary = {
                "session_id": session_id,
                "message_count": len(context.get("conversation_history", [])),
                "relevant_memories_count": len(context.get("relevant_memories", [])),
                "session_info": context.get("session_info", {}),
                "last_updated": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return {"error": str(e)}


# 全局对话图实例将在main.py中初始化
conversation_graph = None