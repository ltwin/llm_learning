from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import asyncio

# 导入自定义模块
from src.models.character import Character, CharacterConfig
from src.models.conversation import ConversationRequest, ConversationResponse
from src.services.memory_manager import MemoryManager
from src.services.character_manager import CharacterManager
from src.services.conversation_service import ConversationService
from src.config import settings

app = FastAPI(
    title="AI陪伴精灵服务",
    description="基于FastAPI+LangChain+LangGraph+Milvus的智能对话系统",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局服务实例
memory_manager = None
character_manager = None
conversation_service = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    global memory_manager, character_manager, conversation_service
    
    # 初始化记忆管理器
    memory_manager = MemoryManager()
    await memory_manager.initialize()
    
    # 初始化角色管理器
    character_manager = CharacterManager()
    
    # 初始化对话服务
    conversation_service = ConversationService(
        memory_manager=memory_manager,
        character_manager=character_manager
    )
    
    print("AI陪伴精灵服务启动成功！")

@app.get("/")
async def root():
    return {"message": "AI陪伴精灵服务运行中", "timestamp": datetime.now()}

@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    """主要的对话接口"""
    try:
        response = await conversation_service.process_conversation(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/character/create")
async def create_character(character_config: CharacterConfig):
    """创建新角色"""
    try:
        character = await character_manager.create_character(character_config)
        return {"message": "角色创建成功", "character_id": character.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/character/{character_id}")
async def get_character(character_id: str):
    """获取角色信息"""
    try:
        character = await character_manager.get_character(character_id)
        if not character:
            raise HTTPException(status_code=404, detail="角色不存在")
        return character
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/conversation/{session_id}")
async def get_conversation_history(session_id: str, limit: int = 50):
    """获取对话历史"""
    try:
        history = await memory_manager.get_conversation_history(session_id, limit)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/save-important")
async def save_important_memory(session_id: str, content: str, importance_score: float = 0.8):
    """手动保存重要记忆到长期记忆"""
    try:
        await memory_manager.save_important_memory(session_id, content, importance_score)
        return {"message": "重要记忆保存成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/search")
async def search_memory(query: str, session_id: Optional[str] = None, limit: int = 10):
    """搜索相关记忆"""
    try:
        results = await memory_manager.search_relevant_memories(query, session_id, limit)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
