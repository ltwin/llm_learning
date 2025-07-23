import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, List, Any
import os

from config import settings
from models import (
    ChatRequest, ChatResponse, CreateCharacterRequest, UpdateCharacterRequest,
    MemorySearchRequest, Memory, CharacterBackground
)
from vector_store import vector_store
from hybrid_memory_manager import HybridMemoryManager
from database import conversation_db

# 初始化混合记忆管理器
memory_manager = HybridMemoryManager(vector_store, conversation_db)
from character_manager import character_manager
from conversation_graph import ConversationGraph

# 初始化对话图
conversation_graph = ConversationGraph(memory_manager)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("Starting AI Companion Spirit Service...")
    try:
        await vector_store.initialize()
        logger.info("Vector store initialized successfully")
        
        await memory_manager.initialize()
        logger.info("Hybrid memory manager initialized successfully")
        
        # 更新conversation_graph的memory_manager引用
        conversation_graph.memory_manager = memory_manager
        logger.info("Conversation graph updated with hybrid memory manager")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # 关闭时清理
    logger.info("Shutting down AI Companion Spirit Service...")
    try:
        await vector_store.close()
        logger.info("Vector store closed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="AI陪伴精灵服务",
    description="基于FastAPI+LangChain+LangGraph+Milvus的AI陪伴精灵服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """根路径 - 返回演示页面"""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    else:
        return {
            "message": "欢迎使用AI陪伴精灵服务",
            "version": "1.0.0",
            "docs": "/docs"
        }


@app.get("/api")
async def api_info():
    """API信息"""
    return {
        "message": "欢迎使用AI陪伴精灵服务API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "demo": "/"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查向量存储连接
        stats = await vector_store.get_memory_stats("health_check", "default")
        return {
            "status": "healthy",
            "timestamp": memory_manager.conversations,
            "vector_store": "connected",
            "characters_loaded": len(character_manager.characters)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/admin/recreate-collection")
async def recreate_collection():
    """重新创建向量集合（管理员功能）"""
    try:
        await vector_store.recreate_collection()
        return {"message": "Collection recreated successfully", "dimension": settings.embedding_dimension}
    except Exception as e:
        logger.error(f"Failed to recreate collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 聊天相关API ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    try:
        logger.info(f"Received chat request from user {request.user_id}")
        response = await conversation_graph.process_chat(request)
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{session_id}/summary")
async def get_conversation_summary(session_id: str):
    """获取对话摘要"""
    try:
        summary = await conversation_graph.get_conversation_summary(session_id)
        return summary
    except Exception as e:
        logger.error(f"Failed to get conversation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{session_id}/context")
async def get_conversation_context(
    session_id: str, 
    user_id: str, 
    current_message: str = "",
    character_id: str = None
):
    """获取对话上下文"""
    try:
        context = await memory_manager.get_conversation_context(
            user_id=user_id,
            session_id=session_id,
            current_message=current_message,
            character_id=character_id
        )
        return {"context": context}
    except Exception as e:
        logger.error(f"Failed to get conversation context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 角色管理API ====================

@app.post("/characters/{character_id}")
async def create_character(character_id: str, request: CreateCharacterRequest):
    """创建角色"""
    try:
        success = character_manager.create_character(character_id, request.background)
        if success:
            return {"message": f"角色 {request.background.name} 创建成功", "character_id": character_id}
        else:
            raise HTTPException(status_code=400, detail="角色创建失败")
    except Exception as e:
        logger.error(f"Failed to create character: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/characters/{character_id}", response_model=CharacterBackground)
async def get_character(character_id: str):
    """获取角色信息"""
    character = character_manager.get_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="角色不存在")
    return character


@app.put("/characters/{character_id}")
async def update_character(character_id: str, request: UpdateCharacterRequest):
    """更新角色信息"""
    try:
        success = character_manager.update_character(character_id, request.background)
        if success:
            return {"message": f"角色 {request.background.name} 更新成功"}
        else:
            raise HTTPException(status_code=404, detail="角色不存在")
    except Exception as e:
        logger.error(f"Failed to update character: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/characters/{character_id}")
async def delete_character(character_id: str):
    """删除角色"""
    try:
        success = character_manager.delete_character(character_id)
        if success:
            return {"message": "角色删除成功"}
        else:
            raise HTTPException(status_code=404, detail="角色不存在或无法删除")
    except Exception as e:
        logger.error(f"Failed to delete character: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/characters")
async def list_characters():
    """列出所有角色"""
    try:
        characters = character_manager.list_characters()
        return {"characters": characters}
    except Exception as e:
        logger.error(f"Failed to list characters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/characters/{character_id}/prompt")
async def get_character_prompt(character_id: str):
    """获取角色提示词"""
    try:
        prompt = character_manager.get_character_prompt(character_id)
        return {"character_id": character_id, "prompt": prompt}
    except Exception as e:
        logger.error(f"Failed to get character prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/characters/{character_id}/export")
async def export_character(character_id: str):
    """导出角色配置"""
    try:
        character_json = character_manager.export_character(character_id)
        if character_json:
            return {"character_id": character_id, "config": character_json}
        else:
            raise HTTPException(status_code=404, detail="角色不存在")
    except Exception as e:
        logger.error(f"Failed to export character: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/characters/{character_id}/import")
async def import_character(character_id: str, config: Dict[str, Any]):
    """导入角色配置"""
    try:
        import json
        character_json = json.dumps(config)
        success = character_manager.import_character(character_id, character_json)
        if success:
            return {"message": "角色导入成功", "character_id": character_id}
        else:
            raise HTTPException(status_code=400, detail="角色导入失败")
    except Exception as e:
        logger.error(f"Failed to import character: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 记忆管理API ====================

@app.post("/conversations/{session_id}/end-with-summary")
async def end_conversation_with_summary(
    session_id: str,
    user_id: str,
    character_id: str,
    location: str = "未知地点"
):
    """结束对话并生成总结"""
    try:
        summary = await memory_manager.end_conversation_with_summary(
            session_id=session_id,
            user_id=user_id,
            character_id=character_id,
            location=location
        )
        
        if summary:
            return {
                "message": "对话已结束，总结已生成",
                "session_id": session_id,
                "summary": summary
            }
        else:
            return {
                "message": "对话已结束，但总结生成失败",
                "session_id": session_id
            }
    except Exception as e:
        logger.error(f"Failed to end conversation with summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversations/{session_id}/generate-summary")
async def generate_conversation_summary(
    session_id: str,
    user_id: str,
    character_id: str,
    location: str = "未知地点"
):
    """为指定对话生成总结（不结束对话）"""
    try:
        summary_memory = await memory_manager.generate_conversation_summary(
            session_id=session_id,
            user_id=user_id,
            character_id=character_id,
            location=location
        )
        
        if summary_memory:
            return {
                "message": "对话总结已生成",
                "session_id": session_id,
                "summary": summary_memory.content,
                "memory_id": summary_memory.id,
                "importance_score": summary_memory.importance_score,
                "created_at": summary_memory.created_at.isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="无法生成对话总结，可能是对话内容不足")
    except Exception as e:
        logger.error(f"Failed to generate conversation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories/search")
async def search_memories(request: MemorySearchRequest):
    """搜索记忆"""
    try:
        memories = await memory_manager.retrieve_relevant_memories(
            query=request.query,
            user_id=request.user_id,
            character_id=request.character_id,
            memory_type=request.memory_type,
            limit=request.limit
        )
        
        return {
            "query": request.query,
            "memories": [
                {
                    "id": memory.id,
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "importance_score": memory.importance_score,
                    "created_at": memory.created_at.isoformat(),
                    "metadata": memory.metadata
                }
                for memory in memories
            ],
            "count": len(memories)
        }
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/stats")
async def get_memory_stats(user_id: str, character_id: str):
    """获取记忆统计信息"""
    try:
        stats = await vector_store.get_memory_stats(user_id, character_id)
        return stats
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """删除记忆"""
    try:
        await vector_store.delete_memory(memory_id)
        return {"message": "记忆删除成功", "memory_id": memory_id}
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 系统管理API ====================

@app.post("/system/cleanup")
async def cleanup_old_data(background_tasks: BackgroundTasks, days: int = 30):
    """清理旧数据"""
    try:
        background_tasks.add_task(memory_manager.clear_old_conversations, days)
        return {"message": f"开始清理{days}天前的旧数据"}
    except Exception as e:
        logger.error(f"Failed to start cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/info")
async def get_system_info():
    """获取系统信息"""
    try:
        return {
            "service_name": "AI陪伴精灵服务",
            "version": "1.0.0",
            "active_conversations": len(memory_manager.conversations),
            "available_characters": len(character_manager.characters),
            "settings": {
                "max_conversation_history": settings.max_conversation_history,
                "max_long_term_memories": settings.max_long_term_memories,
                "memory_importance_threshold": settings.memory_importance_threshold,
                "embedding_model": settings.embedding_model,
                "long_term_memory_extraction_interval": settings.long_term_memory_extraction_interval,
                "character_update_threshold": settings.character_update_threshold,
                "extract_character_changes": settings.extract_character_changes
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """启动应用"""
    logger.info("Starting AI Companion Spirit Service...")
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
