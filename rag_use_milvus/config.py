from tempfile import tempdir
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置类"""
    
    # OpenAI配置
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"
    temperature: float = 0.6
    
    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_db_name: str = "ai_companion"
    
    # 应用配置
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    app_debug: bool = False
    
    # 向量模型配置
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_dimension: int = 2560  # 嵌入向量维度
    embedding_api_url: str = "https://api.siliconflow.cn/v1/embeddings"
    
    # 对话配置
    max_conversation_history: int = 20
    max_long_term_memories: int = 100
    memory_importance_threshold: float = 0.7
    memory_cleanup_days: int = 30
    
    # 对话历史配置
    max_conversation_history: int = 50  # 内存中保持的最大对话历史数量
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()