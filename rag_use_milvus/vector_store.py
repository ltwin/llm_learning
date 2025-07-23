import logging
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import (
    connections, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType,
    utility
)
import uuid
from datetime import datetime
from langchain_openai import OpenAIEmbeddings

from config import settings
from models import Memory, MemoryType

logger = logging.getLogger(__name__)


class VectorStore:
    """Milvus向量存储管理器"""
    
    def __init__(self):
        self.embedding_model = None
        self.collection = None
        self.collection_name = "ai_companion_memories"
        
    async def initialize(self):
        """初始化向量存储"""
        try:
            # 连接Milvus
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
                user=settings.milvus_user,
                password=settings.milvus_password
            )
            logger.info(f"Connected to Milvus at {settings.milvus_host}:{settings.milvus_port}")
            
            # 初始化嵌入模型
            self.embedding_model = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_base_url
            )
            logger.info(f"已配置嵌入模型: {settings.embedding_model}")
            
            # 创建或加载集合
            await self._create_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def _create_collection(self):
        """创建Milvus集合"""
        try:
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                
                # 检查现有集合的维度是否匹配
                schema = self.collection.schema
                embedding_field = None
                for field in schema.fields:
                    if field.name == "embedding":
                        embedding_field = field
                        break
                
                if embedding_field and embedding_field.params.get('dim') != settings.embedding_dimension:
                    logger.warning(f"Existing collection dimension {embedding_field.params.get('dim')} doesn't match config {settings.embedding_dimension}. Recreating collection...")
                    # 删除现有集合
                    utility.drop_collection(self.collection_name)
                    logger.info(f"Dropped existing collection: {self.collection_name}")
                else:
                    logger.info(f"Loaded existing collection: {self.collection_name}")
                    return
            
            # 定义字段模式
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="character_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="importance_score", dtype=DataType.FLOAT),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="accessed_at", dtype=DataType.INT64),
                FieldSchema(name="access_count", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dimension)
            ]
            
            # 创建集合模式
            schema = CollectionSchema(fields, "AI Companion Memories Collection")
            
            # 创建集合
            self.collection = Collection(self.collection_name, schema)
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index("embedding", index_params)
            
            logger.info(f"Created new collection: {self.collection_name} with dimension {settings.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        try:
            embedding = self.embedding_model.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            raise
    
    async def add_memory(self, memory: Memory) -> str:
        """添加记忆到向量存储"""
        try:
            # 生成ID
            memory_id = memory.id or str(uuid.uuid4())
            
            # 生成嵌入向量
            embedding = self._generate_embedding(memory.content)
            
            # 准备数据
            data = [
                [memory_id],
                [memory.user_id],
                [memory.character_id],
                [memory.content],
                [memory.memory_type.value],
                [memory.importance_score],
                [int(memory.created_at.timestamp())],
                [int(memory.accessed_at.timestamp())],
                [memory.access_count],
                [embedding]
            ]
            
            # 插入数据
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Added memory {memory_id} to vector store")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise
    
    async def search_memories(
        self, 
        query: str, 
        user_id: str, 
        character_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        score_threshold: float = 0.3
    ) -> List[Tuple[Memory, float]]:
        """搜索相关记忆"""
        try:
            # 生成查询向量
            query_embedding = self._generate_embedding(query)
            
            # 构建搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 构建过滤表达式
            filter_expr = f'user_id == "{user_id}" and character_id == "{character_id}"'
            if memory_type:
                filter_expr += f' and memory_type == "{memory_type.value}"'
            
            # 加载集合
            self.collection.load()
            
            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=filter_expr,
                output_fields=["id", "user_id", "character_id", "content", 
                             "memory_type", "importance_score", "created_at", 
                             "accessed_at", "access_count"]
            )
            
            # 处理结果
            memories = []
            for hit in results[0]:
                logger.debug(f"Memory candidate: score={hit.score:.3f}, content={hit.entity.get('content', '')[:50]}...")
                if hit.score >= score_threshold:
                    memory_data = hit.entity
                    memory = Memory(
                        id=memory_data.get("id"),
                        user_id=memory_data.get("user_id"),
                        character_id=memory_data.get("character_id"),
                        content=memory_data.get("content"),
                        memory_type=MemoryType(memory_data.get("memory_type")),
                        importance_score=memory_data.get("importance_score"),
                        created_at=datetime.fromtimestamp(memory_data.get("created_at")),
                        accessed_at=datetime.fromtimestamp(memory_data.get("accessed_at")),
                        access_count=memory_data.get("access_count", 0)
                    )
                    memories.append((memory, hit.score))
                    logger.info(f"Selected memory: score={hit.score:.3f}, content={memory.content[:50]}...")
                else:
                    logger.debug(f"Memory filtered out: score={hit.score:.3f} < threshold={score_threshold}")
            
            logger.info(f"Found {len(memories)} relevant memories for query: {query}")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise
    
    async def update_memory_access(self, memory_id: str):
        """更新记忆访问信息"""
        try:
            # 这里可以实现更新访问时间和次数的逻辑
            # Milvus目前不支持直接更新，需要删除后重新插入
            pass
        except Exception as e:
            logger.error(f"Failed to update memory access: {e}")
            raise
    
    async def delete_memory(self, memory_id: str):
        """删除记忆"""
        try:
            expr = f'id == "{memory_id}"'
            self.collection.delete(expr)
            self.collection.flush()
            logger.info(f"Deleted memory {memory_id}")
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            raise
    
    async def get_memory_stats(self, user_id: str, character_id: str) -> Dict[str, Any]:
        """获取记忆统计信息"""
        try:
            # 加载集合
            self.collection.load()
            
            # 查询统计信息
            filter_expr = f'user_id == "{user_id}" and character_id == "{character_id}"'
            
            # 获取总数
            total_count = self.collection.query(
                expr=filter_expr,
                output_fields=["id"]
            )
            
            stats = {
                "total_memories": len(total_count),
                "user_id": user_id,
                "character_id": character_id
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            raise
    
    async def recreate_collection(self):
        """强制重新创建集合"""
        try:
            # 删除现有集合
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection: {self.collection_name}")
            
            # 重新创建集合
            await self._create_collection()
            logger.info(f"Recreated collection with dimension {settings.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to recreate collection: {e}")
            raise
    
    async def close(self):
        """关闭连接"""
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to close vector store: {e}")


# 全局向量存储实例
vector_store = VectorStore()