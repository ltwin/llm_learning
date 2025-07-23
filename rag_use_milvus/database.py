import sqlite3
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import aiosqlite

from models import Message, ConversationSession, MessageRole

logger = logging.getLogger(__name__)


class ConversationDatabase:
    """对话历史数据库管理器"""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
    
    async def initialize(self):
        """初始化数据库"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 创建对话会话表
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        character_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT DEFAULT '{}'
                    )
                """)
                
                # 创建消息表
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT DEFAULT '{}',
                        FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                    )
                """)
                
                # 创建索引
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_session_id 
                    ON messages (session_id)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                    ON messages (timestamp)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_user_character 
                    ON conversation_sessions (user_id, character_id)
                """)
                
                await db.commit()
                logger.info(f"Database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def save_conversation_session(self, session: ConversationSession) -> bool:
        """保存或更新对话会话"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO conversation_sessions 
                    (session_id, user_id, character_id, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.user_id,
                    session.character_id,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    json.dumps(session.metadata)
                ))
                await db.commit()
                logger.debug(f"Saved conversation session {session.session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save conversation session: {e}")
            return False
    
    async def save_message(self, session_id: str, message: Message) -> bool:
        """保存消息"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO messages 
                    (session_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id,
                    message.role.value,
                    message.content,
                    message.timestamp.isoformat(),
                    json.dumps(message.metadata)
                ))
                await db.commit()
                logger.debug(f"Saved message to session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    async def load_conversation_session(self, session_id: str) -> Optional[ConversationSession]:
        """加载对话会话"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 获取会话信息
                async with db.execute("""
                    SELECT session_id, user_id, character_id, created_at, updated_at, metadata
                    FROM conversation_sessions WHERE session_id = ?
                """, (session_id,)) as cursor:
                    session_row = await cursor.fetchone()
                    
                if not session_row:
                    return None
                
                # 获取消息列表
                messages = await self.load_messages(session_id)
                
                # 构建会话对象
                session = ConversationSession(
                    session_id=session_row[0],
                    user_id=session_row[1],
                    character_id=session_row[2],
                    messages=messages,
                    created_at=datetime.fromisoformat(session_row[3]),
                    updated_at=datetime.fromisoformat(session_row[4]),
                    metadata=json.loads(session_row[5])
                )
                
                logger.debug(f"Loaded conversation session {session_id} with {len(messages)} messages")
                return session
                
        except Exception as e:
            logger.error(f"Failed to load conversation session: {e}")
            return None
    
    async def load_messages(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        """加载消息列表"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = """
                    SELECT role, content, timestamp, metadata
                    FROM messages WHERE session_id = ?
                    ORDER BY timestamp ASC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                async with db.execute(query, (session_id,)) as cursor:
                    rows = await cursor.fetchall()
                
                messages = []
                for row in rows:
                    message = Message(
                        role=MessageRole(row[0]),
                        content=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        metadata=json.loads(row[3])
                    )
                    messages.append(message)
                
                return messages
                
        except Exception as e:
            logger.error(f"Failed to load messages: {e}")
            return []
    
    async def get_user_sessions(self, user_id: str, character_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """获取用户的会话列表"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if character_id:
                    query = """
                        SELECT session_id, character_id, created_at, updated_at
                        FROM conversation_sessions 
                        WHERE user_id = ? AND character_id = ?
                        ORDER BY updated_at DESC LIMIT ?
                    """
                    params = (user_id, character_id, limit)
                else:
                    query = """
                        SELECT session_id, character_id, created_at, updated_at
                        FROM conversation_sessions 
                        WHERE user_id = ?
                        ORDER BY updated_at DESC LIMIT ?
                    """
                    params = (user_id, limit)
                
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                
                sessions = []
                for row in rows:
                    sessions.append({
                        "session_id": row[0],
                        "character_id": row[1],
                        "created_at": row[2],
                        "updated_at": row[3]
                    })
                
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    async def delete_old_conversations(self, days: int = 30) -> int:
        """删除旧对话"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.isoformat()
            
            async with aiosqlite.connect(self.db_path) as db:
                # 获取要删除的会话ID
                async with db.execute("""
                    SELECT session_id FROM conversation_sessions 
                    WHERE updated_at < ?
                """, (cutoff_str,)) as cursor:
                    session_ids = [row[0] for row in await cursor.fetchall()]
                
                if not session_ids:
                    return 0
                
                # 删除消息
                placeholders = ','.join(['?' for _ in session_ids])
                await db.execute(f"""
                    DELETE FROM messages WHERE session_id IN ({placeholders})
                """, session_ids)
                
                # 删除会话
                await db.execute(f"""
                    DELETE FROM conversation_sessions WHERE session_id IN ({placeholders})
                """, session_ids)
                
                await db.commit()
                logger.info(f"Deleted {len(session_ids)} old conversations")
                return len(session_ids)
                
        except Exception as e:
            logger.error(f"Failed to delete old conversations: {e}")
            return 0
    
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """获取对话统计信息"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 总会话数
                async with db.execute("SELECT COUNT(*) FROM conversation_sessions") as cursor:
                    total_sessions = (await cursor.fetchone())[0]
                
                # 总消息数
                async with db.execute("SELECT COUNT(*) FROM messages") as cursor:
                    total_messages = (await cursor.fetchone())[0]
                
                # 活跃用户数
                async with db.execute("SELECT COUNT(DISTINCT user_id) FROM conversation_sessions") as cursor:
                    active_users = (await cursor.fetchone())[0]
                
                return {
                    "total_sessions": total_sessions,
                    "total_messages": total_messages,
                    "active_users": active_users
                }
                
        except Exception as e:
            logger.error(f"Failed to get conversation stats: {e}")
            return {}


# 全局数据库实例
conversation_db = ConversationDatabase()