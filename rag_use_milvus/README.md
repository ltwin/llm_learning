# AI陪伴精灵服务

基于FastAPI+LangChain+LangGraph+Milvus的智能AI陪伴精灵服务，具备对话记忆、长期记忆RAG和角色设定功能。

## ✨ 主要功能

### 1. 🧠 智能对话记忆
- **短期记忆**: 保存对话历史，实现连续对话
- **长期记忆**: 自动提取重要信息存储到知识库
- **记忆检索**: 基于向量相似度的智能记忆召回
- **记忆管理**: 支持记忆的增删改查和统计

### 2. 🎭 丰富角色设定
- **角色背景**: 姓名、年龄、职业、背景故事
- **世界观设定**: 自定义角色所处的世界观
- **性格特征**: 多维度性格设定（幽默度、正式度等）
- **人际关系网**: 支持复杂的人物关系设定
- **技能特长**: 角色的专业技能和特长

### 3. 🔄 智能对话流程
- **LangGraph驱动**: 基于状态图的对话流程管理
- **上下文感知**: 结合历史对话和相关记忆
- **个性化回复**: 根据角色设定生成符合人设的回复
- **情感理解**: 理解用户情感并给出合适回应

### 4. 🗄️ 向量知识库
- **Milvus存储**: 高性能向量数据库
- **语义搜索**: 基于语义相似度的知识检索
- **自动索引**: 智能建立向量索引
- **实时更新**: 支持知识库的实时更新

## 🏗️ 技术架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangChain     │    │   LangGraph     │
│   Web服务       │────│   LLM集成       │────│   对话流程      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐
         │   Milvus        │    │   Sentence      │
         │   向量数据库    │────│   Transformers  │
         └─────────────────┘    └─────────────────┘
```

## 📦 安装部署

### 环境要求
- Python 3.12+
- Milvus 2.3+
- OpenAI API Key

### 1. 克隆项目
```bash
cd /root/workspace/learning/llm_learning/rag_use_milvus
```

### 2. 安装依赖
```bash
# 使用uv安装（推荐）
uv sync

# 或使用pip安装
pip install -e .
```

### 3. 配置环境
```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件
vim .env
```

配置示例：
```env
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Milvus配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 应用配置
APP_HOST=0.0.0.0
APP_PORT=8000
```

### 4. 启动Milvus
```bash
# 使用Docker启动Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -v $(pwd)/volumes/milvus:/var/lib/milvus \
  milvusdb/milvus:latest
```

### 5. 启动服务
```bash
# 开发模式
python main.py

# 或使用uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🚀 快速开始

### 1. 访问演示页面
打开浏览器访问: http://localhost:8000

### 2. API文档
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. 基本使用

#### 发送聊天消息
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "character_id": "default",
    "message": "你好，我是小明"
  }'
```

#### 创建自定义角色
```bash
curl -X POST "http://localhost:8000/characters/my_character" \
  -H "Content-Type: application/json" \
  -d '{
    "background": {
      "name": "小雪",
      "occupation": "学习助手",
      "background_story": "我是一个专门帮助学习的AI助手...",
      "world_setting": "现代教育环境",
      "personality": {
        "traits": ["耐心", "博学", "友善"],
        "speaking_style": "温和耐心，循循善诱",
        "emotional_tendency": "积极向上，鼓励学习",
        "humor_level": 0.6,
        "formality_level": 0.4
      }
    }
  }'
```

#### 搜索记忆
```bash
curl -X POST "http://localhost:8000/memories/search" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "character_id": "default",
    "query": "我的爱好",
    "limit": 5
  }'
```

## 📚 API接口

### 聊天接口
- `POST /chat` - 发送聊天消息
- `GET /conversations/{session_id}/summary` - 获取对话摘要
- `GET /conversations/{session_id}/context` - 获取对话上下文

### 角色管理
- `POST /characters/{character_id}` - 创建角色
- `GET /characters/{character_id}` - 获取角色信息
- `PUT /characters/{character_id}` - 更新角色
- `DELETE /characters/{character_id}` - 删除角色
- `GET /characters` - 列出所有角色

### 记忆管理
- `POST /memories/search` - 搜索记忆
- `GET /memories/stats` - 获取记忆统计
- `DELETE /memories/{memory_id}` - 删除记忆

### 系统管理
- `GET /health` - 健康检查
- `GET /system/info` - 系统信息
- `POST /system/cleanup` - 清理旧数据

## 🔧 配置说明

### 环境变量
| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API密钥 | 必填 |
| `OPENAI_BASE_URL` | OpenAI API地址 | https://api.openai.com/v1 |
| `MILVUS_HOST` | Milvus主机地址 | localhost |
| `MILVUS_PORT` | Milvus端口 | 19530 |
| `APP_HOST` | 应用主机地址 | 0.0.0.0 |
| `APP_PORT` | 应用端口 | 8000 |
| `EMBEDDING_MODEL` | 嵌入模型 | sentence-transformers/all-MiniLM-L6-v2 |
| `MAX_CONVERSATION_HISTORY` | 最大对话历史 | 20 |
| `MEMORY_IMPORTANCE_THRESHOLD` | 记忆重要性阈值 | 0.7 |

## 🎯 使用场景

### 1. 个人AI助手
- 日常对话陪伴
- 学习辅导助手
- 情感支持伙伴

### 2. 客服机器人
- 企业客户服务
- 产品咨询助手
- 售后支持系统

### 3. 教育应用
- 虚拟老师
- 学习伙伴
- 知识问答系统

### 4. 娱乐应用
- 角色扮演游戏
- 互动小说
- 虚拟偶像

## 🛠️ 开发指南

### 项目结构
```
rag_use_milvus/
├── main.py              # FastAPI主应用
├── config.py            # 配置管理
├── models.py            # 数据模型
├── vector_store.py      # Milvus向量存储
├── memory_manager.py    # 记忆管理
├── character_manager.py # 角色管理
├── conversation_graph.py # LangGraph对话流程
├── static/              # 静态文件
│   └── index.html       # 演示页面
├── .env.example         # 环境配置示例
├── pyproject.toml       # 项目配置
└── README.md            # 项目文档
```

### 扩展开发
1. **自定义角色类型**: 在`models.py`中扩展角色模型
2. **新增记忆类型**: 在`MemoryType`枚举中添加新类型
3. **自定义对话流程**: 修改`conversation_graph.py`中的流程图
4. **集成新的LLM**: 在配置中添加新的模型支持

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代、快速的Web框架
- [LangChain](https://langchain.com/) - LLM应用开发框架
- [LangGraph](https://langchain-ai.github.io/langgraph/) - 状态图对话流程
- [Milvus](https://milvus.io/) - 向量数据库
- [Sentence Transformers](https://www.sbert.net/) - 文本嵌入模型

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 加入讨论群

---

**享受与AI精灵的美好对话时光！** 🌟