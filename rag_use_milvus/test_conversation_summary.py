#!/usr/bin/env python3
"""
对话总结功能测试脚本

这个脚本演示如何使用新的对话总结功能：
1. 手动生成对话总结
2. 结束对话并生成总结
3. 自动总结功能（每20轮对话自动触发）
"""

import asyncio
import aiohttp
import json
from datetime import datetime


class ConversationSummaryTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_id = "test_user"
        self.character_id = "default"
    
    async def send_chat_message(self, session, message: str):
        """发送聊天消息"""
        url = f"{self.base_url}/chat"
        data = {
            "user_id": self.user_id,
            "character_id": self.character_id,
            "session_id": self.session_id,
            "message": message
        }
        
        async with session.post(url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                print(f"用户: {message}")
                print(f"AI: {result['message']}")
                print("-" * 50)
                return result
            else:
                print(f"发送消息失败: {response.status}")
                return None
    
    async def generate_summary(self, session, location="测试环境"):
        """生成对话总结"""
        url = f"{self.base_url}/conversations/{self.session_id}/generate-summary"
        params = {
            "user_id": self.user_id,
            "character_id": self.character_id,
            "location": location
        }
        
        async with session.post(url, params=params) as response:
            if response.status == 200:
                result = await response.json()
                print("\n📝 对话总结已生成:")
                print(f"总结内容: {result['summary']}")
                print(f"重要性评分: {result['importance_score']}")
                print(f"生成时间: {result['created_at']}")
                print("=" * 60)
                return result
            else:
                error_text = await response.text()
                print(f"生成总结失败: {response.status} - {error_text}")
                return None
    
    async def end_conversation_with_summary(self, session, location="测试环境"):
        """结束对话并生成总结"""
        url = f"{self.base_url}/conversations/{self.session_id}/end-with-summary"
        params = {
            "user_id": self.user_id,
            "character_id": self.character_id,
            "location": location
        }
        
        async with session.post(url, params=params) as response:
            if response.status == 200:
                result = await response.json()
                print("\n🏁 对话已结束:")
                print(f"消息: {result['message']}")
                if 'summary' in result:
                    print(f"最终总结: {result['summary']}")
                print("=" * 60)
                return result
            else:
                error_text = await response.text()
                print(f"结束对话失败: {response.status} - {error_text}")
                return None
    
    async def search_memories(self, session, query: str):
        """搜索记忆"""
        url = f"{self.base_url}/memories/search"
        data = {
            "query": query,
            "user_id": self.user_id,
            "character_id": self.character_id,
            "limit": 5
        }
        
        async with session.post(url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                print(f"\n🔍 搜索记忆 '{query}':")
                print(f"找到 {result['count']} 条相关记忆:")
                for i, memory in enumerate(result['memories'], 1):
                    print(f"{i}. {memory['content'][:100]}...")
                    print(f"   类型: {memory['memory_type']}, 重要性: {memory['importance_score']:.2f}")
                print("-" * 50)
                return result
            else:
                print(f"搜索记忆失败: {response.status}")
                return None
    
    async def run_test(self):
        """运行测试"""
        print(f"🚀 开始测试对话总结功能")
        print(f"会话ID: {self.session_id}")
        print(f"用户ID: {self.user_id}")
        print(f"角色ID: {self.character_id}")
        print("=" * 60)
        
        async with aiohttp.ClientSession() as session:
            # 1. 进行一些对话
            messages = [
                "你好，我是张三，今天是我第一次使用这个AI助手",
                "我是一名软件工程师，在北京工作",
                "我喜欢编程和阅读，特别是科幻小说",
                "今天天气很好，我心情不错",
                "你能记住我刚才说的信息吗？"
            ]
            
            for message in messages:
                await self.send_chat_message(session, message)
                await asyncio.sleep(1)  # 避免请求过快
            
            # 2. 手动生成对话总结
            print("\n🔄 手动生成对话总结...")
            await self.generate_summary(session, "北京办公室")
            
            # 3. 继续对话
            more_messages = [
                "我想了解一下你的记忆功能",
                "你能告诉我刚才我们聊了什么吗？"
            ]
            
            for message in more_messages:
                await self.send_chat_message(session, message)
                await asyncio.sleep(1)
            
            # 4. 搜索记忆
            await self.search_memories(session, "张三 软件工程师")
            await self.search_memories(session, "总结")
            
            # 5. 结束对话并生成最终总结
            print("\n🏁 结束对话并生成最终总结...")
            await self.end_conversation_with_summary(session, "北京办公室")
            
            print("\n✅ 测试完成！")


async def main():
    """主函数"""
    tester = ConversationSummaryTester()
    await tester.run_test()


if __name__ == "__main__":
    print("对话总结功能测试")
    print("确保服务正在运行在 http://localhost:8000")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试出错: {e}")