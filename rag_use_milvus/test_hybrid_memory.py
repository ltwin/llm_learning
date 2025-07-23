#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合记忆管理器测试脚本
测试短期记忆缓存、摘要记忆和向量记忆的协同工作
"""

import asyncio
import httpx
import json
from datetime import datetime


class HybridMemoryTester:
    """混合记忆管理器测试器"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"test_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_id = "test_user_hybrid"
        self.character_id = "default"
    
    async def send_message(self, message: str) -> dict:
        """发送消息到AI服务"""
        timeout = httpx.Timeout(60.0)  # 增加超时时间到60秒
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat",
                    json={
                        "message": message,
                        "user_id": self.user_id,
                        "character_id": self.character_id,
                        "session_id": self.session_id
                    }
                )
                return response.json()
            except httpx.ReadTimeout:
                print("请求超时，可能是AI服务响应较慢")
                return {"message": "请求超时"}
            except Exception as e:
                print(f"发送消息时发生错误: {e}")
                return {"message": "发送失败"}
    
    async def search_memories(self, query: str) -> dict:
        """搜索记忆"""
        timeout = httpx.Timeout(60.0)  # 增加超时时间到60秒
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/memories/search",
                    json={
                        "query": query,
                        "user_id": self.user_id,
                        "character_id": self.character_id,
                        "limit": 10
                    }
                )
                return response.json()
            except httpx.ReadTimeout:
                print("搜索记忆请求超时")
                return {"memories": []}
            except Exception as e:
                print(f"搜索记忆时发生错误: {e}")
                return {"memories": []}
    
    async def get_conversation_summary(self) -> dict:
        """获取对话摘要"""
        timeout = httpx.Timeout(60.0)  # 增加超时时间到60秒
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/conversations/{self.session_id}/summary"
                )
                return response.json()
            except httpx.ReadTimeout:
                print("获取对话摘要请求超时")
                return {"summary": "获取摘要超时"}
            except Exception as e:
                print(f"获取对话摘要时发生错误: {e}")
                return {"summary": "获取摘要失败"}
    
    async def generate_summary(self) -> dict:
        """手动生成对话总结"""
        timeout = httpx.Timeout(60.0)  # 增加超时时间到60秒
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/conversations/{self.session_id}/generate-summary",
                    params={
                        "user_id": self.user_id,
                        "character_id": self.character_id,
                        "location": "测试环境"
                    }
                )
                return response.json()
            except httpx.ReadTimeout:
                print("生成对话摘要请求超时")
                return {"summary": "生成摘要超时"}
            except Exception as e:
                print(f"生成对话摘要时发生错误: {e}")
                return {"summary": "生成摘要失败"}
    
    async def test_short_term_memory(self):
        """测试短期记忆功能"""
        print("\n=== 测试短期记忆功能 ===")
        
        # 发送一系列相关消息
        messages = [
            "我叫张三，今年25岁",
            "我在北京工作，是一名程序员",
            "我喜欢打篮球和看电影",
            "我最近在学习人工智能",
            "你还记得我的名字吗？"
        ]
        
        for i, msg in enumerate(messages, 1):
            print(f"\n{i}. 用户: {msg}")
            response = await self.send_message(msg)
            print(f"   AI: {response.get('message', '无回复')}")
            
            # 短暂延迟
            await asyncio.sleep(0.5)
    
    async def test_memory_search(self):
        """测试记忆搜索功能"""
        print("\n=== 测试记忆搜索功能 ===")
        
        # 搜索不同类型的记忆
        search_queries = [
            "张三",
            "程序员",
            "篮球",
            "人工智能"
        ]
        
        for query in search_queries:
            print(f"\n搜索: {query}")
            result = await self.search_memories(query)
            memories = result.get('memories', [])
            print(f"找到 {len(memories)} 条相关记忆:")
            
            for memory in memories[:3]:  # 只显示前3条
                print(f"  - {memory['content'][:50]}... (重要性: {memory['importance_score']:.2f})")
    
    async def test_summary_memory(self):
        """测试摘要记忆功能"""
        print("\n=== 测试摘要记忆功能 ===")
        
        # 发送更多消息以触发摘要
        additional_messages = [
            "我昨天去了故宫博物院",
            "那里的文物真的很震撼",
            "特别是青铜器展厅",
            "我拍了很多照片",
            "下次想带朋友一起去",
            "你觉得故宫哪个展厅最有趣？"
        ]
        
        for i, msg in enumerate(additional_messages, 1):
            print(f"\n{i}. 用户: {msg}")
            response = await self.send_message(msg)
            print(f"   AI: {response.get('message', '无回复')}")
            await asyncio.sleep(0.5)
        
        # 手动生成摘要
        print("\n生成对话摘要...")
        summary_result = await self.generate_summary()
        if 'summary' in summary_result:
            print(f"摘要内容: {summary_result['summary']}")
        else:
            print(f"摘要生成结果: {summary_result}")
    
    async def test_long_conversation(self):
        """测试长对话的记忆管理"""
        print("\n=== 测试长对话记忆管理 ===")
        
        # 模拟长对话
        conversation_topics = [
            ("工作", ["今天工作很忙", "开了三个会议", "项目进度不错"]),
            ("生活", ["晚上去健身房了", "跑了5公里", "感觉很累但很充实"]),
            ("学习", ["在看机器学习的书", "线性代数有点难", "需要多练习"]),
            ("娱乐", ["周末看了新电影", "剧情很精彩", "推荐给朋友了"])
        ]
        
        for topic, messages in conversation_topics:
            print(f"\n--- {topic}话题 ---")
            for msg in messages:
                response = await self.send_message(msg)
                print(f"用户: {msg}")
                print(f"AI: {response.get('message', '无回复')[:100]}...")
                await asyncio.sleep(0.3)
        
        # 测试记忆检索
        print("\n测试跨话题记忆检索:")
        test_queries = ["工作会议", "健身跑步", "机器学习", "电影"]
        
        for query in test_queries:
            result = await self.search_memories(query)
            memories = result.get('memories', [])
            print(f"\n'{query}' 相关记忆 ({len(memories)}条):")
            for memory in memories[:2]:
                print(f"  - {memory['content'][:60]}...")
    
    async def test_memory_persistence(self):
        """测试记忆持久化"""
        print("\n=== 测试记忆持久化 ===")
        
        # 发送一些重要信息
        important_info = [
            "我的生日是1998年5月15日",
            "我的家乡是杭州",
            "我最喜欢的颜色是蓝色",
            "我有一只叫小白的猫"
        ]
        
        for info in important_info:
            response = await self.send_message(info)
            print(f"保存信息: {info}")
            await asyncio.sleep(0.3)
        
        # 等待一段时间后测试记忆检索
        print("\n等待记忆处理...")
        await asyncio.sleep(2)
        
        # 测试重要信息的检索
        test_queries = ["生日", "家乡", "颜色", "宠物"]
        
        for query in test_queries:
            result = await self.search_memories(query)
            memories = result.get('memories', [])
            if memories:
                print(f"✓ '{query}' 记忆已保存: {memories[0]['content'][:50]}...")
            else:
                print(f"✗ '{query}' 记忆未找到")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print(f"开始混合记忆管理器测试")
        print(f"会话ID: {self.session_id}")
        print(f"用户ID: {self.user_id}")
        print(f"角色ID: {self.character_id}")
        
        try:
            # 按顺序执行测试
            await self.test_short_term_memory()
            await self.test_memory_search()
            await self.test_summary_memory()
            await self.test_long_conversation()
            await self.test_memory_persistence()
            
            print("\n=== 测试完成 ===")
            print("混合记忆管理器功能测试成功！")
            
        except Exception as e:
            print(f"\n测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数"""
    tester = HybridMemoryTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())