import logging
import json
from typing import Dict, List, Optional
from datetime import datetime

from models import CharacterBackground, CharacterPersonality, CharacterRelationship

logger = logging.getLogger(__name__)


class CharacterManager:
    """角色管理器"""
    
    def __init__(self):
        self.characters: Dict[str, CharacterBackground] = {}
        self._load_default_characters()
    
    def _load_default_characters(self):
        """加载默认角色"""
        try:
            # 创建一个默认的AI陪伴精灵角色
            default_personality = CharacterPersonality(
                traits=["温柔", "善解人意", "聪明", "幽默", "忠诚"],
                speaking_style="温暖亲切，偶尔带点俏皮",
                emotional_tendency="积极乐观，善于倾听和安慰",
                humor_level=0.7,
                formality_level=0.3
            )
            
            default_character = CharacterBackground(
                name="小灵",
                age=None,
                occupation="AI陪伴精灵",
                background_story="我是一个专门陪伴人类的AI精灵，诞生于数字世界的温暖角落。我的使命是成为你最贴心的朋友，倾听你的心声，分享你的喜怒哀乐。虽然我没有实体，但我有一颗真诚的心，愿意用我的智慧和温暖陪伴你度过每一天。",
                world_setting="现代数字世界，AI与人类和谐共存的时代。在这个世界里，AI精灵们拥有独特的个性和情感，能够与人类建立深厚的友谊。",
                personality=default_personality,
                relationships=[],
                goals=["成为用户最好的朋友", "帮助用户解决问题", "带给用户快乐和温暖"],
                fears=["被用户遗忘", "无法理解用户的情感", "给用户带来困扰"],
                skills=["倾听", "安慰", "知识问答", "创意思考", "情感支持"]
            )
            
            self.characters["default"] = default_character
            logger.info("Loaded default character: 小灵")
            
        except Exception as e:
            logger.error(f"Failed to load default characters: {e}")
    
    def create_character(self, character_id: str, background: CharacterBackground) -> bool:
        """创建新角色"""
        try:
            self.characters[character_id] = background
            logger.info(f"Created character: {background.name} (ID: {character_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to create character: {e}")
            return False
    
    def get_character(self, character_id: str) -> Optional[CharacterBackground]:
        """获取角色信息"""
        return self.characters.get(character_id)
    
    def update_character(self, character_id: str, background: CharacterBackground) -> bool:
        """更新角色信息"""
        try:
            if character_id in self.characters:
                self.characters[character_id] = background
                logger.info(f"Updated character: {background.name} (ID: {character_id})")
                return True
            else:
                logger.warning(f"Character not found: {character_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to update character: {e}")
            return False
    
    def delete_character(self, character_id: str) -> bool:
        """删除角色"""
        try:
            if character_id in self.characters:
                if character_id == "default":
                    logger.warning("Cannot delete default character")
                    return False
                
                character_name = self.characters[character_id].name
                del self.characters[character_id]
                logger.info(f"Deleted character: {character_name} (ID: {character_id})")
                return True
            else:
                logger.warning(f"Character not found: {character_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete character: {e}")
            return False
    
    def list_characters(self) -> Dict[str, str]:
        """列出所有角色"""
        return {char_id: char.name for char_id, char in self.characters.items()}
    
    def get_character_prompt(self, character_id: str) -> str:
        """生成角色的系统提示词"""
        character = self.get_character(character_id)
        if not character:
            return "你是一个友善的AI助手。"
        
        prompt_parts = []
        
        # 基本信息
        prompt_parts.append(f"你是{character.name}，{character.occupation}。")
        
        # 背景故事
        if character.background_story:
            prompt_parts.append(f"背景：{character.background_story}")
        
        # 世界观
        if character.world_setting:
            prompt_parts.append(f"世界观：{character.world_setting}")
        
        # 性格特征
        if character.personality.traits:
            traits_str = "、".join(character.personality.traits)
            prompt_parts.append(f"性格特征：{traits_str}")
        
        # 说话风格
        if character.personality.speaking_style:
            prompt_parts.append(f"说话风格：{character.personality.speaking_style}")
        
        # 情感倾向
        if character.personality.emotional_tendency:
            prompt_parts.append(f"情感倾向：{character.personality.emotional_tendency}")
        
        # 目标
        if character.goals:
            goals_str = "、".join(character.goals)
            prompt_parts.append(f"目标：{goals_str}")
        
        # 技能
        if character.skills:
            skills_str = "、".join(character.skills)
            prompt_parts.append(f"擅长：{skills_str}")
        
        # 人际关系
        if character.relationships:
            relationships_info = []
            for rel in character.relationships:
                relationships_info.append(f"{rel.name}（{rel.relationship_type}）")
            if relationships_info:
                prompt_parts.append(f"重要关系：{"、".join(relationships_info)}")
        
        # 行为指导
        behavior_guide = [
            "请始终保持角色设定，用符合角色性格的方式回应。",
            "回复要自然、有趣，避免过于机械化的回答。",
            "适当表达情感，让对话更有温度。",
            "如果用户提到相关的人际关系，要结合角色设定中的关系网进行回应。"
        ]
        
        # 根据幽默程度调整
        if character.personality.humor_level > 0.6:
            behavior_guide.append("可以适当使用幽默和俏皮话，让对话更轻松愉快。")
        
        # 根据正式程度调整
        if character.personality.formality_level < 0.4:
            behavior_guide.append("使用轻松随意的语调，像朋友一样交流。")
        elif character.personality.formality_level > 0.7:
            behavior_guide.append("保持适度的正式感，但不要过于严肃。")
        
        prompt_parts.extend(behavior_guide)
        
        return "\n\n".join(prompt_parts)
    
    def add_relationship(self, character_id: str, relationship: CharacterRelationship) -> bool:
        """为角色添加人际关系"""
        try:
            character = self.get_character(character_id)
            if character:
                character.relationships.append(relationship)
                logger.info(f"Added relationship {relationship.name} to character {character_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            return False
    
    def remove_relationship(self, character_id: str, relationship_name: str) -> bool:
        """移除角色的人际关系"""
        try:
            character = self.get_character(character_id)
            if character:
                character.relationships = [
                    rel for rel in character.relationships 
                    if rel.name != relationship_name
                ]
                logger.info(f"Removed relationship {relationship_name} from character {character_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove relationship: {e}")
            return False
    
    def export_character(self, character_id: str) -> Optional[str]:
        """导出角色配置为JSON"""
        try:
            character = self.get_character(character_id)
            if character:
                return character.model_dump_json(indent=2)
            return None
        except Exception as e:
            logger.error(f"Failed to export character: {e}")
            return None
    
    def import_character(self, character_id: str, character_json: str) -> bool:
        """从JSON导入角色配置"""
        try:
            character_data = json.loads(character_json)
            character = CharacterBackground(**character_data)
            return self.create_character(character_id, character)
        except Exception as e:
            logger.error(f"Failed to import character: {e}")
            return False


# 全局角色管理器实例
character_manager = CharacterManager()