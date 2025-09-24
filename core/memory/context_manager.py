"""
上下文管理器
統一管理所有類型的記憶系統，提供智能的上下文感知能力
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .base import MemoryItem, MemoryType, MemoryImportance, MemoryStats
from .session_memory import SessionMemory
from .long_term_memory import LongTermMemory
from .knowledge_graph import KnowledgeGraph
from .personal_memory import PersonalMemory
from config.settings import get_settings


class ContextManager:
    """上下文管理器主類"""
    
    def __init__(self, user_id: str = "default", session_id: str = None):
        self.user_id = user_id
        self.settings = get_settings()
        
        # 初始化各類記憶系統
        self.session_memory = SessionMemory(session_id)
        self.long_term_memory = LongTermMemory()
        self.knowledge_graph = KnowledgeGraph()
        self.personal_memory = PersonalMemory(user_id)
        
        # 上下文配置
        self.max_context_items = 20
        self.context_weight_decay = 0.1
        
        # 當前上下文狀態
        self.current_context: Dict[str, Any] = {}
        self.context_history: List[Dict[str, Any]] = []
        
        # 啟動記憶加載
        asyncio.create_task(self._load_all_memories())
    
    async def _load_all_memories(self):
        """加載所有記憶系統"""
        await asyncio.gather(
            self.session_memory.load_from_disk(),
            self.long_term_memory.load_from_disk(),
            self.personal_memory.load_from_disk()
        )
        self.knowledge_graph.load_from_disk()
    
    async def process_interaction(self, user_input: str, assistant_response: str,
                                context: Dict[str, Any] = None) -> str:
        """處理用戶交互，更新各種記憶"""
        
        # 記錄會話記憶
        turn_id = await self.session_memory.add_conversation_turn(
            user_input, assistant_response, context
        )
        
        # 提取並存儲長期記憶
        await self.long_term_memory.store_from_conversation(
            user_input, assistant_response, MemoryImportance.MEDIUM
        )
        
        # 更新知識圖譜
        combined_text = f"用戶問題：{user_input}\n助手回答：{assistant_response}"
        await self.knowledge_graph.build_knowledge_from_text(combined_text)
        
        # 記錄個人交互
        interaction_details = {
            "user_input": user_input,
            "response_length": len(assistant_response),
            "context": context or {}
        }
        await self.personal_memory.record_interaction("conversation", interaction_details)
        
        # 更新當前上下文
        await self._update_current_context(user_input, assistant_response, context)
        
        return turn_id
    
    async def _update_current_context(self, user_input: str, assistant_response: str,
                                    context: Dict[str, Any] = None):
        """更新當前上下文"""
        now = datetime.now()
        
        # 添加到上下文歷史
        context_item = {
            "timestamp": now,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "context": context or {},
            "importance": self._calculate_context_importance(user_input, assistant_response)
        }
        
        self.context_history.append(context_item)
        
        # 限制上下文歷史大小
        if len(self.context_history) > self.max_context_items:
            self.context_history = self.context_history[-self.max_context_items:]
        
        # 更新當前上下文摘要
        self.current_context = await self._build_context_summary()
    
    def _calculate_context_importance(self, user_input: str, assistant_response: str) -> float:
        """計算上下文重要性"""
        importance = 0.5  # 基礎重要性
        
        # 基於用戶輸入長度
        if len(user_input) > 100:
            importance += 0.1
        
        # 基於回答長度
        if len(assistant_response) > 200:
            importance += 0.1
        
        # 檢查重要關鍵詞
        important_keywords = [
            "重要", "記住", "設置", "配置", "偏好", "remember", "important", "save", "store"
        ]
        for keyword in important_keywords:
            if keyword in user_input.lower() or keyword in assistant_response.lower():
                importance += 0.2
                break
        
        return min(importance, 1.0)
    
    async def _build_context_summary(self) -> Dict[str, Any]:
        """構建當前上下文摘要"""
        if not self.context_history:
            return {}
        
        # 獲取最近的重要上下文
        recent_items = self.context_history[-5:]
        important_items = [item for item in self.context_history if item["importance"] > 0.7]
        
        # 提取關鍵主題
        all_text = " ".join([
            item["user_input"] + " " + item["assistant_response"]
            for item in recent_items
        ])
        
        summary = {
            "recent_topics": await self._extract_topics(all_text),
            "conversation_length": len(self.context_history),
            "last_interaction": recent_items[-1]["timestamp"] if recent_items else None,
            "important_count": len(important_items),
            "current_session_id": self.session_memory.session_id
        }
        
        return summary
    
    async def _extract_topics(self, text: str) -> List[str]:
        """提取文本主題"""
        # 簡化版主題提取
        import re
        
        # 移除標點符號並分詞
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 過濾停用詞
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
            '的', '了', '是', '在', '有', '和', '與', '或', '但', '如果', '因為',
            '我', '你', '他', '她', '它', '我們', '你們', '他們'
        }
        
        # 統計詞頻
        word_freq = {}
        for word in words:
            if len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 返回前5個高頻詞作為主題
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5]]
    
    async def get_relevant_context(self, query: str, max_items: int = 10) -> Dict[str, List[MemoryItem]]:
        """獲取與查詢相關的上下文"""
        
        # 並行檢索各類記憶
        results = await asyncio.gather(
            self.session_memory.retrieve(query, max_items),
            self.long_term_memory.retrieve(query, max_items),
            self.personal_memory.retrieve(query, max_items)
        )
        
        return {
            "session": results[0],
            "long_term": results[1],
            "personal": results[2]
        }
    
    async def get_smart_context(self, query: str, include_kg: bool = True) -> str:
        """獲取智能上下文（格式化為文本）"""
        
        # 獲取相關記憶
        relevant_memories = await self.get_relevant_context(query, 5)
        
        context_parts = []
        
        # 添加會話上下文
        if relevant_memories["session"]:
            context_parts.append("=== 最近對話 ===")
            for item in relevant_memories["session"][:3]:
                context_parts.append(f"- {item.content}")
        
        # 添加相關知識
        if relevant_memories["long_term"]:
            context_parts.append("\n=== 相關知識 ===")
            for item in relevant_memories["long_term"][:3]:
                context_parts.append(f"- {item.content}")
        
        # 添加個人偏好
        if relevant_memories["personal"]:
            context_parts.append("\n=== 個人信息 ===")
            for item in relevant_memories["personal"][:2]:
                context_parts.append(f"- {item.content}")
        
        # 添加知識圖譜信息
        if include_kg:
            kg_entities = self.knowledge_graph.search_entities(query, limit=3)
            if kg_entities:
                context_parts.append("\n=== 相關實體 ===")
                for entity in kg_entities:
                    context_parts.append(f"- {entity.name} ({entity.entity_type})")
        
        return "\n".join(context_parts)
    
    async def store_important_fact(self, fact: str, source: str = "user",
                                 importance: MemoryImportance = MemoryImportance.HIGH) -> str:
        """存儲重要事實"""
        
        metadata = {
            "source": source,
            "stored_at": datetime.now().isoformat(),
            "verified": False
        }
        
        # 存儲到長期記憶
        memory_id = await self.long_term_memory.store(fact, importance, metadata, ["fact"])
        
        # 同時存儲到個人記憶（如果與用戶相關）
        if "用戶" in fact or "我" in fact or "個人" in fact:
            await self.personal_memory.store(fact, importance, metadata, ["personal_fact"])
        
        # 嘗試提取到知識圖譜
        await self.knowledge_graph.build_knowledge_from_text(fact)
        
        return memory_id
    
    async def get_user_preferences(self) -> Dict[str, Any]:
        """獲取用戶偏好摘要"""
        
        # 獲取個人記憶中的偏好
        response_prefs = await self.personal_memory.get_preference("response")
        ui_prefs = await self.personal_memory.get_preference("ui")
        content_prefs = await self.personal_memory.get_preference("content")
        
        # 獲取使用洞察
        insights = await self.personal_memory.get_usage_insights()
        
        return {
            "response_preferences": response_prefs,
            "ui_preferences": ui_prefs,
            "content_preferences": content_prefs,
            "usage_insights": insights,
            "personalization_suggestions": await self.personal_memory.get_personalized_suggestions()
        }
    
    async def learn_from_feedback(self, feedback_type: str, feedback_data: Dict[str, Any],
                                positive: bool) -> bool:
        """從用戶反饋中學習"""
        
        # 記錄到個人記憶
        await self.personal_memory.learn_from_feedback(feedback_type, feedback_data, positive)
        
        # 如果是重要反饋，存儲到長期記憶
        if feedback_data.get("importance", "medium") == "high":
            feedback_content = f"用戶反饋：{feedback_type} - {'正面' if positive else '負面'}"
            await self.long_term_memory.store(
                feedback_content, 
                MemoryImportance.HIGH,
                {"feedback_data": feedback_data},
                ["feedback", "learning"]
            )
        
        return True
    
    async def get_memory_stats(self) -> Dict[str, MemoryStats]:
        """獲取所有記憶系統的統計信息"""
        
        stats = await asyncio.gather(
            self.session_memory.get_session_stats(),
            self.long_term_memory.get_memory_stats(),
            self.personal_memory.get_personal_stats()
        )
        
        kg_stats_data = self.knowledge_graph.get_graph_stats()
        kg_stats = MemoryStats(
            memory_type=MemoryType.KNOWLEDGE,
            total_items=kg_stats_data["total_entities"] + kg_stats_data["total_relations"],
            by_importance={"entities": kg_stats_data["total_entities"], "relations": kg_stats_data["total_relations"]},
            recent_activity=0  # 知識圖譜沒有時間戳，暫時設為0
        )
        
        return {
            "session": stats[0],
            "long_term": stats[1],
            "personal": stats[2],
            "knowledge_graph": kg_stats
        }
    
    async def cleanup_old_memories(self, max_age_days: int = 30) -> Dict[str, int]:
        """清理舊記憶"""
        
        cleanup_results = {}
        
        # 清理會話記憶
        cleaned_session = await self.session_memory.cleanup_old_data(max_age_days * 24)
        cleanup_results["session"] = cleaned_session
        
        # 長期記憶通常不清理，但可以整合
        consolidated = await self.long_term_memory.consolidate_memories()
        cleanup_results["long_term_consolidated"] = consolidated
        
        # 個人記憶保留所有數據
        cleanup_results["personal"] = 0
        
        return cleanup_results
    
    async def export_memories(self, memory_types: List[str] = None) -> Dict[str, Any]:
        """導出記憶數據"""
        
        if memory_types is None:
            memory_types = ["session", "long_term", "personal", "knowledge_graph"]
        
        export_data = {}
        
        if "session" in memory_types:
            export_data["session"] = {
                "conversation_history": [turn.to_dict() for turn in self.session_memory.conversation_history],
                "memory_items": [item.to_dict() for item in await self.session_memory.get_all()]
            }
        
        if "long_term" in memory_types:
            export_data["long_term"] = [item.to_dict() for item in await self.long_term_memory.get_all()]
        
        if "personal" in memory_types:
            export_data["personal"] = {
                "memory_items": [item.to_dict() for item in await self.personal_memory.get_all()],
                "preferences": self.personal_memory.preferences,
                "usage_patterns": dict(self.personal_memory.usage_patterns)
            }
        
        if "knowledge_graph" in memory_types:
            export_data["knowledge_graph"] = {
                "entities": [entity.to_dict() for entity in self.knowledge_graph.entities.values()],
                "relations": [relation.to_dict() for relation in self.knowledge_graph.relations.values()]
            }
        
        export_data["export_timestamp"] = datetime.now().isoformat()
        export_data["user_id"] = self.user_id
        
        return export_data
    
    async def clear_all_memories(self) -> bool:
        """清空所有記憶（謹慎使用）"""
        
        await asyncio.gather(
            self.session_memory.clear(),
            self.long_term_memory.clear(),
            self.personal_memory.clear()
        )
        
        self.knowledge_graph.clear()
        
        # 重置上下文
        self.current_context.clear()
        self.context_history.clear()
        
        return True
    
    async def save_all_memories(self) -> bool:
        """保存所有記憶到磁盤"""
        
        results = await asyncio.gather(
            self.session_memory.save_to_disk(),
            self.long_term_memory.save_to_disk(),
            self.personal_memory.save_to_disk(),
            return_exceptions=True
        )
        
        kg_result = self.knowledge_graph.save_to_disk()
        
        # 檢查是否所有保存操作都成功
        all_success = all(result is True for result in results) and kg_result
        
        return all_success
    
    def get_current_context_summary(self) -> Dict[str, Any]:
        """獲取當前上下文摘要"""
        return {
            "current_context": self.current_context,
            "context_history_length": len(self.context_history),
            "session_id": self.session_memory.session_id,
            "user_id": self.user_id,
            "last_updated": datetime.now().isoformat()
        }

