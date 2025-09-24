"""
個人記憶系統
管理用戶偏好、習慣和個性化信息
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .base import (
    BaseMemory, MemoryItem, MemoryType, MemoryImportance,
    MemoryStats
)
from config.settings import get_settings


class PersonalMemory(BaseMemory):
    """個人記憶實現"""
    
    def __init__(self, user_id: str = "default"):
        super().__init__(MemoryType.PERSONAL)
        self.user_id = user_id
        self.settings = get_settings()
        
        # 存儲目錄
        self.personal_dir = Path(self.settings.storage.data_dir) / "personal_memory"
        self.personal_dir.mkdir(parents=True, exist_ok=True)
        
        # 數據文件
        self.memory_file = self.personal_dir / f"personal_{user_id}.json"
        self.preferences_file = self.personal_dir / f"preferences_{user_id}.json"
        self.patterns_file = self.personal_dir / f"patterns_{user_id}.json"
        
        # 個人數據結構
        self.preferences: Dict[str, Any] = {}
        self.usage_patterns: Dict[str, Any] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        
        # 加載數據
        import asyncio
        try:
            asyncio.create_task(self.load_from_disk())
        except RuntimeError:
            # 如果沒有運行的事件循環，先創建一個
            pass
    
    async def store(self, content: str, importance: MemoryImportance = MemoryImportance.MEDIUM,
                   metadata: Dict[str, Any] = None, tags: List[str] = None) -> str:
        """存儲個人記憶"""
        memory_item = self._create_memory_item(content, importance, metadata, tags)
        self.items[memory_item.id] = memory_item
        
        # 自動保存
        await self.save_to_disk()
        return memory_item.id
    
    async def store_preference(self, category: str, key: str, value: Any, 
                              confidence: float = 1.0) -> bool:
        """存儲用戶偏好"""
        if category not in self.preferences:
            self.preferences[category] = {}
        
        self.preferences[category][key] = {
            "value": value,
            "confidence": confidence,
            "last_updated": datetime.now().isoformat(),
            "update_count": self.preferences[category].get(key, {}).get("update_count", 0) + 1
        }
        
        await self.save_to_disk()
        return True
    
    async def get_preference(self, category: str, key: str = None) -> Any:
        """獲取用戶偏好"""
        if key is None:
            return self.preferences.get(category, {})
        
        category_prefs = self.preferences.get(category, {})
        pref_data = category_prefs.get(key)
        
        if pref_data:
            return pref_data["value"]
        return None
    
    async def update_preference_confidence(self, category: str, key: str, 
                                         confidence_delta: float) -> bool:
        """更新偏好置信度"""
        if category in self.preferences and key in self.preferences[category]:
            current_confidence = self.preferences[category][key]["confidence"]
            new_confidence = max(0.0, min(1.0, current_confidence + confidence_delta))
            self.preferences[category][key]["confidence"] = new_confidence
            self.preferences[category][key]["last_updated"] = datetime.now().isoformat()
            
            await self.save_to_disk()
            return True
        return False
    
    async def record_interaction(self, interaction_type: str, details: Dict[str, Any]) -> str:
        """記錄用戶交互"""
        interaction = {
            "id": self._generate_id(),
            "type": interaction_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        self.interaction_history.append(interaction)
        
        # 限制歷史記錄大小
        max_history = 1000
        if len(self.interaction_history) > max_history:
            self.interaction_history = self.interaction_history[-max_history:]
        
        # 更新使用模式
        await self._update_usage_patterns(interaction)
        
        await self.save_to_disk()
        return interaction["id"]
    
    async def _update_usage_patterns(self, interaction: Dict[str, Any]):
        """更新使用模式"""
        interaction_type = interaction["type"]
        timestamp = datetime.fromisoformat(interaction["timestamp"])
        
        # 時間模式
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        if "time_patterns" not in self.usage_patterns:
            self.usage_patterns["time_patterns"] = {
                "hours": defaultdict(int),
                "weekdays": defaultdict(int)
            }
        
        self.usage_patterns["time_patterns"]["hours"][str(hour)] += 1
        self.usage_patterns["time_patterns"]["weekdays"][str(weekday)] += 1
        
        # 交互類型模式
        if "interaction_types" not in self.usage_patterns:
            self.usage_patterns["interaction_types"] = defaultdict(int)
        
        self.usage_patterns["interaction_types"][interaction_type] += 1
        
        # 頻率模式
        if "frequency" not in self.usage_patterns:
            self.usage_patterns["frequency"] = {
                "daily_counts": defaultdict(int),
                "total_interactions": 0
            }
        
        date_str = timestamp.date().isoformat()
        self.usage_patterns["frequency"]["daily_counts"][date_str] += 1
        self.usage_patterns["frequency"]["total_interactions"] += 1
    
    async def get_usage_insights(self) -> Dict[str, Any]:
        """獲取使用洞察"""
        insights = {}
        
        # 最活躍的時間
        if "time_patterns" in self.usage_patterns:
            hours = self.usage_patterns["time_patterns"]["hours"]
            if hours:
                most_active_hour = max(hours.items(), key=lambda x: x[1])
                insights["most_active_hour"] = {
                    "hour": int(most_active_hour[0]),
                    "count": most_active_hour[1]
                }
            
            weekdays = self.usage_patterns["time_patterns"]["weekdays"]
            if weekdays:
                most_active_weekday = max(weekdays.items(), key=lambda x: x[1])
                weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
                insights["most_active_weekday"] = {
                    "weekday": weekday_names[int(most_active_weekday[0])],
                    "count": most_active_weekday[1]
                }
        
        # 最常用的交互類型
        if "interaction_types" in self.usage_patterns:
            interaction_types = self.usage_patterns["interaction_types"]
            if interaction_types:
                most_used_type = max(interaction_types.items(), key=lambda x: x[1])
                insights["most_used_interaction"] = {
                    "type": most_used_type[0],
                    "count": most_used_type[1]
                }
        
        # 使用頻率趨勢
        if "frequency" in self.usage_patterns:
            daily_counts = self.usage_patterns["frequency"]["daily_counts"]
            if daily_counts:
                # 最近7天的使用量
                recent_days = []
                now = datetime.now()
                for i in range(7):
                    date = (now - timedelta(days=i)).date().isoformat()
                    count = daily_counts.get(date, 0)
                    recent_days.append({"date": date, "count": count})
                
                insights["recent_usage"] = recent_days
                insights["avg_daily_usage"] = sum(day["count"] for day in recent_days) / len(recent_days)
        
        return insights
    
    async def learn_from_feedback(self, feedback_type: str, context: Dict[str, Any], 
                                 positive: bool) -> bool:
        """從用戶反饋中學習"""
        
        feedback_record = {
            "type": feedback_type,
            "context": context,
            "positive": positive,
            "timestamp": datetime.now().isoformat()
        }
        
        # 記錄反饋
        await self.record_interaction("feedback", feedback_record)
        
        # 根據反饋調整偏好
        if feedback_type == "response_style":
            style = context.get("style")
            if style:
                confidence_delta = 0.1 if positive else -0.1
                await self.update_preference_confidence("response", "style", confidence_delta)
        
        elif feedback_type == "content_relevance":
            topic = context.get("topic")
            if topic:
                category = "interests"
                confidence_delta = 0.05 if positive else -0.05
                await self.update_preference_confidence(category, topic, confidence_delta)
        
        return True
    
    async def get_personalized_suggestions(self) -> List[str]:
        """獲取個性化建議"""
        suggestions = []
        
        # 基於使用模式的建議
        insights = await self.get_usage_insights()
        
        if "most_active_hour" in insights:
            hour = insights["most_active_hour"]["hour"]
            if 9 <= hour <= 17:
                suggestions.append("您通常在工作時間較活躍，建議設置工作相關的快捷命令")
            elif 18 <= hour <= 22:
                suggestions.append("您通常在晚間較活躍，建議設置學習相關的功能")
        
        # 基於偏好的建議
        response_prefs = await self.get_preference("response")
        if response_prefs and "detail_level" in response_prefs:
            if response_prefs["detail_level"]["value"] == "detailed":
                suggestions.append("您偏好詳細回答，建議啟用深度分析功能")
            elif response_prefs["detail_level"]["value"] == "brief":
                suggestions.append("您偏好簡潔回答，建議啟用快速模式")
        
        # 基於交互歷史的建議
        if "most_used_interaction" in insights:
            interaction_type = insights["most_used_interaction"]["type"]
            if interaction_type == "query":
                suggestions.append("您經常進行查詢，建議優化搜索關鍵詞收藏功能")
            elif interaction_type == "agent":
                suggestions.append("您經常使用代理功能，建議設置常用任務模板")
        
        return suggestions
    
    async def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """檢索個人記憶"""
        query_lower = query.lower()
        matching_items = []
        
        for item in self.items.values():
            relevance = 0.0
            
            # 內容匹配
            if query_lower in item.content.lower():
                relevance += 1.0
            
            # 標籤匹配
            for tag in item.tags:
                if query_lower in tag.lower():
                    relevance += 0.5
            
            # 元數據匹配
            for key, value in item.metadata.items():
                if query_lower in str(value).lower():
                    relevance += 0.3
            
            if relevance > 0:
                item.update_access()
                matching_items.append((item, relevance))
        
        # 按相關性排序
        matching_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in matching_items[:limit]]
    
    async def update(self, memory_id: str, content: str = None,
                    importance: MemoryImportance = None, metadata: Dict[str, Any] = None) -> bool:
        """更新記憶"""
        if memory_id not in self.items:
            return False
        
        item = self.items[memory_id]
        
        if content is not None:
            item.content = content
        if importance is not None:
            item.importance = importance
        if metadata is not None:
            item.metadata.update(metadata)
        
        item.last_accessed = datetime.now()
        await self.save_to_disk()
        return True
    
    async def delete(self, memory_id: str) -> bool:
        """刪除記憶"""
        if memory_id in self.items:
            del self.items[memory_id]
            await self.save_to_disk()
            return True
        return False
    
    async def get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """根據ID獲取記憶"""
        return self.items.get(memory_id)
    
    async def get_all(self) -> List[MemoryItem]:
        """獲取所有記憶"""
        return list(self.items.values())
    
    async def clear(self) -> bool:
        """清空個人記憶"""
        self.items.clear()
        self.preferences.clear()
        self.usage_patterns.clear()
        self.interaction_history.clear()
        await self.save_to_disk()
        return True
    
    async def save_to_disk(self) -> bool:
        """保存到磁盤"""
        try:
            # 保存記憶項目
            memory_data = {
                "user_id": self.user_id,
                "items": [item.to_dict() for item in self.items.values()],
                "interaction_history": self.interaction_history,
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
            # 保存偏好
            with open(self.preferences_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, ensure_ascii=False, indent=2)
            
            # 保存使用模式
            patterns_data = dict(self.usage_patterns)
            # 轉換defaultdict為普通dict
            for key, value in patterns_data.items():
                if isinstance(value, defaultdict):
                    patterns_data[key] = dict(value)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, defaultdict):
                            patterns_data[key][subkey] = dict(subvalue)
            
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to save personal memory: {e}")
            return False
    
    async def load_from_disk(self) -> bool:
        """從磁盤加載"""
        try:
            # 加載記憶項目
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                
                self.user_id = memory_data.get("user_id", self.user_id)
                self.interaction_history = memory_data.get("interaction_history", [])
                
                self.items.clear()
                for item_data in memory_data.get("items", []):
                    item = MemoryItem.from_dict(item_data)
                    self.items[item.id] = item
            
            # 加載偏好
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r', encoding='utf-8') as f:
                    self.preferences = json.load(f)
            
            # 加載使用模式
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)
                    
                    # 重建defaultdict結構
                    self.usage_patterns = {}
                    for key, value in patterns_data.items():
                        if key == "time_patterns":
                            self.usage_patterns[key] = {
                                "hours": defaultdict(int, value.get("hours", {})),
                                "weekdays": defaultdict(int, value.get("weekdays", {}))
                            }
                        elif key == "interaction_types":
                            self.usage_patterns[key] = defaultdict(int, value)
                        elif key == "frequency":
                            self.usage_patterns[key] = {
                                "daily_counts": defaultdict(int, value.get("daily_counts", {})),
                                "total_interactions": value.get("total_interactions", 0)
                            }
                        else:
                            self.usage_patterns[key] = value
            
            return True
        except Exception as e:
            print(f"Failed to load personal memory: {e}")
            return False
    
    async def get_personal_stats(self) -> MemoryStats:
        """獲取個人記憶統計"""
        by_importance = {}
        for importance in MemoryImportance:
            count = sum(1 for item in self.items.values() if item.importance == importance)
            by_importance[importance.name.lower()] = count
        
        # 計算最近活動
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_activity = len([
            interaction for interaction in self.interaction_history
            if datetime.fromisoformat(interaction["timestamp"]) > recent_cutoff
        ])
        
        return MemoryStats(
            memory_type=MemoryType.PERSONAL,
            total_items=len(self.items),
            by_importance=by_importance,
            recent_activity=recent_activity
        )
