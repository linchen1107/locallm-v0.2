"""
會話記憶系統
管理當前會話的上下文和對話歷史
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from .base import (
    BaseMemory, MemoryItem, MemoryType, MemoryImportance, 
    ConversationTurn, MemorySearchResult, MemoryStats
)
from config.settings import get_settings


class SessionMemory(BaseMemory):
    """會話記憶實現"""
    
    def __init__(self, session_id: str = None):
        super().__init__(MemoryType.SESSION)
        self.session_id = session_id or self._generate_session_id()
        self.conversation_history: List[ConversationTurn] = []
        self.session_context: Dict[str, Any] = {}
        self.settings = get_settings()
        self.max_history_size = 100  # 最大對話歷史數量
        self.session_started = datetime.now()
        
        # 創建會話目錄
        self.session_dir = Path(self.settings.storage.data_dir) / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.session_dir / f"session_{self.session_id}.json"
    
    def _generate_session_id(self) -> str:
        """生成會話ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async def store(self, content: str, importance: MemoryImportance = MemoryImportance.MEDIUM,
                   metadata: Dict[str, Any] = None, tags: List[str] = None) -> str:
        """存儲會話記憶"""
        memory_item = self._create_memory_item(content, importance, metadata, tags)
        self.items[memory_item.id] = memory_item
        
        # 自動保存
        await self.save_to_disk()
        return memory_item.id
    
    async def add_conversation_turn(self, user_input: str, assistant_response: str,
                                   context: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> str:
        """添加對話輪次"""
        turn = ConversationTurn(
            id=self._generate_id(),
            user_input=user_input,
            assistant_response=assistant_response,
            timestamp=datetime.now(),
            context=context or {},
            metadata=metadata or {}
        )
        
        self.conversation_history.append(turn)
        
        # 限制歷史大小
        if len(self.conversation_history) > self.max_history_size:
            self.conversation_history = self.conversation_history[-self.max_history_size:]
        
        # 自動保存
        await self.save_to_disk()
        return turn.id
    
    async def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """檢索會話記憶"""
        query_lower = query.lower()
        matching_items = []
        
        # 搜索記憶項目
        for item in self.items.values():
            if query_lower in item.content.lower():
                item.update_access()
                matching_items.append((item, self._calculate_relevance_score(item, query)))
        
        # 搜索對話歷史
        for turn in self.conversation_history:
            if (query_lower in turn.user_input.lower() or 
                query_lower in turn.assistant_response.lower()):
                
                # 將對話輪次轉換為記憶項目
                memory_item = MemoryItem(
                    id=turn.id,
                    content=f"用戶: {turn.user_input}\n助手: {turn.assistant_response}",
                    memory_type=MemoryType.SESSION,
                    importance=MemoryImportance.MEDIUM,
                    created_at=turn.timestamp,
                    last_accessed=datetime.now(),
                    metadata=turn.metadata,
                    tags=["conversation"]
                )
                matching_items.append((memory_item, self._calculate_relevance_score(memory_item, query)))
        
        # 按相關性排序
        matching_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in matching_items[:limit]]
    
    def _calculate_relevance_score(self, item: MemoryItem, query: str) -> float:
        """計算相關性分數"""
        query_lower = query.lower()
        content_lower = item.content.lower()
        
        # 基礎匹配分數
        if query_lower in content_lower:
            base_score = 1.0
        else:
            base_score = 0.0
        
        # 重要性加權
        importance_weight = item.importance.value / 4.0
        
        # 時間衰減（最近的記憶分數更高）
        time_diff = datetime.now() - item.created_at
        time_decay = max(0.1, 1.0 - (time_diff.total_seconds() / (24 * 3600)))  # 24小時衰減
        
        # 訪問頻率加權
        access_weight = min(1.0, item.access_count / 10.0)
        
        return base_score * importance_weight * time_decay * (1 + access_weight)
    
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
        
        # 檢查是否是對話歷史
        self.conversation_history = [turn for turn in self.conversation_history if turn.id != memory_id]
        await self.save_to_disk()
        return True
    
    async def get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """根據ID獲取記憶"""
        return self.items.get(memory_id)
    
    async def get_all(self) -> List[MemoryItem]:
        """獲取所有記憶"""
        return list(self.items.values())
    
    async def get_conversation_history(self, limit: int = None) -> List[ConversationTurn]:
        """獲取對話歷史"""
        if limit is None:
            return self.conversation_history.copy()
        return self.conversation_history[-limit:]
    
    async def get_recent_context(self, turns: int = 5) -> str:
        """獲取最近的對話上下文"""
        recent_turns = self.conversation_history[-turns:]
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"用戶: {turn.user_input}")
            context_parts.append(f"助手: {turn.assistant_response}")
        
        return "\n".join(context_parts)
    
    async def update_session_context(self, key: str, value: Any):
        """更新會話上下文"""
        self.session_context[key] = value
        await self.save_to_disk()
    
    async def get_session_context(self, key: str = None) -> Any:
        """獲取會話上下文"""
        if key is None:
            return self.session_context.copy()
        return self.session_context.get(key)
    
    async def clear(self) -> bool:
        """清空會話記憶"""
        self.items.clear()
        self.conversation_history.clear()
        self.session_context.clear()
        await self.save_to_disk()
        return True
    
    async def save_to_disk(self) -> bool:
        """保存到磁盤"""
        try:
            session_data = {
                "session_id": self.session_id,
                "session_started": self.session_started.isoformat(),
                "session_context": self.session_context,
                "memory_items": [item.to_dict() for item in self.items.values()],
                "conversation_history": [turn.to_dict() for turn in self.conversation_history],
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to save session memory: {e}")
            return False
    
    async def load_from_disk(self) -> bool:
        """從磁盤加載"""
        try:
            if not self.session_file.exists():
                return True
            
            with open(self.session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.session_id = session_data.get("session_id", self.session_id)
            self.session_started = datetime.fromisoformat(session_data.get("session_started", datetime.now().isoformat()))
            self.session_context = session_data.get("session_context", {})
            
            # 加載記憶項目
            self.items.clear()
            for item_data in session_data.get("memory_items", []):
                item = MemoryItem.from_dict(item_data)
                self.items[item.id] = item
            
            # 加載對話歷史
            self.conversation_history.clear()
            for turn_data in session_data.get("conversation_history", []):
                turn = ConversationTurn.from_dict(turn_data)
                self.conversation_history.append(turn)
            
            return True
        except Exception as e:
            print(f"Failed to load session memory: {e}")
            return False
    
    async def get_session_stats(self) -> MemoryStats:
        """獲取會話統計信息"""
        by_importance = {}
        for importance in MemoryImportance:
            count = sum(1 for item in self.items.values() if item.importance == importance)
            by_importance[importance.name.lower()] = count
        
        # 計算最近活動（最近1小時的記憶）
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_activity = sum(1 for item in self.items.values() if item.created_at > recent_cutoff)
        recent_activity += sum(1 for turn in self.conversation_history if turn.timestamp > recent_cutoff)
        
        return MemoryStats(
            memory_type=MemoryType.SESSION,
            total_items=len(self.items) + len(self.conversation_history),
            by_importance=by_importance,
            recent_activity=recent_activity
        )
    
    async def cleanup_old_data(self, max_age_hours: int = 24):
        """清理舊數據"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # 清理舊的記憶項目
        old_items = [item_id for item_id, item in self.items.items() 
                    if item.created_at < cutoff_time and item.importance.value < MemoryImportance.HIGH.value]
        
        for item_id in old_items:
            del self.items[item_id]
        
        # 清理舊的對話歷史（保留重要的）
        self.conversation_history = [turn for turn in self.conversation_history 
                                   if turn.timestamp > cutoff_time or 
                                   turn.metadata.get("important", False)]
        
        await self.save_to_disk()
        return len(old_items)

