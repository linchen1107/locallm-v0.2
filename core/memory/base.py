"""
記憶系統基礎類
定義記憶系統的通用接口和數據結構
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid


class MemoryType(Enum):
    """記憶類型枚舉"""
    SESSION = "session"           # 會話記憶
    LONG_TERM = "long_term"      # 長期記憶
    KNOWLEDGE = "knowledge"       # 知識圖譜
    PERSONAL = "personal"        # 個人記憶


class MemoryImportance(Enum):
    """記憶重要性等級"""
    LOW = 1      # 低重要性
    MEDIUM = 2   # 中等重要性
    HIGH = 3     # 高重要性
    CRITICAL = 4 # 關鍵重要性


@dataclass
class MemoryItem:
    """記憶項目基礎類"""
    id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['importance'] = self.importance.value
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """從字典創建記憶項目"""
        data['memory_type'] = MemoryType(data['memory_type'])
        data['importance'] = MemoryImportance(data['importance'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)
    
    def update_access(self):
        """更新訪問信息"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class ConversationTurn:
    """對話輪次"""
    id: str
    user_input: str
    assistant_response: str
    timestamp: datetime
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """從字典創建對話輪次"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class KnowledgeEntity:
    """知識實體"""
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntity':
        """從字典創建知識實體"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


@dataclass
class KnowledgeRelation:
    """知識關係"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    strength: float  # 關係強度 0.0-1.0
    properties: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeRelation':
        """從字典創建知識關係"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class BaseMemory(ABC):
    """記憶基礎抽象類"""
    
    def __init__(self, memory_type: MemoryType):
        self.memory_type = memory_type
        self.items: Dict[str, MemoryItem] = {}
    
    @abstractmethod
    async def store(self, content: str, importance: MemoryImportance = MemoryImportance.MEDIUM, 
                   metadata: Dict[str, Any] = None, tags: List[str] = None) -> str:
        """存儲記憶"""
        pass
    
    @abstractmethod
    async def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """檢索記憶"""
        pass
    
    @abstractmethod
    async def update(self, memory_id: str, content: str = None, 
                    importance: MemoryImportance = None, metadata: Dict[str, Any] = None) -> bool:
        """更新記憶"""
        pass
    
    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """刪除記憶"""
        pass
    
    @abstractmethod
    async def get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """根據ID獲取記憶"""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[MemoryItem]:
        """獲取所有記憶"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """清空記憶"""
        pass
    
    @abstractmethod
    async def save_to_disk(self) -> bool:
        """保存到磁盤"""
        pass
    
    @abstractmethod
    async def load_from_disk(self) -> bool:
        """從磁盤加載"""
        pass
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        return str(uuid.uuid4())
    
    def _create_memory_item(self, content: str, importance: MemoryImportance,
                          metadata: Dict[str, Any] = None, tags: List[str] = None) -> MemoryItem:
        """創建記憶項目"""
        now = datetime.now()
        return MemoryItem(
            id=self._generate_id(),
            content=content,
            memory_type=self.memory_type,
            importance=importance,
            created_at=now,
            last_accessed=now,
            metadata=metadata or {},
            tags=tags or []
        )


class MemorySearchResult:
    """記憶搜索結果"""
    
    def __init__(self, items: List[MemoryItem], query: str, total_count: int):
        self.items = items
        self.query = query
        self.total_count = total_count
        self.search_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "items": [item.to_dict() for item in self.items],
            "query": self.query,
            "total_count": self.total_count,
            "search_time": self.search_time.isoformat(),
            "returned_count": len(self.items)
        }


class MemoryStats:
    """記憶統計信息"""
    
    def __init__(self, memory_type: MemoryType, total_items: int, 
                 by_importance: Dict[str, int], recent_activity: int):
        self.memory_type = memory_type
        self.total_items = total_items
        self.by_importance = by_importance
        self.recent_activity = recent_activity
        self.generated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "memory_type": self.memory_type.value,
            "total_items": self.total_items,
            "by_importance": self.by_importance,
            "recent_activity": self.recent_activity,
            "generated_at": self.generated_at.isoformat()
        }

