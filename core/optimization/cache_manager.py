"""
高性能緩存管理器
支持多級緩存、智能失效和持久化存儲
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import aiofiles
import sqlite3
from threading import RLock

import numpy as np


@dataclass
class CacheEntry:
    """緩存條目"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    expires_at: Optional[datetime] = None
    size_bytes: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class CacheStats:
    """緩存統計"""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_size = 0
        self.start_time = datetime.now()
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "total_size_mb": self.total_size / (1024 * 1024),
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
        }


class MemoryCache:
    """內存緩存層"""
    
    def __init__(self, max_size_mb: int = 512, max_entries: int = 10000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self._lock = RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """獲取緩存項"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # 檢查過期
                if entry.expires_at and datetime.now() > entry.expires_at:
                    self._evict(key)
                    self.stats.misses += 1
                    return None
                
                # 更新訪問信息
                entry.accessed_at = datetime.now()
                entry.access_count += 1
                self.stats.hits += 1
                
                return entry.value
            
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            tags: List[str] = None) -> bool:
        """設置緩存項"""
        with self._lock:
            # 計算大小
            size_bytes = self._calculate_size(value)
            
            # 檢查是否需要清理空間
            if self._needs_eviction(size_bytes):
                self._evict_lru()
            
            # 創建緩存條目
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                expires_at=expires_at,
                size_bytes=size_bytes,
                tags=tags or []
            )
            
            # 如果key已存在，先移除舊條目
            if key in self.cache:
                self.stats.total_size -= self.cache[key].size_bytes
            
            self.cache[key] = entry
            self.stats.total_size += size_bytes
            
            return True
    
    def delete(self, key: str) -> bool:
        """刪除緩存項"""
        with self._lock:
            if key in self.cache:
                self.stats.total_size -= self.cache[key].size_bytes
                del self.cache[key]
                return True
            return False
    
    def clear_by_tags(self, tags: List[str]):
        """根據標籤清理緩存"""
        with self._lock:
            keys_to_delete = []
            for key, entry in self.cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                self.delete(key)
    
    def _calculate_size(self, value: Any) -> int:
        """估算對象大小"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value[:100])  # 樣本估算
            elif isinstance(value, dict):
                sample_items = list(value.items())[:50]  # 樣本估算
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in sample_items)
            else:
                # 使用pickle估算
                return len(pickle.dumps(value))
        except:
            return 1024  # 默認1KB
    
    def _needs_eviction(self, new_size: int) -> bool:
        """檢查是否需要清理"""
        return (self.stats.total_size + new_size > self.max_size_bytes or 
                len(self.cache) >= self.max_entries)
    
    def _evict_lru(self):
        """LRU淘汰策略"""
        if not self.cache:
            return
        
        # 找到最少最近使用的條目
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].accessed_at)
        self._evict(lru_key)
    
    def _evict(self, key: str):
        """淘汰指定條目"""
        if key in self.cache:
            self.stats.total_size -= self.cache[key].size_bytes
            del self.cache[key]
            self.stats.evictions += 1


class DiskCache:
    """磁盤緩存層"""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 2048):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # 初始化SQLite索引
        self.db_path = self.cache_dir / 'cache_index.db'
        self._init_db()
    
    def _init_db(self):
        """初始化數據庫"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP,
                    accessed_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    size_bytes INTEGER,
                    tags TEXT,
                    access_count INTEGER DEFAULT 1
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)')
    
    async def get(self, key: str) -> Optional[Any]:
        """異步獲取緩存項"""
        file_path = await self._get_file_path(key)
        if not file_path or not file_path.exists():
            return None
        
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()
                value = pickle.loads(data)
            
            # 更新訪問時間
            await self._update_access_time(key)
            
            return value
        except Exception:
            # 文件損壞，刪除記錄
            await self._remove_entry(key)
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                 tags: List[str] = None) -> bool:
        """異步設置緩存項"""
        try:
            # 序列化數據
            data = pickle.dumps(value)
            size_bytes = len(data)
            
            # 檢查磁盤空間
            if await self._needs_cleanup(size_bytes):
                await self._cleanup_old_entries()
            
            # 生成文件路徑
            file_path = self._generate_file_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 寫入文件
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
            
            # 更新索引
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (key, file_path, created_at, accessed_at, expires_at, size_bytes, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    key, str(file_path), datetime.now(), datetime.now(),
                    expires_at, size_bytes, json.dumps(tags or [])
                ))
            
            return True
        except Exception as e:
            print(f"Disk cache set error: {e}")
            return False
    
    async def _get_file_path(self, key: str) -> Optional[Path]:
        """獲取文件路徑"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT file_path, expires_at FROM cache_entries WHERE key = ?',
                (key,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            file_path, expires_at = row
            
            # 檢查過期
            if expires_at:
                expires_dt = datetime.fromisoformat(expires_at)
                if datetime.now() > expires_dt:
                    await self._remove_entry(key)
                    return None
            
            return Path(file_path)
    
    def _generate_file_path(self, key: str) -> Path:
        """生成文件路徑"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / hash_key[:2] / f"{hash_key}.cache"
    
    async def _needs_cleanup(self, new_size: int) -> bool:
        """檢查是否需要清理"""
        current_size = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT SUM(size_bytes) FROM cache_entries')
            result = cursor.fetchone()
            current_size = result[0] or 0
        
        return current_size + new_size > self.max_size_bytes
    
    async def _cleanup_old_entries(self):
        """清理舊條目"""
        with sqlite3.connect(self.db_path) as conn:
            # 刪除過期條目
            conn.execute(
                'DELETE FROM cache_entries WHERE expires_at < ?',
                (datetime.now(),)
            )
            
            # 按訪問時間刪除舊條目（保留最近訪問的80%）
            cursor = conn.execute('SELECT COUNT(*) FROM cache_entries')
            total_count = cursor.fetchone()[0]
            
            if total_count > 1000:  # 如果條目太多
                delete_count = total_count // 5  # 刪除20%
                conn.execute('''
                    DELETE FROM cache_entries WHERE key IN (
                        SELECT key FROM cache_entries 
                        ORDER BY accessed_at ASC 
                        LIMIT ?
                    )
                ''', (delete_count,))
    
    async def _remove_entry(self, key: str):
        """移除緩存條目"""
        try:
            # 從數據庫刪除記錄
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT file_path FROM cache_entries WHERE key = ?', (key,))
                row = cursor.fetchone()
                
                if row:
                    file_path = Path(row[0])
                    # 刪除文件
                    if file_path.exists():
                        file_path.unlink()
                    
                    # 刪除數據庫記錄
                    conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                    
        except Exception as e:
            # 靜默處理刪除錯誤，避免影響主流程
            pass


class MultiLevelCache:
    """多級緩存管理器"""
    
    def __init__(self, memory_cache_mb: int = 512, disk_cache_mb: int = 2048,
                 cache_dir: str = None):
        
        if cache_dir is None:
            cache_dir = str(Path.home() / '.locallm' / 'cache')
        
        self.memory_cache = MemoryCache(memory_cache_mb)
        self.disk_cache = DiskCache(cache_dir, disk_cache_mb)
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """多級緩存獲取"""
        # 先查內存緩存
        value = self.memory_cache.get(key)
        if value is not None:
            self.stats.hits += 1
            return value
        
        # 查磁盤緩存
        value = await self.disk_cache.get(key)
        if value is not None:
            # 回寫到內存緩存
            self.memory_cache.set(key, value)
            self.stats.hits += 1
            return value
        
        self.stats.misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                 tags: List[str] = None, cache_level: str = "both") -> bool:
        """多級緩存設置"""
        success = True
        
        if cache_level in ["memory", "both"]:
            success &= self.memory_cache.set(key, value, ttl_seconds, tags)
        
        if cache_level in ["disk", "both"]:
            success &= await self.disk_cache.set(key, value, ttl_seconds, tags)
        
        return success
    
    def delete(self, key: str):
        """刪除緩存項"""
        self.memory_cache.delete(key)
        # 磁盤緩存的刪除需要異步處理
        asyncio.create_task(self._remove_from_disk(key))
    
    async def _remove_from_disk(self, key: str):
        """從磁盤緩存移除"""
        await self.disk_cache._remove_entry(key)
    
    def clear_by_tags(self, tags: List[str]):
        """根據標籤清理緩存"""
        self.memory_cache.clear_by_tags(tags)
        # 磁盤緩存清理需要異步處理
        asyncio.create_task(self._clear_disk_by_tags(tags))
    
    async def _clear_disk_by_tags(self, tags: List[str]):
        """磁盤緩存標籤清理"""
        # 實現磁盤緩存的標籤清理邏輯
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        combined_stats = {
            "memory_cache": self.memory_cache.stats.to_dict(),
            "disk_cache_entries": self._get_disk_cache_count(),
            "total_hits": self.stats.hits + self.memory_cache.stats.hits,
            "total_misses": self.stats.misses + self.memory_cache.stats.misses,
        }
        
        total_requests = combined_stats["total_hits"] + combined_stats["total_misses"]
        if total_requests > 0:
            combined_stats["overall_hit_rate"] = combined_stats["total_hits"] / total_requests * 100
        
        return combined_stats
    
    def _get_disk_cache_count(self) -> int:
        """獲取磁盤緩存條目數"""
        try:
            with sqlite3.connect(self.disk_cache.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM cache_entries')
                return cursor.fetchone()[0]
        except:
            return 0


class CacheKeyGenerator:
    """緩存鍵生成器"""
    
    @staticmethod
    def generate_key(prefix: str, **kwargs) -> str:
        """生成緩存鍵"""
        # 將參數排序以確保一致性，過濾掉不可序列化的對象
        serializable_params = []
        for key, value in sorted(kwargs.items()):
            try:
                # 嘗試序列化，如果失敗則使用對象的字符串表示
                json.dumps(value)
                serializable_params.append((key, value))
            except (TypeError, ValueError):
                # 對於不可序列化的對象，使用其類型和ID
                serializable_params.append((key, f"{type(value).__name__}_{id(value)}"))
        
        param_str = json.dumps(serializable_params, sort_keys=True, ensure_ascii=False)
        
        # 生成哈希
        hash_value = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        
        return f"{prefix}:{hash_value}"
    
    @staticmethod
    def generate_vector_key(text: str, model: str) -> str:
        """為向量生成緩存鍵"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        return f"vector:{model}:{text_hash}"
    
    @staticmethod
    def generate_search_key(query: str, filters: Dict[str, Any], limit: int) -> str:
        """為搜索結果生成緩存鍵"""
        return CacheKeyGenerator.generate_key(
            "search",
            query=query,
            filters=filters,
            limit=limit
        )


# 全局緩存實例
_global_cache = None


def get_cache_manager() -> MultiLevelCache:
    """獲取全局緩存管理器"""
    global _global_cache
    if _global_cache is None:
        _global_cache = MultiLevelCache()
    return _global_cache


# 緩存裝飾器
def cached(ttl_seconds: Optional[int] = None, cache_level: str = "both", 
          key_generator: Optional[Callable] = None):
    """緩存裝飾器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # 生成緩存鍵
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = CacheKeyGenerator.generate_key(
                    func.__name__, 
                    args=args, 
                    kwargs=kwargs
                )
            
            # 嘗試從緩存獲取
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 執行函數
            result = await func(*args, **kwargs)
            
            # 緩存結果
            await cache.set(cache_key, result, ttl_seconds, cache_level=cache_level)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # 生成緩存鍵
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = CacheKeyGenerator.generate_key(
                    func.__name__,
                    args=args,
                    kwargs=kwargs
                )
            
            # 嘗試從內存緩存獲取（同步版本只用內存緩存）
            cached_result = cache.memory_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 執行函數
            result = func(*args, **kwargs)
            
            # 緩存到內存
            cache.memory_cache.set(cache_key, result, ttl_seconds)
            
            return result
        
        # 根據函數類型返回相應的包裝器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
