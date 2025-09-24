"""
長期記憶系統
存儲和檢索重要的知識、經驗和學習內容
"""

import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .base import (
    BaseMemory, MemoryItem, MemoryType, MemoryImportance,
    MemorySearchResult, MemoryStats
)
from config.settings import get_settings
from core.models.ollama_client import get_ollama_client


class LongTermMemory(BaseMemory):
    """長期記憶實現"""
    
    def __init__(self):
        super().__init__(MemoryType.LONG_TERM)
        self.settings = get_settings()
        
        # 存儲目錄
        self.memory_dir = Path(self.settings.storage.data_dir) / "long_term_memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # 數據文件
        self.memory_file = self.memory_dir / "long_term_memories.json"
        self.embeddings_file = self.memory_dir / "embeddings.json"
        
        # 內存中的嵌入向量映射
        self.embeddings: Dict[str, List[float]] = {}
        
        # 加載數據
        asyncio.create_task(self.load_from_disk())
    
    async def store(self, content: str, importance: MemoryImportance = MemoryImportance.MEDIUM,
                   metadata: Dict[str, Any] = None, tags: List[str] = None) -> str:
        """存儲長期記憶"""
        
        # 檢查是否是重要記憶（只存儲中等以上重要性的記憶）
        if importance.value < MemoryImportance.MEDIUM.value:
            return ""
        
        # 檢查重複內容
        existing_memory = await self._find_similar_memory(content)
        if existing_memory:
            # 更新現有記憶而不是創建新的
            await self.update(existing_memory.id, content, importance, metadata)
            return existing_memory.id
        
        # 創建新記憶
        memory_item = self._create_memory_item(content, importance, metadata, tags)
        self.items[memory_item.id] = memory_item
        
        # 生成嵌入向量
        try:
            client = await get_ollama_client()
            embedding = await client.generate_embeddings(content)
            self.embeddings[memory_item.id] = embedding
        except Exception as e:
            print(f"Failed to generate embedding for memory: {e}")
        
        # 保存到磁盤
        await self.save_to_disk()
        return memory_item.id
    
    async def store_from_conversation(self, user_input: str, assistant_response: str,
                                    importance: MemoryImportance = MemoryImportance.MEDIUM) -> str:
        """從對話中提取並存儲長期記憶"""
        
        # 使用LLM提取重要信息
        extraction_prompt = f"""從以下對話中提取值得長期記住的重要信息、知識點或事實：

用戶: {user_input}
助手: {assistant_response}

請提取：
1. 重要的事實信息
2. 學習到的新知識
3. 用戶的重要需求或偏好
4. 有價值的問題解決方案

如果沒有值得長期記住的信息，請回答"無"。
如果有，請簡潔地總結要記住的內容。

提取結果:"""
        
        try:
            client = await get_ollama_client()
            extracted_info = await client.generate_text(extraction_prompt, temperature=0.1)
            
            if extracted_info.strip().lower() not in ["無", "没有", "none", ""]:
                metadata = {
                    "source": "conversation",
                    "original_user_input": user_input,
                    "original_assistant_response": assistant_response,
                    "extracted_at": datetime.now().isoformat()
                }
                
                return await self.store(extracted_info, importance, metadata, ["conversation", "extracted"])
        except Exception as e:
            print(f"Failed to extract information from conversation: {e}")
        
        return ""
    
    async def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """檢索長期記憶"""
        
        # 生成查詢嵌入
        try:
            client = await get_ollama_client()
            query_embedding = await client.generate_embeddings(query)
        except Exception as e:
            print(f"Failed to generate query embedding: {e}")
            return await self._fallback_text_search(query, limit)
        
        # 計算相似度
        similarities = []
        for memory_id, memory_embedding in self.embeddings.items():
            if memory_id in self.items:
                similarity = self._cosine_similarity(query_embedding, memory_embedding)
                similarities.append((self.items[memory_id], similarity))
        
        # 排序並返回結果
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 更新訪問信息
        results = []
        for memory_item, similarity in similarities[:limit]:
            memory_item.update_access()
            results.append(memory_item)
        
        # 保存更新的訪問信息
        await self.save_to_disk()
        return results
    
    async def _fallback_text_search(self, query: str, limit: int) -> List[MemoryItem]:
        """文本搜索後備方案"""
        query_lower = query.lower()
        matching_items = []
        
        for item in self.items.values():
            relevance_score = self._calculate_text_relevance(item, query_lower)
            if relevance_score > 0:
                matching_items.append((item, relevance_score))
        
        matching_items.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for item, score in matching_items[:limit]:
            item.update_access()
            results.append(item)
        
        return results
    
    def _calculate_text_relevance(self, item: MemoryItem, query_lower: str) -> float:
        """計算文本相關性"""
        content_lower = item.content.lower()
        
        # 精確匹配
        if query_lower in content_lower:
            base_score = 1.0
        else:
            # 詞語匹配
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words.intersection(content_words))
            base_score = overlap / len(query_words) if query_words else 0
        
        # 重要性加權
        importance_weight = item.importance.value / 4.0
        
        # 訪問頻率加權
        access_weight = min(1.0, item.access_count / 5.0)
        
        return base_score * importance_weight * (1 + access_weight)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """計算餘弦相似度"""
        try:
            import numpy as np
            
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            # 簡單的點積計算作為後備
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    async def _find_similar_memory(self, content: str, threshold: float = 0.85) -> Optional[MemoryItem]:
        """查找相似的記憶"""
        try:
            client = await get_ollama_client()
            content_embedding = await client.generate_embeddings(content)
            
            for memory_id, memory_embedding in self.embeddings.items():
                if memory_id in self.items:
                    similarity = self._cosine_similarity(content_embedding, memory_embedding)
                    if similarity > threshold:
                        return self.items[memory_id]
        except Exception:
            pass
        
        return None
    
    async def update(self, memory_id: str, content: str = None,
                    importance: MemoryImportance = None, metadata: Dict[str, Any] = None) -> bool:
        """更新記憶"""
        if memory_id not in self.items:
            return False
        
        item = self.items[memory_id]
        
        # 更新內容
        if content is not None:
            item.content = content
            # 重新生成嵌入
            try:
                client = await get_ollama_client()
                embedding = await client.generate_embeddings(content)
                self.embeddings[memory_id] = embedding
            except Exception as e:
                print(f"Failed to update embedding: {e}")
        
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
            
        if memory_id in self.embeddings:
            del self.embeddings[memory_id]
        
        await self.save_to_disk()
        return True
    
    async def get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """根據ID獲取記憶"""
        return self.items.get(memory_id)
    
    async def get_all(self) -> List[MemoryItem]:
        """獲取所有記憶"""
        return list(self.items.values())
    
    async def get_by_importance(self, importance: MemoryImportance) -> List[MemoryItem]:
        """根據重要性獲取記憶"""
        return [item for item in self.items.values() if item.importance == importance]
    
    async def get_by_tags(self, tags: List[str]) -> List[MemoryItem]:
        """根據標籤獲取記憶"""
        results = []
        for item in self.items.values():
            if any(tag in item.tags for tag in tags):
                results.append(item)
        return results
    
    async def get_recent_memories(self, days: int = 7, limit: int = 20) -> List[MemoryItem]:
        """獲取最近的記憶"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_memories = [
            item for item in self.items.values() 
            if item.created_at > cutoff_date
        ]
        
        # 按創建時間排序
        recent_memories.sort(key=lambda x: x.created_at, reverse=True)
        return recent_memories[:limit]
    
    async def consolidate_memories(self) -> int:
        """整合和去重記憶"""
        consolidated_count = 0
        
        # 查找相似記憶並合併
        memory_ids = list(self.items.keys())
        processed = set()
        
        for i, memory_id in enumerate(memory_ids):
            if memory_id in processed or memory_id not in self.items:
                continue
            
            current_memory = self.items[memory_id]
            similar_memories = []
            
            # 查找相似記憶
            for j in range(i + 1, len(memory_ids)):
                other_id = memory_ids[j]
                if other_id in processed or other_id not in self.items:
                    continue
                
                if other_id in self.embeddings and memory_id in self.embeddings:
                    similarity = self._cosine_similarity(
                        self.embeddings[memory_id], 
                        self.embeddings[other_id]
                    )
                    
                    if similarity > 0.8:  # 高相似度閾值
                        similar_memories.append(self.items[other_id])
                        processed.add(other_id)
            
            # 合併相似記憶
            if similar_memories:
                consolidated_content = await self._merge_memories(current_memory, similar_memories)
                await self.update(memory_id, consolidated_content)
                
                # 刪除被合併的記憶
                for similar_memory in similar_memories:
                    await self.delete(similar_memory.id)
                
                consolidated_count += len(similar_memories)
            
            processed.add(memory_id)
        
        await self.save_to_disk()
        return consolidated_count
    
    async def _merge_memories(self, primary: MemoryItem, similar: List[MemoryItem]) -> str:
        """合併記憶內容"""
        contents = [primary.content] + [mem.content for mem in similar]
        
        merge_prompt = f"""請將以下相似的記憶內容合併成一個更完整、準確的記憶：

記憶內容：
{chr(10).join([f'{i+1}. {content}' for i, content in enumerate(contents)])}

要求：
1. 保留所有重要信息
2. 去除重複內容
3. 整合相關信息
4. 保持簡潔明確

合併後的記憶："""
        
        try:
            client = await get_ollama_client()
            merged_content = await client.generate_text(merge_prompt, temperature=0.1)
            return merged_content.strip()
        except Exception:
            # 如果LLM不可用，簡單連接內容
            return " | ".join(contents)
    
    async def clear(self) -> bool:
        """清空長期記憶"""
        self.items.clear()
        self.embeddings.clear()
        await self.save_to_disk()
        return True
    
    async def save_to_disk(self) -> bool:
        """保存到磁盤"""
        try:
            # 保存記憶項目
            memory_data = {
                "items": [item.to_dict() for item in self.items.values()],
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
            # 保存嵌入向量
            with open(self.embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(self.embeddings, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to save long-term memory: {e}")
            return False
    
    async def load_from_disk(self) -> bool:
        """從磁盤加載"""
        try:
            # 加載記憶項目
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                
                self.items.clear()
                for item_data in memory_data.get("items", []):
                    item = MemoryItem.from_dict(item_data)
                    self.items[item.id] = item
            
            # 加載嵌入向量
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    self.embeddings = json.load(f)
            
            return True
        except Exception as e:
            print(f"Failed to load long-term memory: {e}")
            return False
    
    async def get_memory_stats(self) -> MemoryStats:
        """獲取記憶統計信息"""
        by_importance = {}
        for importance in MemoryImportance:
            count = sum(1 for item in self.items.values() if item.importance == importance)
            by_importance[importance.name.lower()] = count
        
        # 計算最近活動（最近7天的記憶）
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_activity = sum(1 for item in self.items.values() if item.created_at > recent_cutoff)
        
        return MemoryStats(
            memory_type=MemoryType.LONG_TERM,
            total_items=len(self.items),
            by_importance=by_importance,
            recent_activity=recent_activity
        )

