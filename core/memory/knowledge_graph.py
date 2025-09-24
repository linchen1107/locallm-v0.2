"""
知識圖譜系統
管理實體、關係和概念的結構化知識
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

from .base import (
    KnowledgeEntity, KnowledgeRelation, MemoryType, MemoryStats
)
from config.settings import get_settings
from core.models.ollama_client import get_ollama_client


class KnowledgeGraph:
    """知識圖譜實現"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # 存儲目錄
        self.kg_dir = Path(self.settings.storage.data_dir) / "knowledge_graph"
        self.kg_dir.mkdir(parents=True, exist_ok=True)
        
        # 數據文件
        self.entities_file = self.kg_dir / "entities.json"
        self.relations_file = self.kg_dir / "relations.json"
        
        # 數據結構
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relations: Dict[str, KnowledgeRelation] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # 實體類型索引
        self.relation_index: Dict[str, Set[str]] = defaultdict(set)  # 關係類型索引
        
        # 加載數據
        self.load_from_disk()
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        import uuid
        return str(uuid.uuid4())
    
    async def add_entity(self, name: str, entity_type: str, 
                        properties: Dict[str, Any] = None) -> str:
        """添加實體"""
        
        # 檢查是否已存在
        existing_entity = self.find_entity_by_name(name, entity_type)
        if existing_entity:
            # 更新現有實體
            if properties:
                existing_entity.properties.update(properties)
                existing_entity.last_updated = datetime.now()
            return existing_entity.id
        
        # 創建新實體
        entity = KnowledgeEntity(
            id=self._generate_id(),
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.entities[entity.id] = entity
        self.entity_index[entity_type].add(entity.id)
        
        await self.save_to_disk()
        return entity.id
    
    async def add_relation(self, source_entity_id: str, target_entity_id: str,
                          relation_type: str, strength: float = 1.0,
                          properties: Dict[str, Any] = None) -> str:
        """添加關係"""
        
        # 驗證實體是否存在
        if source_entity_id not in self.entities or target_entity_id not in self.entities:
            raise ValueError("Source or target entity does not exist")
        
        # 檢查是否已存在相同關係
        existing_relation = self.find_relation(source_entity_id, target_entity_id, relation_type)
        if existing_relation:
            # 更新關係強度
            existing_relation.strength = max(existing_relation.strength, strength)
            if properties:
                existing_relation.properties.update(properties)
            return existing_relation.id
        
        # 創建新關係
        relation = KnowledgeRelation(
            id=self._generate_id(),
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            strength=min(max(strength, 0.0), 1.0),  # 限制在0-1之間
            properties=properties or {},
            created_at=datetime.now()
        )
        
        self.relations[relation.id] = relation
        self.relation_index[relation_type].add(relation.id)
        
        await self.save_to_disk()
        return relation.id
    
    def find_entity_by_name(self, name: str, entity_type: str = None) -> Optional[KnowledgeEntity]:
        """根據名稱查找實體"""
        for entity in self.entities.values():
            if entity.name.lower() == name.lower():
                if entity_type is None or entity.entity_type == entity_type:
                    return entity
        return None
    
    def find_relation(self, source_id: str, target_id: str, relation_type: str = None) -> Optional[KnowledgeRelation]:
        """查找關係"""
        for relation in self.relations.values():
            if (relation.source_entity_id == source_id and 
                relation.target_entity_id == target_id):
                if relation_type is None or relation.relation_type == relation_type:
                    return relation
        return None
    
    def get_entity_relations(self, entity_id: str, direction: str = "both") -> List[KnowledgeRelation]:
        """獲取實體的所有關係"""
        relations = []
        
        for relation in self.relations.values():
            if direction in ["both", "outgoing"] and relation.source_entity_id == entity_id:
                relations.append(relation)
            elif direction in ["both", "incoming"] and relation.target_entity_id == entity_id:
                relations.append(relation)
        
        return relations
    
    def get_related_entities(self, entity_id: str, relation_type: str = None, 
                           max_depth: int = 2) -> List[Tuple[KnowledgeEntity, int]]:
        """獲取相關實體（包括間接關係）"""
        visited = set()
        result = []
        queue = [(entity_id, 0)]
        
        while queue and max_depth > 0:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth >= max_depth:
                continue
            
            visited.add(current_id)
            
            # 獲取直接相關的實體
            for relation in self.get_entity_relations(current_id):
                if relation_type and relation.relation_type != relation_type:
                    continue
                
                related_id = None
                if relation.source_entity_id == current_id:
                    related_id = relation.target_entity_id
                elif relation.target_entity_id == current_id:
                    related_id = relation.source_entity_id
                
                if related_id and related_id in self.entities and related_id not in visited:
                    result.append((self.entities[related_id], depth + 1))
                    queue.append((related_id, depth + 1))
        
        return result
    
    async def extract_entities_from_text(self, text: str) -> List[Tuple[str, str]]:
        """從文本中提取實體"""
        
        # 使用LLM提取實體
        extraction_prompt = f"""從以下文本中提取重要的實體，包括人名、地名、組織、概念、技術等。

文本：
{text}

請以JSON格式返回實體列表，每個實體包含名稱和類型：
[
  {{"name": "實體名稱", "type": "實體類型"}},
  ...
]

實體類型可以是：PERSON（人物）、PLACE（地點）、ORGANIZATION（組織）、CONCEPT（概念）、TECHNOLOGY（技術）、EVENT（事件）等。

提取結果："""
        
        try:
            client = await get_ollama_client()
            result = await client.generate_text(extraction_prompt, temperature=0.1)
            
            # 嘗試解析JSON
            import json
            entities_data = json.loads(result.strip())
            return [(entity["name"], entity["type"]) for entity in entities_data]
            
        except Exception as e:
            print(f"Failed to extract entities: {e}")
            return self._fallback_entity_extraction(text)
    
    def _fallback_entity_extraction(self, text: str) -> List[Tuple[str, str]]:
        """後備實體提取方法"""
        entities = []
        
        # 簡單的正則表達式提取
        # 提取可能的人名（大寫字母開頭的詞組）
        person_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        persons = re.findall(person_pattern, text)
        for person in persons:
            if len(person.split()) <= 3:  # 限制長度
                entities.append((person, "PERSON"))
        
        # 提取可能的組織（包含特定關鍵詞）
        org_keywords = ["公司", "Corporation", "Inc", "Ltd", "組織", "機構"]
        for keyword in org_keywords:
            pattern = rf'\b\w*{keyword}\w*\b'
            orgs = re.findall(pattern, text, re.IGNORECASE)
            for org in orgs:
                entities.append((org, "ORGANIZATION"))
        
        return list(set(entities))  # 去重
    
    async def extract_relations_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """從文本中提取關係"""
        
        extraction_prompt = f"""從以下文本中提取實體之間的關係。

文本：
{text}

請以JSON格式返回關係列表：
[
  {{"source": "源實體", "target": "目標實體", "relation": "關係類型"}},
  ...
]

關係類型可以是：WORKS_FOR（工作於）、LOCATED_IN（位於）、PART_OF（屬於）、RELATED_TO（相關於）、CAUSES（導致）、USES（使用）等。

提取結果："""
        
        try:
            client = await get_ollama_client()
            result = await client.generate_text(extraction_prompt, temperature=0.1)
            
            import json
            relations_data = json.loads(result.strip())
            return [(rel["source"], rel["target"], rel["relation"]) for rel in relations_data]
            
        except Exception as e:
            print(f"Failed to extract relations: {e}")
            return []
    
    async def build_knowledge_from_text(self, text: str) -> Dict[str, Any]:
        """從文本構建知識圖譜"""
        
        # 提取實體
        entities = await self.extract_entities_from_text(text)
        entity_ids = {}
        
        # 添加實體
        for name, entity_type in entities:
            entity_id = await self.add_entity(name, entity_type, {"source": "text_extraction"})
            entity_ids[name] = entity_id
        
        # 提取關係
        relations = await self.extract_relations_from_text(text)
        relation_ids = []
        
        # 添加關係
        for source_name, target_name, relation_type in relations:
            if source_name in entity_ids and target_name in entity_ids:
                relation_id = await self.add_relation(
                    entity_ids[source_name],
                    entity_ids[target_name],
                    relation_type,
                    strength=0.7,
                    properties={"source": "text_extraction"}
                )
                relation_ids.append(relation_id)
        
        return {
            "entities_added": len(entities),
            "relations_added": len(relation_ids),
            "entity_ids": list(entity_ids.values()),
            "relation_ids": relation_ids
        }
    
    def search_entities(self, query: str, entity_type: str = None, limit: int = 10) -> List[KnowledgeEntity]:
        """搜索實體"""
        query_lower = query.lower()
        results = []
        
        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            
            # 名稱匹配
            if query_lower in entity.name.lower():
                relevance = 1.0 if query_lower == entity.name.lower() else 0.5
                results.append((entity, relevance))
                continue
            
            # 屬性匹配
            for key, value in entity.properties.items():
                if query_lower in str(value).lower():
                    results.append((entity, 0.3))
                    break
        
        # 按相關性排序
        results.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, score in results[:limit]]
    
    def get_entity_by_type(self, entity_type: str) -> List[KnowledgeEntity]:
        """根據類型獲取實體"""
        entity_ids = self.entity_index.get(entity_type, set())
        return [self.entities[entity_id] for entity_id in entity_ids if entity_id in self.entities]
    
    def get_relations_by_type(self, relation_type: str) -> List[KnowledgeRelation]:
        """根據類型獲取關係"""
        relation_ids = self.relation_index.get(relation_type, set())
        return [self.relations[relation_id] for relation_id in relation_ids if relation_id in self.relations]
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """獲取圖譜統計信息"""
        entity_types = defaultdict(int)
        relation_types = defaultdict(int)
        
        for entity in self.entities.values():
            entity_types[entity.entity_type] += 1
        
        for relation in self.relations.values():
            relation_types[relation.relation_type] += 1
        
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "avg_relations_per_entity": len(self.relations) / len(self.entities) if self.entities else 0
        }
    
    def get_path_between_entities(self, source_id: str, target_id: str, max_depth: int = 3) -> List[List[str]]:
        """查找實體間的路徑"""
        if source_id not in self.entities or target_id not in self.entities:
            return []
        
        paths = []
        visited = set()
        
        def dfs(current_id: str, path: List[str], depth: int):
            if depth > max_depth or current_id in visited:
                return
            
            if current_id == target_id:
                paths.append(path + [current_id])
                return
            
            visited.add(current_id)
            
            # 探索所有相鄰實體
            for relation in self.get_entity_relations(current_id):
                next_id = None
                if relation.source_entity_id == current_id:
                    next_id = relation.target_entity_id
                elif relation.target_entity_id == current_id:
                    next_id = relation.source_entity_id
                
                if next_id and next_id not in visited:
                    dfs(next_id, path + [current_id], depth + 1)
            
            visited.remove(current_id)
        
        dfs(source_id, [], 0)
        return paths
    
    def delete_entity(self, entity_id: str) -> bool:
        """刪除實體"""
        if entity_id not in self.entities:
            return False
        
        entity = self.entities[entity_id]
        
        # 刪除相關的關係
        relations_to_delete = []
        for relation_id, relation in self.relations.items():
            if relation.source_entity_id == entity_id or relation.target_entity_id == entity_id:
                relations_to_delete.append(relation_id)
        
        for relation_id in relations_to_delete:
            self.delete_relation(relation_id)
        
        # 從索引中移除
        self.entity_index[entity.entity_type].discard(entity_id)
        
        # 刪除實體
        del self.entities[entity_id]
        
        return True
    
    def delete_relation(self, relation_id: str) -> bool:
        """刪除關係"""
        if relation_id not in self.relations:
            return False
        
        relation = self.relations[relation_id]
        
        # 從索引中移除
        self.relation_index[relation.relation_type].discard(relation_id)
        
        # 刪除關係
        del self.relations[relation_id]
        
        return True
    
    def clear(self) -> bool:
        """清空知識圖譜"""
        self.entities.clear()
        self.relations.clear()
        self.entity_index.clear()
        self.relation_index.clear()
        return True
    
    def save_to_disk(self) -> bool:
        """保存到磁盤"""
        try:
            # 保存實體
            entities_data = {
                "entities": [entity.to_dict() for entity in self.entities.values()],
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, ensure_ascii=False, indent=2)
            
            # 保存關係
            relations_data = {
                "relations": [relation.to_dict() for relation in self.relations.values()],
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.relations_file, 'w', encoding='utf-8') as f:
                json.dump(relations_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to save knowledge graph: {e}")
            return False
    
    def load_from_disk(self) -> bool:
        """從磁盤加載"""
        try:
            # 加載實體
            if self.entities_file.exists():
                with open(self.entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)
                
                self.entities.clear()
                self.entity_index.clear()
                
                for entity_data in entities_data.get("entities", []):
                    entity = KnowledgeEntity.from_dict(entity_data)
                    self.entities[entity.id] = entity
                    self.entity_index[entity.entity_type].add(entity.id)
            
            # 加載關係
            if self.relations_file.exists():
                with open(self.relations_file, 'r', encoding='utf-8') as f:
                    relations_data = json.load(f)
                
                self.relations.clear()
                self.relation_index.clear()
                
                for relation_data in relations_data.get("relations", []):
                    relation = KnowledgeRelation.from_dict(relation_data)
                    self.relations[relation.id] = relation
                    self.relation_index[relation.relation_type].add(relation.id)
            
            return True
        except Exception as e:
            print(f"Failed to load knowledge graph: {e}")
            return False

