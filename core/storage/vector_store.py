"""
向量存儲模塊
提供向量數據庫的統一接口
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np

# 向量數據庫
import chromadb
from chromadb.config import Settings

from config.settings import get_settings
from core.rag.document_processing import DocumentChunk
from core.models.ollama_client import get_ollama_client


class VectorStore(ABC):
    """向量存儲抽象基類"""
    
    @abstractmethod
    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """添加文檔塊"""
        pass
    
    @abstractmethod
    async def similarity_search(self, 
                              query: str, 
                              k: int = 5, 
                              filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """相似度搜索"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """刪除文檔"""
        pass
    
    @abstractmethod
    async def update_document(self, document_id: str, chunk: DocumentChunk) -> bool:
        """更新文檔"""
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """獲取文檔數量"""
        pass
    
    @abstractmethod
    async def clear_collection(self) -> bool:
        """清空集合"""
        pass


class ChromaVectorStore(VectorStore):
    """Chroma向量存儲實現"""
    
    def __init__(self, collection_name: str = None, persist_directory: str = None):
        self.settings = get_settings()
        self.collection_name = collection_name or self.settings.storage.vector_collection_name
        self.persist_directory = persist_directory or str(self.settings.storage.vector_db_path)
        
        # 確保目錄存在
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 初始化Chroma客戶端
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 獲取或創建集合
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings collection"}
            )
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """生成嵌入向量"""
        client = await get_ollama_client()
        return await client.generate_embeddings(text)
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """添加文檔塊"""
        if not chunks:
            return []
        
        # 準備數據
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        # 批量生成嵌入
        texts = [chunk.content for chunk in chunks]
        client = await get_ollama_client()
        
        try:
            batch_embeddings = await client.generate_embeddings(texts)
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {e}")
        
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            
            # 準備元數據
            metadata = {
                "file_path": str(chunk.metadata.file_path),
                "file_name": chunk.metadata.file_name,
                "file_type": chunk.metadata.file_type,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
                "created_at": chunk.metadata.created_at.isoformat(),
                "chunk_hash": chunk.hash
            }
            
            documents.append(chunk.content)
            metadatas.append(metadata)
            ids.append(chunk_id)
            embeddings.append(batch_embeddings[i])
        
        # 添加到集合
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            return ids
        except Exception as e:
            raise ValueError(f"Failed to add documents to vector store: {e}")
    
    async def similarity_search(self, 
                              query: str, 
                              k: int = 5, 
                              filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """相似度搜索"""
        try:
            # 生成查詢嵌入
            query_embedding = await self._generate_embedding(query)
            
            # 執行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            
            # 轉換結果
            search_results = []
            
            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    # 重建DocumentChunk
                    from core.rag.document_processing import DocumentMetadata
                    
                    # 重建元數據
                    doc_metadata = DocumentMetadata.__new__(DocumentMetadata)
                    doc_metadata.file_path = Path(metadata["file_path"])
                    doc_metadata.file_name = metadata["file_name"]
                    doc_metadata.file_type = metadata["file_type"]
                    doc_metadata.created_at = datetime.fromisoformat(metadata["created_at"])
                    
                    # 重建文檔塊
                    chunk = DocumentChunk(
                        content=doc,
                        metadata=doc_metadata,
                        chunk_index=metadata["chunk_index"],
                        chunk_type=metadata["chunk_type"]
                    )
                    chunk.word_count = metadata["word_count"]
                    chunk.char_count = metadata["char_count"]
                    chunk.hash = metadata["chunk_hash"]
                    
                    # 轉換距離為相似度（距離越小，相似度越高）
                    similarity = 1.0 - distance
                    
                    search_results.append((chunk, similarity))
            
            return search_results
            
        except Exception as e:
            raise ValueError(f"Failed to perform similarity search: {e}")
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """刪除文檔"""
        try:
            self.collection.delete(ids=document_ids)
            return True
        except Exception as e:
            print(f"Failed to delete documents: {e}")
            return False
    
    async def delete_by_metadata(self, filter_dict: Dict[str, Any]) -> bool:
        """根據元數據刪除文檔"""
        try:
            self.collection.delete(where=filter_dict)
            return True
        except Exception as e:
            print(f"Failed to delete documents by metadata: {e}")
            return False
    
    async def update_document(self, document_id: str, chunk: DocumentChunk) -> bool:
        """更新文檔"""
        try:
            # 生成新的嵌入
            embedding = await self._generate_embedding(chunk.content)
            
            # 準備元數據
            metadata = {
                "file_path": str(chunk.metadata.file_path),
                "file_name": chunk.metadata.file_name,
                "file_type": chunk.metadata.file_type,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
                "created_at": chunk.metadata.created_at.isoformat(),
                "chunk_hash": chunk.hash
            }
            
            # 更新文檔
            self.collection.update(
                ids=[document_id],
                documents=[chunk.content],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            return True
        except Exception as e:
            print(f"Failed to update document: {e}")
            return False
    
    async def get_document_count(self) -> int:
        """獲取文檔數量"""
        try:
            return self.collection.count()
        except:
            return 0
    
    async def clear_collection(self) -> bool:
        """清空集合"""
        try:
            # 獲取所有文檔ID
            results = self.collection.get(include=[])
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
            return True
        except Exception as e:
            print(f"Failed to clear collection: {e}")
            return False
    
    async def get_documents_by_file(self, file_path: str) -> List[Tuple[str, DocumentChunk]]:
        """獲取特定文件的所有文檔塊"""
        try:
            results = self.collection.get(
                where={"file_path": file_path},
                include=["documents", "metadatas"]
            )
            
            file_chunks = []
            if results["documents"]:
                for doc_id, doc, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
                    # 重建DocumentChunk
                    from core.rag.document_processing import DocumentMetadata
                    
                    doc_metadata = DocumentMetadata.__new__(DocumentMetadata)
                    doc_metadata.file_path = Path(metadata["file_path"])
                    doc_metadata.file_name = metadata["file_name"]
                    doc_metadata.file_type = metadata["file_type"]
                    doc_metadata.created_at = datetime.fromisoformat(metadata["created_at"])
                    
                    chunk = DocumentChunk(
                        content=doc,
                        metadata=doc_metadata,
                        chunk_index=metadata["chunk_index"],
                        chunk_type=metadata["chunk_type"]
                    )
                    
                    file_chunks.append((doc_id, chunk))
            
            return file_chunks
        except Exception as e:
            raise ValueError(f"Failed to get documents by file: {e}")


class VectorStoreManager:
    """向量存儲管理器"""
    
    def __init__(self, store_type: str = "chroma"):
        self.settings = get_settings()
        self.store_type = store_type.lower()
        self._store = None
    
    def _create_store(self) -> VectorStore:
        """創建向量存儲實例"""
        if self.store_type == "chroma":
            return ChromaVectorStore(
                collection_name=self.settings.storage.vector_collection_name,
                persist_directory=str(self.settings.storage.vector_db_path)
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    def get_store(self) -> VectorStore:
        """獲取向量存儲實例"""
        if self._store is None:
            self._store = self._create_store()
        return self._store
    
    async def initialize_store(self) -> bool:
        """初始化向量存儲"""
        try:
            store = self.get_store()
            count = await store.get_document_count()
            print(f"Vector store initialized with {count} documents")
            return True
        except Exception as e:
            print(f"Failed to initialize vector store: {e}")
            return False
    
    async def add_documents_from_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """從文檔塊添加文檔"""
        store = self.get_store()
        return await store.add_documents(chunks)
    
    async def search_similar_documents(self, 
                                     query: str, 
                                     k: int = 5,
                                     file_type_filter: Optional[str] = None,
                                     similarity_threshold: Optional[float] = None) -> List[Tuple[DocumentChunk, float]]:
        """搜索相似文檔"""
        store = self.get_store()
        
        # 構建過濾條件
        filter_dict = {}
        if file_type_filter:
            filter_dict["file_type"] = file_type_filter
        
        results = await store.similarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict if filter_dict else None
        )
        
        # 應用相似度閾值過濾
        if similarity_threshold is not None:
            results = [(chunk, score) for chunk, score in results if score >= similarity_threshold]
        
        return results
    
    async def remove_file_documents(self, file_path: str) -> bool:
        """移除特定文件的所有文檔"""
        store = self.get_store()
        
        if isinstance(store, ChromaVectorStore):
            return await store.delete_by_metadata({"file_path": file_path})
        else:
            # 對於其他存儲類型，先獲取文檔ID再刪除
            file_chunks = await store.get_documents_by_file(file_path)
            if file_chunks:
                doc_ids = [doc_id for doc_id, _ in file_chunks]
                return await store.delete_documents(doc_ids)
            return True
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """獲取集合統計信息"""
        store = self.get_store()
        
        total_docs = await store.get_document_count()
        
        # 獲取文件類型統計（僅對Chroma）
        file_type_stats = {}
        if isinstance(store, ChromaVectorStore):
            try:
                # 獲取所有文檔的元數據
                results = store.collection.get(include=["metadatas"])
                
                if results["metadatas"]:
                    for metadata in results["metadatas"]:
                        file_type = metadata.get("file_type", "unknown")
                        file_type_stats[file_type] = file_type_stats.get(file_type, 0) + 1
            except:
                pass
        
        return {
            "total_documents": total_docs,
            "file_type_distribution": file_type_stats,
            "store_type": self.store_type,
            "collection_name": self.settings.storage.vector_collection_name
        }


# 全局向量存儲管理器實例
_vector_store_manager = None


def get_vector_store_manager() -> VectorStoreManager:
    """獲取向量存儲管理器實例"""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager(
            store_type=get_settings().storage.vector_db_type
        )
    return _vector_store_manager

