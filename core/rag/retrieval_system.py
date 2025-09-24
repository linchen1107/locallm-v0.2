"""
RAG檢索系統
實現混合檢索和答案生成
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

from config.settings import get_settings
from core.models.ollama_client import get_ollama_client
from core.storage.vector_store import get_vector_store_manager
from core.rag.document_processing import DocumentChunk
from core.memory.context_manager import ContextManager


@dataclass
class RetrievalResult:
    """檢索結果"""
    answer: str
    sources: List[DocumentChunk]
    confidence: float
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: Dict[str, Any]


@dataclass
class SearchQuery:
    """搜索查詢"""
    text: str
    filters: Optional[Dict[str, Any]] = None
    max_results: int = 5
    similarity_threshold: float = 0.7
    include_metadata: bool = True


class KeywordRetriever:
    """關鍵詞檢索器"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def extract_keywords(self, query: str) -> List[str]:
        """提取關鍵詞"""
        # 移除標點符號並分詞
        words = re.findall(r'\b\w+\b', query.lower())
        
        # 過濾停用詞（簡化版）
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
            '的', '了', '是', '在', '有', '和', '與', '或', '但', '如果', '因為'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def score_chunk_relevance(self, chunk: DocumentChunk, keywords: List[str]) -> float:
        """計算文檔塊與關鍵詞的相關性"""
        content_lower = chunk.content.lower()
        
        # 計算關鍵詞匹配分數
        keyword_score = 0
        for keyword in keywords:
            if keyword in content_lower:
                # 精確匹配得分更高
                exact_matches = content_lower.count(keyword)
                keyword_score += exact_matches * 2
                
                # 部分匹配
                for word in content_lower.split():
                    if keyword in word:
                        keyword_score += 0.5
        
        # 正規化分數
        max_possible_score = len(keywords) * 2
        return min(keyword_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0


class HybridRetriever:
    """混合檢索器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.vector_store_manager = get_vector_store_manager()
        self.keyword_retriever = KeywordRetriever()
    
    async def retrieve(self, query: SearchQuery) -> List[Tuple[DocumentChunk, float]]:
        """執行混合檢索"""
        start_time = datetime.now()
        
        # 向量檢索
        vector_results = await self._vector_retrieval(query)
        
        # 關鍵詞檢索
        keyword_results = await self._keyword_retrieval(query, vector_results)
        
        # 混合重新排序
        final_results = await self._hybrid_rerank(query, vector_results, keyword_results)
        
        retrieval_time = (datetime.now() - start_time).total_seconds()
        
        # 添加檢索時間到元數據
        for chunk, score in final_results:
            if not hasattr(chunk, 'retrieval_metadata'):
                chunk.retrieval_metadata = {}
            chunk.retrieval_metadata['retrieval_time'] = retrieval_time
        
        return final_results[:query.max_results]
    
    async def _vector_retrieval(self, query: SearchQuery) -> List[Tuple[DocumentChunk, float]]:
        """向量檢索"""
        try:
            results = await self.vector_store_manager.search_similar_documents(
                query=query.text,
                k=query.max_results * 2,  # 獲取更多結果用於重新排序
                similarity_threshold=query.similarity_threshold
            )
            return results
        except Exception as e:
            print(f"Vector retrieval failed: {e}")
            return []
    
    async def _keyword_retrieval(self, query: SearchQuery, 
                               vector_results: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:
        """關鍵詞檢索"""
        keywords = self.keyword_retriever.extract_keywords(query.text)
        if not keywords:
            return vector_results
        
        # 重新計算基於關鍵詞的相關性分數
        keyword_scored_results = []
        for chunk, vector_score in vector_results:
            keyword_score = self.keyword_retriever.score_chunk_relevance(chunk, keywords)
            keyword_scored_results.append((chunk, keyword_score))
        
        return keyword_scored_results
    
    async def _hybrid_rerank(self, query: SearchQuery,
                           vector_results: List[Tuple[DocumentChunk, float]],
                           keyword_results: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:
        """混合重新排序"""
        # 創建分數字典
        vector_scores = {chunk.hash: score for chunk, score in vector_results}
        keyword_scores = {chunk.hash: score for chunk, score in keyword_results}
        
        # 計算混合分數
        hybrid_results = []
        processed_hashes = set()
        
        for chunk, vector_score in vector_results:
            if chunk.hash in processed_hashes:
                continue
            
            keyword_score = keyword_scores.get(chunk.hash, 0.0)
            
            # 混合分數計算（可以調整權重）
            hybrid_score = 0.7 * vector_score + 0.3 * keyword_score
            
            hybrid_results.append((chunk, hybrid_score))
            processed_hashes.add(chunk.hash)
        
        # 按分數排序
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_results


class ContextBuilder:
    """上下文構建器"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def build_context(self, query: str, chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """構建檢索上下文"""
        if not chunks:
            return ""
        
        context_parts = []
        total_length = 0
        max_length = self.settings.rag.max_context_length
        
        for i, (chunk, score) in enumerate(chunks):
            # 格式化來源信息
            source_info = f"[Source {i+1}: {chunk.metadata.file_name}]"
            chunk_content = f"{source_info}\n{chunk.content}\n"
            
            # 檢查長度限制
            if total_length + len(chunk_content) > max_length:
                break
            
            context_parts.append(chunk_content)
            total_length += len(chunk_content)
        
        return "\n".join(context_parts)
    
    def format_sources(self, chunks: List[Tuple[DocumentChunk, float]]) -> List[Dict[str, Any]]:
        """格式化來源信息"""
        sources = []
        for i, (chunk, score) in enumerate(chunks):
            source = {
                "index": i + 1,
                "file_name": chunk.metadata.file_name,
                "file_path": str(chunk.metadata.file_path),
                "file_type": chunk.metadata.file_type,
                "chunk_index": chunk.chunk_index,
                "relevance_score": round(score, 3),
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            }
            sources.append(source)
        return sources


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.context_builder = ContextBuilder()
    
    async def generate_answer(self, query: str, 
                            retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> RetrievalResult:
        """生成答案"""
        start_time = datetime.now()
        
        # 構建上下文
        context = self.context_builder.build_context(query, retrieved_chunks)
        
        # 構建提示
        prompt = self._build_prompt(query, context)
        
        # 生成答案
        generation_start = datetime.now()
        try:
            client = await get_ollama_client()
            answer = await client.generate_text(
                prompt=prompt,
                temperature=0.1,  # 較低溫度確保答案更準確
                max_tokens=1000
            )
        except Exception as e:
            answer = f"抱歉，生成答案時發生錯誤：{str(e)}"
        
        generation_time = (datetime.now() - generation_start).total_seconds()
        total_time = (datetime.now() - start_time).total_seconds()
        
        # 計算置信度
        confidence = self._calculate_confidence(retrieved_chunks, answer)
        
        # 格式化來源
        sources = [chunk for chunk, _ in retrieved_chunks]
        
        # 構建元數據
        metadata = {
            "query": query,
            "num_sources": len(sources),
            "context_length": len(context),
            "sources_info": self.context_builder.format_sources(retrieved_chunks)
        }
        
        return RetrievalResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            retrieval_time=0.0,  # 將在RAGSystem中設置
            generation_time=generation_time,
            total_time=total_time,
            metadata=metadata
        )
    
    def _build_prompt(self, query: str, context: str) -> str:
        """構建生成提示"""
        if not context.strip():
            return f"""請回答以下問題：

問題：{query}

由於沒有找到相關的文檔內容，請基於你的知識回答這個問題，並說明這是基於一般知識的回答。
"""
        
        prompt = f"""基於以下文檔內容回答問題。請確保答案準確、相關且有用。

文檔內容：
{context}

問題：{query}

請基於上述文檔內容回答問題。如果文檔中沒有足夠信息，請說明並提供可能的建議。

回答："""
        
        return prompt
    
    def _calculate_confidence(self, chunks: List[Tuple[DocumentChunk, float]], answer: str) -> float:
        """計算答案置信度"""
        if not chunks:
            return 0.1
        
        # 基於檢索分數計算基礎置信度
        avg_score = sum(score for _, score in chunks) / len(chunks)
        
        # 基於答案長度調整（太短或太長都降低置信度）
        answer_length_score = 1.0
        if len(answer) < 50:
            answer_length_score = 0.5
        elif len(answer) > 2000:
            answer_length_score = 0.8
        
        # 檢查答案中是否包含不確定性詞語
        uncertainty_words = ['不確定', '可能', '也許', '大概', '似乎', '抱歉', '沒有足夠信息']
        uncertainty_penalty = 1.0
        for word in uncertainty_words:
            if word in answer:
                uncertainty_penalty -= 0.1
        
        uncertainty_penalty = max(uncertainty_penalty, 0.3)
        
        # 計算最終置信度
        confidence = avg_score * answer_length_score * uncertainty_penalty
        return min(max(confidence, 0.0), 1.0)


class RAGSystem:
    """RAG系統主類"""
    
    def __init__(self, user_id: str = "default"):
        self.settings = get_settings()
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()
        self.vector_store_manager = get_vector_store_manager()
        
        # 記憶系統集成
        self.context_manager = ContextManager(user_id)
        self.user_id = user_id
    
    async def initialize(self) -> bool:
        """初始化RAG系統"""
        try:
            # 初始化向量存儲
            await self.vector_store_manager.initialize_store()
            return True
        except Exception as e:
            print(f"Failed to initialize RAG system: {e}")
            return False
    
    async def query(self, question: str, 
                   filters: Optional[Dict[str, Any]] = None,
                   max_results: int = None,
                   similarity_threshold: float = None,
                   include_memory_context: bool = True) -> RetrievalResult:
        """執行RAG查詢"""
        start_time = datetime.now()
        
        # 獲取記憶上下文（如果啟用）
        enhanced_question = question
        memory_context = ""
        if include_memory_context:
            memory_context = await self.context_manager.get_smart_context(question)
            if memory_context:
                enhanced_question = f"{question}\n\n相關上下文：\n{memory_context}"
        
        # 構建搜索查詢
        search_query = SearchQuery(
            text=enhanced_question,
            filters=filters,
            max_results=max_results or self.settings.rag.top_k,
            similarity_threshold=similarity_threshold or self.settings.rag.similarity_threshold
        )
        
        # 檢索相關文檔
        retrieval_start = datetime.now()
        retrieved_chunks = await self.retriever.retrieve(search_query)
        retrieval_time = (datetime.now() - retrieval_start).total_seconds()
        
        # 生成答案
        result = await self.generator.generate_answer(question, retrieved_chunks)
        
        # 設置檢索時間
        result.retrieval_time = retrieval_time
        result.total_time = (datetime.now() - start_time).total_seconds()
        
        # 將查詢和回答保存到記憶系統
        if include_memory_context:
            await self._save_query_to_memory(question, result.answer, memory_context, result.confidence)
        
        return result
    
    async def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """添加文檔到知識庫"""
        from core.rag.document_processing import DocumentProcessingPipeline
        
        try:
            # 處理文檔
            processor = DocumentProcessingPipeline()
            chunks = await processor.process_batch(file_paths)
            
            if not chunks:
                return {"success": False, "message": "No valid documents processed", "added_count": 0}
            
            # 添加到向量存儲
            doc_ids = await self.vector_store_manager.add_documents_from_chunks(chunks)
            
            return {
                "success": True,
                "message": f"Successfully added {len(chunks)} document chunks",
                "added_count": len(chunks),
                "processed_files": len(file_paths),
                "document_ids": doc_ids
            }
            
        except Exception as e:
            return {"success": False, "message": f"Failed to add documents: {str(e)}", "added_count": 0}
    
    async def remove_documents(self, file_path: str) -> Dict[str, Any]:
        """從知識庫移除文檔"""
        try:
            success = await self.vector_store_manager.remove_file_documents(file_path)
            
            if success:
                return {"success": True, "message": f"Successfully removed documents from {file_path}"}
            else:
                return {"success": False, "message": f"Failed to remove documents from {file_path}"}
                
        except Exception as e:
            return {"success": False, "message": f"Error removing documents: {str(e)}"}
    
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """獲取知識庫統計信息"""
        try:
            stats = await self.vector_store_manager.get_collection_stats()
            return {"success": True, "stats": stats}
        except Exception as e:
            return {"success": False, "message": f"Failed to get stats: {str(e)}"}
    
    async def search_documents(self, query: str, 
                             k: int = 10,
                             file_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索文檔（不生成答案）"""
        try:
            results = await self.vector_store_manager.search_similar_documents(
                query=query,
                k=k,
                file_type_filter=file_type_filter,
                similarity_threshold=self.settings.rag.similarity_threshold
            )
            
            formatted_results = []
            for chunk, score in results:
                formatted_results.append({
                    "content": chunk.content,
                    "file_name": chunk.metadata.file_name,
                    "file_path": str(chunk.metadata.file_path),
                    "file_type": chunk.metadata.file_type,
                    "chunk_index": chunk.chunk_index,
                    "similarity_score": round(score, 3),
                    "word_count": chunk.word_count
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []
    
    async def _save_query_to_memory(self, question: str, answer: str, 
                                   memory_context: str, confidence: float):
        """將查詢結果保存到記憶系統"""
        try:
            # 處理查詢交互
            await self.context_manager.process_interaction(question, answer, {
                "confidence": confidence,
                "memory_context_used": bool(memory_context),
                "query_type": "rag_query"
            })
            
            # 如果答案質量高，存儲到長期記憶
            if confidence > 0.8:
                from core.memory.base import MemoryImportance
                
                fact_content = f"問題：{question}\n答案：{answer}"
                await self.context_manager.long_term_memory.store(
                    fact_content,
                    MemoryImportance.MEDIUM,
                    {
                        "source": "rag_query",
                        "confidence": confidence,
                        "query_timestamp": datetime.now().isoformat()
                    },
                    ["query", "knowledge", "rag"]
                )
            
        except Exception as e:
            print(f"保存查詢到記憶系統失敗: {e}")
    
    async def learn_from_user_feedback(self, question: str, answer: str, 
                                      feedback_positive: bool, feedback_details: str = None):
        """從用戶反饋中學習"""
        try:
            feedback_data = {
                "question": question,
                "answer": answer,
                "feedback_details": feedback_details,
                "rag_system": True
            }
            
            await self.context_manager.learn_from_feedback(
                "rag_answer_quality",
                feedback_data,
                feedback_positive
            )
            
            # 如果是負面反饋，記錄以改進
            if not feedback_positive:
                await self.context_manager.personal_memory.record_interaction(
                    "rag_negative_feedback",
                    {
                        "question": question,
                        "answer_snippet": answer[:100],
                        "feedback_details": feedback_details
                    }
                )
                
        except Exception as e:
            print(f"學習用戶反饋失敗: {e}")
    
    async def get_memory_enhanced_suggestions(self, query: str) -> List[str]:
        """獲取記憶增強的建議"""
        try:
            # 獲取相關記憶
            relevant_context = await self.context_manager.get_relevant_context(query, max_items=5)
            
            suggestions = []
            
            # 基於會話記憶的建議
            if relevant_context["session"]:
                suggestions.append("基於最近的對話，您可能也對以下內容感興趣...")
            
            # 基於長期記憶的建議
            if relevant_context["long_term"]:
                suggestions.append("根據您之前學習的知識，建議深入了解...")
            
            # 基於個人偏好的建議
            preferences = await self.context_manager.get_user_preferences()
            if preferences.get("content_preferences"):
                suggestions.append("根據您的偏好，推薦相關內容...")
            
            return suggestions
            
        except Exception as e:
            print(f"獲取記憶增強建議失敗: {e}")
            return []


# 全局RAG系統實例
_rag_system = None


async def get_rag_system(user_id: str = "default") -> RAGSystem:
    """獲取RAG系統實例"""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem(user_id)
        await _rag_system.initialize()
    return _rag_system
