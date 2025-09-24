"""
集成優化的RAG系統
將所有優化組件整合到現有RAG系統中
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from config.settings import get_settings
from core.models.ollama_client import get_ollama_client
from core.storage.vector_store import get_vector_store_manager
from core.rag.document_processing import DocumentChunk
from core.memory.context_manager import ContextManager

# 導入優化組件
from .cache_manager import get_cache_manager, cached
from .batch_processor import create_batch_processor, BatchConfig
from .parallel_executor import ParallelExecutor, ExecutorConfig, Task
from .recovery_manager import get_retry_executor, smart_retry
from .resource_monitor import get_resource_monitor, monitor_performance
from .enhanced_retrieval import EnhancedRetriever, SearchQuery as OptimizedSearchQuery, SearchResult


@dataclass
class OptimizedRetrievalResult:
    """優化的檢索結果"""
    answer: str
    sources: List[DocumentChunk]
    confidence: float
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: Dict[str, Any]
    
    # 新增優化相關指標
    cache_hit: bool = False
    retry_count: int = 0
    batch_processed: bool = False
    memory_usage_mb: float = 0.0
    optimization_stats: Dict[str, Any] = None


class OptimizedRAGSystem:
    """優化的RAG系統"""
    
    def __init__(self, user_id: str = "default"):
        self.settings = get_settings()
        self.ollama_client = get_ollama_client()
        self.vector_store_manager = get_vector_store_manager()
        self.context_manager = ContextManager(user_id)
        self.user_id = user_id
        
        # 優化組件
        self.cache = get_cache_manager()
        self.monitor = get_resource_monitor()
        self.retry_executor = get_retry_executor()
        self.enhanced_retriever = EnhancedRetriever()
        
        # 批處理配置
        self.batch_processor = create_batch_processor(
            processing_mode="async",
            batch_size=32,
            adaptive=False  # 使用普通的BatchProcessor而不是AdaptiveBatchProcessor
        )
        
        # 並行執行器
        self.parallel_executor = ParallelExecutor(ExecutorConfig(
            max_workers=4,
            timeout_seconds=60
        ))
        
        # 啟動監控
        if not getattr(self.monitor, '_monitoring', False):
            self.monitor.start_monitoring()
    
    @monitor_performance("rag_query")
    @smart_retry(max_retries=3, base_delay=1.0)
    async def query(self, 
                   question: str,
                   filters: Optional[Dict[str, Any]] = None,
                   max_results: int = 5,
                   similarity_threshold: float = 0.7,
                   include_memory_context: bool = True,
                   use_batch_processing: bool = False,
                   search_strategy: str = "hybrid") -> OptimizedRetrievalResult:
        """優化的查詢方法"""
        
        start_time = time.time()
        optimization_stats = {
            "cache_operations": 0,
            "retry_attempts": 0,
            "batch_size": 1,
            "parallel_tasks": 0
        }
        
        try:
            # 獲取記憶上下文
            enhanced_question = question
            memory_context = ""
            if include_memory_context:
                memory_context = await self.context_manager.get_smart_context(question)
                if memory_context:
                    enhanced_question = f"{question}\n\n相關上下文：\n{memory_context}"
            
            # 檢查緩存
            cache_key = f"rag_query:{hash(enhanced_question + str(filters))}"
            cached_result = await self.cache.get(cache_key)
            optimization_stats["cache_operations"] += 1
            
            if cached_result:
                cached_result.cache_hit = True
                cached_result.total_time = time.time() - start_time
                return cached_result
            
            # 執行檢索
            retrieval_start = time.time()
            
            # 創建優化的搜索查詢
            search_query = OptimizedSearchQuery(
                text=enhanced_question,
                limit=max_results * 2,  # 獲取更多候選進行重排序
                similarity_threshold=similarity_threshold,
                filters=filters or {},
                search_strategy=search_strategy,
                rerank=True,
                use_cache=True
            )
            
            # 執行增強檢索
            search_results, retrieval_metrics = await self.enhanced_retriever.search(search_query)
            
            retrieval_time = time.time() - retrieval_start
            
            # 轉換搜索結果為DocumentChunk格式
            sources = []
            for result in search_results[:max_results]:
                chunk = DocumentChunk(
                    id=result.id,
                    content=result.content,
                    chunk_index=result.chunk_index,
                    source_file=result.source,
                    metadata=result.metadata
                )
                sources.append(chunk)
            
            # 生成答案
            generation_start = time.time()
            
            if use_batch_processing and len(sources) > 5:
                # 使用批處理生成答案
                answer = await self._batch_generate_answer(enhanced_question, sources)
                optimization_stats["batch_size"] = len(sources)
            else:
                # 單次生成答案
                answer = await self._generate_answer(enhanced_question, sources)
            
            generation_time = time.time() - generation_start
            
            # 計算置信度
            confidence = self._calculate_confidence(search_results, answer)
            
            # 獲取內存使用情況
            memory_usage = self.monitor.get_current_status().get('resources', {}).get('memory_used_gb', 0) * 1024
            
            # 創建結果
            result = OptimizedRetrievalResult(
                answer=answer,
                sources=sources,
                confidence=confidence,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=time.time() - start_time,
                metadata={
                    "query_enhanced": bool(memory_context),
                    "memory_context_length": len(memory_context),
                    "search_strategy": search_strategy,
                    "retrieval_metrics": retrieval_metrics.__dict__
                },
                cache_hit=False,
                retry_count=optimization_stats["retry_attempts"],
                batch_processed=use_batch_processing,
                memory_usage_mb=memory_usage,
                optimization_stats=optimization_stats
            )
            
            # 緩存結果
            await self.cache.set(cache_key, result, ttl_seconds=3600)
            optimization_stats["cache_operations"] += 1
            
            # 保存到記憶系統
            if include_memory_context:
                await self._save_query_to_memory(question, result, memory_context)
            
            return result
            
        except Exception as e:
            optimization_stats["retry_attempts"] += 1
            # 重新拋出異常，讓重試機制處理
            raise e
    
    async def batch_query(self, 
                         questions: List[str],
                         **kwargs) -> List[OptimizedRetrievalResult]:
        """批量查詢"""
        
        # 使用批處理器處理多個查詢
        results = await self.batch_processor.process_batch(
            questions,
            lambda q: self.query(q, use_batch_processing=True, **kwargs)
        )
        
        return results.results
    
    async def parallel_query(self,
                           questions: List[str],
                           **kwargs) -> List[OptimizedRetrievalResult]:
        """並行查詢"""
        
        # 創建並行任務
        tasks = []
        for i, question in enumerate(questions):
            task = Task(
                id=f"query_{i}",
                func=self.query,
                args=(question,),
                kwargs=kwargs
            )
            tasks.append(task)
        
        # 執行並行任務
        results = await self.parallel_executor.execute_parallel(tasks)
        
        return [r.result for r in results if r.success]
    
    @cached(ttl_seconds=1800, cache_level="memory")
    async def _generate_answer(self, question: str, sources: List[DocumentChunk]) -> str:
        """生成答案"""
        
        if not sources:
            return "抱歉，我找不到相關的信息來回答您的問題。"
        
        # 構建上下文
        context_parts = []
        for i, source in enumerate(sources[:3]):  # 限制上下文長度
            context_parts.append(f"參考資料{i+1}：{source.content[:500]}...")
        
        context = "\n\n".join(context_parts)
        
        # 構建提示
        prompt = f"""基於以下參考資料回答問題：

{context}

問題：{question}

請根據參考資料提供準確、詳細的回答。如果參考資料中沒有足夠信息，請說明並提供你能確定的部分信息。

回答："""
        
        try:
            response = await self.ollama_client.generate(
                model=self.settings.ollama.chat_model,
                prompt=prompt,
                temperature=self.settings.ollama.chat_temperature,
                max_tokens=self.settings.ollama.chat_max_tokens
            )
            
            return response.get('response', '抱歉，生成回答時出現錯誤。')
            
        except Exception as e:
            return f"生成回答時出現錯誤：{str(e)}"
    
    async def _batch_generate_answer(self, question: str, sources: List[DocumentChunk]) -> str:
        """批處理生成答案"""
        
        # 將源文檔分批處理
        batch_size = 3
        source_batches = [sources[i:i+batch_size] for i in range(0, len(sources), batch_size)]
        
        # 為每個批次生成部分答案
        partial_answers = []
        for batch in source_batches:
            partial_answer = await self._generate_answer(question, batch)
            partial_answers.append(partial_answer)
        
        # 合併部分答案
        if len(partial_answers) == 1:
            return partial_answers[0]
        
        # 使用LLM合併多個部分答案
        combined_prompt = f"""請將以下多個部分答案合併成一個完整、連貫的回答：

問題：{question}

部分答案：
{chr(10).join(f'{i+1}. {answer}' for i, answer in enumerate(partial_answers))}

請提供一個合併後的完整答案："""
        
        try:
            response = await self.ollama_client.generate(
                model=self.settings.ollama.chat_model,
                prompt=combined_prompt,
                temperature=0.3,  # 較低溫度確保一致性
                max_tokens=self.settings.ollama.chat_max_tokens
            )
            
            return response.get('response', partial_answers[0])
            
        except Exception:
            # 如果合併失敗，返回第一個部分答案
            return partial_answers[0]
    
    def _calculate_confidence(self, search_results: List[SearchResult], answer: str) -> float:
        """計算答案置信度"""
        if not search_results:
            return 0.0
        
        # 基於檢索結果的平均分數
        avg_score = sum(r.score for r in search_results) / len(search_results)
        
        # 基於答案長度的調整
        answer_length_factor = min(len(answer) / 200, 1.0)  # 200字符為基準
        
        # 基於源數量的調整
        source_factor = min(len(search_results) / 3, 1.0)  # 3個源為基準
        
        # 綜合置信度
        confidence = avg_score * 0.6 + answer_length_factor * 0.2 + source_factor * 0.2
        
        return min(confidence, 1.0)
    
    async def _save_query_to_memory(self, question: str, result: OptimizedRetrievalResult, 
                                  memory_context: str):
        """保存查詢到記憶系統"""
        try:
            await self.context_manager.process_interaction(question, result.answer, {
                "confidence": result.confidence,
                "memory_context_used": bool(memory_context),
                "query_type": "optimized_rag_query",
                "optimization_stats": result.optimization_stats
            })
            
            # 如果答案質量高，存儲到長期記憶
            if result.confidence > 0.8:
                from core.memory.base import MemoryImportance
                
                fact_content = f"問題：{question}\n答案：{result.answer}"
                await self.context_manager.long_term_memory.store(
                    fact_content,
                    MemoryImportance.MEDIUM,
                    {
                        "source": "optimized_rag_query",
                        "confidence": result.confidence,
                        "query_timestamp": datetime.now().isoformat(),
                        "optimization_used": True
                    },
                    ["query", "knowledge", "optimized_rag"]
                )
                
        except Exception as e:
            print(f"保存查詢到記憶系統失敗: {e}")
    
    async def ingest_documents(self, documents: List[Dict[str, Any]]):
        """優化的文檔攝入"""
        
        # 使用批處理攝入文檔
        batch_result = await self.batch_processor.process_batch(
            documents,
            self._process_single_document
        )
        
        # 添加到增強檢索器
        await self.enhanced_retriever.add_documents(documents)
        
        print(f"✅ 已攝入 {batch_result.items_processed} 個文檔，失敗 {batch_result.items_failed} 個")
        
        return batch_result
    
    async def _process_single_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """處理單個文檔"""
        # 這裡可以添加文檔預處理邏輯
        return {
            "id": document.get("id"),
            "processed": True,
            "content_length": len(document.get("content", ""))
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """獲取優化統計信息"""
        return {
            "cache_stats": self.cache.get_stats(),
            "monitor_stats": self.monitor.get_current_status(),
            "retry_stats": self.retry_executor.get_statistics(),
            "retrieval_stats": self.enhanced_retriever.get_statistics(),
            "executor_stats": self.parallel_executor.get_execution_stats()
        }
    
    def optimize_system(self):
        """優化系統性能"""
        print("🔧 開始系統優化...")
        
        # 優化檢索索引
        self.enhanced_retriever.optimize_indices()
        
        # 清理過期緩存
        # 這裡可以添加緩存清理邏輯
        
        # 獲取系統建議
        recommendations = self.monitor.get_resource_recommendations()
        if recommendations:
            print("💡 系統優化建議:")
            for rec in recommendations:
                print(f"  - {rec}")
        
        print("✅ 系統優化完成")
    
    def __del__(self):
        """清理資源"""
        try:
            if hasattr(self, 'parallel_executor'):
                self.parallel_executor.shutdown()
        except:
            pass


# 全局優化RAG實例
_global_optimized_rag = None


async def get_optimized_rag_system(user_id: str = "default") -> OptimizedRAGSystem:
    """獲取優化的RAG系統實例"""
    global _global_optimized_rag
    if _global_optimized_rag is None:
        _global_optimized_rag = OptimizedRAGSystem(user_id)
        # 可以在這裡添加初始化邏輯
    return _global_optimized_rag


# 便捷函數
async def optimized_query(question: str, **kwargs) -> OptimizedRetrievalResult:
    """便捷的優化查詢函數"""
    rag_system = await get_optimized_rag_system()
    return await rag_system.query(question, **kwargs)


async def batch_optimized_query(questions: List[str], **kwargs) -> List[OptimizedRetrievalResult]:
    """便捷的批量優化查詢函數"""
    rag_system = await get_optimized_rag_system()
    return await rag_system.batch_query(questions, **kwargs)
