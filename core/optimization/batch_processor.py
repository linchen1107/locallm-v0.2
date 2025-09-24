"""
高性能批處理器
支持並行處理、流式處理和智能批次分割
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Callable, Union, AsyncGenerator, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from queue import Queue
import threading
import psutil

from .cache_manager import get_cache_manager, CacheKeyGenerator


@dataclass
class BatchConfig:
    """批處理配置"""
    batch_size: int = 32
    max_workers: int = None
    processing_mode: str = "thread"  # thread, process, async
    memory_limit_mb: int = 1024
    enable_caching: bool = True
    progress_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        if self.max_workers is None:
            if self.processing_mode == "process":
                self.max_workers = min(mp.cpu_count(), 8)
            else:
                self.max_workers = min(mp.cpu_count() * 2, 16)


@dataclass  
class BatchResult:
    """批處理結果"""
    batch_id: str
    items_processed: int
    items_failed: int
    processing_time: float
    results: List[Any] = field(default_factory=list)
    errors: List[Exception] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        total = self.items_processed + self.items_failed
        return (self.items_processed / total * 100) if total > 0 else 0.0


class MemoryMonitor:
    """內存監控器"""
    
    def __init__(self, limit_mb: int = 1024):
        self.limit_bytes = limit_mb * 1024 * 1024
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """獲取當前內存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def is_memory_available(self, required_mb: float = 100) -> bool:
        """檢查是否有足夠內存"""
        current_mb = self.get_memory_usage()
        return (current_mb + required_mb) * 1024 * 1024 < self.limit_bytes
    
    def wait_for_memory(self, required_mb: float = 100, timeout: float = 60):
        """等待內存可用"""
        start_time = time.time()
        while not self.is_memory_available(required_mb):
            if time.time() - start_time > timeout:
                raise RuntimeError(f"Memory timeout: required {required_mb}MB")
            time.sleep(0.1)


class BatchProcessor:
    """高性能批處理器"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_mb)
        self.cache = get_cache_manager() if self.config.enable_caching else None
        
        # 執行器池
        self._thread_pool = None
        self._process_pool = None
    
    async def process_batch(self, items: List[Any], 
                          processor_func: Callable,
                          **kwargs) -> BatchResult:
        """處理批次數據"""
        
        batch_id = f"batch_{int(time.time() * 1000)}"
        start_time = datetime.now()
        
        try:
            # 根據處理模式選擇執行策略
            if self.config.processing_mode == "async":
                results = await self._process_async_batch(items, processor_func, **kwargs)
            elif self.config.processing_mode == "thread":
                results = await self._process_threaded_batch(items, processor_func, **kwargs)
            elif self.config.processing_mode == "process":
                results = await self._process_multiprocess_batch(items, processor_func, **kwargs)
            else:
                raise ValueError(f"Unsupported processing mode: {self.config.processing_mode}")
            
            # 統計結果
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return BatchResult(
                batch_id=batch_id,
                items_processed=len(successful_results),
                items_failed=len(failed_results),
                processing_time=processing_time,
                results=successful_results,
                errors=failed_results,
                start_time=start_time,
                end_time=end_time
            )
            
        except Exception as e:
            # 處理批次級別的錯誤
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return BatchResult(
                batch_id=batch_id,
                items_processed=0,
                items_failed=len(items),
                processing_time=processing_time,
                results=[],
                errors=[e],
                start_time=start_time,
                end_time=end_time
            )
    
    async def _process_async_batch(self, items: List[Any], 
                                  processor_func: Callable, **kwargs) -> List[Any]:
        """異步批處理"""
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_with_semaphore(item):
            async with semaphore:
                # 檢查緩存
                if self.cache:
                    cache_key = CacheKeyGenerator.generate_key(
                        processor_func.__name__, item=str(item)[:100], **kwargs
                    )
                    cached_result = await self.cache.get(cache_key)
                    if cached_result is not None:
                        return cached_result
                
                # 處理項目
                try:
                    result = await self._execute_with_retry(processor_func, item, **kwargs)
                    
                    # 緩存結果
                    if self.cache and result is not None:
                        await self.cache.set(cache_key, result, ttl_seconds=3600)
                    
                    return result
                    
                except Exception as e:
                    if self.config.error_callback:
                        self.config.error_callback(item, e)
                    return e
        
        # 並發處理所有項目
        tasks = [process_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _process_threaded_batch(self, items: List[Any],
                                    processor_func: Callable, **kwargs) -> List[Any]:
        """線程池批處理"""
        if not self._thread_pool:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        def process_item_sync(item):
            # 檢查緩存（同步版本）
            if self.cache:
                cache_key = CacheKeyGenerator.generate_key(
                    processor_func.__name__, item=str(item)[:100], **kwargs
                )
                cached_result = self.cache.memory_cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            try:
                # 如果是異步函數，在線程中運行事件循環
                if asyncio.iscoroutinefunction(processor_func):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            self._execute_with_retry(processor_func, item, **kwargs)
                        )
                    finally:
                        loop.close()
                else:
                    result = self._execute_with_retry_sync(processor_func, item, **kwargs)
                
                # 緩存結果
                if self.cache and result is not None:
                    self.cache.memory_cache.set(cache_key, result, ttl_seconds=3600)
                
                return result
                
            except Exception as e:
                if self.config.error_callback:
                    self.config.error_callback(item, e)
                return e
        
        # 提交任務到線程池
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(self._thread_pool, process_item_sync, item)
            for item in items
        ]
        
        results = await asyncio.gather(*futures, return_exceptions=True)
        return results
    
    async def _process_multiprocess_batch(self, items: List[Any],
                                        processor_func: Callable, **kwargs) -> List[Any]:
        """多進程批處理"""
        if not self._process_pool:
            self._process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
        
        # 將處理函數和參數打包
        def process_item_mp(args):
            item, func_name, func_kwargs = args
            try:
                # 在子進程中重新導入和執行函數
                # 注意：這裡需要確保processor_func是可序列化的
                result = self._execute_with_retry_sync(processor_func, item, **func_kwargs)
                return result
            except Exception as e:
                return e
        
        # 準備參數
        args_list = [(item, processor_func.__name__, kwargs) for item in items]
        
        # 提交到進程池
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(self._process_pool, process_item_mp, args)
            for args in args_list
        ]
        
        results = await asyncio.gather(*futures, return_exceptions=True)
        return results
    
    async def _execute_with_retry(self, func: Callable, *args, **kwargs):
        """異步重試執行"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    # 對於同步函數，在線程池中執行以避免阻塞事件循環
                    loop = asyncio.get_event_loop()
                    if not self._thread_pool:
                        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
                    
                    # 使用partial來正確傳遞參數，避免lambda閉包問題
                    from functools import partial
                    wrapped_func = partial(func, *args, **kwargs)
                    return await loop.run_in_executor(self._thread_pool, wrapped_func)
                    
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    
        raise last_exception
    
    def _execute_with_retry_sync(self, func: Callable, *args, **kwargs):
        """同步重試執行"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                    
        raise last_exception
    
    def __del__(self):
        """清理資源"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
        if self._process_pool:
            self._process_pool.shutdown(wait=False)


class StreamProcessor:
    """流式處理器"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_mb)
    
    async def process_stream(self, item_generator: AsyncGenerator[Any, None],
                           processor_func: Callable,
                           **kwargs) -> AsyncGenerator[Any, None]:
        """流式處理數據"""
        
        current_batch = []
        batch_processor = BatchProcessor(self.config)
        
        async for item in item_generator:
            current_batch.append(item)
            
            # 檢查批次大小或內存限制
            if (len(current_batch) >= self.config.batch_size or 
                not self.memory_monitor.is_memory_available(50)):
                
                # 處理當前批次
                if current_batch:
                    batch_result = await batch_processor.process_batch(
                        current_batch, processor_func, **kwargs
                    )
                    
                    # 返回結果
                    for result in batch_result.results:
                        yield result
                    
                    # 清空批次
                    current_batch = []
                    
                    # 進度回調
                    if self.config.progress_callback:
                        self.config.progress_callback(batch_result)
        
        # 處理剩餘項目
        if current_batch:
            batch_result = await batch_processor.process_batch(
                current_batch, processor_func, **kwargs
            )
            
            for result in batch_result.results:
                yield result


class AdaptiveBatchProcessor:
    """自適應批處理器"""
    
    def __init__(self, initial_config: BatchConfig = None):
        self.config = initial_config or BatchConfig()
        self.performance_history = []
        self.optimal_batch_size = self.config.batch_size
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_mb)
    
    async def process_with_adaptation(self, items: List[Any],
                                    processor_func: Callable,
                                    **kwargs) -> List[BatchResult]:
        """自適應批處理"""
        results = []
        remaining_items = items.copy()
        
        while remaining_items:
            # 動態調整批次大小
            current_batch_size = self._calculate_optimal_batch_size()
            batch_items = remaining_items[:current_batch_size]
            remaining_items = remaining_items[current_batch_size:]
            
            # 更新配置
            adapted_config = BatchConfig(
                batch_size=current_batch_size,
                max_workers=self.config.max_workers,
                processing_mode=self.config.processing_mode,
                memory_limit_mb=self.config.memory_limit_mb,
                enable_caching=self.config.enable_caching,
                progress_callback=self.config.progress_callback,
                error_callback=self.config.error_callback,
                retry_attempts=self.config.retry_attempts,
                retry_delay=self.config.retry_delay
            )
            
            # 處理批次
            processor = BatchProcessor(adapted_config)
            batch_result = await processor.process_batch(batch_items, processor_func, **kwargs)
            results.append(batch_result)
            
            # 記錄性能數據
            self._record_performance(batch_result)
            
            # 進度回調
            if self.config.progress_callback:
                self.config.progress_callback(batch_result)
        
        return results
    
    def _calculate_optimal_batch_size(self) -> int:
        """計算最優批次大小"""
        if len(self.performance_history) < 3:
            return self.config.batch_size
        
        # 分析最近的性能數據
        recent_performance = self.performance_history[-5:]
        
        # 計算每個批次大小的平均處理速度
        batch_size_performance = {}
        for result in recent_performance:
            batch_size = result.items_processed + result.items_failed
            if batch_size > 0:
                speed = batch_size / result.processing_time  # 項目/秒
                if batch_size not in batch_size_performance:
                    batch_size_performance[batch_size] = []
                batch_size_performance[batch_size].append(speed)
        
        # 找到最優批次大小
        if batch_size_performance:
            avg_speeds = {
                size: sum(speeds) / len(speeds)
                for size, speeds in batch_size_performance.items()
            }
            optimal_size = max(avg_speeds.keys(), key=lambda k: avg_speeds[k])
            
            # 考慮內存限制
            available_memory_mb = self.memory_monitor.get_memory_usage()
            max_safe_batch_size = min(
                optimal_size,
                int(self.config.memory_limit_mb * 0.8 / max(available_memory_mb / optimal_size, 1))
            )
            
            return max(1, min(max_safe_batch_size, self.config.batch_size * 2))
        
        return self.config.batch_size
    
    def _record_performance(self, result: BatchResult):
        """記錄性能數據"""
        self.performance_history.append(result)
        
        # 只保留最近的性能數據
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]


# 工廠函數
def create_batch_processor(processing_mode: str = "async",
                         batch_size: int = 32,
                         max_workers: int = None,
                         memory_limit_mb: int = 1024,
                         adaptive: bool = False) -> Union[BatchProcessor, AdaptiveBatchProcessor]:
    """創建批處理器"""
    
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        processing_mode=processing_mode,
        memory_limit_mb=memory_limit_mb,
        enable_caching=True
    )
    
    if adaptive:
        return AdaptiveBatchProcessor(config)
    else:
        return BatchProcessor(config)


# 便捷裝飾器
def batch_process(batch_size: int = 32, processing_mode: str = "async"):
    """批處理裝飾器"""
    def decorator(func):
        async def wrapper(items: List[Any], **kwargs):
            processor = create_batch_processor(
                processing_mode=processing_mode,
                batch_size=batch_size
            )
            
            result = await processor.process_batch(items, func, **kwargs)
            return result.results
        
        return wrapper
    return decorator
