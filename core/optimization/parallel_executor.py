"""
並行執行器
支持多種並行模式、資源調度和故障恢復
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import multiprocessing as mp
from queue import Queue, Empty
import psutil
import signal
import sys

from .cache_manager import get_cache_manager
from .batch_processor import MemoryMonitor


class ExecutionMode(Enum):
    """執行模式"""
    SEQUENTIAL = "sequential"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC_CONCURRENT = "async_concurrent"
    HYBRID = "hybrid"


class TaskPriority(Enum):
    """任務優先級"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """任務定義"""
    id: str
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"task_{int(time.time() * 1000000)}"


@dataclass
class ExecutionResult:
    """執行結果"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    memory_used: float = 0.0
    worker_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ExecutorConfig:
    """執行器配置"""
    mode: ExecutionMode = ExecutionMode.ASYNC_CONCURRENT
    max_workers: int = None
    max_memory_mb: int = 2048
    timeout_seconds: float = 300.0
    enable_monitoring: bool = True
    enable_caching: bool = True
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.max_workers is None:
            if self.mode == ExecutionMode.PROCESS_POOL:
                self.max_workers = min(mp.cpu_count(), 8)
            else:
                self.max_workers = min(mp.cpu_count() * 2, 16)


class ResourceMonitor:
    """資源監控器"""
    
    def __init__(self, memory_limit_mb: int = 2048):
        self.memory_limit_mb = memory_limit_mb
        self.process = psutil.Process()
        self._monitoring = False
        self._stats = {
            'peak_memory_mb': 0.0,
            'avg_cpu_percent': 0.0,
            'total_execution_time': 0.0,
            'task_count': 0
        }
    
    def start_monitoring(self):
        """開始監控"""
        self._monitoring = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def stop_monitoring(self):
        """停止監控"""
        self._monitoring = False
    
    def _monitor_loop(self):
        """監控循環"""
        cpu_samples = []
        
        while self._monitoring:
            try:
                # 監控內存
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self._stats['peak_memory_mb'] = max(self._stats['peak_memory_mb'], memory_mb)
                
                # 監控CPU
                cpu_percent = self.process.cpu_percent()
                cpu_samples.append(cpu_percent)
                if len(cpu_samples) > 60:  # 保留最近60個樣本
                    cpu_samples.pop(0)
                
                self._stats['avg_cpu_percent'] = sum(cpu_samples) / len(cpu_samples)
                
                time.sleep(1)
                
            except Exception:
                continue
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取監控統計"""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        return {
            **self._stats,
            'current_memory_mb': current_memory,
            'memory_usage_percent': (current_memory / self.memory_limit_mb) * 100,
            'memory_available': self.memory_limit_mb - current_memory > 100
        }
    
    def is_resource_available(self, required_memory_mb: float = 100) -> bool:
        """檢查資源是否可用"""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        return (current_memory + required_memory_mb) < self.memory_limit_mb * 0.9


class ParallelExecutor:
    """並行執行器"""
    
    def __init__(self, config: ExecutorConfig = None):
        self.config = config or ExecutorConfig()
        self.resource_monitor = ResourceMonitor(self.config.max_memory_mb)
        self.cache = get_cache_manager() if self.config.enable_caching else None
        
        # 執行器池
        self._thread_pool = None
        self._process_pool = None
        
        # 任務管理
        self.pending_tasks: Queue = Queue()
        self.completed_tasks: Dict[str, ExecutionResult] = {}
        self.failed_tasks: Dict[str, ExecutionResult] = {}
        
        # 依賴圖
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # 監控
        if self.config.enable_monitoring:
            self.resource_monitor.start_monitoring()
    
    async def submit_task(self, task: Task) -> str:
        """提交任務"""
        # 註冊依賴關係
        if task.dependencies:
            self.dependency_graph[task.id] = task.dependencies
        
        # 檢查是否可以立即執行
        if self._can_execute_task(task):
            return await self._execute_task(task)
        else:
            # 加入待執行隊列
            self.pending_tasks.put(task)
            return task.id
    
    async def submit_tasks(self, tasks: List[Task]) -> List[str]:
        """批量提交任務"""
        task_ids = []
        for task in tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        return task_ids
    
    async def execute_parallel(self, tasks: List[Task]) -> List[ExecutionResult]:
        """並行執行任務列表"""
        if not tasks:
            return []
        
        # 根據執行模式選擇策略
        if self.config.mode == ExecutionMode.SEQUENTIAL:
            return await self._execute_sequential(tasks)
        elif self.config.mode == ExecutionMode.ASYNC_CONCURRENT:
            return await self._execute_async_concurrent(tasks)
        elif self.config.mode == ExecutionMode.THREAD_POOL:
            return await self._execute_thread_pool(tasks)
        elif self.config.mode == ExecutionMode.PROCESS_POOL:
            return await self._execute_process_pool(tasks)
        elif self.config.mode == ExecutionMode.HYBRID:
            return await self._execute_hybrid(tasks)
        else:
            raise ValueError(f"Unsupported execution mode: {self.config.mode}")
    
    async def _execute_sequential(self, tasks: List[Task]) -> List[ExecutionResult]:
        """順序執行"""
        results = []
        for task in tasks:
            result = await self._execute_single_task(task)
            results.append(result)
        return results
    
    async def _execute_async_concurrent(self, tasks: List[Task]) -> List[ExecutionResult]:
        """異步並發執行"""
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def execute_with_semaphore(task: Task):
            async with semaphore:
                return await self._execute_single_task(task)
        
        # 按優先級排序
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value, reverse=True)
        
        # 並發執行
        coroutines = [execute_with_semaphore(task) for task in sorted_tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # 處理異常結果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ExecutionResult(
                    task_id=sorted_tasks[i].id,
                    success=False,
                    error=result,
                    execution_time=0.0
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_thread_pool(self, tasks: List[Task]) -> List[ExecutionResult]:
        """線程池執行"""
        if not self._thread_pool:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        def execute_sync_task(task: Task) -> ExecutionResult:
            return asyncio.run(self._execute_single_task(task))
        
        # 提交任務到線程池
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(self._thread_pool, execute_sync_task, task)
            for task in tasks
        ]
        
        # 等待所有任務完成
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 處理結果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ExecutionResult(
                    task_id=tasks[i].id,
                    success=False,
                    error=result,
                    execution_time=0.0
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_process_pool(self, tasks: List[Task]) -> List[ExecutionResult]:
        """進程池執行"""
        if not self._process_pool:
            self._process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
        
        def execute_process_task(task_data):
            task_id, func, args, kwargs = task_data
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    execution_time=execution_time
                )
            except Exception as e:
                return ExecutionResult(
                    task_id=task_id,
                    success=False,
                    error=e,
                    execution_time=0.0
                )
        
        # 準備任務數據
        task_data_list = [
            (task.id, task.func, task.args, task.kwargs)
            for task in tasks
        ]
        
        # 提交到進程池
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(self._process_pool, execute_process_task, task_data)
            for task_data in task_data_list
        ]
        
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 處理結果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ExecutionResult(
                    task_id=tasks[i].id,
                    success=False,
                    error=result,
                    execution_time=0.0
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_hybrid(self, tasks: List[Task]) -> List[ExecutionResult]:
        """混合執行模式"""
        # 根據任務特性分類
        cpu_intensive_tasks = []
        io_intensive_tasks = []
        
        for task in tasks:
            # 根據任務標籤或函數特性分類
            if 'cpu_intensive' in task.tags:
                cpu_intensive_tasks.append(task)
            else:
                io_intensive_tasks.append(task)
        
        results = []
        
        # CPU密集型任務使用進程池
        if cpu_intensive_tasks:
            cpu_results = await self._execute_process_pool(cpu_intensive_tasks)
            results.extend(cpu_results)
        
        # IO密集型任務使用異步並發
        if io_intensive_tasks:
            io_results = await self._execute_async_concurrent(io_intensive_tasks)
            results.extend(io_results)
        
        return results
    
    async def _execute_single_task(self, task: Task) -> ExecutionResult:
        """執行單個任務"""
        start_time = datetime.now()
        start_memory = self.resource_monitor.process.memory_info().rss / 1024 / 1024
        
        try:
            # 檢查緩存
            if self.cache:
                cache_key = f"task:{task.func.__name__}:{hash(str(task.args) + str(task.kwargs))}"
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    return ExecutionResult(
                        task_id=task.id,
                        success=True,
                        result=cached_result,
                        execution_time=0.0,
                        start_time=start_time,
                        end_time=datetime.now()
                    )
            
            # 執行任務
            task.started_at = start_time
            
            if asyncio.iscoroutinefunction(task.func):
                # 異步函數
                if task.timeout:
                    result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    result = await task.func(*task.args, **task.kwargs)
            else:
                # 同步函數
                result = task.func(*task.args, **task.kwargs)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            end_memory = self.resource_monitor.process.memory_info().rss / 1024 / 1024
            
            task.completed_at = end_time
            
            # 緩存結果
            if self.cache and result is not None:
                await self.cache.set(cache_key, result, ttl_seconds=3600)
            
            execution_result = ExecutionResult(
                task_id=task.id,
                success=True,
                result=result,
                execution_time=execution_time,
                memory_used=end_memory - start_memory,
                start_time=start_time,
                end_time=end_time
            )
            
            self.completed_tasks[task.id] = execution_result
            return execution_result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 重試邏輯
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                await asyncio.sleep(2 ** task.retry_count)  # 指數退避
                return await self._execute_single_task(task)
            
            execution_result = ExecutionResult(
                task_id=task.id,
                success=False,
                error=e,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time
            )
            
            self.failed_tasks[task.id] = execution_result
            return execution_result
    
    async def _execute_task(self, task: Task) -> str:
        """執行任務並返回任務ID"""
        result = await self._execute_single_task(task)
        
        # 檢查依賴任務
        await self._check_and_execute_dependent_tasks(task.id)
        
        return task.id
    
    def _can_execute_task(self, task: Task) -> bool:
        """檢查任務是否可以執行"""
        if not task.dependencies:
            return True
        
        # 檢查所有依賴是否已完成
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    async def _check_and_execute_dependent_tasks(self, completed_task_id: str):
        """檢查並執行依賴任務"""
        tasks_to_execute = []
        
        # 檢查待執行隊列中的任務
        remaining_tasks = []
        
        while not self.pending_tasks.empty():
            try:
                task = self.pending_tasks.get_nowait()
                if self._can_execute_task(task):
                    tasks_to_execute.append(task)
                else:
                    remaining_tasks.append(task)
            except Empty:
                break
        
        # 將無法執行的任務放回隊列
        for task in remaining_tasks:
            self.pending_tasks.put(task)
        
        # 執行可以執行的任務
        for task in tasks_to_execute:
            await self._execute_task(task)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """獲取執行統計"""
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        successful_tasks = len(self.completed_tasks)
        
        if total_tasks > 0:
            success_rate = (successful_tasks / total_tasks) * 100
        else:
            success_rate = 0.0
        
        # 計算平均執行時間
        if self.completed_tasks:
            avg_execution_time = sum(
                result.execution_time for result in self.completed_tasks.values()
            ) / len(self.completed_tasks)
        else:
            avg_execution_time = 0.0
        
        stats = {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': len(self.failed_tasks),
            'success_rate': success_rate,
            'avg_execution_time': avg_execution_time,
            'pending_tasks': self.pending_tasks.qsize(),
            'resource_stats': self.resource_monitor.get_stats() if self.config.enable_monitoring else {}
        }
        
        return stats
    
    async def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> List[ExecutionResult]:
        """等待任務完成"""
        start_time = time.time()
        results = []
        
        while len(results) < len(task_ids):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            for task_id in task_ids:
                if task_id in self.completed_tasks:
                    if task_id not in [r.task_id for r in results]:
                        results.append(self.completed_tasks[task_id])
                elif task_id in self.failed_tasks:
                    if task_id not in [r.task_id for r in results]:
                        results.append(self.failed_tasks[task_id])
            
            if len(results) < len(task_ids):
                await asyncio.sleep(0.1)
        
        return results
    
    def shutdown(self):
        """關閉執行器"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        
        if self.config.enable_monitoring:
            self.resource_monitor.stop_monitoring()


# 便捷函數
async def execute_parallel_functions(functions: List[Callable],
                                   args_list: List[Tuple] = None,
                                   kwargs_list: List[Dict] = None,
                                   mode: ExecutionMode = ExecutionMode.ASYNC_CONCURRENT,
                                   max_workers: int = None) -> List[ExecutionResult]:
    """並行執行函數列表"""
    
    if args_list is None:
        args_list = [() for _ in functions]
    if kwargs_list is None:
        kwargs_list = [{} for _ in functions]
    
    # 創建任務
    tasks = []
    for i, func in enumerate(functions):
        task = Task(
            id=f"func_{i}",
            func=func,
            args=args_list[i] if i < len(args_list) else (),
            kwargs=kwargs_list[i] if i < len(kwargs_list) else {}
        )
        tasks.append(task)
    
    # 創建執行器並執行
    config = ExecutorConfig(mode=mode, max_workers=max_workers)
    executor = ParallelExecutor(config)
    
    try:
        results = await executor.execute_parallel(tasks)
        return results
    finally:
        executor.shutdown()


# 裝飾器
def parallel_task(priority: TaskPriority = TaskPriority.NORMAL,
                 timeout: Optional[float] = None,
                 max_retries: int = 3,
                 tags: List[str] = None):
    """並行任務裝飾器"""
    def decorator(func):
        func._parallel_config = {
            'priority': priority,
            'timeout': timeout,
            'max_retries': max_retries,
            'tags': tags or []
        }
        return func
    return decorator

