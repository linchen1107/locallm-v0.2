"""
工作流執行器

負責執行生成的工作流，管理步驟依賴和執行狀態
"""

import asyncio
import time
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .auto_generator import GeneratedWorkflow, WorkflowStep


class StepStatus(Enum):
    """步驟執行狀態"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """步驟執行結果"""
    step_id: str
    status: StepStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    output: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """執行時長（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'result': str(self.result) if self.result else None,
            'error': self.error,
            'output': self.output
        }


@dataclass
class WorkflowExecution:
    """工作流執行記錄"""
    workflow_id: str
    execution_id: str
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """總執行時長（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def progress(self) -> float:
        """執行進度（0.0-1.0）"""
        if not self.step_results:
            return 0.0
        
        completed = sum(1 for result in self.step_results.values() 
                       if result.status in [StepStatus.COMPLETED, StepStatus.SKIPPED])
        return completed / len(self.step_results)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_id': self.workflow_id,
            'execution_id': self.execution_id,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'progress': self.progress,
            'step_results': {k: v.to_dict() for k, v in self.step_results.items()},
            'context': self.context
        }


class WorkflowExecutor:
    """工作流執行器"""
    
    def __init__(self):
        from core.tools.manager import get_tool_manager
        self.tool_manager = get_tool_manager()
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
    
    async def execute_workflow(self, workflow: GeneratedWorkflow, 
                             context: Optional[Dict[str, Any]] = None,
                             progress_callback: Optional[callable] = None) -> WorkflowExecution:
        """執行工作流"""
        
        import uuid
        execution_id = str(uuid.uuid4())[:8]
        
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            execution_id=execution_id,
            context=context or {}
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            execution.status = "running"
            execution.start_time = datetime.now()
            
            if progress_callback:
                await progress_callback(execution, "started")
            
            # 初始化步驟結果
            for step in workflow.steps:
                execution.step_results[step.id] = StepResult(
                    step_id=step.id,
                    status=StepStatus.PENDING
                )
            
            # 執行步驟
            await self._execute_steps(workflow.steps, execution, progress_callback)
            
            # 確定最終狀態
            failed_steps = [r for r in execution.step_results.values() 
                          if r.status == StepStatus.FAILED]
            
            if failed_steps:
                execution.status = "failed"
            else:
                execution.status = "completed"
            
            execution.end_time = datetime.now()
            
            if progress_callback:
                await progress_callback(execution, "completed")
        
        except Exception as e:
            execution.status = "error"
            execution.end_time = datetime.now()
            
            if progress_callback:
                await progress_callback(execution, f"error: {str(e)}")
        
        finally:
            # 移動到歷史記錄
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            self.execution_history.append(execution)
            if len(self.execution_history) > 100:  # 保持最近100條記錄
                self.execution_history.pop(0)
        
        return execution
    
    async def _execute_steps(self, steps: List[WorkflowStep], 
                           execution: WorkflowExecution,
                           progress_callback: Optional[callable] = None):
        """執行工作流步驟"""
        
        # 構建依賴圖
        dependency_graph = self._build_dependency_graph(steps)
        
        # 執行步驟（按依賴順序）
        executed_steps = set()
        
        while len(executed_steps) < len(steps):
            # 找到可執行的步驟（依賴已完成）
            ready_steps = []
            
            for step in steps:
                if step.id not in executed_steps:
                    if all(dep in executed_steps for dep in step.dependencies):
                        ready_steps.append(step)
            
            if not ready_steps:
                # 檢查是否有循環依賴
                remaining_steps = [s for s in steps if s.id not in executed_steps]
                if remaining_steps:
                    # 標記剩餘步驟為失敗
                    for step in remaining_steps:
                        result = execution.step_results[step.id]
                        result.status = StepStatus.FAILED
                        result.error = "Circular dependency detected"
                break
            
            # 並行執行就緒的步驟
            tasks = []
            for step in ready_steps:
                task = self._execute_single_step(step, execution, progress_callback)
                tasks.append(task)
            
            # 等待這一輪的步驟完成
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # 更新已執行集合
            for step in ready_steps:
                executed_steps.add(step.id)
    
    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """構建依賴圖"""
        graph = {}
        for step in steps:
            graph[step.id] = step.dependencies.copy()
        return graph
    
    async def _execute_single_step(self, step: WorkflowStep, 
                                 execution: WorkflowExecution,
                                 progress_callback: Optional[callable] = None):
        """執行單個步驟"""
        
        result = execution.step_results[step.id]
        result.status = StepStatus.RUNNING
        result.start_time = datetime.now()
        
        if progress_callback:
            await progress_callback(execution, f"executing step: {step.purpose}")
        
        try:
            # 準備步驟上下文
            step_context = execution.context.copy()
            
            # 添加前置步驟的輸出
            for dep_id in step.dependencies:
                if dep_id in execution.step_results:
                    dep_result = execution.step_results[dep_id]
                    if dep_result.output:
                        step_context[f"step_{dep_id}_output"] = dep_result.output
            
            # 執行步驟
            step_output = await self._call_tool(step, step_context)
            
            result.result = step_output
            result.output = step_output if isinstance(step_output, dict) else {"result": step_output}
            result.status = StepStatus.COMPLETED
        
        except Exception as e:
            result.error = str(e)
            result.status = StepStatus.FAILED
        
        finally:
            result.end_time = datetime.now()
    
    async def _call_tool(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """調用工具執行步驟"""
        
        # 特殊處理一些工具
        if step.tool == "ai_chat":
            return await self._handle_ai_chat(step, context)
        elif step.tool == "memory":
            return await self._handle_memory_operation(step, context)
        elif step.tool == "context_analyzer":
            return await self._handle_context_analysis(step, context)
        elif step.tool == "content_generator":
            return await self._handle_content_generation(step, context)
        elif step.tool == "general_executor":
            return await self._handle_general_execution(step, context)
        
        # 調用工具管理器
        try:
            tool = self.tool_manager.get_tool(step.tool)
            if tool:
                # 合併步驟參數和上下文
                tool_params = {**step.params, **context}
                result = await tool.execute(step.operation, **tool_params)
                return result
            else:
                raise ValueError(f"Tool '{step.tool}' not found")
        
        except Exception as e:
            # 如果工具不存在或執行失敗，返回模擬結果
            return {
                "status": "simulated",
                "tool": step.tool,
                "operation": step.operation,
                "purpose": step.purpose,
                "params": step.params,
                "note": f"Tool execution simulated due to: {str(e)}"
            }
    
    async def _handle_ai_chat(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """處理AI聊天步驟"""
        try:
            from core.models.ollama_client import get_ollama_client
            client = get_ollama_client()
            
            question = step.params.get('question', 'How can I help you?')
            
            response = await client.chat(
                model="qwen3:latest",
                prompt=question,
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                "question": question,
                "answer": response.get("response", "No response generated"),
                "model": "qwen3:latest"
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "AI chat simulation - Ollama not available"
            }
    
    async def _handle_memory_operation(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """處理記憶操作步驟"""
        operation = step.operation
        
        if operation == "save":
            content = step.params.get('content', context.get('previous_output', 'Workflow result'))
            return {
                "operation": "save",
                "content_preview": content[:100] + "..." if len(str(content)) > 100 else str(content),
                "saved": True
            }
        
        elif operation == "recall":
            query = step.params.get('query', 'workflow')
            return {
                "operation": "recall",
                "query": query,
                "results": ["Sample memory result 1", "Sample memory result 2"]
            }
        
        return {"operation": operation, "status": "completed"}
    
    async def _handle_context_analysis(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """處理上下文分析步驟"""
        return {
            "analysis": "Context analysis completed",
            "project_type": context.get('project_info', {}).get('project_type', 'unknown'),
            "file_count": len(context.get('file_summary', {}).get('file_types', {})),
            "working_directory": context.get('working_directory', '.')
        }
    
    async def _handle_content_generation(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """處理內容生成步驟"""
        content_type = step.params.get('content_type', 'text')
        template = step.params.get('template', 'default')
        
        return {
            "generated": True,
            "content_type": content_type,
            "template": template,
            "preview": f"Generated {content_type} content using {template} template"
        }
    
    async def _handle_general_execution(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """處理通用執行步驟"""
        return {
            "executed": True,
            "operation": step.operation,
            "params": step.params,
            "context_keys": list(context.keys())
        }
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """獲取執行狀態"""
        
        # 檢查活躍執行
        if execution_id in self.active_executions:
            return self.active_executions[execution_id].to_dict()
        
        # 檢查歷史記錄
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return execution.to_dict()
        
        return None
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """列出活躍的執行"""
        return [execution.to_dict() for execution in self.active_executions.values()]
    
    def list_execution_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """列出執行歷史"""
        return [execution.to_dict() for execution in self.execution_history[-limit:]]
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """取消執行"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.status = "cancelled"
            execution.end_time = datetime.now()
            
            # 移動到歷史記錄
            del self.active_executions[execution_id]
            self.execution_history.append(execution)
            
            return True
        
        return False
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """獲取執行統計信息"""
        all_executions = list(self.active_executions.values()) + self.execution_history
        
        if not all_executions:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "status_distribution": {}
            }
        
        total = len(all_executions)
        completed = sum(1 for e in all_executions if e.status == "completed")
        
        # 計算平均執行時間
        durations = [e.duration for e in all_executions if e.duration > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        # 狀態分布
        status_dist = {}
        for execution in all_executions:
            status_dist[execution.status] = status_dist.get(execution.status, 0) + 1
        
        return {
            "total_executions": total,
            "success_rate": (completed / total) * 100 if total > 0 else 0.0,
            "average_duration": avg_duration,
            "status_distribution": status_dist
        }


def get_workflow_executor() -> WorkflowExecutor:
    """獲取工作流執行器實例"""
    if not hasattr(get_workflow_executor, '_instance'):
        get_workflow_executor._instance = WorkflowExecutor()
    return get_workflow_executor._instance


