"""
增強的工具系統基礎架構
支持工具間協作、流水線處理和動態組合
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum
import inspect

from config.settings import get_settings


class ToolCategory(Enum):
    """工具分類"""
    BASIC = "basic"           # 基礎工具
    ADVANCED = "advanced"     # 高級工具
    DOMAIN = "domain"         # 領域專用工具
    CUSTOM = "custom"         # 自定義工具


class ToolStatus(Enum):
    """工具狀態"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CooperationMode(Enum):
    """協作模式"""
    SEQUENTIAL = "sequential"     # 順序執行
    PARALLEL = "parallel"         # 並行執行
    PIPELINE = "pipeline"         # 流水線處理
    CONDITIONAL = "conditional"   # 條件執行


@dataclass
class ToolDependency:
    """工具依賴關係"""
    tool_name: str
    required_output_keys: List[str]
    optional: bool = False
    condition: Optional[Callable] = None


@dataclass
class ToolMetadata:
    """工具元數據"""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = "System"
    tags: List[str] = None
    dependencies: List[ToolDependency] = None
    output_schema: Dict[str, Any] = None
    input_schema: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ToolExecution:
    """工具執行記錄"""
    id: str
    tool_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = None
    status: ToolStatus = ToolStatus.IDLE
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


class ToolResult:
    """增強的工具執行結果"""
    
    def __init__(self, success: bool, data: Any = None, error: str = None, 
                 metadata: Dict[str, Any] = None, execution_id: str = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.execution_id = execution_id or str(uuid.uuid4())
        self.timestamp = datetime.now()
        
        # 協作相關屬性
        self.shareable_data = {}  # 可分享給其他工具的數據
        self.next_suggestions = []  # 建議的下一步工具
    
    def add_shareable_data(self, key: str, value: Any):
        """添加可分享的數據"""
        self.shareable_data[key] = value
    
    def suggest_next_tool(self, tool_name: str, reason: str = ""):
        """建議下一個工具"""
        self.next_suggestions.append({
            "tool": tool_name,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "shareable_data": self.shareable_data,
            "next_suggestions": self.next_suggestions
        }


class BaseTool(ABC):
    """增強的工具基類"""
    
    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
        self.settings = get_settings()
        self.status = ToolStatus.IDLE
        self.execution_history: List[ToolExecution] = []
        
        # 協作機制
        self.shared_context = {}  # 工具間共享上下文
        self.event_listeners = {}  # 事件監聽器
        
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """執行工具主邏輯"""
        pass
    
    async def pre_execute(self, **kwargs) -> Dict[str, Any]:
        """執行前預處理"""
        return kwargs
    
    async def post_execute(self, result: ToolResult) -> ToolResult:
        """執行後處理"""
        return result
    
    async def run(self, **kwargs) -> ToolResult:
        """完整的工具執行流程"""
        execution = ToolExecution(
            id=str(uuid.uuid4()),
            tool_name=self.metadata.name,
            input_data=kwargs,
            started_at=datetime.now()
        )
        
        try:
            self.status = ToolStatus.RUNNING
            execution.status = ToolStatus.RUNNING
            
            # 預處理
            processed_kwargs = await self.pre_execute(**kwargs)
            
            # 主執行
            result = await self.execute(**processed_kwargs)
            result.execution_id = execution.id
            
            # 後處理
            result = await self.post_execute(result)
            
            # 更新執行記錄
            execution.output_data = result.data
            execution.status = ToolStatus.COMPLETED if result.success else ToolStatus.FAILED
            execution.completed_at = datetime.now()
            execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            if not result.success:
                execution.error_message = result.error
            
            self.status = ToolStatus.COMPLETED if result.success else ToolStatus.FAILED
            
        except Exception as e:
            result = ToolResult(False, error=str(e), execution_id=execution.id)
            execution.status = ToolStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            self.status = ToolStatus.FAILED
        
        finally:
            self.execution_history.append(execution)
        
        return result
    
    def share_data(self, key: str, value: Any):
        """分享數據到共享上下文"""
        self.shared_context[key] = {
            "value": value,
            "source": self.metadata.name,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_shared_data(self, key: str) -> Any:
        """獲取共享數據"""
        if key in self.shared_context:
            return self.shared_context[key]["value"]
        return None
    
    def emit_event(self, event_type: str, data: Any):
        """發出事件"""
        if event_type in self.event_listeners:
            for listener in self.event_listeners[event_type]:
                asyncio.create_task(listener(data))
    
    def listen_to_event(self, event_type: str, listener: Callable):
        """監聽事件"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(listener)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """獲取工具能力描述"""
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "category": self.metadata.category.value,
            "version": self.metadata.version,
            "tags": self.metadata.tags,
            "input_schema": self.metadata.input_schema,
            "output_schema": self.metadata.output_schema,
            "dependencies": [asdict(dep) for dep in self.metadata.dependencies]
        }
    
    def can_cooperate_with(self, other_tool: 'BaseTool') -> bool:
        """檢查是否能與其他工具協作"""
        # 檢查依賴關係
        for dep in self.metadata.dependencies:
            if dep.tool_name == other_tool.metadata.name:
                return True
        
        # 檢查輸出輸入匹配
        if (self.metadata.output_schema and other_tool.metadata.input_schema):
            output_keys = set(self.metadata.output_schema.keys())
            input_keys = set(other_tool.metadata.input_schema.keys())
            return bool(output_keys.intersection(input_keys))
        
        return False


class ToolWorkflow:
    """工具工作流管理器"""
    
    def __init__(self, name: str = None):
        self.name = name or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tools: List[BaseTool] = []
        self.execution_plan: List[Dict[str, Any]] = []
        self.shared_context = {}
        self.workflow_id = str(uuid.uuid4())
    
    def add_tool(self, tool: BaseTool, config: Dict[str, Any] = None):
        """添加工具到工作流"""
        self.tools.append(tool)
        # 同步共享上下文
        tool.shared_context = self.shared_context
    
    def create_sequential_plan(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """創建順序執行計劃"""
        plan = []
        for tool_name in tool_names:
            plan.append({
                "tool": tool_name,
                "mode": CooperationMode.SEQUENTIAL,
                "depends_on": plan[-1]["tool"] if plan else None
            })
        return plan
    
    def create_parallel_plan(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """創建並行執行計劃"""
        return [{
            "tool": tool_name,
            "mode": CooperationMode.PARALLEL,
            "depends_on": None
        } for tool_name in tool_names]
    
    def create_pipeline_plan(self, pipeline_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """創建流水線執行計劃"""
        plan = []
        for i, config in enumerate(pipeline_config):
            plan.append({
                "tool": config["tool"],
                "mode": CooperationMode.PIPELINE,
                "input_mapping": config.get("input_mapping", {}),
                "output_mapping": config.get("output_mapping", {}),
                "depends_on": plan[-1]["tool"] if plan else None
            })
        return plan
    
    async def execute_sequential(self, tool_configs: List[Dict[str, Any]]) -> List[ToolResult]:
        """順序執行工具"""
        results = []
        context = {}
        
        for config in tool_configs:
            tool_name = config["tool"]
            tool = self._find_tool(tool_name)
            if not tool:
                continue
            
            # 準備輸入參數
            kwargs = config.get("params", {})
            if context:
                kwargs.update(context)
            
            result = await tool.run(**kwargs)
            results.append(result)
            
            # 更新上下文
            if result.success and result.shareable_data:
                context.update(result.shareable_data)
        
        return results
    
    async def execute_parallel(self, tool_configs: List[Dict[str, Any]]) -> List[ToolResult]:
        """並行執行工具"""
        tasks = []
        
        for config in tool_configs:
            tool_name = config["tool"]
            tool = self._find_tool(tool_name)
            if tool:
                kwargs = config.get("params", {})
                tasks.append(tool.run(**kwargs))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 處理異常結果
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append(ToolResult(False, error=str(result)))
                else:
                    processed_results.append(result)
            return processed_results
        
        return []
    
    async def execute_pipeline(self, pipeline_config: List[Dict[str, Any]]) -> List[ToolResult]:
        """流水線執行工具"""
        results = []
        pipeline_data = {}
        
        for config in pipeline_config:
            tool_name = config["tool"]
            tool = self._find_tool(tool_name)
            if not tool:
                continue
            
            # 準備輸入
            kwargs = config.get("params", {})
            
            # 應用輸入映射
            input_mapping = config.get("input_mapping", {})
            for source_key, target_key in input_mapping.items():
                if source_key in pipeline_data:
                    kwargs[target_key] = pipeline_data[source_key]
            
            result = await tool.run(**kwargs)
            results.append(result)
            
            # 應用輸出映射
            if result.success:
                output_mapping = config.get("output_mapping", {})
                if output_mapping:
                    for source_key, target_key in output_mapping.items():
                        if hasattr(result.data, source_key):
                            pipeline_data[target_key] = getattr(result.data, source_key)
                        elif isinstance(result.data, dict) and source_key in result.data:
                            pipeline_data[target_key] = result.data[source_key]
                
                # 添加可分享數據
                if result.shareable_data:
                    pipeline_data.update(result.shareable_data)
        
        return results
    
    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        """查找工具"""
        for tool in self.tools:
            if tool.metadata.name == tool_name:
                return tool
        return None
    
    async def suggest_next_tools(self, current_result: ToolResult) -> List[str]:
        """基於當前結果建議下一個工具"""
        suggestions = []
        
        # 基於工具的建議
        for suggestion in current_result.next_suggestions:
            suggestions.append(suggestion["tool"])
        
        # 基於依賴關係的建議
        for tool in self.tools:
            for dep in tool.metadata.dependencies:
                if current_result.shareable_data and any(
                    key in current_result.shareable_data for key in dep.required_output_keys
                ):
                    suggestions.append(tool.metadata.name)
        
        return list(set(suggestions))


class ToolRegistry:
    """增強的工具註冊表"""
    
    def __init__(self):
        self.tools: Dict[str, Type[BaseTool]] = {}
        self.tool_instances: Dict[str, BaseTool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }
    
    def register_tool(self, tool_class: Type[BaseTool]):
        """註冊工具類"""
        # 創建臨時實例以獲取元數據
        temp_instance = tool_class()
        tool_name = temp_instance.metadata.name
        
        self.tools[tool_name] = tool_class
        self.categories[temp_instance.metadata.category].append(tool_name)
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """獲取工具實例"""
        if tool_name not in self.tool_instances:
            if tool_name in self.tools:
                self.tool_instances[tool_name] = self.tools[tool_name]()
            else:
                return None
        
        return self.tool_instances[tool_name]
    
    def list_tools(self, category: ToolCategory = None) -> List[Dict[str, Any]]:
        """列出工具"""
        if category:
            tool_names = self.categories.get(category, [])
        else:
            tool_names = list(self.tools.keys())
        
        tools_info = []
        for tool_name in tool_names:
            tool = self.get_tool(tool_name)
            if tool:
                tools_info.append(tool.get_capabilities())
        
        return tools_info
    
    def find_compatible_tools(self, output_schema: Dict[str, Any]) -> List[str]:
        """查找兼容的工具"""
        compatible_tools = []
        
        for tool_name in self.tools:
            tool = self.get_tool(tool_name)
            if tool and tool.metadata.input_schema:
                output_keys = set(output_schema.keys())
                input_keys = set(tool.metadata.input_schema.keys())
                if output_keys.intersection(input_keys):
                    compatible_tools.append(tool_name)
        
        return compatible_tools
    
    def create_workflow(self, workflow_name: str = None) -> ToolWorkflow:
        """創建工具工作流"""
        workflow = ToolWorkflow(workflow_name)
        
        # 添加所有已註冊的工具
        for tool_name in self.tools:
            tool = self.get_tool(tool_name)
            if tool:
                workflow.add_tool(tool)
        
        return workflow


# 全局工具註冊表實例
_tool_registry = None


def get_tool_registry() -> ToolRegistry:
    """獲取工具註冊表實例"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


# 工具裝飾器
def tool(category: ToolCategory, name: str = None, description: str = None, 
         version: str = "1.0.0", tags: List[str] = None):
    """工具註冊裝飾器"""
    def decorator(cls):
        tool_name = name or cls.__name__.replace("Tool", "").lower()
        tool_description = description or cls.__doc__ or f"{tool_name} tool"
        
        # 為類添加元數據
        if not hasattr(cls, '__init_metadata__'):
            original_init = cls.__init__
            
            def new_init(self, *args, **kwargs):
                metadata = ToolMetadata(
                    name=tool_name,
                    description=tool_description,
                    category=category,
                    version=version,
                    tags=tags or []
                )
                original_init(self, metadata, *args, **kwargs)
            
            cls.__init__ = new_init
            cls.__init_metadata__ = True
        
        # 註冊到全局註冊表
        registry = get_tool_registry()
        registry.register_tool(cls)
        
        return cls
    
    return decorator

