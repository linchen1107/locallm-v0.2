"""
統一工具管理系統
整合所有工具，提供統一接口和協作機制
"""

import asyncio
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import json

from .base import (
    BaseTool, ToolResult, ToolRegistry, ToolWorkflow, ToolCategory,
    get_tool_registry, CooperationMode
)


class EnhancedToolManager:
    """增強的工具管理器"""
    
    def __init__(self):
        self.registry = get_tool_registry()
        self.workflows: Dict[str, ToolWorkflow] = {}
        self.tool_usage_stats = {}
        self._load_all_tools()
    
    def _load_all_tools(self):
        """加載所有工具"""
        try:
            # 導入基礎工具
            from .basic.file_tools import FileOperationTool, PathUtilityTool
            
            # 導入高級工具
            from .advanced.data_analysis import DataAnalysisTool, WebScrapingTool
            
            # 導入領域工具
            from .domain.development_tools import GitOperationTool, ProjectManagementTool, TestGenerationTool
            
            print("✅ All tools loaded successfully")
            
        except ImportError as e:
            print(f"⚠️ Some tools failed to load: {e}")
    
    def get_available_tools(self, category: ToolCategory = None) -> List[Dict[str, Any]]:
        """獲取可用工具列表"""
        return self.registry.list_tools(category)
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """獲取工具實例"""
        return self.registry.get_tool(tool_name)
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """執行單個工具"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(False, error=f"Tool not found: {tool_name}")
        
        # 記錄使用統計
        self._record_usage(tool_name)
        
        try:
            result = await tool.run(**kwargs)
            
            # 記錄結果統計
            self._record_result(tool_name, result.success)
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Tool execution failed: {str(e)}")
    
    def _record_usage(self, tool_name: str):
        """記錄工具使用統計"""
        if tool_name not in self.tool_usage_stats:
            self.tool_usage_stats[tool_name] = {
                "usage_count": 0,
                "success_count": 0,
                "failure_count": 0
            }
        
        self.tool_usage_stats[tool_name]["usage_count"] += 1
    
    def _record_result(self, tool_name: str, success: bool):
        """記錄執行結果統計"""
        if success:
            self.tool_usage_stats[tool_name]["success_count"] += 1
        else:
            self.tool_usage_stats[tool_name]["failure_count"] += 1
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """獲取使用統計"""
        return self.tool_usage_stats.copy()
    
    def create_workflow(self, name: str, tools: List[str], mode: CooperationMode = CooperationMode.SEQUENTIAL) -> str:
        """創建工具工作流"""
        workflow = self.registry.create_workflow(name)
        
        if mode == CooperationMode.SEQUENTIAL:
            workflow.execution_plan = workflow.create_sequential_plan(tools)
        elif mode == CooperationMode.PARALLEL:
            workflow.execution_plan = workflow.create_parallel_plan(tools)
        
        workflow_id = workflow.workflow_id
        self.workflows[workflow_id] = workflow
        
        return workflow_id
    
    def create_pipeline_workflow(self, name: str, pipeline_config: List[Dict[str, Any]]) -> str:
        """創建流水線工作流"""
        workflow = self.registry.create_workflow(name)
        workflow.execution_plan = workflow.create_pipeline_plan(pipeline_config)
        
        workflow_id = workflow.workflow_id
        self.workflows[workflow_id] = workflow
        
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str, initial_params: Dict[str, Any] = None) -> List[ToolResult]:
        """執行工作流"""
        if workflow_id not in self.workflows:
            return [ToolResult(False, error=f"Workflow not found: {workflow_id}")]
        
        workflow = self.workflows[workflow_id]
        
        try:
            # 根據工作流類型執行
            first_step = workflow.execution_plan[0] if workflow.execution_plan else {}
            mode = first_step.get("mode", CooperationMode.SEQUENTIAL)
            
            if mode == CooperationMode.SEQUENTIAL:
                tool_configs = [{"tool": step["tool"], "params": initial_params or {}} 
                              for step in workflow.execution_plan]
                return await workflow.execute_sequential(tool_configs)
            
            elif mode == CooperationMode.PARALLEL:
                tool_configs = [{"tool": step["tool"], "params": initial_params or {}} 
                              for step in workflow.execution_plan]
                return await workflow.execute_parallel(tool_configs)
            
            elif mode == CooperationMode.PIPELINE:
                return await workflow.execute_pipeline(workflow.execution_plan)
            
            else:
                return [ToolResult(False, error=f"Unsupported workflow mode: {mode}")]
                
        except Exception as e:
            return [ToolResult(False, error=f"Workflow execution failed: {str(e)}")]
    
    async def suggest_tool_chain(self, initial_tool: str, goal: str) -> List[str]:
        """建議工具鏈"""
        tool_chain = [initial_tool]
        
        # 獲取初始工具
        current_tool = self.get_tool(initial_tool)
        if not current_tool:
            return tool_chain
        
        # 模擬執行以獲取輸出結構
        try:
            # 這裡應該基於工具的輸出結構和目標來建議後續工具
            # 簡化實現：基於工具類別和標籤
            
            if "file" in goal.lower():
                if "file_operation" not in tool_chain:
                    tool_chain.append("file_operation")
            
            if "data" in goal.lower() or "analysis" in goal.lower():
                if "data_analysis" not in tool_chain:
                    tool_chain.append("data_analysis")
            
            if "git" in goal.lower() or "version" in goal.lower():
                if "git_operation" not in tool_chain:
                    tool_chain.append("git_operation")
            
            if "test" in goal.lower():
                if "test_generation" not in tool_chain:
                    tool_chain.append("test_generation")
            
            if "web" in goal.lower() or "scrape" in goal.lower():
                if "web_scraping" not in tool_chain:
                    tool_chain.append("web_scraping")
            
        except Exception:
            pass
        
        return tool_chain
    
    def get_tool_compatibility_matrix(self) -> Dict[str, List[str]]:
        """獲取工具兼容性矩陣"""
        compatibility = {}
        tools = self.get_available_tools()
        
        for tool_info in tools:
            tool_name = tool_info["name"]
            compatible_tools = []
            
            tool = self.get_tool(tool_name)
            if tool:
                for other_tool_info in tools:
                    if other_tool_info["name"] != tool_name:
                        other_tool = self.get_tool(other_tool_info["name"])
                        if other_tool and tool.can_cooperate_with(other_tool):
                            compatible_tools.append(other_tool_info["name"])
            
            compatibility[tool_name] = compatible_tools
        
        return compatibility
    
    async def auto_workflow_from_goal(self, goal: str, context: Dict[str, Any] = None) -> str:
        """根據目標自動創建工作流"""
        
        # 分析目標關鍵詞
        goal_lower = goal.lower()
        suggested_tools = []
        
        # 基於關鍵詞建議工具
        keyword_tool_mapping = {
            "file": ["file_operation"],
            "read": ["file_operation"],
            "write": ["file_operation"],
            "data": ["data_analysis"],
            "csv": ["data_analysis"],
            "json": ["data_analysis"],
            "analyze": ["data_analysis"],
            "chart": ["data_analysis"],
            "plot": ["data_analysis"],
            "git": ["git_operation"],
            "commit": ["git_operation"],
            "branch": ["git_operation"],
            "test": ["test_generation"],
            "unit test": ["test_generation"],
            "web": ["web_scraping"],
            "scrape": ["web_scraping"],
            "url": ["web_scraping"],
            "project": ["project_management"],
            "structure": ["project_management"]
        }
        
        for keyword, tools in keyword_tool_mapping.items():
            if keyword in goal_lower:
                suggested_tools.extend(tools)
        
        # 去重並保持順序
        unique_tools = []
        for tool in suggested_tools:
            if tool not in unique_tools:
                unique_tools.append(tool)
        
        if not unique_tools:
            unique_tools = ["file_operation"]  # 默認工具
        
        # 創建工作流
        workflow_name = f"auto_workflow_{len(self.workflows) + 1}"
        return self.create_workflow(workflow_name, unique_tools, CooperationMode.SEQUENTIAL)
    
    def export_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """導出工作流配置"""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        return {
            "name": workflow.name,
            "workflow_id": workflow_id,
            "execution_plan": workflow.execution_plan,
            "tools": [tool.metadata.name for tool in workflow.tools]
        }
    
    def import_workflow(self, workflow_config: Dict[str, Any]) -> str:
        """導入工作流配置"""
        workflow = self.registry.create_workflow(workflow_config["name"])
        workflow.execution_plan = workflow_config["execution_plan"]
        
        workflow_id = workflow.workflow_id
        self.workflows[workflow_id] = workflow
        
        return workflow_id
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """列出所有工作流"""
        return [
            {
                "workflow_id": workflow_id,
                "name": workflow.name,
                "tools_count": len(workflow.tools),
                "steps_count": len(workflow.execution_plan)
            }
            for workflow_id, workflow in self.workflows.items()
        ]
    
    async def demonstrate_cooperation(self) -> Dict[str, Any]:
        """演示工具協作功能"""
        demo_results = {}
        
        try:
            # 演示1：文件操作 + 數據分析
            print("🔧 演示工具協作：文件操作 + 數據分析")
            
            # 創建示例CSV數據
            sample_data = "name,age,city\nAlice,25,New York\nBob,30,London\nCharlie,35,Tokyo"
            
            # 1. 使用文件工具創建CSV文件
            file_result = await self.execute_tool(
                "file_operation",
                operation="write",
                path="demo_data.csv",
                content=sample_data
            )
            
            demo_results["step1_file_creation"] = file_result.to_dict()
            
            if file_result.success:
                # 2. 使用數據分析工具分析文件
                analysis_result = await self.execute_tool(
                    "data_analysis",
                    operation="describe",
                    data_source="demo_data.csv",
                    data_type="csv"
                )
                
                demo_results["step2_data_analysis"] = analysis_result.to_dict()
            
            # 演示2：項目管理 + Git操作
            print("🔧 演示工具協作：項目管理 + Git操作")
            
            # 1. 分析當前項目
            project_result = await self.execute_tool(
                "project_management",
                operation="analyze",
                project_path="."
            )
            
            demo_results["step3_project_analysis"] = project_result.to_dict()
            
            # 2. 檢查Git狀態
            git_result = await self.execute_tool(
                "git_operation",
                operation="status",
                repository_path="."
            )
            
            demo_results["step4_git_status"] = git_result.to_dict()
            
            # 清理演示文件
            await self.execute_tool(
                "file_operation",
                operation="delete",
                path="demo_data.csv"
            )
            
            demo_results["demo_completed"] = True
            demo_results["cooperation_features"] = [
                "工具間數據共享",
                "自動建議下一步工具", 
                "流水線式處理",
                "並行工具執行",
                "智能工作流生成"
            ]
            
        except Exception as e:
            demo_results["error"] = str(e)
            demo_results["demo_completed"] = False
        
        return demo_results


# 全局工具管理器實例
_tool_manager = None


def get_tool_manager() -> EnhancedToolManager:
    """獲取工具管理器實例"""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = EnhancedToolManager()
    return _tool_manager

