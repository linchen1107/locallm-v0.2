"""
智能代理核心
提供任務規劃、執行和管理功能
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from config.settings import get_settings
from config.security import get_security_auditor
from core.models.ollama_client import get_ollama_client
from core.agent.tools import get_tool_registry, ToolResult
from core.tools.manager import get_tool_manager
from core.tools.base import ToolResult as NewToolResult
from core.memory.context_manager import ContextManager


class TaskStatus(Enum):
    """任務狀態"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskStep:
    """任務步驟"""
    id: int
    description: str
    tool_name: str
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[ToolResult] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class Task:
    """任務"""
    id: str
    description: str
    steps: List[TaskStep]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class TaskPlanner:
    """任務規劃器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.tool_registry = get_tool_registry()
    
    async def plan_task(self, description: str) -> List[TaskStep]:
        """規劃任務步驟"""
        try:
            # 獲取可用工具信息
            available_tools = self.tool_registry.list_tools()
            tools_info = self._format_tools_info(available_tools)
            
            # 構建規劃提示
            planning_prompt = self._build_planning_prompt(description, tools_info)
            
            # 使用LLM進行任務規劃
            client = await get_ollama_client()
            planning_result = await client.generate_text(
                prompt=planning_prompt,
                temperature=0.1,
                max_tokens=2000
            )
            
            # 解析規劃結果
            steps = self._parse_planning_result(planning_result)
            
            return steps
            
        except Exception as e:
            print(f"Task planning failed: {e}")
            # 返回一個基本的單步驟計劃
            return [TaskStep(
                id=1,
                description=description,
                tool_name="knowledge_base",
                parameters={"query": description}
            )]
    
    def _format_tools_info(self, tools: List[Dict[str, Any]]) -> str:
        """格式化工具信息"""
        tools_description = "Available tools:\n"
        for tool in tools:
            tools_description += f"\n- {tool['name']}: {tool['description']}\n"
            if 'parameters' in tool and 'properties' in tool['parameters']:
                tools_description += "  Parameters:\n"
                for param, info in tool['parameters']['properties'].items():
                    required = param in tool['parameters'].get('required', [])
                    tools_description += f"    - {param} ({'required' if required else 'optional'}): {info.get('description', '')}\n"
        
        return tools_description
    
    def _build_planning_prompt(self, description: str, tools_info: str) -> str:
        """構建規劃提示"""
        prompt = f"""You are an intelligent task planner. Given a user request, break it down into specific, executable steps using the available tools.

{tools_info}

User Request: {description}

Please plan the task by breaking it into steps. For each step, specify:
1. A clear description of what the step does
2. Which tool to use
3. The parameters for that tool

Format your response as a JSON array of steps:
[
  {{
    "description": "Step description",
    "tool_name": "tool_name",
    "parameters": {{
      "param1": "value1",
      "param2": "value2"
    }}
  }}
]

Important guidelines:
- Use specific, actionable steps
- Choose the most appropriate tool for each step
- Ensure parameters match the tool's requirements
- Consider dependencies between steps
- Be practical and realistic about what can be accomplished

Task Plan:"""
        
        return prompt
    
    def _parse_planning_result(self, planning_result: str) -> List[TaskStep]:
        """解析規劃結果"""
        try:
            # 嘗試提取JSON數組
            json_match = re.search(r'\[.*\]', planning_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                steps_data = json.loads(json_str)
                
                steps = []
                for i, step_data in enumerate(steps_data):
                    step = TaskStep(
                        id=i + 1,
                        description=step_data.get("description", ""),
                        tool_name=step_data.get("tool_name", ""),
                        parameters=step_data.get("parameters", {})
                    )
                    steps.append(step)
                
                return steps
            else:
                # 如果無法解析JSON，嘗試手動解析
                return self._fallback_parse(planning_result)
                
        except Exception as e:
            print(f"Failed to parse planning result: {e}")
            return self._fallback_parse(planning_result)
    
    def _fallback_parse(self, planning_result: str) -> List[TaskStep]:
        """備用解析方法"""
        # 簡單的文本解析，創建一個基本步驟
        return [TaskStep(
            id=1,
            description="Execute user request",
            tool_name="knowledge_base",
            parameters={"query": planning_result[:200]}
        )]


class TaskExecutor:
    """任務執行器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.tool_registry = get_tool_registry()
        self.running_tasks: Dict[str, Task] = {}
    
    async def execute_task(self, task: Task) -> Task:
        """執行任務"""
        self.running_tasks[task.id] = task
        
        try:
            task.status = TaskStatus.EXECUTING
            task.started_at = datetime.now()
            
            # 記錄任務開始
            get_security_auditor().log_action(
                "task_started",
                details={"task_id": task.id, "description": task.description}
            )
            
            # 執行每個步驟
            for step in task.steps:
                if task.status == TaskStatus.CANCELLED:
                    break
                
                await self._execute_step(step)
                
                # 如果步驟失敗且是關鍵步驟，停止執行
                if step.result and not step.result.success:
                    error_msg = f"Step {step.id} failed: {step.result.error}"
                    task.error_message = error_msg
                    task.status = TaskStatus.FAILED
                    break
            
            # 確定最終狀態
            if task.status == TaskStatus.EXECUTING:
                if all(step.result and step.result.success for step in task.steps):
                    task.status = TaskStatus.COMPLETED
                else:
                    task.status = TaskStatus.FAILED
            
            task.completed_at = datetime.now()
            
            # 記錄任務完成
            get_security_auditor().log_action(
                "task_completed",
                details={
                    "task_id": task.id,
                    "status": task.status.value,
                    "execution_time": (task.completed_at - task.started_at).total_seconds()
                }
            )
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            get_security_auditor().log_security_event(
                "task_execution_error",
                severity="error",
                details={"task_id": task.id, "error": str(e)}
            )
        
        finally:
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
        
        return task
    
    async def _execute_step(self, step: TaskStep):
        """執行單個步驟"""
        step.status = TaskStatus.EXECUTING
        step.start_time = datetime.now()
        
        try:
            # 檢查工具是否存在
            tool = self.tool_registry.get_tool(step.tool_name)
            if not tool:
                step.result = ToolResult(False, error=f"Tool '{step.tool_name}' not found")
                step.status = TaskStatus.FAILED
                return
            
            # 執行工具
            result = await self.tool_registry.execute_tool(step.tool_name, **step.parameters)
            step.result = result
            
            if result.success:
                step.status = TaskStatus.COMPLETED
            else:
                step.status = TaskStatus.FAILED
            
        except Exception as e:
            step.result = ToolResult(False, error=str(e))
            step.status = TaskStatus.FAILED
        
        finally:
            step.end_time = datetime.now()
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任務"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            return True
        return False
    
    def get_running_tasks(self) -> List[Task]:
        """獲取正在運行的任務"""
        return list(self.running_tasks.values())


class IntelligentAgent:
    """智能代理主類"""
    
    def __init__(self, user_id: str = "default"):
        self.settings = get_settings()
        self.planner = TaskPlanner()
        self.executor = TaskExecutor()
        self.task_history: List[Task] = []
        self._task_counter = 0
        
        # 記憶系統集成
        self.context_manager = ContextManager(user_id)
        self.user_id = user_id
        
        # 新工具系統集成
        self.tool_manager = get_tool_manager()
        self.active_workflows: Dict[str, Any] = {}
    
    async def execute_complex_task(self, description: str, 
                                 auto_confirm: bool = True) -> Task:
        """執行複雜任務"""
        try:
            # 生成任務ID
            self._task_counter += 1
            task_id = f"task_{self._task_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 獲取相關上下文
            relevant_context = await self.context_manager.get_smart_context(description)
            
            # 規劃任務（包含上下文）
            print(f"Planning task: {description}")
            enhanced_description = description
            if relevant_context:
                enhanced_description = f"{description}\n\n上下文信息：\n{relevant_context}"
            
            steps = await self.planner.plan_task(enhanced_description)
            
            # 創建任務
            task = Task(
                id=task_id,
                description=description,
                steps=steps,
                metadata={"auto_confirm": auto_confirm}
            )
            
            # 顯示執行計劃
            print(f"\nTask Plan ({len(steps)} steps):")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step.description}")
                print(f"     Tool: {step.tool_name}")
                print(f"     Parameters: {step.parameters}")
            
            # 確認執行（如果需要）
            if not auto_confirm:
                confirm = input("\nProceed with execution? (y/n): ")
                if confirm.lower() != 'y':
                    task.status = TaskStatus.CANCELLED
                    return task
            
            # 執行任務
            print("\nExecuting task...")
            task = await self.executor.execute_task(task)
            
            # 添加到歷史記錄
            self.task_history.append(task)
            
            # 保存任務結果到記憶系統
            if task.status == TaskStatus.COMPLETED:
                await self._save_task_to_memory(task)
            
            # 顯示結果
            self._display_task_result(task)
            
            return task
            
        except Exception as e:
            print(f"Task execution failed: {e}")
            # 創建失敗的任務記錄
            task = Task(
                id=f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=description,
                steps=[],
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
            self.task_history.append(task)
            return task
    
    def _display_task_result(self, task: Task):
        """顯示任務結果"""
        print(f"\n{'='*50}")
        print(f"Task Status: {task.status.value.upper()}")
        print(f"Task ID: {task.id}")
        print(f"Description: {task.description}")
        
        if task.started_at and task.completed_at:
            duration = (task.completed_at - task.started_at).total_seconds()
            print(f"Execution Time: {duration:.2f} seconds")
        
        if task.error_message:
            print(f"Error: {task.error_message}")
        
        print(f"\nStep Results:")
        for step in task.steps:
            status_icon = "✓" if step.status == TaskStatus.COMPLETED else "✗"
            print(f"  {status_icon} Step {step.id}: {step.description}")
            
            if step.result:
                if step.result.success:
                    # 顯示部分結果數據
                    if isinstance(step.result.data, dict):
                        for key, value in list(step.result.data.items())[:3]:
                            if isinstance(value, str) and len(value) > 100:
                                value = value[:100] + "..."
                            print(f"     {key}: {value}")
                    else:
                        result_str = str(step.result.data)
                        if len(result_str) > 200:
                            result_str = result_str[:200] + "..."
                        print(f"     Result: {result_str}")
                else:
                    print(f"     Error: {step.result.error}")
        
        print(f"{'='*50}")
    
    async def simple_query(self, question: str) -> str:
        """簡單查詢（直接使用知識庫）"""
        try:
            tool_registry = get_tool_registry()
            result = await tool_registry.execute_tool(
                "knowledge_base",
                query=question
            )
            
            if result.success:
                return result.data.get("answer", "No answer found")
            else:
                return f"Query failed: {result.error}"
                
        except Exception as e:
            return f"Query error: {str(e)}"
    
    def get_task_history(self, limit: int = 10) -> List[Task]:
        """獲取任務歷史"""
        return self.task_history[-limit:]
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """根據ID獲取任務"""
        for task in self.task_history:
            if task.id == task_id:
                return task
        return None
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """獲取代理狀態"""
        running_tasks = self.executor.get_running_tasks()
        
        return {
            "status": "active",
            "running_tasks": len(running_tasks),
            "total_tasks": len(self.task_history),
            "completed_tasks": len([t for t in self.task_history if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in self.task_history if t.status == TaskStatus.FAILED]),
            "available_tools": len(self.get_available_tools()),
            "settings": {
                "max_execution_time": self.settings.agent.max_execution_time,
                "max_parallel_tasks": self.settings.agent.max_parallel_tasks,
                "code_execution_enabled": self.settings.agent.enable_code_execution,
                "sandbox_enabled": self.settings.agent.sandbox_enabled
            }
        }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """獲取可用工具列表"""
        return self.executor.tool_registry.list_tools()
    
    async def _save_task_to_memory(self, task: Task):
        """保存任務到記憶系統"""
        try:
            from core.memory.base import MemoryImportance
            
            # 構建任務摘要
            task_summary = f"完成任務：{task.description}"
            
            # 添加步驟信息
            if task.steps:
                completed_steps = [step for step in task.steps if step.status == TaskStatus.COMPLETED]
                if completed_steps:
                    step_summaries = [f"- {step.description}" for step in completed_steps]
                    task_summary += f"\n執行步驟：\n" + "\n".join(step_summaries)
            
            # 添加結果信息
            if hasattr(task, 'result') and task.result:
                task_summary += f"\n任務結果：{task.result}"
            
            # 根據任務複雜度確定重要性
            importance = MemoryImportance.MEDIUM
            if len(task.steps) > 5:
                importance = MemoryImportance.HIGH
            elif any("創建" in step.description or "生成" in step.description for step in task.steps):
                importance = MemoryImportance.HIGH
            
            # 創建元數據
            metadata = {
                "task_id": task.id,
                "task_type": "agent_execution",
                "steps_count": len(task.steps),
                "execution_time": (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else None,
                "tools_used": list(set(step.tool for step in task.steps if hasattr(step, 'tool')))
            }
            
            # 存儲到長期記憶
            await self.context_manager.long_term_memory.store(
                task_summary,
                importance,
                metadata,
                ["task", "completed", "agent"]
            )
            
            print(f"任務 {task.id} 已保存到記憶系統")
            
        except Exception as e:
            print(f"保存任務到記憶系統失敗: {e}")
    
    async def get_relevant_experience(self, task_description: str) -> str:
        """獲取相關的執行經驗"""
        try:
            # 搜索相關的任務記憶
            relevant_memories = await self.context_manager.long_term_memory.retrieve(task_description, limit=3)
            
            if not relevant_memories:
                return ""
            
            experience_parts = ["基於過往經驗："]
            for memory in relevant_memories:
                if "task" in memory.tags:
                    experience_parts.append(f"- {memory.content}")
            
            return "\n".join(experience_parts) if len(experience_parts) > 1 else ""
            
        except Exception as e:
            print(f"獲取相關經驗失敗: {e}")
            return ""
    
    async def learn_from_task_execution(self, task: Task):
        """從任務執行中學習"""
        try:
            # 分析任務執行模式
            if task.status == TaskStatus.COMPLETED:
                # 記錄成功模式
                await self.context_manager.personal_memory.record_interaction(
                    "task_success",
                    {
                        "task_description": task.description,
                        "steps_count": len(task.steps),
                        "execution_time": (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else None
                    }
                )
                
                # 更新偏好
                await self.context_manager.personal_memory.store_preference(
                    "task_execution",
                    "preferred_complexity",
                    len(task.steps),
                    confidence=0.1
                )
            
            elif task.status == TaskStatus.FAILED:
                # 記錄失敗模式以避免重複
                await self.context_manager.personal_memory.record_interaction(
                    "task_failure",
                    {
                        "task_description": task.description,
                        "error_message": task.error_message
                    }
                )
                
        except Exception as e:
            print(f"任務學習失敗: {e}")
    
    async def get_personalized_suggestions(self) -> List[str]:
        """獲取個性化建議"""
        try:
            # 獲取用戶偏好
            preferences = await self.context_manager.get_user_preferences()
            
            suggestions = []
            
            # 基於使用模式的建議
            if preferences.get("usage_insights"):
                insights = preferences["usage_insights"]
                if "most_used_interaction" in insights:
                    interaction_type = insights["most_used_interaction"]["type"]
                    if interaction_type == "task_success":
                        suggestions.append("您經常成功完成任務，建議嘗試更複雜的挑戰")
            
            # 基於任務歷史的建議
            recent_tasks = self.task_history[-5:] if self.task_history else []
            if recent_tasks:
                completed_count = sum(1 for task in recent_tasks if task.status == TaskStatus.COMPLETED)
                if completed_count == len(recent_tasks):
                    suggestions.append("您的任務執行成功率很高，可以嘗試並行執行多個任務")
                elif completed_count < len(recent_tasks) * 0.5:
                    suggestions.append("建議將複雜任務分解為更小的步驟")
            
            # 基於工具使用統計的建議
            tool_stats = self.tool_manager.get_usage_stats()
            if tool_stats:
                most_used_tools = sorted(tool_stats.items(), key=lambda x: x[1]["usage_count"], reverse=True)
                if most_used_tools:
                    suggestions.append(f"您最常使用 {most_used_tools[0][0]} 工具，建議探索相關的協作工具")
            
            return suggestions
            
        except Exception as e:
            print(f"獲取個性化建議失敗: {e}")
            return []
    
    async def execute_enhanced_task(self, description: str, use_workflows: bool = True) -> Dict[str, Any]:
        """使用增強工具系統執行任務"""
        try:
            # 分析任務並建議工具鏈
            tool_chain = await self.tool_manager.suggest_tool_chain("file_operation", description)
            
            if use_workflows and len(tool_chain) > 1:
                # 創建並執行工作流
                workflow_id = await self.tool_manager.auto_workflow_from_goal(description)
                workflow_results = await self.tool_manager.execute_workflow(workflow_id)
                
                # 記錄工作流
                self.active_workflows[workflow_id] = {
                    "description": description,
                    "created_at": datetime.now(),
                    "results": workflow_results
                }
                
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "tool_chain": tool_chain,
                    "results": [result.to_dict() for result in workflow_results],
                    "execution_mode": "workflow"
                }
            else:
                # 執行單個工具
                if tool_chain:
                    result = await self.tool_manager.execute_tool(tool_chain[0])
                    return {
                        "success": result.success,
                        "tool_used": tool_chain[0],
                        "result": result.to_dict(),
                        "execution_mode": "single_tool"
                    }
                else:
                    return {
                        "success": False,
                        "error": "No suitable tools found for the task",
                        "execution_mode": "failed"
                    }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_mode": "error"
            }
    
    async def get_tool_recommendations(self, current_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """獲取工具推薦"""
        try:
            recommendations = []
            
            # 獲取所有可用工具
            available_tools = self.tool_manager.get_available_tools()
            
            # 基於上下文推薦工具
            if current_context:
                for tool_info in available_tools:
                    relevance_score = self._calculate_tool_relevance(tool_info, current_context)
                    if relevance_score > 0.5:
                        recommendations.append({
                            "tool": tool_info["name"],
                            "description": tool_info["description"],
                            "relevance_score": relevance_score,
                            "category": tool_info["category"],
                            "tags": tool_info.get("tags", [])
                        })
            
            # 按相關性排序
            recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return recommendations[:5]  # 返回前5個推薦
            
        except Exception as e:
            print(f"獲取工具推薦失敗: {e}")
            return []
    
    def _calculate_tool_relevance(self, tool_info: Dict[str, Any], context: Dict[str, Any]) -> float:
        """計算工具相關性分數"""
        score = 0.0
        
        # 基於標籤匹配
        tool_tags = tool_info.get("tags", [])
        context_keywords = str(context).lower().split()
        
        for tag in tool_tags:
            if any(tag.lower() in keyword for keyword in context_keywords):
                score += 0.3
        
        # 基於描述匹配
        description = tool_info.get("description", "").lower()
        for keyword in context_keywords:
            if keyword in description:
                score += 0.2
        
        # 基於類別相關性
        category = tool_info.get("category", "")
        if "file" in str(context).lower() and category == "basic":
            score += 0.2
        elif "data" in str(context).lower() and category == "advanced":
            score += 0.2
        elif "git" in str(context).lower() and category == "domain":
            score += 0.2
        
        return min(score, 1.0)
    
    async def demonstrate_tool_cooperation(self) -> Dict[str, Any]:
        """演示工具協作功能"""
        try:
            print("🚀 開始演示工具協作功能...")
            
            # 使用工具管理器的演示功能
            demo_results = await self.tool_manager.demonstrate_cooperation()
            
            # 添加Agent層面的統計
            demo_results["agent_integration"] = {
                "active_workflows": len(self.active_workflows),
                "tool_manager_ready": True,
                "memory_integration": True,
                "enhanced_features": [
                    "智能工具鏈建議",
                    "自動工作流生成", 
                    "上下文感知推薦",
                    "記憶增強執行"
                ]
            }
            
            return demo_results
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_integration": {
                    "active_workflows": len(self.active_workflows),
                    "tool_manager_ready": False
                }
            }


# 全局智能代理實例
_intelligent_agent = None


def get_intelligent_agent(user_id: str = "default") -> IntelligentAgent:
    """獲取智能代理實例"""
    global _intelligent_agent
    if _intelligent_agent is None:
        _intelligent_agent = IntelligentAgent(user_id)
    return _intelligent_agent
