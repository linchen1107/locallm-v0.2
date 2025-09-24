"""
工作流自動生成器

根據解析的命令和上下文自動生成智能工作流
"""

import re
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..intelligence.command_parser import ParsedCommand


@dataclass
class WorkflowStep:
    """工作流步驟"""
    id: str
    tool: str
    operation: str
    purpose: str
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_time: int = 10  # 預估時間（秒）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tool': self.tool,
            'operation': self.operation,
            'purpose': self.purpose,
            'params': self.params,
            'dependencies': self.dependencies,
            'estimated_time': self.estimated_time
        }


@dataclass
class WorkflowTemplate:
    """工作流模板"""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    triggers: List[str]
    category: str = "general"
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'steps': self.steps,
            'triggers': self.triggers,
            'category': self.category,
            'priority': self.priority
        }


@dataclass
class GeneratedWorkflow:
    """生成的工作流"""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    template_based: bool
    template_name: Optional[str] = None
    estimated_time: int = 0
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'steps': [step.to_dict() for step in self.steps],
            'template_based': self.template_based,
            'template_name': self.template_name,
            'estimated_time': self.estimated_time,
            'confidence': self.confidence,
            'created_at': self.created_at
        }


class WorkflowAutoGenerator:
    """工作流自動生成器"""
    
    # 預定義工作流模板
    WORKFLOW_TEMPLATES = {
        'data_analysis_pipeline': WorkflowTemplate(
            name='數據分析流水線',
            description='完整的數據分析工作流程',
            steps=[
                {'tool': 'file_operation', 'operation': 'read', 'purpose': '讀取數據文件'},
                {'tool': 'data_analysis', 'operation': 'describe', 'purpose': '生成描述性統計'},
                {'tool': 'data_analysis', 'operation': 'visualize', 'purpose': '創建數據可視化'},
                {'tool': 'file_operation', 'operation': 'write', 'purpose': '保存分析報告'}
            ],
            triggers=['分析.*數據', r'.*\.csv.*分析', 'data.*analysis', '數據.*報告'],
            category='data_science',
            priority=3
        ),
        
        'project_inspection': WorkflowTemplate(
            name='項目檢查流水線',
            description='全面的項目結構和質量檢查',
            steps=[
                {'tool': 'project_management', 'operation': 'analyze', 'purpose': '分析項目結構'},
                {'tool': 'project_management', 'operation': 'dependencies', 'purpose': '檢查依賴關係'},
                {'tool': 'git_operation', 'operation': 'status', 'purpose': '檢查版本控制狀態'},
                {'tool': 'file_operation', 'operation': 'write', 'purpose': '生成檢查報告'}
            ],
            triggers=['檢查項目', '項目分析', 'project.*check', '項目.*質量'],
            category='development',
            priority=2
        ),
        
        'document_processing': WorkflowTemplate(
            name='文檔處理流水線',
            description='文檔分析、總結和處理工作流程',
            steps=[
                {'tool': 'file_operation', 'operation': 'list', 'purpose': '掃描文檔文件'},
                {'tool': 'document_processing', 'operation': 'extract', 'purpose': '提取文檔內容'},
                {'tool': 'ai_processing', 'operation': 'summarize', 'purpose': '生成文檔摘要'},
                {'tool': 'file_operation', 'operation': 'write', 'purpose': '保存處理結果'}
            ],
            triggers=['處理文檔', '分析文檔', 'document.*process', '文檔.*總結'],
            category='documentation',
            priority=2
        ),
        
        'code_quality_audit': WorkflowTemplate(
            name='代碼質量審核',
            description='代碼質量分析和測試生成工作流程',
            steps=[
                {'tool': 'project_management', 'operation': 'structure', 'purpose': '分析代碼結構'},
                {'tool': 'test_generation', 'operation': 'generate', 'purpose': '生成測試用例'},
                {'tool': 'git_operation', 'operation': 'log', 'purpose': '檢查提交歷史'},
                {'tool': 'file_operation', 'operation': 'write', 'purpose': '生成質量報告'}
            ],
            triggers=['代碼質量', 'code.*quality', '測試.*生成', '質量.*審核'],
            category='development',
            priority=2
        ),
        
        'knowledge_base_build': WorkflowTemplate(
            name='知識庫構建',
            description='從文檔構建智能知識庫',
            steps=[
                {'tool': 'file_operation', 'operation': 'list', 'purpose': '掃描所有文檔'},
                {'tool': 'document_processing', 'operation': 'extract', 'purpose': '提取文檔內容'},
                {'tool': 'knowledge_base', 'operation': 'ingest', 'purpose': '建立知識庫'},
                {'tool': 'memory', 'operation': 'save', 'purpose': '保存到長期記憶'},
                {'tool': 'file_operation', 'operation': 'write', 'purpose': '生成知識庫報告'}
            ],
            triggers=['建立知識庫', 'knowledge.*base', '知識.*整理', '文檔.*索引'],
            category='knowledge_management',
            priority=3
        )
    }
    
    def __init__(self):
        from core.tools.manager import get_tool_manager
        self.tool_manager = get_tool_manager()
    
    async def generate_workflow_from_command(self, parsed_command: ParsedCommand, 
                                           context: Dict[str, Any]) -> GeneratedWorkflow:
        """根據解析的命令生成工作流"""
        
        # 1. 嘗試匹配預定義模板
        matched_template = self._match_template(parsed_command.natural_language, context)
        
        if matched_template:
            workflow = await self._customize_template(matched_template, parsed_command, context)
        else:
            # 2. 動態生成工作流
            workflow = await self._generate_dynamic_workflow(parsed_command, context)
        
        return workflow
    
    def _match_template(self, command: str, context: Dict[str, Any]) -> Optional[WorkflowTemplate]:
        """匹配最佳的預定義模板"""
        best_match = None
        best_score = 0
        
        for template_name, template in self.WORKFLOW_TEMPLATES.items():
            score = self._calculate_template_match_score(command, template, context)
            
            if score > best_score and score > 0.3:  # 最低匹配閾值
                best_score = score
                best_match = template
        
        return best_match
    
    def _calculate_template_match_score(self, command: str, template: WorkflowTemplate, 
                                      context: Dict[str, Any]) -> float:
        """計算模板匹配分數"""
        score = 0.0
        command_lower = command.lower()
        
        # 觸發詞匹配
        trigger_matches = 0
        for trigger in template.triggers:
            if re.search(trigger, command_lower, re.IGNORECASE):
                trigger_matches += 1
        
        if trigger_matches > 0:
            score += min(0.6, trigger_matches * 0.2)
        
        # 上下文相關性
        project_type = context.get('project_info', {}).get('project_type', 'unknown')
        file_summary = context.get('file_summary', {})
        
        # 基於項目類型的相關性
        if template.category == 'data_science':
            if project_type == 'data_science' or any(ext in file_summary.get('file_types', {}) 
                                                   for ext in ['.csv', '.json', '.xlsx']):
                score += 0.2
        
        elif template.category == 'development':
            if project_type in ['python', 'javascript', 'java', 'rust', 'go']:
                score += 0.2
        
        elif template.category == 'documentation':
            if any(ext in file_summary.get('file_types', {}) 
                   for ext in ['.md', '.txt', '.pdf', '.docx']):
                score += 0.2
        
        # 模板優先級影響
        score += template.priority * 0.05
        
        return min(1.0, score)
    
    async def _customize_template(self, template: WorkflowTemplate, 
                                parsed_command: ParsedCommand, 
                                context: Dict[str, Any]) -> GeneratedWorkflow:
        """自定義模板參數"""
        
        customized_steps = []
        total_time = 0
        
        for i, step_template in enumerate(template.steps):
            step_id = f"step_{i+1}"
            step = WorkflowStep(
                id=step_id,
                tool=step_template['tool'],
                operation=step_template['operation'],
                purpose=step_template['purpose'],
                params=step_template.get('params', {}),
                dependencies=[f"step_{i}"] if i > 0 else []
            )
            
            # 根據目標對象自定義參數
            self._customize_step_params(step, parsed_command, context, template.category)
            
            customized_steps.append(step)
            total_time += step.estimated_time
        
        workflow_id = str(uuid.uuid4())[:8]
        
        return GeneratedWorkflow(
            id=workflow_id,
            name=template.name,
            description=f"基於模板 '{template.name}' 為命令 '{parsed_command.natural_language}' 生成的工作流",
            steps=customized_steps,
            template_based=True,
            template_name=template.name,
            estimated_time=total_time,
            confidence=0.9
        )
    
    def _customize_step_params(self, step: WorkflowStep, parsed_command: ParsedCommand, 
                              context: Dict[str, Any], category: str):
        """自定義步驟參數"""
        
        # 基於目標文件自定義
        if parsed_command.targets:
            primary_target = parsed_command.targets[0]
            
            if step.tool == 'file_operation':
                if step.operation == 'read':
                    step.params['path'] = primary_target
                elif step.operation == 'write':
                    # 生成輸出文件名
                    output_name = self._generate_output_filename(parsed_command, category)
                    step.params['path'] = output_name
                    step.params['content_type'] = parsed_command.parameters.get('output_format', 'markdown')
                elif step.operation == 'list':
                    step.params['path'] = primary_target if primary_target != '.' else '.'
                    step.params['recursive'] = True
            
            elif step.tool == 'data_analysis':
                if primary_target.endswith(('.csv', '.json', '.xlsx')):
                    step.params['data_source'] = primary_target
                    step.params['data_type'] = primary_target.split('.')[-1]
            
            elif step.tool == 'project_management':
                step.params['project_path'] = primary_target if primary_target != '.' else '.'
        
        # 基於解析參數自定義
        for param_key, param_value in parsed_command.parameters.items():
            if param_key == 'verbose':
                step.params['detail_level'] = 'high' if param_value else 'normal'
            elif param_key == 'output_format':
                step.params['output_format'] = param_value
            elif param_key == 'limit':
                step.params['max_items'] = param_value
            elif param_key == 'max_depth':
                step.params['max_depth'] = param_value
        
        # 基於工具類型設置默認參數
        if step.tool == 'data_analysis' and step.operation == 'visualize':
            step.params.setdefault('chart_types', ['histogram', 'correlation', 'distribution'])
        
        elif step.tool == 'test_generation':
            step.params.setdefault('test_types', ['unit', 'integration'])
            step.params.setdefault('coverage_target', 80)
    
    def _generate_output_filename(self, parsed_command: ParsedCommand, category: str) -> str:
        """生成輸出文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        base_names = {
            'data_science': 'data_analysis_report',
            'development': 'project_analysis_report',
            'documentation': 'document_processing_report',
            'knowledge_management': 'knowledge_base_report'
        }
        
        base_name = base_names.get(category, 'analysis_report')
        output_format = parsed_command.parameters.get('output_format', 'md')
        
        return f"{base_name}_{timestamp}.{output_format}"
    
    async def _generate_dynamic_workflow(self, parsed_command: ParsedCommand, 
                                       context: Dict[str, Any]) -> GeneratedWorkflow:
        """動態生成工作流"""
        
        steps = []
        
        # 基於意圖生成步驟
        if parsed_command.intent == 'analyze':
            steps.extend(self._generate_analysis_steps(parsed_command, context))
        elif parsed_command.intent == 'chat':
            steps.extend(self._generate_chat_steps(parsed_command, context))
        elif parsed_command.intent == 'execute':
            steps.extend(self._generate_execution_steps(parsed_command, context))
        elif parsed_command.intent == 'read':
            steps.extend(self._generate_read_steps(parsed_command, context))
        elif parsed_command.intent == 'write':
            steps.extend(self._generate_write_steps(parsed_command, context))
        
        # 如果沒有生成步驟，創建基本分析步驟
        if not steps:
            steps.extend(self._generate_default_steps(parsed_command, context))
        
        total_time = sum(step.estimated_time for step in steps)
        workflow_id = str(uuid.uuid4())[:8]
        
        return GeneratedWorkflow(
            id=workflow_id,
            name=f"智能工作流 - {parsed_command.intent}",
            description=f"為命令 '{parsed_command.natural_language}' 動態生成的工作流",
            steps=steps,
            template_based=False,
            estimated_time=total_time,
            confidence=parsed_command.confidence * 0.7
        )
    
    def _generate_analysis_steps(self, parsed_command: ParsedCommand, 
                               context: Dict[str, Any]) -> List[WorkflowStep]:
        """生成分析類步驟"""
        steps = []
        project_type = context.get('project_info', {}).get('project_type', 'unknown')
        
        if project_type == 'data_science' or any(target.endswith(('.csv', '.json', '.xlsx')) 
                                               for target in parsed_command.targets):
            # 數據分析工作流
            steps.append(WorkflowStep(
                id="step_1",
                tool="data_analysis",
                operation="load",
                purpose="加載數據文件",
                params={'data_source': parsed_command.targets[0] if parsed_command.targets else '.'}
            ))
            
            steps.append(WorkflowStep(
                id="step_2", 
                tool="data_analysis",
                operation="describe",
                purpose="生成描述性統計",
                dependencies=["step_1"]
            ))
            
            if 'visualize' in parsed_command.actions:
                steps.append(WorkflowStep(
                    id="step_3",
                    tool="data_analysis",
                    operation="visualize", 
                    purpose="創建數據可視化",
                    dependencies=["step_2"]
                ))
        
        elif project_type in ['python', 'javascript', 'java', 'rust', 'go']:
            # 代碼項目分析
            steps.append(WorkflowStep(
                id="step_1",
                tool="project_management",
                operation="analyze",
                purpose="分析項目結構",
                params={'project_path': '.'}
            ))
            
            steps.append(WorkflowStep(
                id="step_2",
                tool="project_management", 
                operation="dependencies",
                purpose="分析項目依賴",
                dependencies=["step_1"]
            ))
        
        else:
            # 通用分析
            steps.append(WorkflowStep(
                id="step_1",
                tool="file_operation",
                operation="list",
                purpose="掃描文件結構",
                params={'path': '.', 'recursive': True}
            ))
        
        # 添加報告生成步驟
        if steps:
            report_step = WorkflowStep(
                id=f"step_{len(steps)+1}",
                tool="file_operation",
                operation="write",
                purpose="生成分析報告",
                dependencies=[steps[-1].id],
                params={
                    'path': self._generate_output_filename(parsed_command, 'analysis'),
                    'content_type': parsed_command.parameters.get('output_format', 'markdown')
                }
            )
            steps.append(report_step)
        
        return steps
    
    def _generate_chat_steps(self, parsed_command: ParsedCommand, 
                           context: Dict[str, Any]) -> List[WorkflowStep]:
        """生成對話類步驟"""
        steps = []
        
        steps.append(WorkflowStep(
            id="step_1",
            tool="ai_chat",
            operation="query",
            purpose="AI對話回答",
            params={
                'question': parsed_command.natural_language,
                'context': context,
                'include_memory': True
            }
        ))
        
        if parsed_command.parameters.get('save_to_memory', True):
            steps.append(WorkflowStep(
                id="step_2",
                tool="memory",
                operation="save",
                purpose="保存對話到記憶",
                dependencies=["step_1"]
            ))
        
        return steps
    
    def _generate_execution_steps(self, parsed_command: ParsedCommand, 
                                context: Dict[str, Any]) -> List[WorkflowStep]:
        """生成執行類步驟"""
        steps = []
        
        # 基於動作生成執行步驟
        for i, action in enumerate(parsed_command.actions):
            step_id = f"step_{i+1}"
            
            if action == 'execute':
                steps.append(WorkflowStep(
                    id=step_id,
                    tool="general_executor",
                    operation="execute",
                    purpose=f"執行 {action} 操作",
                    dependencies=[f"step_{i}"] if i > 0 else []
                ))
            
            elif action == 'generate':
                steps.append(WorkflowStep(
                    id=step_id,
                    tool="content_generator",
                    operation="generate",
                    purpose="生成內容",
                    dependencies=[f"step_{i}"] if i > 0 else []
                ))
        
        return steps
    
    def _generate_read_steps(self, parsed_command: ParsedCommand, 
                           context: Dict[str, Any]) -> List[WorkflowStep]:
        """生成讀取類步驟"""
        steps = []
        
        for i, target in enumerate(parsed_command.targets):
            steps.append(WorkflowStep(
                id=f"step_{i+1}",
                tool="file_operation",
                operation="read",
                purpose=f"讀取 {target}",
                params={'path': target}
            ))
        
        return steps
    
    def _generate_write_steps(self, parsed_command: ParsedCommand, 
                            context: Dict[str, Any]) -> List[WorkflowStep]:
        """生成寫入類步驟"""
        steps = []
        
        steps.append(WorkflowStep(
            id="step_1",
            tool="file_operation",
            operation="write",
            purpose="寫入文件",
            params={
                'path': parsed_command.targets[0] if parsed_command.targets else 'output.txt',
                'content_type': parsed_command.parameters.get('output_format', 'text')
            }
        ))
        
        return steps
    
    def _generate_default_steps(self, parsed_command: ParsedCommand, 
                              context: Dict[str, Any]) -> List[WorkflowStep]:
        """生成默認步驟"""
        steps = []
        
        steps.append(WorkflowStep(
            id="step_1",
            tool="file_operation",
            operation="list",
            purpose="掃描當前目錄",
            params={'path': '.', 'recursive': False}
        ))
        
        steps.append(WorkflowStep(
            id="step_2",
            tool="context_analyzer",
            operation="analyze",
            purpose="分析上下文",
            dependencies=["step_1"]
        ))
        
        return steps
    
    def list_available_templates(self) -> List[Dict[str, Any]]:
        """列出所有可用的工作流模板"""
        return [template.to_dict() for template in self.WORKFLOW_TEMPLATES.values()]
    
    def get_template_by_name(self, name: str) -> Optional[WorkflowTemplate]:
        """根據名稱獲取模板"""
        return self.WORKFLOW_TEMPLATES.get(name)


def get_workflow_auto_generator() -> WorkflowAutoGenerator:
    """獲取工作流自動生成器實例"""
    if not hasattr(get_workflow_auto_generator, '_instance'):
        get_workflow_auto_generator._instance = WorkflowAutoGenerator()
    return get_workflow_auto_generator._instance
