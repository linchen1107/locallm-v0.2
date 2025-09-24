"""
智能化功能管理器

統一管理和協調所有智能化功能：自然語言解析、工作流生成、命令補全等
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .command_parser import NaturalLanguageParser, ParsedCommand, get_natural_language_parser
from ..workflows.auto_generator import WorkflowAutoGenerator, GeneratedWorkflow, get_workflow_auto_generator
from ..workflows.executor import WorkflowExecutor, WorkflowExecution, get_workflow_executor
from ..completion.auto_complete import AutoCompleter, get_auto_completer
from ..completion.shortcuts import QuickCommands, get_quick_commands


@dataclass
class IntelligenceResult:
    """智能化處理結果"""
    success: bool
    result_type: str  # 'answer', 'workflow', 'command', 'error'
    content: Any
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'result_type': self.result_type,
            'content': self.content,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class IntelligenceManager:
    """智能化功能管理器"""
    
    def __init__(self):
        self.nlp_parser = get_natural_language_parser()
        self.workflow_generator = get_workflow_auto_generator()
        self.workflow_executor = get_workflow_executor()
        self.auto_completer = get_auto_completer()
        self.quick_commands = get_quick_commands()
        
        # 處理統計
        self.processing_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'natural_language_requests': 0,
            'workflow_executions': 0,
            'shortcut_expansions': 0
        }
    
    async def process_user_input(self, user_input: str, 
                               context: Optional[Dict[str, Any]] = None,
                               processing_mode: str = 'auto') -> IntelligenceResult:
        """
        智能處理用戶輸入
        
        Args:
            user_input: 用戶輸入的命令或自然語言
            context: 上下文信息
            processing_mode: 處理模式 ('auto', 'command', 'natural', 'workflow')
        """
        if context is None:
            context = {}
        
        self.processing_stats['total_requests'] += 1
        
        try:
            # 1. 預處理：展開快捷方式
            expanded_input = self._expand_shortcuts(user_input, context)
            
            # 2. 確定處理模式
            if processing_mode == 'auto':
                processing_mode = self._determine_processing_mode(expanded_input, context)
            
            # 3. 根據模式處理
            if processing_mode == 'command':
                result = await self._process_as_command(expanded_input, context)
            elif processing_mode == 'natural':
                result = await self._process_as_natural_language(expanded_input, context)
            elif processing_mode == 'workflow':
                result = await self._process_as_workflow(expanded_input, context)
            else:
                # 默認：嘗試智能解析
                result = await self._process_intelligently(expanded_input, context)
            
            if result.success:
                self.processing_stats['successful_requests'] += 1
            
            return result
        
        except Exception as e:
            return IntelligenceResult(
                success=False,
                result_type='error',
                content=f"處理用戶輸入時發生錯誤: {str(e)}",
                confidence=0.0,
                metadata={'error_type': type(e).__name__, 'original_input': user_input}
            )
    
    def _expand_shortcuts(self, user_input: str, context: Dict[str, Any]) -> str:
        """展開快捷方式"""
        expanded = self.quick_commands.expand_shortcut(user_input, context)
        
        if expanded != user_input:
            self.processing_stats['shortcut_expansions'] += 1
            context['shortcut_expanded'] = True
            context['original_input'] = user_input
        
        return expanded
    
    def _determine_processing_mode(self, user_input: str, context: Dict[str, Any]) -> str:
        """確定處理模式"""
        
        # 明確的命令模式指示器
        command_indicators = ['analyze', 'tools', 'memory', 'config', 'workflow', 'help']
        
        # 自然語言指示器
        natural_indicators = ['?', '？', '怎麼', '如何', '為什麼', '什麼是', '解釋', '說明']
        
        # 工作流指示器
        workflow_indicators = ['然後', '接著', '並且', '同時', '先.*再', '首先.*然後']
        
        user_input_lower = user_input.lower()
        
        # 檢查命令模式
        if any(user_input_lower.startswith(cmd) for cmd in command_indicators):
            return 'command'
        
        # 檢查自然語言模式
        if any(indicator in user_input for indicator in natural_indicators):
            return 'natural'
        
        # 檢查工作流模式
        if any(indicator in user_input for indicator in workflow_indicators):
            return 'workflow'
        
        # 基於長度和復雜度判斷
        word_count = len(user_input.split())
        
        if word_count >= 6 and ',' in user_input:
            return 'workflow'
        elif word_count >= 4:
            return 'natural'
        else:
            return 'command'
    
    async def _process_as_command(self, user_input: str, context: Dict[str, Any]) -> IntelligenceResult:
        """作為命令處理"""
        
        # 直接執行命令（這裡應該調用原有的命令處理邏輯）
        return IntelligenceResult(
            success=True,
            result_type='command',
            content={
                'command': user_input,
                'execution_mode': 'direct',
                'note': '命令已準備執行'
            },
            confidence=0.9,
            metadata={'processing_mode': 'command'}
        )
    
    async def _process_as_natural_language(self, user_input: str, context: Dict[str, Any]) -> IntelligenceResult:
        """作為自然語言處理"""
        
        self.processing_stats['natural_language_requests'] += 1
        
        # 1. 解析自然語言
        parsed_command = self.nlp_parser.parse(user_input, context)
        
        # 2. 根據意圖處理
        if parsed_command.intent == 'chat':
            # 直接作為聊天處理
            return IntelligenceResult(
                success=True,
                result_type='answer',
                content={
                    'question': user_input,
                    'intent': 'chat',
                    'suggested_action': 'forward_to_ai_chat'
                },
                confidence=parsed_command.confidence,
                metadata={'parsed_command': parsed_command.to_dict()}
            )
        
        else:
            # 生成並執行工作流
            workflow = await self.workflow_generator.generate_workflow_from_command(
                parsed_command, context
            )
            
            return IntelligenceResult(
                success=True,
                result_type='workflow',
                content={
                    'workflow': workflow.to_dict(),
                    'parsed_command': parsed_command.to_dict(),
                    'ready_to_execute': True
                },
                confidence=workflow.confidence,
                metadata={'processing_mode': 'natural_language'}
            )
    
    async def _process_as_workflow(self, user_input: str, context: Dict[str, Any]) -> IntelligenceResult:
        """作為工作流處理"""
        
        # 解析為多步驟工作流
        parsed_command = self.nlp_parser.parse(user_input, context)
        workflow = await self.workflow_generator.generate_workflow_from_command(
            parsed_command, context
        )
        
        return IntelligenceResult(
            success=True,
            result_type='workflow',
            content={
                'workflow': workflow.to_dict(),
                'multi_step': True,
                'ready_to_execute': True
            },
            confidence=workflow.confidence,
            metadata={'processing_mode': 'workflow'}
        )
    
    async def _process_intelligently(self, user_input: str, context: Dict[str, Any]) -> IntelligenceResult:
        """智能處理（自動判斷最佳方式）"""
        
        # 1. 解析命令
        parsed_command = self.nlp_parser.parse(user_input, context)
        
        # 2. 基於置信度和意圖決定處理方式
        if parsed_command.confidence < 0.5:
            # 置信度低，建議明確命令
            suggestions = self.get_command_suggestions(user_input, context)
            
            return IntelligenceResult(
                success=False,
                result_type='suggestion',
                content={
                    'message': '無法確定您的意圖，請嘗試以下建議:',
                    'suggestions': suggestions[:5],
                    'original_input': user_input
                },
                confidence=parsed_command.confidence,
                metadata={'needs_clarification': True}
            )
        
        # 3. 置信度足夠，生成工作流
        workflow = await self.workflow_generator.generate_workflow_from_command(
            parsed_command, context
        )
        
        return IntelligenceResult(
            success=True,
            result_type='workflow',
            content={
                'workflow': workflow.to_dict(),
                'parsed_command': parsed_command.to_dict(),
                'auto_generated': True
            },
            confidence=workflow.confidence,
            metadata={'processing_mode': 'intelligent'}
        )
    
    async def execute_workflow(self, workflow: GeneratedWorkflow, 
                             context: Dict[str, Any],
                             progress_callback: Optional[callable] = None) -> WorkflowExecution:
        """執行工作流"""
        
        self.processing_stats['workflow_executions'] += 1
        
        return await self.workflow_executor.execute_workflow(
            workflow, context, progress_callback
        )
    
    def get_command_completions(self, incomplete: str, 
                              current_args: List[str] = None,
                              context: Dict[str, Any] = None) -> List[str]:
        """獲取命令補全"""
        return self.auto_completer.complete_command(incomplete, current_args)
    
    def get_smart_completions(self, incomplete: str, 
                            context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """獲取智能補全"""
        return self.auto_completer.get_smart_completions(incomplete, context)
    
    def get_command_suggestions(self, user_input: str, 
                              context: Dict[str, Any]) -> List[Dict[str, str]]:
        """獲取命令建議"""
        
        # 結合自動補全和快捷方式的建議
        completions = self.auto_completer.get_command_suggestions(context)
        shortcuts = self.quick_commands.get_shortcut_suggestions(user_input, context)
        
        # 合併建議
        all_suggestions = []
        
        # 添加補全建議
        for suggestion in completions:
            all_suggestions.append({
                'type': 'completion',
                'command': suggestion['command'],
                'description': suggestion['description']
            })
        
        # 添加快捷方式建議
        for suggestion in shortcuts:
            all_suggestions.append({
                'type': 'shortcut',
                'command': suggestion['command'],
                'description': suggestion['description']
            })
        
        return all_suggestions[:10]  # 最多返回10個建議
    
    def get_workflow_templates(self) -> List[Dict[str, Any]]:
        """獲取可用的工作流模板"""
        return self.workflow_generator.list_available_templates()
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """獲取活躍的工作流執行"""
        return self.workflow_executor.list_active_executions()
    
    def get_workflow_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """獲取工作流執行歷史"""
        return self.workflow_executor.list_execution_history(limit)
    
    def add_user_shortcut(self, shortcut: str, command: str, description: str = "") -> bool:
        """添加用戶自定義快捷方式"""
        return self.quick_commands.add_user_shortcut(shortcut, command, description)
    
    def list_shortcuts(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """列出可用快捷方式"""
        return self.quick_commands.list_available_shortcuts(context)
    
    def get_intelligence_statistics(self) -> Dict[str, Any]:
        """獲取智能化功能統計"""
        
        return {
            'processing_stats': self.processing_stats.copy(),
            'nlp_parser_stats': {
                'command_history_length': len(self.nlp_parser.command_history)
            },
            'workflow_stats': self.workflow_executor.get_execution_statistics(),
            'completion_stats': self.auto_completer.get_completion_statistics(),
            'shortcut_stats': self.quick_commands.get_usage_statistics()
        }
    
    def reset_statistics(self):
        """重置統計信息"""
        self.processing_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'natural_language_requests': 0,
            'workflow_executions': 0,
            'shortcut_expansions': 0
        }
        
        # 清理各組件的統計
        self.nlp_parser.clear_history()
        self.auto_completer.clear_history()
        self.quick_commands.clear_usage_stats()


def get_intelligence_manager() -> IntelligenceManager:
    """獲取智能化管理器實例"""
    if not hasattr(get_intelligence_manager, '_instance'):
        get_intelligence_manager._instance = IntelligenceManager()
    return get_intelligence_manager._instance


