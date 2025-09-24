"""
命令路由器
智能解析和路由用戶命令到相應的處理器
"""

import re
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# 嘗試導入核心RAG系統
try:
    from core.optimization.integrated_rag import get_optimized_rag_system
    from core.tools.manager import get_tool_manager
    from core.memory.context_manager import ContextManager
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# 導入智能化功能
try:
    from locallm.intelligence import get_intelligence_manager, IntelligenceResult
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False


@dataclass
class CommandResult:
    """命令執行結果"""
    success: bool
    output: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'output': self.output,
            'data': self.data,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata or {}
        }


class CommandRouter:
    """智能命令路由器"""
    
    def __init__(self, config):
        self.config = config
        
        # 初始化核心組件（如果可用）
        if RAG_AVAILABLE:
            self.rag_system = None  # 延遲初始化
            self.tool_manager = None  # 延遲初始化
            self.context_manager = None  # 延遲初始化
        
        # 初始化智能化管理器（如果可用）
        if INTELLIGENCE_AVAILABLE:
            self.intelligence_manager = get_intelligence_manager()
        else:
            self.intelligence_manager = None
        
        # 命令模式匹配
        self.command_patterns = {
            # 分析命令
            'analyze': [
                r'^analyze\s*(.*)?$',
                r'^分析\s*(.*)?$',
                r'^檢查\s*(.*)?$'
            ],
            
            # 聊天命令
            'chat': [
                r'^chat\s+(.+)$',
                r'^ask\s+(.+)$',
                r'^問\s+(.+)$',
                r'^請問\s+(.+)$'
            ],
            
            # 工具命令
            'tools': [
                r'^tools?\s+(.+)$',
                r'^tool\s+(.+)$',
                r'^工具\s+(.+)$'
            ],
            
            # 記憶命令
            'memory': [
                r'^memory\s+(.+)$',
                r'^記憶\s+(.+)$',
                r'^記住\s+(.+)$'
            ],
            
            # 配置命令
            'config': [
                r'^config\s+(.+)$',
                r'^設定\s+(.+)$',
                r'^配置\s+(.+)$'
            ],
            
            # 幫助命令
            'help': [
                r'^help\s*(.*)?$',
                r'^幫助\s*(.*)?$',
                r'^説明\s*(.*)?$'
            ]
        }
        
        # 自然語言模式
        self.natural_language_patterns = [
            r'^.*\.csv.*$',  # 包含CSV的描述
            r'^.*分析.*報告.*$',  # 分析和報告
            r'^.*總結.*$',  # 總結類
            r'^.*生成.*$',  # 生成類
            r'^.*處理.*$',  # 處理類
            r'^.*讀取.*$',  # 讀取類
        ]
    
    async def route_command(self, args, context: Dict[str, Any], working_dir: Path) -> Dict[str, Any]:
        """路由命令到相應處理器"""
        start_time = datetime.now()
        
        try:
            # 組合命令文本
            command_text = ' '.join(args.command) if args.command else ''
            
            # 如果沒有命令，顯示幫助
            if not command_text:
                return await self._handle_help([], context)
            
            # 如果智能化管理器可用，優先使用智能處理
            if self.intelligence_manager:
                intelligence_result = await self.intelligence_manager.process_user_input(
                    command_text, context
                )
                
                if intelligence_result.success:
                    result = await self._handle_intelligence_result(
                        intelligence_result, context, working_dir
                    )
                else:
                    # 智能處理失敗，回退到傳統處理
                    result = await self._fallback_to_traditional_processing(
                        command_text, context, working_dir
                    )
            else:
                # 沒有智能化管理器，使用傳統處理
                result = await self._fallback_to_traditional_processing(
                    command_text, context, working_dir
                )
            
            # 計算執行時間
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time'] = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return CommandResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            ).to_dict()
    
    def _match_structured_command(self, command_text: str) -> tuple[Optional[str], Optional[str]]:
        """匹配結構化命令"""
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.match(pattern, command_text, re.IGNORECASE)
                if match:
                    args = match.group(1) if match.groups() else ''
                    return command_type, args.strip()
        
        return None, None
    
    async def _execute_structured_command(self, command_type: str, command_args: str, 
                                        context: Dict[str, Any], working_dir: Path) -> Dict[str, Any]:
        """執行結構化命令"""
        
        if command_type == 'analyze':
            return await self._handle_analyze(command_args, context, working_dir)
        elif command_type == 'chat':
            return await self._handle_chat(command_args, context)
        elif command_type == 'tools':
            return await self._handle_tools(command_args, context)
        elif command_type == 'memory':
            return await self._handle_memory(command_args, context)
        elif command_type == 'config':
            return await self._handle_config(command_args, context)
        elif command_type == 'help':
            return await self._handle_help(command_args.split() if command_args else [], context)
        else:
            return CommandResult(
                success=False,
                error=f"未支持的命令類型: {command_type}"
            ).to_dict()
    
    async def _execute_natural_language_command(self, command_text: str, 
                                              context: Dict[str, Any], working_dir: Path) -> Dict[str, Any]:
        """執行自然語言命令"""
        if not RAG_AVAILABLE:
            return CommandResult(
                success=False,
                error="自然語言處理功能不可用，請檢查RAG系統是否正確安裝"
            ).to_dict()
        
        try:
            # 初始化RAG系統（如果還未初始化）
            if self.rag_system is None:
                self.rag_system = await get_optimized_rag_system()
            
            # 使用RAG系統處理自然語言命令
            result = await self.rag_system.query(
                question=command_text,
                include_memory_context=True,
                search_strategy="hybrid"
            )
            
            return CommandResult(
                success=True,
                output=result.answer,
                data={
                    'confidence': result.confidence,
                    'sources': [{'id': s.id, 'content': s.content[:100]} for s in result.sources],
                    'retrieval_time': result.retrieval_time,
                    'generation_time': result.generation_time
                },
                metadata={'type': 'natural_language', 'strategy': 'rag'}
            ).to_dict()
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"自然語言處理失敗: {str(e)}"
            ).to_dict()
    
    async def _handle_analyze(self, args: str, context: Dict[str, Any], working_dir: Path) -> Dict[str, Any]:
        """處理分析命令"""
        target = args or '.'  # 默認分析當前目錄
        
        # 基於上下文提供分析
        if 'project_info' in context:
            project_info = context['project_info']
            analysis_result = {
                'project_type': project_info.get('primary_type', 'unknown'),
                'confidence': project_info.get('confidence_scores', {}),
                'detected_features': project_info.get('detected_features', []),
                'working_directory': str(working_dir)
            }
            
            if 'file_summary' in context:
                analysis_result['file_summary'] = context['file_summary']
            
            if 'git_info' in context:
                analysis_result['git_info'] = context['git_info']
            
            # 生成分析報告
            report = self._generate_analysis_report(analysis_result)
            
            return CommandResult(
                success=True,
                output=report,
                data=analysis_result,
                metadata={'type': 'analysis', 'target': target}
            ).to_dict()
        else:
            return CommandResult(
                success=False,
                error="無法獲取項目上下文信息"
            ).to_dict()
    
    def _generate_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """生成分析報告"""
        report_lines = [
            "[統計] 項目分析報告",
            "=" * 40,
            f"[文件夾] 目錄: {analysis['working_directory']}",
            f"[目標] 項目類型: {analysis['project_type']}"
        ]
        
        # 文件統計
        if 'file_summary' in analysis:
            file_summary = analysis['file_summary']
            report_lines.extend([
                "",
                "[列表] 文件統計:",
                f"  • 文件總數: {file_summary.get('total_files', 0)}",
                f"  • 目錄總數: {file_summary.get('total_directories', 0)}"
            ])
            
            # 文件類型分布
            if 'file_types' in file_summary:
                report_lines.append("  • 文件類型分布:")
                for ext, count in sorted(file_summary['file_types'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
                    ext_name = ext if ext else '無擴展名'
                    report_lines.append(f"    - {ext_name}: {count}")
        
        # Git信息
        if 'git_info' in analysis:
            git_info = analysis['git_info']
            if git_info.get('is_git_repo'):
                report_lines.extend([
                    "",
                    "[並行] Git 信息:",
                    f"  • 當前分支: {git_info.get('current_branch', 'unknown')}",
                    f"  • 是否有未提交更改: {'是' if git_info.get('has_uncommitted_changes') else '否'}"
                ])
        
        return "\n".join(report_lines)
    
    async def _handle_chat(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """處理聊天命令"""
        if not RAG_AVAILABLE:
            return CommandResult(
                success=False,
                error="聊天功能不可用，請檢查RAG系統是否正確安裝"
            ).to_dict()
        
        try:
            # 初始化RAG系統
            if self.rag_system is None:
                self.rag_system = await get_optimized_rag_system()
            
            # 增強問題上下文
            enhanced_question = f"{question}\n\n當前環境: {context.get('working_directory', '未知')}"
            
            result = await self.rag_system.query(
                question=enhanced_question,
                include_memory_context=True,
                search_strategy="hybrid"
            )
            
            return CommandResult(
                success=True,
                output=result.answer,
                data={
                    'confidence': result.confidence,
                    'sources': len(result.sources)
                },
                metadata={'type': 'chat', 'question': question}
            ).to_dict()
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"聊天處理失敗: {str(e)}"
            ).to_dict()
    
    async def _handle_tools(self, args: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """處理工具命令"""
        if not RAG_AVAILABLE:
            return CommandResult(
                success=False,
                error="工具功能不可用，請檢查工具系統是否正確安裝"
            ).to_dict()
        
        try:
            # 初始化工具管理器
            if self.tool_manager is None:
                self.tool_manager = get_tool_manager()
            
            if args == 'list' or args == '列表':
                # 列出所有工具
                tools = self.tool_manager.get_available_tools()
                
                tool_list = []
                for tool in tools:
                    tool_list.append(f"• {tool['name']}: {tool['description']}")
                
                return CommandResult(
                    success=True,
                    output=f"[工具] 可用工具:\n" + "\n".join(tool_list),
                    data={'tools': tools},
                    metadata={'type': 'tools', 'action': 'list'}
                ).to_dict()
            
            else:
                # 嘗試解析和執行工具命令
                parts = args.split()
                if len(parts) >= 2:
                    tool_name = parts[0]
                    operation = parts[1]
                    
                    result = await self.tool_manager.execute_tool(tool_name, operation=operation)
                    
                    return CommandResult(
                        success=result.success,
                        output=f"工具執行結果: {result.data}" if result.success else f"執行失敗: {result.error}",
                        data=result.data,
                        error=result.error if not result.success else None,
                        metadata={'type': 'tools', 'action': 'execute', 'tool': tool_name}
                    ).to_dict()
                else:
                    return CommandResult(
                        success=False,
                        error="工具命令格式錯誤。使用: tools <工具名> <操作>"
                    ).to_dict()
                    
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"工具處理失敗: {str(e)}"
            ).to_dict()
    
    async def _handle_memory(self, args: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """處理記憶命令"""
        if not RAG_AVAILABLE:
            return CommandResult(
                success=False,
                error="記憶功能不可用，請檢查記憶系統是否正確安裝"
            ).to_dict()
        
        try:
            # 初始化上下文管理器
            if self.context_manager is None:
                self.context_manager = ContextManager()
            
            parts = args.split(' ', 1)
            action = parts[0]
            content = parts[1] if len(parts) > 1 else ''
            
            if action in ['save', '保存']:
                if not content:
                    return CommandResult(
                        success=False,
                        error="請提供要保存的內容"
                    ).to_dict()
                
                # 保存到記憶
                from core.memory.base import MemoryImportance
                await self.context_manager.long_term_memory.store(
                    content,
                    MemoryImportance.MEDIUM,
                    {'source': 'user_command', 'context': context.get('working_directory')},
                    ['user_input', 'important']
                )
                
                return CommandResult(
                    success=True,
                    output=f"[成功] 已保存到記憶: {content[:50]}{'...' if len(content) > 50 else ''}",
                    metadata={'type': 'memory', 'action': 'save'}
                ).to_dict()
            
            elif action in ['stats', '統計']:
                # 獲取記憶統計
                session_count = len(self.context_manager.session_memory.interactions)
                
                return CommandResult(
                    success=True,
                    output=f"[統計] 記憶統計:\n• 會話記憶: {session_count} 條交互記錄",
                    data={'session_interactions': session_count},
                    metadata={'type': 'memory', 'action': 'stats'}
                ).to_dict()
            
            else:
                return CommandResult(
                    success=False,
                    error=f"未知的記憶操作: {action}"
                ).to_dict()
                
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"記憶處理失敗: {str(e)}"
            ).to_dict()
    
    async def _handle_config(self, args: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """處理配置命令"""
        parts = args.split()
        action = parts[0] if parts else 'show'
        
        if action in ['show', '顯示']:
            config_info = {
                'ollama_host': getattr(self.config, 'ollama_host', 'http://localhost:11434'),
                'chat_model': getattr(self.config, 'chat_model', 'qwen3:latest'),
                'embedding_model': getattr(self.config, 'embedding_model', 'embeddinggemma:latest'),
                'default_output_format': getattr(self.config, 'default_output_format', 'rich'),
                'auto_detect_project': getattr(self.config, 'auto_detect_project', True)
            }
            
            config_text = "[配置] 當前配置:\n"
            for key, value in config_info.items():
                config_text += f"• {key}: {value}\n"
            
            return CommandResult(
                success=True,
                output=config_text,
                data=config_info,
                metadata={'type': 'config', 'action': 'show'}
            ).to_dict()
        
        else:
            return CommandResult(
                success=False,
                error=f"未知的配置操作: {action}"
            ).to_dict()
    
    async def _handle_help(self, args: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """處理幫助命令"""
        help_text = """
🤖 LocalLM - 本地AI智能助理

[列表] 基本命令:
• analyze [目標]           - 分析項目或文件
• chat "問題"             - 與AI對話
• tools list              - 列出可用工具
• memory save "內容"      - 保存到記憶
• config show             - 顯示配置

🌟 自然語言支持:
• "讀取data.csv並生成報告"
• "分析這個Python項目"
• "總結當前目錄的文檔"

[修復] 選項:
• --show-context          - 顯示環境上下文
• --output json           - JSON格式輸出
• --help                  - 顯示幫助
• --version               - 顯示版本

[提示] 提示: LocalLM會自動檢測您的項目類型和上下文，提供更智能的協助。
        """
        
        return CommandResult(
            success=True,
            output=help_text.strip(),
            metadata={'type': 'help'}
        ).to_dict()
    
    async def _handle_intelligence_result(self, intelligence_result: 'IntelligenceResult', 
                                        context: Dict[str, Any], working_dir: Path) -> Dict[str, Any]:
        """處理智能化結果"""
        
        if intelligence_result.result_type == 'answer':
            # 直接回答類型
            return CommandResult(
                success=True,
                output=intelligence_result.content.get('answer', '處理完成'),
                data=intelligence_result.content,
                metadata={'intelligence_used': True, 'confidence': intelligence_result.confidence}
            ).to_dict()
        
        elif intelligence_result.result_type == 'workflow':
            # 工作流類型
            workflow_data = intelligence_result.content.get('workflow', {})
            
            if intelligence_result.content.get('ready_to_execute', False):
                # 詢問用戶是否執行工作流
                workflow_summary = self._generate_workflow_summary(workflow_data)
                
                return CommandResult(
                    success=True,
                    output=f"已為您生成智能工作流:\n\n{workflow_summary}\n\n輸入 'y' 執行工作流，或 'n' 取消",
                    data={
                        'workflow': workflow_data,
                        'requires_confirmation': True,
                        'action': 'execute_workflow'
                    },
                    metadata={'intelligence_used': True, 'confidence': intelligence_result.confidence}
                ).to_dict()
            else:
                return CommandResult(
                    success=True,
                    output="工作流已生成但需要進一步配置",
                    data=workflow_data,
                    metadata={'intelligence_used': True}
                ).to_dict()
        
        elif intelligence_result.result_type == 'command':
            # 命令類型，重新路由執行
            command_data = intelligence_result.content
            return CommandResult(
                success=True,
                output=f"智能解析為命令: {command_data.get('command', '')}",
                data=command_data,
                metadata={'intelligence_used': True, 'confidence': intelligence_result.confidence}
            ).to_dict()
        
        elif intelligence_result.result_type == 'suggestion':
            # 建議類型
            suggestions = intelligence_result.content.get('suggestions', [])
            suggestion_text = "\n".join([f"• {s.get('command', '')}: {s.get('description', '')}" 
                                       for s in suggestions])
            
            return CommandResult(
                success=True,
                output=f"{intelligence_result.content.get('message', '以下是建議:')}\n\n{suggestion_text}",
                data=intelligence_result.content,
                metadata={'intelligence_used': True, 'needs_clarification': True}
            ).to_dict()
        
        else:
            # 其他類型
            return CommandResult(
                success=True,
                output=str(intelligence_result.content),
                data=intelligence_result.content,
                metadata={'intelligence_used': True}
            ).to_dict()
    
    def _generate_workflow_summary(self, workflow_data: Dict[str, Any]) -> str:
        """生成工作流摘要"""
        summary_lines = []
        
        summary_lines.append(f"🔧 工作流: {workflow_data.get('name', 'Unknown')}")
        summary_lines.append(f"📝 描述: {workflow_data.get('description', 'No description')}")
        
        steps = workflow_data.get('steps', [])
        if steps:
            summary_lines.append(f"📋 步驟 ({len(steps)}個):")
            for i, step in enumerate(steps, 1):
                step_name = step.get('purpose', step.get('operation', 'Unknown'))
                summary_lines.append(f"   {i}. {step_name}")
        
        estimated_time = workflow_data.get('estimated_time', 0)
        if estimated_time > 0:
            summary_lines.append(f"⏱️ 預計時間: {estimated_time}秒")
        
        confidence = workflow_data.get('confidence', 0)
        summary_lines.append(f"🎯 置信度: {confidence:.1%}")
        
        return "\n".join(summary_lines)
    
    async def _fallback_to_traditional_processing(self, command_text: str, 
                                                context: Dict[str, Any], working_dir: Path) -> Dict[str, Any]:
        """回退到傳統處理方式"""
        
        # 嘗試匹配結構化命令
        command_type, command_args = self._match_structured_command(command_text)
        
        if command_type:
            result = await self._execute_structured_command(
                command_type, command_args, context, working_dir
            )
        else:
            # 嘗試作為自然語言處理
            result = await self._execute_natural_language_command(
                command_text, context, working_dir
            )
        
        # 添加回退標記
        if isinstance(result, dict) and 'metadata' in result:
            result['metadata']['fallback_used'] = True
        
        return result
    
    def get_command_completions(self, incomplete: str, current_args: List[str] = None) -> List[str]:
        """獲取命令補全建議"""
        
        if self.intelligence_manager:
            return self.intelligence_manager.get_command_completions(incomplete, current_args)
        else:
            # 基本補全
            completions = []
            
            # 如果沒有當前參數，補全主命令
            if not current_args:
                for command in self.command_patterns.keys():
                    if command.startswith(incomplete.lower()):
                        completions.append(command)
            
            return sorted(completions)
    
    def get_command_suggestions(self, user_input: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """獲取命令建議"""
        
        if self.intelligence_manager:
            return self.intelligence_manager.get_command_suggestions(user_input, context)
        else:
            # 基本建議
            return [
                {'command': 'analyze .', 'description': '分析當前目錄'},
                {'command': 'help', 'description': '顯示幫助信息'}
            ]
    
    def get_intelligence_statistics(self) -> Dict[str, Any]:
        """獲取智能化統計信息"""
        
        if self.intelligence_manager:
            return self.intelligence_manager.get_intelligence_statistics()
        else:
            return {'intelligence_available': False}
