"""
智能命令補全系統

提供基於上下文的智能命令補全和建議
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class AutoCompleter:
    """智能命令補全系統"""
    
    # 主命令補全
    MAIN_COMMANDS = {
        'analyze': '分析項目、文件或數據',
        'chat': '與AI智能對話',
        'tools': '工具管理和執行',
        'memory': '記憶系統管理',
        'config': '配置管理',
        'workflow': '工作流管理',
        'help': '顯示幫助信息'
    }
    
    # 子命令補全
    SUBCOMMANDS = {
        'analyze': {
            'project': '分析項目結構',
            'data': '分析數據文件',
            'files': '分析文件結構',
            'structure': '分析目錄結構',
            'dependencies': '分析依賴關係',
            'quality': '代碼質量分析',
            'performance': '性能分析'
        },
        'chat': {
            'ask': '提問',
            'explain': '解釋說明',
            'help': '獲取幫助',
            'translate': '翻譯內容',
            'summarize': '總結內容'
        },
        'tools': {
            'list': '列出所有工具',
            'execute': '執行工具',
            'stats': '工具使用統計',
            'demo': '工具演示',
            'workflow': '工具工作流',
            'export': '導出工具結果'
        },
        'memory': {
            'save': '保存到記憶',
            'recall': '回憶內容',
            'stats': '記憶統計',
            'export': '導出記憶',
            'clear': '清空記憶',
            'search': '搜索記憶'
        },
        'config': {
            'show': '顯示配置',
            'set': '設置配置',
            'reset': '重置配置',
            'edit': '編輯配置',
            'validate': '驗證配置',
            'backup': '備份配置'
        },
        'workflow': {
            'create': '創建工作流',
            'run': '運行工作流',
            'list': '列出工作流',
            'edit': '編輯工作流',
            'delete': '刪除工作流',
            'status': '工作流狀態',
            'history': '執行歷史'
        }
    }
    
    # 參數補全
    PARAMETERS = {
        '--output': ['json', 'yaml', 'csv', 'markdown', 'plain'],
        '--format': ['json', 'yaml', 'csv', 'markdown', 'plain'],
        '--depth': ['1', '2', '3', '5', '10'],
        '--limit': ['5', '10', '20', '50', '100'],
        '--sort': ['name', 'size', 'time', 'type'],
        '--filter': ['*.py', '*.js', '*.md', '*.txt', '*.csv'],
        '--verbose': [],
        '--quiet': [],
        '--help': [],
        '--version': []
    }
    
    # 文件擴展名
    COMMON_EXTENSIONS = [
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.rs', '.go',
        '.csv', '.json', '.xml', '.yaml', '.yml', '.toml',
        '.md', '.txt', '.pdf', '.docx', '.html', '.css',
        '.png', '.jpg', '.jpeg', '.gif', '.svg',
        '.zip', '.tar', '.gz', '.sql', '.db'
    ]
    
    def __init__(self, context_detector=None):
        self.context_detector = context_detector
        self.completion_cache = {}
        self.user_history = []
    
    def complete_command(self, incomplete: str, current_args: List[str] = None) -> List[str]:
        """主要的命令補全入口"""
        current_args = current_args or []
        incomplete = incomplete.strip()
        
        # 參數補全優先（不論上下文）
        if incomplete.startswith('-'):
            main_command = current_args[0].lower() if current_args else None
            return self._complete_parameter(incomplete, main_command)
        
        # 如果沒有參數，補全主命令
        if not current_args:
            return self._complete_main_command(incomplete)
        
        main_command = current_args[0].lower()
        
        # 如果只有一個參數，補全子命令
        if len(current_args) == 1:
            return self._complete_subcommand(main_command, incomplete)
        
        # 文件路徑補全
        return self._complete_file_path(incomplete, main_command)
    
    def _complete_main_command(self, incomplete: str) -> List[str]:
        """補全主命令"""
        matches = []
        incomplete_lower = incomplete.lower()
        
        for command, description in self.MAIN_COMMANDS.items():
            if command.startswith(incomplete_lower):
                matches.append(command)
        
        # 如果沒有匹配，提供模糊匹配
        if not matches:
            for command in self.MAIN_COMMANDS.keys():
                if incomplete_lower in command:
                    matches.append(command)
        
        return sorted(matches)
    
    def _complete_subcommand(self, main_command: str, incomplete: str) -> List[str]:
        """補全子命令"""
        if main_command not in self.SUBCOMMANDS:
            return []
        
        matches = []
        incomplete_lower = incomplete.lower()
        subcommands = self.SUBCOMMANDS[main_command]
        
        for subcommand, description in subcommands.items():
            if subcommand.startswith(incomplete_lower):
                matches.append(subcommand)
        
        return sorted(matches)
    
    def _complete_parameter(self, incomplete: str, main_command: str) -> List[str]:
        """補全參數"""
        matches = []
        
        for param, values in self.PARAMETERS.items():
            if param.startswith(incomplete):
                if values:
                    # 如果參數有預定義值，返回參數=值的形式
                    matches.extend([f"{param}={value}" for value in values])
                else:
                    # 否則只返回參數名
                    matches.append(param)
        
        return sorted(matches)
    
    def _complete_file_path(self, incomplete: str, main_command: str = None) -> List[str]:
        """補全文件路徑"""
        try:
            # 處理路徑
            if incomplete.startswith(('/', '\\', '~')):
                # 絕對路徑
                base_path = Path(incomplete).expanduser().parent
                prefix = Path(incomplete).name
            else:
                # 相對路徑
                base_path = Path.cwd()
                prefix = incomplete
            
            # 如果路徑包含目錄分隔符，調整base_path
            if '/' in incomplete or '\\' in incomplete:
                parts = incomplete.replace('\\', '/').split('/')
                if len(parts) > 1:
                    base_path = Path(base_path / '/'.join(parts[:-1]))
                    prefix = parts[-1]
            
            if not base_path.exists():
                return []
            
            matches = []
            
            # 遍歷目錄內容
            for item in base_path.iterdir():
                if item.name.startswith('.') and not prefix.startswith('.'):
                    continue  # 跳過隱藏文件，除非用戶明確輸入
                
                if item.name.lower().startswith(prefix.lower()):
                    if item.is_dir():
                        matches.append(f"{item.name}/")
                    else:
                        # 根據主命令過濾文件類型
                        if self._should_include_file(item, main_command):
                            matches.append(item.name)
            
            return sorted(matches)
        
        except (PermissionError, OSError):
            return []
    
    def _should_include_file(self, file_path: Path, main_command: str = None) -> bool:
        """判斷是否應該包含該文件"""
        
        # 如果沒有主命令，包含所有文件
        if not main_command:
            return True
        
        file_ext = file_path.suffix.lower()
        
        # 基於命令類型過濾
        if main_command == 'analyze':
            if file_ext in ['.csv', '.json', '.xlsx', '.py', '.js', '.md']:
                return True
        
        elif main_command == 'tools':
            if file_ext in ['.csv', '.json', '.py', '.txt']:
                return True
        
        elif main_command == 'chat':
            if file_ext in ['.md', '.txt', '.pdf', '.docx']:
                return True
        
        # 默認包含常見文件類型
        return file_ext in self.COMMON_EXTENSIONS
    
    def get_command_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """基於上下文獲取命令建議"""
        suggestions = []
        
        # 基於項目類型的建議
        project_type = context.get('project_info', {}).get('project_type', 'unknown')
        
        if project_type == 'data_science':
            suggestions.extend([
                {'command': 'analyze data', 'description': '分析數據文件'},
                {'command': 'tools execute --tool data_analysis', 'description': '執行數據分析工具'},
                {'command': 'workflow create data_analysis_pipeline', 'description': '創建數據分析工作流'}
            ])
        
        elif project_type == 'python':
            suggestions.extend([
                {'command': 'analyze project', 'description': '分析Python項目'},
                {'command': 'analyze dependencies', 'description': '檢查依賴關係'},
                {'command': 'tools execute --tool test_generation', 'description': '生成測試用例'}
            ])
        
        elif project_type == 'javascript':
            suggestions.extend([
                {'command': 'analyze project', 'description': '分析JavaScript項目'},
                {'command': 'analyze dependencies', 'description': '檢查npm依賴'}
            ])
        
        # 基於文件類型的建議
        file_summary = context.get('file_summary', {})
        file_types = file_summary.get('file_types', {})
        
        if '.csv' in file_types:
            suggestions.append({
                'command': 'analyze data *.csv',
                'description': '分析CSV數據文件'
            })
        
        if '.md' in file_types:
            suggestions.append({
                'command': 'chat "總結這些Markdown文檔"',
                'description': '總結Markdown文檔'
            })
        
        if '.py' in file_types:
            suggestions.append({
                'command': 'analyze quality',
                'description': '分析Python代碼質量'
            })
        
        # 基於最近命令的建議
        if self.user_history:
            last_command = self.user_history[-1]
            if 'analyze' in last_command:
                suggestions.append({
                    'command': 'memory save "分析結果"',
                    'description': '保存分析結果到記憶'
                })
        
        # 去重並限制數量
        unique_suggestions = []
        seen_commands = set()
        
        for suggestion in suggestions:
            if suggestion['command'] not in seen_commands:
                unique_suggestions.append(suggestion)
                seen_commands.add(suggestion['command'])
        
        return unique_suggestions[:8]  # 最多返回8個建議
    
    def get_smart_completions(self, incomplete: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """獲取智能補全建議"""
        completions = []
        
        # 獲取基本補全
        basic_completions = self.complete_command(incomplete)
        
        for completion in basic_completions:
            completion_info = {
                'text': completion,
                'type': 'command',
                'description': self._get_completion_description(completion),
                'priority': self._calculate_completion_priority(completion, context)
            }
            completions.append(completion_info)
        
        # 添加上下文相關的補全
        context_completions = self._get_context_completions(incomplete, context)
        completions.extend(context_completions)
        
        # 排序並返回
        completions.sort(key=lambda x: x['priority'], reverse=True)
        return completions[:10]  # 最多返回10個
    
    def _get_completion_description(self, completion: str) -> str:
        """獲取補全項的描述"""
        
        # 主命令描述
        if completion in self.MAIN_COMMANDS:
            return self.MAIN_COMMANDS[completion]
        
        # 子命令描述
        for main_cmd, subcmds in self.SUBCOMMANDS.items():
            if completion in subcmds:
                return subcmds[completion]
        
        # 文件/目錄描述
        if completion.endswith('/'):
            return '目錄'
        elif '.' in completion:
            ext = completion.split('.')[-1].lower()
            ext_descriptions = {
                'py': 'Python文件',
                'js': 'JavaScript文件',
                'csv': 'CSV數據文件',
                'json': 'JSON文件',
                'md': 'Markdown文檔',
                'txt': '文本文件'
            }
            return ext_descriptions.get(ext, '文件')
        
        return '項目'
    
    def _calculate_completion_priority(self, completion: str, context: Dict[str, Any]) -> float:
        """計算補全項的優先級"""
        priority = 1.0
        
        # 基於項目類型調整優先級
        project_type = context.get('project_info', {}).get('project_type', 'unknown')
        
        if project_type == 'data_science':
            if completion in ['analyze', 'data', 'tools']:
                priority += 0.5
        
        elif project_type == 'python':
            if completion in ['analyze', 'project', 'dependencies']:
                priority += 0.5
        
        # 基於使用歷史調整優先級
        if completion in self.user_history:
            priority += 0.3
        
        # 基於文件存在性調整優先級
        if not completion.startswith('-') and Path(completion).exists():
            priority += 0.2
        
        return priority
    
    def _get_context_completions(self, incomplete: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """獲取上下文相關的補全"""
        completions = []
        
        # 如果用戶輸入看起來像自然語言，提供自然語言建議
        if len(incomplete.split()) > 1 and not incomplete.startswith('-'):
            natural_suggestions = [
                {
                    'text': f'chat "{incomplete}"',
                    'type': 'natural_language',
                    'description': '作為問題向AI提問',
                    'priority': 0.8
                },
                {
                    'text': f'workflow create --from-text "{incomplete}"',
                    'type': 'natural_language',
                    'description': '從自然語言創建工作流',
                    'priority': 0.6
                }
            ]
            completions.extend(natural_suggestions)
        
        return completions
    
    def add_to_history(self, command: str):
        """添加命令到歷史記錄"""
        self.user_history.append(command)
        if len(self.user_history) > 50:  # 保持最近50條記錄
            self.user_history.pop(0)
    
    def clear_history(self):
        """清空歷史記錄"""
        self.user_history.clear()
    
    def get_completion_statistics(self) -> Dict[str, Any]:
        """獲取補全統計信息"""
        return {
            'total_commands': len(self.user_history),
            'unique_commands': len(set(self.user_history)),
            'most_used_commands': self._get_most_used_commands(),
            'cache_size': len(self.completion_cache)
        }
    
    def _get_most_used_commands(self, limit: int = 5) -> List[Dict[str, Any]]:
        """獲取最常用的命令"""
        from collections import Counter
        
        # 提取主命令
        main_commands = []
        for cmd in self.user_history:
            parts = cmd.split()
            if parts:
                main_commands.append(parts[0])
        
        command_counts = Counter(main_commands)
        most_common = command_counts.most_common(limit)
        
        return [
            {'command': cmd, 'count': count, 'percentage': (count / len(main_commands)) * 100}
            for cmd, count in most_common
        ]


def get_auto_completer() -> AutoCompleter:
    """獲取自動補全器實例"""
    if not hasattr(get_auto_completer, '_instance'):
        get_auto_completer._instance = AutoCompleter()
    return get_auto_completer._instance

