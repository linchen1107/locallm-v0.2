"""
快捷命令系統

提供快捷命令、別名和智能展開功能
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class QuickCommands:
    """快捷命令系統"""
    
    # 靜態快捷方式
    STATIC_SHORTCUTS = {
        # 分析快捷方式
        'a': 'analyze .',
        'ap': 'analyze project',
        'ad': 'analyze data',
        'af': 'analyze files',
        'as': 'analyze structure',
        'aq': 'analyze quality',
        'adep': 'analyze dependencies',
        
        # 工具快捷方式
        't': 'tools list',
        'te': 'tools execute',
        'td': 'tools demo',
        'ts': 'tools stats',
        'tw': 'tools workflow',
        
        # 記憶快捷方式
        'm': 'memory stats',
        'ms': 'memory save',
        'mr': 'memory recall',
        'me': 'memory export',
        'mc': 'memory clear',
        'mse': 'memory search',
        
        # 聊天快捷方式
        'c': 'chat',
        'ca': 'chat ask',
        'ce': 'chat explain',
        'ch': 'chat help',
        'cs': 'chat summarize',
        
        # 配置快捷方式
        'cfg': 'config show',
        'cfgs': 'config set',
        'cfgr': 'config reset',
        'cfge': 'config edit',
        
        # 工作流快捷方式
        'w': 'workflow list',
        'wc': 'workflow create',
        'wr': 'workflow run',
        'ws': 'workflow status',
        'wh': 'workflow history',
        
        # 通用快捷方式
        'h': 'help',
        '?': 'help',
        'q': 'quit',
        'exit': 'quit',
        'clear': 'clear',
        'cls': 'clear',
        
        # 輸出格式快捷方式
        'json': '--output json',
        'yaml': '--output yaml',
        'csv': '--output csv',
        'md': '--output markdown',
        
        # 常用參數組合
        'v': '--verbose',
        'vv': '--verbose --depth 5',
        'q': '--quiet',
        'all': '--depth 999 --limit 999'
    }
    
    # 文件類型處理器
    FILE_TYPE_HANDLERS = {
        'csv': lambda file: f'tools execute --tool data_analysis --operation describe "{file}"',
        'json': lambda file: f'tools execute --tool data_analysis --operation load "{file}"',
        'py': lambda file: f'analyze quality "{file}"',
        'js': lambda file: f'analyze project',
        'md': lambda file: f'chat "總結這個文檔：{file}"',
        'txt': lambda file: f'chat "分析這個文件：{file}"',
        'pdf': lambda file: f'chat "總結這個PDF：{file}"',
        'xlsx': lambda file: f'tools execute --tool data_analysis --operation load "{file}"',
        'sql': lambda file: f'tools execute --tool data_analysis --operation query "{file}"'
    }
    
    # 智能模式
    SMART_PATTERNS = {
        # 數字快捷方式 - 重複歷史命令
        r'^\d+$': '_handle_history_number',
        
        # 文件擴展名快捷方式
        r'^\.(\w+)$': '_handle_file_extension',
        
        # 目錄快捷方式
        r'^/(.*)': '_handle_directory_shortcut',
        r'^~/(.*)': '_handle_home_directory',
        
        # 搜索模式
        r'^find\s+(.+)': '_handle_find_command',
        r'^search\s+(.+)': '_handle_search_command',
        
        # 快速執行模式
        r'^!(.+)': '_handle_quick_execute',
        
        # 管道模式
        r'^(.+)\s*\|\s*(.+)': '_handle_pipe_command'
    }
    
    def __init__(self):
        self.user_shortcuts = {}  # 用戶自定義快捷方式
        self.command_history = []
        self.usage_stats = {}
        self.context_cache = {}
    
    def expand_shortcut(self, command: str, context: Optional[Dict[str, Any]] = None) -> str:
        """展開快捷命令"""
        if context is None:
            context = {}
        
        original_command = command
        command = command.strip()
        
        # 記錄使用統計
        self._record_usage(command)
        
        # 1. 檢查用戶自定義快捷方式
        if command in self.user_shortcuts:
            expanded = self.user_shortcuts[command]
            return self._apply_context_variables(expanded, context)
        
        # 2. 檢查靜態快捷方式
        if command in self.STATIC_SHORTCUTS:
            expanded = self.STATIC_SHORTCUTS[command]
            return self._apply_context_variables(expanded, context)
        
        # 3. 檢查智能模式
        for pattern, handler_name in self.SMART_PATTERNS.items():
            match = re.match(pattern, command)
            if match:
                handler = getattr(self, handler_name)
                expanded = handler(match, context)
                if expanded != command:
                    return expanded
        
        # 4. 檢查組合快捷方式
        if ' ' in command:
            expanded = self._handle_composite_shortcut(command, context)
            if expanded != command:
                return expanded
        
        # 5. 檢查部分匹配
        expanded = self._handle_partial_match(command, context)
        if expanded != command:
            return expanded
        
        # 6. 如果沒有匹配，返回原始命令
        return original_command
    
    def _apply_context_variables(self, command: str, context: Dict[str, Any]) -> str:
        """應用上下文變量"""
        
        # 替換上下文變量
        variables = {
            '{pwd}': context.get('working_directory', '.'),
            '{project}': context.get('project_info', {}).get('name', '.'),
            '{user}': context.get('environment_info', {}).get('user', 'user'),
            '{date}': datetime.now().strftime('%Y-%m-%d'),
            '{time}': datetime.now().strftime('%H:%M:%S'),
            '{timestamp}': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        for var, value in variables.items():
            command = command.replace(var, str(value))
        
        return command
    
    def _handle_history_number(self, match, context: Dict[str, Any]) -> str:
        """處理數字快捷方式（歷史命令）"""
        number = int(match.group(0))
        
        if 1 <= number <= len(self.command_history):
            return self.command_history[-(number)]
        
        return match.group(0)  # 返回原始命令
    
    def _handle_file_extension(self, match, context: Dict[str, Any]) -> str:
        """處理文件擴展名快捷方式"""
        extension = match.group(1).lower()
        
        # 查找指定擴展名的文件
        files = self._find_files_by_extension(extension, context)
        
        if files and extension in self.FILE_TYPE_HANDLERS:
            # 如果只有一個文件，直接處理
            if len(files) == 1:
                return self.FILE_TYPE_HANDLERS[extension](files[0])
            
            # 如果有多個文件，處理第一個並列出所有
            primary_file = files[0]
            command = self.FILE_TYPE_HANDLERS[extension](primary_file)
            
            # 添加注釋顯示其他文件
            other_files = files[1:3]  # 最多顯示3個其他文件
            if other_files:
                comment = f" # 其他文件: {', '.join(other_files)}"
                command += comment
            
            return command
        
        return match.group(0)
    
    def _handle_directory_shortcut(self, match, context: Dict[str, Any]) -> str:
        """處理目錄快捷方式"""
        path = match.group(1)
        return f'analyze "{path}"'
    
    def _handle_home_directory(self, match, context: Dict[str, Any]) -> str:
        """處理家目錄快捷方式"""
        path = match.group(1)
        full_path = str(Path.home() / path)
        return f'analyze "{full_path}"'
    
    def _handle_find_command(self, match, context: Dict[str, Any]) -> str:
        """處理查找命令"""
        search_term = match.group(1)
        return f'tools execute --tool file_operation --operation search --pattern "{search_term}"'
    
    def _handle_search_command(self, match, context: Dict[str, Any]) -> str:
        """處理搜索命令"""
        search_term = match.group(1)
        return f'memory search "{search_term}"'
    
    def _handle_quick_execute(self, match, context: Dict[str, Any]) -> str:
        """處理快速執行模式"""
        command = match.group(1)
        return f'tools execute --command "{command}"'
    
    def _handle_pipe_command(self, match, context: Dict[str, Any]) -> str:
        """處理管道命令"""
        first_cmd = match.group(1).strip()
        second_cmd = match.group(2).strip()
        
        # 展開兩個命令
        expanded_first = self.expand_shortcut(first_cmd, context)
        expanded_second = self.expand_shortcut(second_cmd, context)
        
        return f'workflow create --steps "{expanded_first}" "{expanded_second}"'
    
    def _handle_composite_shortcut(self, command: str, context: Dict[str, Any]) -> str:
        """處理組合快捷方式"""
        parts = command.split()
        
        # 嘗試展開第一個部分
        first_part = parts[0]
        if first_part in self.STATIC_SHORTCUTS:
            expanded_first = self.STATIC_SHORTCUTS[first_part]
            remaining_parts = ' '.join(parts[1:])
            return f"{expanded_first} {remaining_parts}"
        
        return command
    
    def _handle_partial_match(self, command: str, context: Dict[str, Any]) -> str:
        """處理部分匹配"""
        
        # 查找以輸入開頭的快捷方式
        matches = []
        for shortcut in self.STATIC_SHORTCUTS.keys():
            if shortcut.startswith(command.lower()):
                matches.append(shortcut)
        
        # 如果只有一個匹配，使用它
        if len(matches) == 1:
            return self.STATIC_SHORTCUTS[matches[0]]
        
        return command
    
    def _find_files_by_extension(self, extension: str, context: Dict[str, Any]) -> List[str]:
        """查找指定擴展名的文件"""
        working_dir = Path(context.get('working_directory', '.'))
        
        try:
            # 查找文件
            pattern = f"*.{extension}"
            files = list(working_dir.glob(pattern))
            
            # 排序並轉換為字符串
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)  # 按修改時間排序
            return [str(f.name) for f in files[:10]]  # 最多返回10個文件
        
        except (PermissionError, OSError):
            return []
    
    def _record_usage(self, command: str):
        """記錄快捷方式使用統計"""
        self.usage_stats[command] = self.usage_stats.get(command, 0) + 1
    
    def add_user_shortcut(self, shortcut: str, command: str, description: str = "") -> bool:
        """添加用戶自定義快捷方式"""
        
        # 檢查是否與系統快捷方式衝突
        if shortcut in self.STATIC_SHORTCUTS:
            return False
        
        self.user_shortcuts[shortcut] = command
        return True
    
    def remove_user_shortcut(self, shortcut: str) -> bool:
        """移除用戶自定義快捷方式"""
        if shortcut in self.user_shortcuts:
            del self.user_shortcuts[shortcut]
            return True
        return False
    
    def list_available_shortcuts(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """列出當前可用的快捷方式"""
        if context is None:
            context = {}
        
        shortcuts = {
            'static': {},
            'dynamic': {},
            'user_defined': {},
            'context_specific': {}
        }
        
        # 靜態快捷方式
        shortcuts['static'] = self.STATIC_SHORTCUTS.copy()
        
        # 用戶定義快捷方式
        shortcuts['user_defined'] = self.user_shortcuts.copy()
        
        # 動態快捷方式（基於文件）
        for ext in self.FILE_TYPE_HANDLERS:
            files = self._find_files_by_extension(ext, context)
            if files:
                shortcuts['dynamic'][f'.{ext}'] = f'處理 {ext.upper()} 文件 ({len(files)} 個)'
        
        # 上下文特定快捷方式
        project_type = context.get('project_info', {}).get('project_type')
        if project_type:
            shortcuts['context_specific'][f'@{project_type}'] = f'針對 {project_type} 項目的快捷方式'
        
        return shortcuts
    
    def get_shortcut_suggestions(self, partial_input: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """獲取快捷方式建議"""
        suggestions = []
        
        # 基於部分輸入匹配
        for shortcut, command in self.STATIC_SHORTCUTS.items():
            if shortcut.startswith(partial_input.lower()):
                suggestions.append({
                    'shortcut': shortcut,
                    'command': command,
                    'type': 'static',
                    'description': self._get_shortcut_description(shortcut, command)
                })
        
        # 用戶定義的快捷方式
        for shortcut, command in self.user_shortcuts.items():
            if shortcut.startswith(partial_input.lower()):
                suggestions.append({
                    'shortcut': shortcut,
                    'command': command,
                    'type': 'user_defined',
                    'description': f'用戶定義：{command}'
                })
        
        # 動態快捷方式
        if partial_input.startswith('.'):
            ext = partial_input[1:]
            if ext in self.FILE_TYPE_HANDLERS:
                files = self._find_files_by_extension(ext, context)
                if files:
                    suggestions.append({
                        'shortcut': partial_input,
                        'command': f'處理 {ext.upper()} 文件',
                        'type': 'dynamic',
                        'description': f'找到 {len(files)} 個 {ext} 文件'
                    })
        
        return suggestions[:10]  # 最多返回10個建議
    
    def _get_shortcut_description(self, shortcut: str, command: str) -> str:
        """獲取快捷方式描述"""
        descriptions = {
            'a': '分析當前目錄',
            'ap': '分析項目結構',
            'ad': '分析數據文件',
            't': '列出所有工具',
            'm': '查看記憶統計',
            'c': '開始AI對話',
            'w': '列出工作流',
            'h': '顯示幫助',
            'cfg': '顯示配置'
        }
        
        return descriptions.get(shortcut, command)
    
    def add_to_history(self, command: str):
        """添加命令到歷史記錄"""
        # 只記錄原始命令，不記錄展開後的命令
        if command not in self.command_history or self.command_history[-1] != command:
            self.command_history.append(command)
            
        # 保持歷史記錄在合理大小
        if len(self.command_history) > 100:
            self.command_history.pop(0)
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """獲取使用統計信息"""
        total_usage = sum(self.usage_stats.values())
        
        # 最常用的快捷方式
        sorted_shortcuts = sorted(
            self.usage_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'total_shortcuts_used': total_usage,
            'unique_shortcuts': len(self.usage_stats),
            'most_used': sorted_shortcuts[:10],
            'user_shortcuts_count': len(self.user_shortcuts),
            'static_shortcuts_count': len(self.STATIC_SHORTCUTS),
            'average_usage_per_shortcut': total_usage / len(self.usage_stats) if self.usage_stats else 0
        }
    
    def export_user_shortcuts(self, file_path: str) -> bool:
        """導出用戶自定義快捷方式"""
        try:
            import json
            
            export_data = {
                'user_shortcuts': self.user_shortcuts,
                'usage_stats': self.usage_stats,
                'export_date': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
        
        except Exception:
            return False
    
    def import_user_shortcuts(self, file_path: str) -> bool:
        """導入用戶自定義快捷方式"""
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 合併用戶快捷方式
            imported_shortcuts = data.get('user_shortcuts', {})
            for shortcut, command in imported_shortcuts.items():
                if shortcut not in self.STATIC_SHORTCUTS:  # 避免覆蓋系統快捷方式
                    self.user_shortcuts[shortcut] = command
            
            return True
        
        except Exception:
            return False
    
    def clear_usage_stats(self):
        """清空使用統計"""
        self.usage_stats.clear()
    
    def reset_user_shortcuts(self):
        """重置用戶快捷方式"""
        self.user_shortcuts.clear()


def get_quick_commands() -> QuickCommands:
    """獲取快捷命令實例"""
    if not hasattr(get_quick_commands, '_instance'):
        get_quick_commands._instance = QuickCommands()
    return get_quick_commands._instance


