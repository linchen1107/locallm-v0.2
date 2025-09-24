"""
全局CLI用戶界面
提供美觀的命令行交互體驗
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.columns import Columns
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


class CLIInterface:
    """全局CLI用戶界面"""
    
    def __init__(self, config):
        self.config = config
        self.output_format = getattr(config, 'default_output_format', 'rich')
        
        if RICH_AVAILABLE and self.output_format == 'rich':
            self.console = Console()
            self.use_rich = True
        else:
            self.console = None
            self.use_rich = False
    
    def show_welcome(self):
        """顯示歡迎信息"""
        if self.use_rich:
            self._show_rich_welcome()
        else:
            self._show_plain_welcome()
    
    def _show_rich_welcome(self):
        """Rich格式的歡迎信息"""
        welcome_text = """
🤖 [bold blue]LocalLM[/bold blue] - 本地AI智能助理 [dim](v2.0)[/dim]

現在您可以在[bold green]任何目錄[/bold green]下使用AI助理！

[yellow]✨ 快速開始:[/yellow]
• [cyan]locallm analyze[/cyan]              - 分析當前項目
• [cyan]locallm "處理data.csv"[/cyan]       - 自然語言任務  
• [cyan]locallm chat "問題"[/cyan]          - AI對話
• [cyan]locallm tools list[/cyan]           - 查看工具
• [cyan]locallm --help[/cyan]               - 查看幫助

[dim][提示] 提示: 使用 locallm --show-context 查看當前環境信息[/dim]
        """
        
        panel = Panel(
            welcome_text,
            title="[啟動] 歡迎使用 LocalLM",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        self.console.print()
    
    def _show_plain_welcome(self):
        """純文本格式的歡迎信息"""
        print("=" * 60)
        print("🤖 LocalLM - 本地AI智能助理 (v2.0)")
        print("=" * 60)
        print()
        print("現在您可以在任何目錄下使用AI助理！")
        print()
        print("✨ 快速開始:")
        print("• locallm analyze              - 分析當前項目")
        print("• locallm \"處理data.csv\"       - 自然語言任務")
        print("• locallm chat \"問題\"          - AI對話")
        print("• locallm tools list           - 查看工具")
        print("• locallm --help               - 查看幫助")
        print()
        print("[提示] 提示: 使用 locallm --show-context 查看當前環境信息")
        print("=" * 60)
        print()
    
    def show_context_info(self, context: Dict[str, Any]):
        """顯示上下文信息"""
        if self.use_rich:
            self._show_rich_context(context)
        else:
            self._show_plain_context(context)
    
    def _show_rich_context(self, context: Dict[str, Any]):
        """Rich格式的上下文信息"""
        # 基本信息表格
        info_table = Table(title="[搜索] 當前環境上下文", show_header=True, header_style="bold magenta")
        info_table.add_column("項目", style="cyan", no_wrap=True)
        info_table.add_column("值", style="white")
        
        # 基本信息
        info_table.add_row("當前目錄", context.get('working_directory', 'Unknown'))
        info_table.add_row("檢測時間", context.get('timestamp', 'Unknown'))
        
        # 項目信息
        if 'project_info' in context:
            project_info = context['project_info']
            project_type = project_info.get('primary_type', 'unknown')
            confidence = project_info.get('confidence_scores', {}).get(project_type, 0)
            info_table.add_row("項目類型", f"{project_type} (置信度: {confidence:.2f})")
        
        # 文件統計
        if 'file_summary' in context:
            file_summary = context['file_summary']
            total_files = file_summary.get('total_files', 0)
            total_dirs = file_summary.get('total_directories', 0)
            info_table.add_row("文件統計", f"{total_files} 文件, {total_dirs} 目錄")
        
        # Git信息
        if 'git_info' in context:
            git_info = context['git_info']
            if git_info.get('is_git_repo'):
                branch = git_info.get('current_branch', 'unknown')
                info_table.add_row("Git狀態", f"分支: {branch}")
        
        self.console.print(info_table)
        
        # 環境信息
        if 'environment_info' in context:
            env_info = context['environment_info']
            env_table = Table(title="💻 系統環境", show_header=True, header_style="bold green")
            env_table.add_column("項目", style="cyan")
            env_table.add_column("值", style="white")
            
            env_table.add_row("操作系統", env_info.get('os', 'Unknown'))
            env_table.add_row("Python版本", env_info.get('python_version', 'Unknown').split()[0])
            env_table.add_row("用戶", env_info.get('user', 'Unknown'))
            env_table.add_row("主機名", env_info.get('hostname', 'Unknown'))
            
            self.console.print(env_table)
    
    def _show_plain_context(self, context: Dict[str, Any]):
        """純文本格式的上下文信息"""
        print("[搜索] 當前環境上下文")
        print("-" * 40)
        print(f"當前目錄: {context.get('working_directory', 'Unknown')}")
        print(f"檢測時間: {context.get('timestamp', 'Unknown')}")
        
        if 'project_info' in context:
            project_info = context['project_info']
            project_type = project_info.get('primary_type', 'unknown')
            print(f"項目類型: {project_type}")
        
        if 'file_summary' in context:
            file_summary = context['file_summary']
            total_files = file_summary.get('total_files', 0)
            total_dirs = file_summary.get('total_directories', 0)
            print(f"文件統計: {total_files} 文件, {total_dirs} 目錄")
        
        if 'git_info' in context:
            git_info = context['git_info']
            if git_info.get('is_git_repo'):
                branch = git_info.get('current_branch', 'unknown')
                print(f"Git分支: {branch}")
        
        print("-" * 40)
    
    def show_result(self, result: Dict[str, Any]):
        """顯示命令執行結果"""
        if self.output_format == 'json':
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif self.use_rich:
            self._show_rich_result(result)
        else:
            self._show_plain_result(result)
    
    def _show_rich_result(self, result: Dict[str, Any]):
        """Rich格式的結果顯示"""
        if result.get('success', True):
            if 'output' in result:
                self.console.print(result['output'])
            if 'data' in result:
                self._display_data(result['data'])
        else:
            error_panel = Panel(
                f"[失敗] {result.get('error', '未知錯誤')}",
                title="執行失敗",
                border_style="red"
            )
            self.console.print(error_panel)
    
    def _show_plain_result(self, result: Dict[str, Any]):
        """純文本格式的結果顯示"""
        if result.get('success', True):
            if 'output' in result:
                print(result['output'])
            if 'data' in result:
                print(f"結果: {result['data']}")
        else:
            print(f"[失敗] 執行失敗: {result.get('error', '未知錯誤')}")
    
    def _display_data(self, data: Any):
        """顯示數據"""
        if isinstance(data, dict):
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("鍵", style="cyan")
            table.add_column("值", style="white")
            
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)[:100] + "..."
                table.add_row(str(key), str(value))
            
            self.console.print(table)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self.console.print(f"{i+1}. {item}")
        else:
            self.console.print(str(data))
    
    def show_error(self, error_message: str, suggestions: List[str] = None):
        """顯示錯誤信息"""
        if self.use_rich:
            error_panel = Panel(
                f"[失敗] {error_message}",
                title="錯誤",
                border_style="red"
            )
            self.console.print(error_panel)
            
            if suggestions:
                suggestion_text = "\n".join([f"[提示] {s}" for s in suggestions])
                suggestion_panel = Panel(
                    suggestion_text,
                    title="建議",
                    border_style="yellow"
                )
                self.console.print(suggestion_panel)
        else:
            print(f"[失敗] 錯誤: {error_message}")
            if suggestions:
                print("\n建議:")
                for suggestion in suggestions:
                    print(f"[提示] {suggestion}")
    
    def show_progress(self, description: str):
        """顯示進度指示器"""
        if self.use_rich:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            )
        else:
            print(f"⏳ {description}...")
            return None
    
    def confirm(self, question: str) -> bool:
        """確認對話"""
        if self.use_rich:
            return Confirm.ask(question)
        else:
            response = input(f"{question} (y/N): ")
            return response.lower() in ['y', 'yes', 'true', '1']
    
    def prompt(self, question: str, default: str = None) -> str:
        """輸入提示"""
        if self.use_rich:
            return Prompt.ask(question, default=default)
        else:
            prompt_text = question
            if default:
                prompt_text += f" (默認: {default})"
            prompt_text += ": "
            
            response = input(prompt_text)
            return response if response else default


