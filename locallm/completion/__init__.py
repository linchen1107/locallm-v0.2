"""
LocalLM 命令補全和快捷方式模塊

提供智能命令補全、快捷命令和命令建議功能
"""

from .auto_complete import AutoCompleter, get_auto_completer
from .shortcuts import QuickCommands, get_quick_commands

__all__ = [
    'AutoCompleter',
    'QuickCommands',
    'get_auto_completer',
    'get_quick_commands'
]
