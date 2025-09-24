"""
LocalLM 智能化功能模塊

提供自然語言處理、命令解析、工作流生成等智能化功能
"""

from .command_parser import NaturalLanguageParser, ParsedCommand
from .intelligence_manager import IntelligenceManager, IntelligenceResult, get_intelligence_manager

__all__ = [
    'NaturalLanguageParser',
    'ParsedCommand',
    'IntelligenceManager',
    'IntelligenceResult',
    'get_intelligence_manager'
]
