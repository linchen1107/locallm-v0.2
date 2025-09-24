#!/usr/bin/env python3
"""
LocalLM 全局CLI入口點
支持在任何目錄下使用 `locallm` 命令
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import List, Optional
import argparse
import signal

# 添加項目根目錄到Python路徑
current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from locallm.config.global_config import ConfigManager
from locallm.context.detector import ContextDetector
from locallm.commands.router import CommandRouter
from locallm.cli.interface import CLIInterface


class GlobalCLI:
    """全局CLI主類"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        self.working_dir = Path.cwd()
        self.context_detector = ContextDetector(self.working_dir)
        self.command_router = CommandRouter(self.config)
        self.interface = CLIInterface(self.config)
        
        # 設置信號處理
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """處理中斷信號"""
        print("\n🛑 操作已取消")
        sys.exit(0)
    
    async def run(self, args: List[str]):
        """運行CLI命令"""
        try:
            # 顯示歡迎信息（首次運行或版本更新時）
            if self._should_show_welcome():
                self.interface.show_welcome()
            
            # 解析命令行參數
            parsed_args = self._parse_arguments(args)
            
            # 檢測當前上下文
            if parsed_args.detect_context:
                print("[搜索] 檢測當前環境上下文...")
                context = self.context_detector.detect_full_context()
                
                if parsed_args.show_context:
                    self.interface.show_context_info(context)
            else:
                context = self._get_minimal_context()
            
            # 路由並執行命令
            result = await self.command_router.route_command(
                parsed_args, context, self.working_dir
            )
            
            # 顯示結果
            self.interface.show_result(result)
            
            return result.get('success', True)
            
        except KeyboardInterrupt:
            print("\n🛑 操作已取消")
            return False
        except Exception as e:
            self.interface.show_error(f"執行失敗: {str(e)}")
            if self.config.debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    def _parse_arguments(self, args: List[str]) -> argparse.Namespace:
        """解析命令行參數"""
        parser = argparse.ArgumentParser(
            prog='locallm',
            description='LocalLM - 本地AI智能助理',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  locallm analyze                    # 分析當前項目
  locallm "讀取data.csv生成報告"       # 自然語言任務
  locallm chat "這個錯誤怎麼解決?"      # AI對話
  locallm tools list                 # 查看可用工具
  locallm memory save "重要信息"      # 保存到記憶
  locallm config show                # 顯示配置
  locallm --help                     # 顯示幫助
            """
        )
        
        # 主要參數
        parser.add_argument(
            'command',
            nargs='*',
            help='要執行的命令或自然語言描述'
        )
        
        # 全局選項
        parser.add_argument(
            '--version',
            action='version',
            version=f'LocalLM {self._get_version()}'
        )
        
        parser.add_argument(
            '--config',
            metavar='PATH',
            help='指定配置文件路徑'
        )
        
        parser.add_argument(
            '--no-context',
            dest='detect_context',
            action='store_false',
            default=True,
            help='跳過上下文檢測（提高啟動速度）'
        )
        
        parser.add_argument(
            '--show-context',
            action='store_true',
            help='顯示檢測到的上下文信息'
        )
        
        parser.add_argument(
            '--output',
            choices=['rich', 'json', 'plain'],
            default=None,
            help='輸出格式'
        )
        
        parser.add_argument(
            '--debug',
            action='store_true',
            help='啟用調試模式'
        )
        
        parser.add_argument(
            '--safe-mode',
            action='store_true',
            help='啟用安全模式（限制文件操作）'
        )
        
        # 快捷選項
        parser.add_argument(
            '-q', '--quick',
            action='store_true',
            help='快速模式（跳過詳細檢測）'
        )
        
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='詳細輸出模式'
        )
        
        return parser.parse_args(args)
    
    def _should_show_welcome(self) -> bool:
        """檢查是否應該顯示歡迎信息"""
        welcome_file = Path.home() / '.locallm' / '.welcome_shown'
        
        if not welcome_file.exists():
            welcome_file.parent.mkdir(parents=True, exist_ok=True)
            welcome_file.touch()
            return True
        
        return False
    
    def _get_minimal_context(self) -> dict:
        """獲取最小上下文信息"""
        return {
            'working_directory': str(self.working_dir),
            'timestamp': self.context_detector._get_timestamp(),
            'minimal_mode': True
        }
    
    def _get_version(self) -> str:
        """獲取版本信息"""
        try:
            from locallm import __version__
            return __version__
        except ImportError:
            return "2.0.0"


def main():
    """主入口函數"""
    # 獲取命令行參數
    args = sys.argv[1:]
    
    # 創建全局CLI實例
    cli = GlobalCLI()
    
    # 運行命令
    try:
        success = asyncio.run(cli.run(args))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"[失敗] 意外錯誤: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


