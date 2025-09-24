#!/usr/bin/env python3
"""
個人AI知識庫管理系統主入口
"""

import sys
import asyncio
from pathlib import Path

# 添加項目根目錄到Python路徑
sys.path.insert(0, str(Path(__file__).parent))

from core.cli.commands import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 再見！")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 程序錯誤: {e}")
        sys.exit(1)

