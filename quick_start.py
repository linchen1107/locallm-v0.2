#!/usr/bin/env python3
"""
快速啟動腳本
幫助用戶快速設置和開始使用AI知識庫助手
"""

import asyncio
import sys
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()


def check_python_version():
    """檢查Python版本"""
    if sys.version_info < (3, 8):
        console.print("❌ 需要Python 3.8或更高版本", style="red")
        sys.exit(1)
    console.print(f"✅ Python版本: {sys.version.split()[0]}", style="green")


def check_ollama_installation():
    """檢查Ollama是否安裝"""
    try:
        result = subprocess.run(
            ["ollama", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            console.print("✅ Ollama已安裝", style="green")
            return True
        else:
            console.print("❌ Ollama未正確安裝", style="red")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        console.print("❌ Ollama未安裝", style="red")
        return False


def install_dependencies():
    """安裝Python依賴"""
    console.print("📦 安裝Python依賴...", style="blue")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True
        )
        console.print("✅ 依賴安裝完成", style="green")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"❌ 依賴安裝失敗: {e}", style="red")
        return False


async def download_models():
    """下載必要的模型"""
    models = [
        ("qwen3:latest", "聊天模型"),
        ("embeddinggemma:latest", "嵌入模型"),
        ("qwen3:latest", "代碼模型")
    ]
    
    console.print("📥 下載Ollama模型...", style="blue")
    
    for model_name, description in models:
        console.print(f"正在下載 {description} ({model_name})...")
        
        try:
            process = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                console.print(f"✅ {description}下載完成", style="green")
            else:
                console.print(f"❌ {description}下載失敗: {stderr.decode()}", style="red")
                return False
                
        except Exception as e:
            console.print(f"❌ 下載{description}時出錯: {e}", style="red")
            return False
    
    return True


def create_directories():
    """創建必要的目錄"""
    directories = [
        "data",
        "data/documents",
        "data/knowledge_base", 
        "data/cache",
        "data/vector_db",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    console.print("✅ 目錄結構創建完成", style="green")


def create_example_env():
    """創建示例環境配置文件"""
    env_content = """# AI知識庫助手配置文件
# 複製此文件為 .env 並根據需要修改

# 應用基本配置
APP_NAME="Personal AI Knowledge Assistant"
DEBUG=false

# Ollama配置
OLLAMA_HOST=http://localhost:11434
OLLAMA_CHAT_MODEL=llama2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_CODE_MODEL=codellama
OLLAMA_TIMEOUT=120

# 存儲配置
STORAGE_DATA_DIR=./data
STORAGE_MAX_FILE_SIZE=104857600

# Agent配置
AGENT_ENABLE_CODE_EXECUTION=false
AGENT_SANDBOX_ENABLED=true
AGENT_MAX_EXECUTION_TIME=300

# RAG配置
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=5

# 安全配置
SECURITY_LOG_LEVEL=INFO
SECURITY_SANDBOX_TIMEOUT=30
"""
    
    env_file = Path(".env.example")
    env_file.write_text(env_content, encoding='utf-8')
    console.print("✅ 示例配置文件已創建 (.env.example)", style="green")


async def test_system():
    """測試系統功能"""
    console.print("🧪 測試系統功能...", style="blue")
    
    try:
        # 導入main模塊
        sys.path.insert(0, str(Path.cwd()))
        
        # 測試初始化
        from core.models.ollama_client import OllamaModelManager
        
        async with OllamaModelManager() as model_manager:
            validation_result = await model_manager.validate_configuration()
            
            if validation_result["connection"]:
                console.print("✅ Ollama連接測試通過", style="green")
            else:
                console.print("❌ Ollama連接測試失敗", style="red")
                return False
        
        # 測試向量數據庫
        from core.storage.vector_store import get_vector_store_manager
        vector_store_manager = get_vector_store_manager()
        await vector_store_manager.initialize_store()
        console.print("✅ 向量數據庫初始化成功", style="green")
        
        return True
        
    except Exception as e:
        console.print(f"❌ 系統測試失敗: {e}", style="red")
        return False


def show_usage_examples():
    """顯示使用示例"""
    table = Table(title="🎯 常用命令示例")
    table.add_column("命令", style="cyan")
    table.add_column("說明", style="magenta")
    
    examples = [
        ("python main.py init", "初始化系統"),
        ("python main.py status", "查看系統狀態"),
        ("python main.py ingest ./documents", "導入文檔到知識庫"),
        ("python main.py query '如何使用Python？'", "查詢知識庫"),
        ("python main.py agent '總結當前目錄的文檔'", "使用智能代理執行任務"),
        ("python main.py tools", "查看可用工具"),
    ]
    
    for command, description in examples:
        table.add_row(command, description)
    
    console.print(table)


async def main():
    """主函數"""
    console.print(Panel.fit(
        "🚀 AI知識庫助手快速啟動",
        style="bold blue"
    ))
    
    # 檢查環境
    console.print("\n📋 檢查系統環境...", style="bold")
    check_python_version()
    
    if not check_ollama_installation():
        console.print("\n❌ Ollama未安裝。請先安裝Ollama:", style="red")
        console.print("   訪問: https://ollama.ai", style="dim")
        console.print("   或運行: curl -fsSL https://ollama.ai/install.sh | sh", style="dim")
        return
    
    # 安裝依賴
    if not install_dependencies():
        console.print("❌ 依賴安裝失敗，請手動運行: pip install -r requirements.txt", style="red")
        return
    
    # 創建目錄結構
    console.print("\n📁 創建目錄結構...", style="bold")
    create_directories()
    create_example_env()
    
    # 下載模型
    if Confirm.ask("\n📥 是否下載必要的Ollama模型？(推薦)"):
        success = await download_models()
        if not success:
            console.print("❌ 模型下載失敗，請手動下載", style="red")
            return
    
    # 測試系統
    if Confirm.ask("\n🧪 是否測試系統功能？"):
        success = await test_system()
        if not success:
            console.print("❌ 系統測試失敗", style="red")
            return
    
    # 顯示完成信息
    console.print("\n🎉 設置完成！", style="bold green")
    
    # 顯示使用示例
    show_usage_examples()
    
    # 詢問是否初始化系統
    if Confirm.ask("\n🚀 是否現在初始化AI助手系統？"):
        console.print("正在初始化...")
        try:
            result = subprocess.run(
                [sys.executable, "main.py", "init"],
                check=True,
                capture_output=True,
                text=True
            )
            console.print("✅ 系統初始化完成！", style="green")
            console.print("\n現在您可以開始使用AI助手了:", style="bold")
            console.print("  python main.py query '你好，請介紹一下你的功能'", style="cyan")
        except subprocess.CalledProcessError as e:
            console.print(f"❌ 初始化失敗: {e}", style="red")
            console.print("請手動運行: python main.py init", style="dim")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 安裝已取消", style="yellow")
    except Exception as e:
        console.print(f"\n❌ 安裝過程中出錯: {e}", style="red")
