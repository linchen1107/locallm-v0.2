"""
CLI命令定義
提供所有主要功能的命令行接口
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Confirm

from config.settings import get_settings
from core.models.ollama_client import OllamaModelManager
from core.rag.retrieval_system import get_rag_system
from core.agent.intelligent_agent import get_intelligent_agent

# 創建Typer應用
app = typer.Typer(
    name="ai-assistant",
    help="個人AI知識庫管理助手",
    add_completion=False
)

# 控制台實例
console = Console()


def run_async(coro):
    """運行異步函數的輔助函數"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="強制重新初始化")
):
    """初始化AI助手系統"""
    
    console.print("🚀 初始化AI知識庫管理系統...", style="bold blue")
    
    async def _init():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                # 檢查Ollama連接和模型
                task1 = progress.add_task("檢查Ollama服務和模型...", total=None)
                
                async with OllamaModelManager() as model_manager:
                    validation_result = await model_manager.validate_configuration()
                    
                    if not validation_result["connection"]:
                        progress.update(task1, description="❌ Ollama服務連接失敗")
                        console.print(f"錯誤: {validation_result.get('error', 'Ollama服務不可用')}", style="red")
                        return False
                    
                    if validation_result["status"] != "ready":
                        progress.update(task1, description="📥 下載缺失的模型...")
                        await model_manager.ensure_models_ready()
                
                progress.update(task1, description="✅ Ollama服務和模型就緒")
                
                # 初始化RAG系統
                task2 = progress.add_task("初始化RAG系統...", total=None)
                rag_system = await get_rag_system()
                progress.update(task2, description="✅ RAG系統初始化完成")
                
                # 獲取系統狀態
                task3 = progress.add_task("檢查系統狀態...", total=None)
                agent = get_intelligent_agent()
                status = await agent.get_agent_status()
                progress.update(task3, description="✅ 系統狀態檢查完成")
            
            # 顯示初始化結果
            console.print("\n🎉 系統初始化完成！", style="bold green")
            
            # 創建狀態表
            table = Table(title="系統狀態")
            table.add_column("項目", style="cyan")
            table.add_column("狀態", style="magenta")
            
            table.add_row("Ollama連接", "✅ 已連接")
            table.add_row("模型狀態", "✅ 就緒")
            table.add_row("RAG系統", "✅ 已初始化")
            table.add_row("可用工具", f"✅ {status['available_tools']} 個")
            
            console.print(table)
            
            console.print("\n使用 'ai-assistant --help' 查看可用命令", style="dim")
            return True
            
        except Exception as e:
            console.print(f"❌ 初始化失敗: {e}", style="red")
            return False
    
    success = run_async(_init())
    if not success:
        raise typer.Exit(1)


@app.command()
def ingest(
    paths: List[str] = typer.Argument(..., help="要處理的文件或目錄路徑"),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="遞歸處理目錄"),
    force: bool = typer.Option(False, "--force", "-f", help="強制重新處理已存在的文件")
):
    """將文檔導入知識庫"""
    
    console.print(f"📚 正在處理 {len(paths)} 個路徑...", style="bold blue")
    
    async def _ingest():
        try:
            rag_system = await get_rag_system()
            
            # 收集所有文件路徑
            all_files = []
            for path_str in paths:
                path = Path(path_str)
                if not path.exists():
                    console.print(f"⚠️  路徑不存在: {path_str}", style="yellow")
                    continue
                
                if path.is_file():
                    all_files.append(str(path))
                elif path.is_dir() and recursive:
                    settings = get_settings()
                    for ext in settings.storage.supported_formats:
                        all_files.extend([str(p) for p in path.rglob(f"*{ext}")])
                elif path.is_dir():
                    settings = get_settings()
                    for ext in settings.storage.supported_formats:
                        all_files.extend([str(p) for p in path.glob(f"*{ext}")])
            
            if not all_files:
                console.print("❌ 沒有找到可處理的文件", style="red")
                return
            
            console.print(f"📄 找到 {len(all_files)} 個文件")
            
            # 處理文檔
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("處理文檔...", total=None)
                
                result = await rag_system.add_documents(all_files)
                
                if result["success"]:
                    progress.update(task, description="✅ 文檔處理完成")
                    
                    # 顯示結果
                    panel = Panel(
                        f"✅ 成功處理 {result['processed_files']} 個文件\n"
                        f"📝 創建 {result['added_count']} 個文檔塊\n"
                        f"💾 已添加到知識庫",
                        title="處理完成",
                        border_style="green"
                    )
                    console.print(panel)
                else:
                    progress.update(task, description="❌ 文檔處理失敗")
                    console.print(f"❌ 處理失敗: {result['message']}", style="red")
                    
        except Exception as e:
            console.print(f"❌ 導入失敗: {e}", style="red")
    
    run_async(_ingest())


@app.command()
def query(
    question: str = typer.Argument(..., help="要查詢的問題"),
    max_results: int = typer.Option(5, "--max-results", "-k", help="最大檢索結果數"),
    show_sources: bool = typer.Option(True, "--show-sources", "-s", help="顯示來源信息"),
    file_type: Optional[str] = typer.Option(None, "--file-type", "-t", help="過濾文件類型")
):
    """查詢知識庫"""
    
    console.print(f"🔍 查詢: {question}", style="bold blue")
    
    async def _query():
        try:
            rag_system = await get_rag_system()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("檢索相關信息...", total=None)
                
                # 構建過濾條件
                filters = {}
                if file_type:
                    filters["file_type"] = file_type
                
                result = await rag_system.query(
                    question=question,
                    filters=filters if filters else None,
                    max_results=max_results
                )
                
                progress.update(task, description="✅ 檢索完成")
            
            # 顯示答案
            answer_panel = Panel(
                result.answer,
                title=f"回答 (置信度: {result.confidence:.2f})",
                border_style="green" if result.confidence > 0.7 else "yellow"
            )
            console.print(answer_panel)
            
            # 顯示來源信息
            if show_sources and result.sources:
                console.print("\n📚 相關來源:", style="bold")
                
                sources_table = Table()
                sources_table.add_column("檔案", style="cyan")
                sources_table.add_column("類型", style="magenta")
                sources_table.add_column("預覽", style="dim")
                
                for i, source in enumerate(result.sources[:5]):
                    preview = source.content[:100] + "..." if len(source.content) > 100 else source.content
                    sources_table.add_row(
                        source.metadata.file_name,
                        source.metadata.file_type,
                        preview
                    )
                
                console.print(sources_table)
            
            # 顯示性能指標
            console.print(f"\n⏱️  檢索時間: {result.retrieval_time:.2f}s | 生成時間: {result.generation_time:.2f}s", style="dim")
            
        except Exception as e:
            console.print(f"❌ 查詢失敗: {e}", style="red")
    
    run_async(_query())


@app.command()
def agent(
    task: str = typer.Argument(..., help="要執行的任務描述"),
    auto_confirm: bool = typer.Option(False, "--auto", "-y", help="自動確認執行計劃")
):
    """使用智能代理執行複雜任務"""
    
    console.print(f"🤖 智能代理任務: {task}", style="bold blue")
    
    async def _agent():
        try:
            agent = get_intelligent_agent()
            
            # 執行任務
            result_task = await agent.execute_complex_task(
                description=task,
                auto_confirm=auto_confirm
            )
            
            # 任務已經在agent中顯示了結果，這裡只需要處理退出代碼
            if result_task.status.value == "completed":
                console.print("🎉 任務執行成功!", style="bold green")
            else:
                console.print("❌ 任務執行失敗", style="bold red")
                if result_task.error_message:
                    console.print(f"錯誤: {result_task.error_message}", style="red")
                raise typer.Exit(1)
                
        except Exception as e:
            console.print(f"❌ 代理執行失敗: {e}", style="red")
            raise typer.Exit(1)
    
    run_async(_agent())


@app.command()
def status():
    """顯示系統狀態"""
    
    async def _status():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("檢查系統狀態...", total=None)
                
                # 獲取各組件狀態
                agent = get_intelligent_agent()
                agent_status = await agent.get_agent_status()
                
                rag_system = await get_rag_system()
                kb_stats = await rag_system.get_knowledge_base_stats()
                
                # 檢查Ollama
                async with OllamaModelManager() as model_manager:
                    ollama_status = await model_manager.validate_configuration()
                
                progress.update(task, description="✅ 狀態檢查完成")
            
            # 系統狀態表
            status_table = Table(title="🚀 AI助手系統狀態")
            status_table.add_column("組件", style="cyan")
            status_table.add_column("狀態", style="magenta")
            status_table.add_column("詳情", style="green")
            
            # Ollama狀態
            ollama_status_text = "✅ 已連接" if ollama_status["connection"] else "❌ 未連接"
            status_table.add_row("Ollama服務", ollama_status_text, f"模型: {len(ollama_status['models'])} 個")
            
            # RAG系統狀態
            if kb_stats["success"]:
                kb_info = kb_stats["stats"]
                status_table.add_row(
                    "知識庫", 
                    "✅ 已初始化", 
                    f"文檔: {kb_info['total_documents']} 個"
                )
            else:
                status_table.add_row("知識庫", "❌ 未初始化", "")
            
            # Agent狀態
            status_table.add_row(
                "智能代理", 
                f"✅ {agent_status['status']}", 
                f"工具: {agent_status['available_tools']} 個"
            )
            
            # 任務統計
            status_table.add_row(
                "任務統計",
                f"總計: {agent_status['total_tasks']}",
                f"成功: {agent_status['completed_tasks']}, 失敗: {agent_status['failed_tasks']}"
            )
            
            console.print(status_table)
            
            # 配置信息
            settings_table = Table(title="⚙️  系統配置")
            settings_table.add_column("設置", style="cyan")
            settings_table.add_column("值", style="magenta")
            
            settings = get_settings()
            settings_table.add_row("聊天模型", settings.ollama.chat_model)
            settings_table.add_row("嵌入模型", settings.ollama.embedding_model)
            settings_table.add_row("代碼模型", settings.ollama.code_model)
            settings_table.add_row("沙箱模式", "✅ 啟用" if settings.agent.sandbox_enabled else "❌ 禁用")
            settings_table.add_row("代碼執行", "✅ 啟用" if settings.agent.enable_code_execution else "❌ 禁用")
            
            console.print(settings_table)
            
        except Exception as e:
            console.print(f"❌ 狀態檢查失敗: {e}", style="red")
    
    run_async(_status())


@app.command()
def search(
    query: str = typer.Argument(..., help="搜索查詢"),
    limit: int = typer.Option(10, "--limit", "-l", help="結果數量限制"),
    file_type: Optional[str] = typer.Option(None, "--file-type", "-t", help="過濾文件類型")
):
    """搜索文檔（不生成答案）"""
    
    async def _search():
        try:
            rag_system = await get_rag_system()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("搜索文檔...", total=None)
                
                results = await rag_system.search_documents(
                    query=query,
                    k=limit,
                    file_type_filter=file_type
                )
                
                progress.update(task, description="✅ 搜索完成")
            
            if not results:
                console.print("❌ 未找到相關文檔", style="yellow")
                return
            
            # 顯示搜索結果
            results_table = Table(title=f"🔍 搜索結果 ({len(results)} 個)")
            results_table.add_column("檔案", style="cyan")
            results_table.add_column("類型", style="magenta")
            results_table.add_column("相似度", style="green")
            results_table.add_column("內容預覽", style="dim")
            
            for result in results:
                results_table.add_row(
                    result["file_name"],
                    result["file_type"],
                    f"{result['similarity_score']:.3f}",
                    result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"]
                )
            
            console.print(results_table)
            
        except Exception as e:
            console.print(f"❌ 搜索失敗: {e}", style="red")
    
    run_async(_search())


@app.command()
def remove(
    file_path: str = typer.Argument(..., help="要移除的文件路徑"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="跳過確認")
):
    """從知識庫移除文檔"""
    
    if not confirm:
        if not Confirm.ask(f"確定要從知識庫移除 {file_path} 嗎？"):
            console.print("操作已取消", style="yellow")
            return
    
    async def _remove():
        try:
            rag_system = await get_rag_system()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("移除文檔...", total=None)
                
                result = await rag_system.remove_documents(file_path)
                
                if result["success"]:
                    progress.update(task, description="✅ 移除完成")
                    console.print(f"✅ {result['message']}", style="green")
                else:
                    progress.update(task, description="❌ 移除失敗")
                    console.print(f"❌ {result['message']}", style="red")
                    
        except Exception as e:
            console.print(f"❌ 移除失敗: {e}", style="red")
    
    run_async(_remove())


@app.command()
def tools():
    """列出可用的工具"""
    
    agent = get_intelligent_agent()
    available_tools = agent.get_available_tools()
    
    # 創建工具表格
    tools_table = Table(title="🛠️  可用工具")
    tools_table.add_column("工具名稱", style="cyan")
    tools_table.add_column("描述", style="magenta")
    tools_table.add_column("主要參數", style="dim")
    
    for tool in available_tools:
        # 提取主要參數
        params = []
        if 'parameters' in tool and 'properties' in tool['parameters']:
            required_params = tool['parameters'].get('required', [])
            for param in required_params[:3]:  # 只顯示前3個必需參數
                params.append(f"{param} (必需)")
        
        params_str = ", ".join(params) if params else "無"
        
        tools_table.add_row(
            tool['name'],
            tool['description'],
            params_str
        )
    
    console.print(tools_table)


@app.command()
def memory(
    action: str = typer.Argument(..., help="操作類型: stats/export/clear/consolidate"),
    memory_type: Optional[str] = typer.Option(None, "--type", "-t", help="記憶類型: session/long_term/personal/knowledge_graph"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="輸出文件路徑"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="跳過確認")
):
    """記憶系統管理"""
    
    async def _handle_memory():
        nonlocal output_file
        try:
            from core.memory.context_manager import ContextManager
            
            context_manager = ContextManager()
            
            if action == "stats":
                # 顯示記憶統計信息
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("獲取記憶統計...", total=None)
                    
                    stats = await context_manager.get_memory_stats()
                    
                    progress.update(task, description="✅ 統計完成")
                
                # 創建統計表格
                stats_table = Table(title="📊 記憶系統統計")
                stats_table.add_column("記憶類型", style="cyan")
                stats_table.add_column("總項目數", justify="right", style="magenta")
                stats_table.add_column("最近活動", justify="right", style="green")
                stats_table.add_column("詳細信息", style="dim")
                
                for mem_type, stat in stats.items():
                    details = []
                    if hasattr(stat, 'by_importance') and stat.by_importance:
                        for importance, count in stat.by_importance.items():
                            if count > 0:
                                details.append(f"{importance}: {count}")
                    
                    details_str = ", ".join(details) if details else "無詳細信息"
                    
                    stats_table.add_row(
                        mem_type,
                        str(stat.total_items),
                        str(stat.recent_activity),
                        details_str
                    )
                
                console.print(stats_table)
                
            elif action == "export":
                # 導出記憶數據
                if not output_file:
                    output_file = f"memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                export_types = []
                if memory_type:
                    export_types = [memory_type]
                else:
                    export_types = ["session", "long_term", "personal", "knowledge_graph"]
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("導出記憶數據...", total=None)
                    
                    export_data = await context_manager.export_memories(export_types)
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, ensure_ascii=False, indent=2)
                    
                    progress.update(task, description="✅ 導出完成")
                
                    console.print(f"✅ 記憶數據已導出到: {output_file}", style="green")
                
            elif action == "clear":
                # 清空記憶
                if not confirm:
                    if not Confirm.ask("⚠️  確定要清空記憶系統嗎？此操作不可逆！"):
                        console.print("操作已取消", style="yellow")
                        return
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("清空記憶系統...", total=None)
                    
                    success = await context_manager.clear_all_memories()
                    
                    if success:
                        progress.update(task, description="✅ 清空完成")
                        console.print("✅ 記憶系統已清空", style="green")
                    else:
                        progress.update(task, description="❌ 清空失敗")
                        console.print("❌ 清空記憶系統失敗", style="red")
                
            elif action == "consolidate":
                # 整合記憶
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("整合記憶數據...", total=None)
                    
                    consolidated_count = await context_manager.long_term_memory.consolidate_memories()
                    cleanup_results = await context_manager.cleanup_old_memories()
                    
                    progress.update(task, description="✅ 整合完成")
                
                console.print(f"✅ 已整合 {consolidated_count} 條重複記憶", style="green")
                console.print(f"✅ 清理統計: {cleanup_results}", style="green")
                
            else:
                console.print(f"❌ 未知操作: {action}", style="red")
                console.print("可用操作: stats, export, clear, consolidate", style="yellow")
                
        except Exception as e:
            console.print(f"❌ 記憶操作失敗: {e}", style="red")
    
    run_async(_handle_memory())


@app.command()
def context(
    action: str = typer.Argument(..., help="操作類型: show/save/load"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="查詢內容"),
    fact: Optional[str] = typer.Option(None, "--fact", "-f", help="要保存的事實"),
    importance: str = typer.Option("medium", "--importance", "-i", help="重要性: low/medium/high/critical")
):
    """上下文和個人記憶管理"""
    
    async def _handle_context():
        try:
            from core.memory.context_manager import ContextManager
            from core.memory.base import MemoryImportance
            
            context_manager = ContextManager()
            
            if action == "show":
                if query:
                    # 顯示相關上下文
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        task = progress.add_task("搜索相關上下文...", total=None)
                        
                        smart_context = await context_manager.get_smart_context(query)
                        
                        progress.update(task, description="✅ 搜索完成")
                    
                    if smart_context:
                        console.print(Panel(smart_context, title="🧠 相關上下文", border_style="blue"))
                    else:
                        console.print("未找到相關上下文", style="yellow")
                else:
                    # 顯示當前上下文摘要
                    summary = context_manager.get_current_context_summary()
                    console.print(Panel(json.dumps(summary, ensure_ascii=False, indent=2), 
                                      title="📊 當前上下文摘要", border_style="green"))
                
            elif action == "save":
                if not fact:
                    console.print("❌ 請使用 --fact 參數提供要保存的事實", style="red")
                    return
                
                # 轉換重要性等級
                importance_map = {
                    "low": MemoryImportance.LOW,
                    "medium": MemoryImportance.MEDIUM,
                    "high": MemoryImportance.HIGH,
                    "critical": MemoryImportance.CRITICAL
                }
                
                importance_level = importance_map.get(importance.lower(), MemoryImportance.MEDIUM)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("保存重要事實...", total=None)
                    
                    memory_id = await context_manager.store_important_fact(fact, "user", importance_level)
                    
                    progress.update(task, description="✅ 保存完成")
                
                console.print(f"✅ 事實已保存，記憶ID: {memory_id}", style="green")
                
            elif action == "load":
                # 顯示個人偏好和使用模式
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("加載個人偏好...", total=None)
                    
                    preferences = await context_manager.get_user_preferences()
                    
                    progress.update(task, description="✅ 加載完成")
                
                console.print(Panel(json.dumps(preferences, ensure_ascii=False, indent=2), 
                              title="👤 個人偏好和使用模式", border_style="magenta"))
                
            else:
                console.print(f"❌ 未知操作: {action}", style="red")
                console.print("可用操作: show, save, load", style="yellow")
                
        except Exception as e:
            console.print(f"❌ 上下文操作失敗: {e}", style="red")
    
    run_async(_handle_context())


@app.command()
def tools(
    action: str = typer.Argument(..., help="操作類型: list/execute/workflow/demo/stats"),
    tool_name: Optional[str] = typer.Option(None, "--tool", "-t", help="工具名稱"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="工具類別"),
    operation: Optional[str] = typer.Option(None, "--operation", "-o", help="工具操作"),
    workflow_name: Optional[str] = typer.Option(None, "--workflow", "-w", help="工作流名稱"),
    goal: Optional[str] = typer.Option(None, "--goal", "-g", help="任務目標")
):
    """增強工具系統管理"""
    
    async def _handle_tools():
        nonlocal tool_name, category, operation, workflow_name, goal
        try:
            from core.tools.manager import get_tool_manager
            from core.tools.base import ToolCategory
            
            tool_manager = get_tool_manager()
            
            if action == "list":
                # 列出可用工具
                category_filter = None
                if category:
                    try:
                        category_filter = ToolCategory(category.lower())
                    except ValueError:
                        console.print(f"❌ 無效的類別: {category}", style="red")
                        return
                
                tools_list = tool_manager.get_available_tools(category_filter)
                
                # 創建工具表格
                tools_table = Table(title="🛠️ 可用工具列表")
                tools_table.add_column("工具名稱", style="cyan")
                tools_table.add_column("類別", style="magenta") 
                tools_table.add_column("版本", style="green")
                tools_table.add_column("描述", style="white")
                tools_table.add_column("標籤", style="dim")
                
                for tool_info in tools_list:
                    tags_str = ", ".join(tool_info.get("tags", []))
                    tools_table.add_row(
                        tool_info["name"],
                        tool_info["category"],
                        tool_info.get("version", "1.0.0"),
                        tool_info["description"][:50] + "..." if len(tool_info["description"]) > 50 else tool_info["description"],
                        tags_str[:30] + "..." if len(tags_str) > 30 else tags_str
                    )
                
                console.print(tools_table)
                
                # 顯示統計
                stats = tool_manager.get_usage_stats()
                if stats:
                    console.print(f"\n📊 工具使用統計: {len(stats)} 個工具已使用")
                    for tool, stat in sorted(stats.items(), key=lambda x: x[1]["usage_count"], reverse=True)[:5]:
                        console.print(f"  • {tool}: {stat['usage_count']} 次使用, {stat['success_count']} 次成功")
                
            elif action == "execute":
                # 執行工具
                if not tool_name:
                    console.print("❌ 請指定工具名稱 --tool", style="red")
                    return
                
                if not operation:
                    console.print("❌ 請指定操作 --operation", style="red")
                    return
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task(f"執行工具 {tool_name}...", total=None)
                    
                    result = await tool_manager.execute_tool(tool_name, operation=operation)
                    
                    progress.update(task, description="✅ 執行完成")
                
                if result.success:
                    console.print(f"✅ 工具執行成功", style="green")
                    console.print(Panel(
                        json.dumps(result.data, ensure_ascii=False, indent=2),
                        title="執行結果",
                        border_style="green"
                    ))
                    
                    # 顯示建議的下一步工具
                    if result.next_suggestions:
                        console.print("\n💡 建議的下一步工具:")
                        for suggestion in result.next_suggestions:
                            console.print(f"  • {suggestion['tool']}: {suggestion['reason']}")
                else:
                    console.print(f"❌ 工具執行失敗: {result.error}", style="red")
            
            elif action == "workflow":
                # 工作流管理
                if workflow_name == "create" and goal:
                    # 創建自動工作流
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        task = progress.add_task("創建工作流...", total=None)
                        
                        workflow_id = await tool_manager.auto_workflow_from_goal(goal)
                        
                        progress.update(task, description="✅ 工作流創建完成")
                    
                    console.print(f"✅ 工作流創建成功", style="green")
                    console.print(f"工作流ID: {workflow_id}")
                    
                    # 顯示工作流詳情
                    workflow_config = tool_manager.export_workflow(workflow_id)
                    if workflow_config:
                        console.print(Panel(
                            json.dumps(workflow_config, ensure_ascii=False, indent=2),
                            title="工作流配置",
                            border_style="blue"
                        ))
                
                elif workflow_name == "list":
                    # 列出所有工作流
                    workflows = tool_manager.list_workflows()
                    
                    if workflows:
                        workflow_table = Table(title="📋 工作流列表")
                        workflow_table.add_column("工作流ID", style="cyan")
                        workflow_table.add_column("名稱", style="magenta")
                        workflow_table.add_column("工具數量", justify="right", style="green")
                        workflow_table.add_column("步驟數量", justify="right", style="yellow")
                        
                        for workflow in workflows:
                            workflow_table.add_row(
                                workflow["workflow_id"][:8] + "...",
                                workflow["name"],
                                str(workflow["tools_count"]),
                                str(workflow["steps_count"])
                            )
                        
                        console.print(workflow_table)
                    else:
                        console.print("📋 暫無工作流", style="yellow")
                
                elif workflow_name and workflow_name not in ["create", "list"]:
                    # 執行指定工作流
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        task = progress.add_task(f"執行工作流 {workflow_name}...", total=None)
                        
                        results = await tool_manager.execute_workflow(workflow_name)
                        
                        progress.update(task, description="✅ 工作流執行完成")
                    
                    console.print(f"✅ 工作流執行完成", style="green")
                    console.print(f"執行了 {len(results)} 個步驟")
                    
                    for i, result in enumerate(results):
                        status = "✅" if result.success else "❌"
                        console.print(f"  {status} 步驟 {i+1}: {'成功' if result.success else '失敗'}")
                
                else:
                    console.print("❌ 請指定有效的工作流操作", style="red")
                    console.print("可用操作: --workflow create --goal '目標'", style="yellow")
                    console.print("         --workflow list", style="yellow")
                    console.print("         --workflow <workflow_id>", style="yellow")
            
            elif action == "demo":
                # 演示工具協作
                console.print("🚀 開始演示工具協作功能...", style="blue")
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("運行協作演示...", total=None)
                    
                    demo_results = await tool_manager.demonstrate_cooperation()
                    
                    progress.update(task, description="✅ 演示完成")
                
                if demo_results.get("demo_completed"):
                    console.print("✅ 工具協作演示成功完成！", style="green")
                    
                    # 顯示協作功能
                    features = demo_results.get("cooperation_features", [])
                    if features:
                        console.print("\n🎯 協作功能展示:")
                        for feature in features:
                            console.print(f"  ✓ {feature}")
                    
                    # 顯示演示結果摘要
                    console.print(f"\n📊 演示結果統計:")
                    for key, value in demo_results.items():
                        if key.startswith("step") and isinstance(value, dict):
                            success = value.get("success", False)
                            status = "✅" if success else "❌"
                            console.print(f"  {status} {key}: {'成功' if success else '失敗'}")
                
                else:
                    console.print("❌ 演示執行失敗", style="red")
                    if "error" in demo_results:
                        console.print(f"錯誤: {demo_results['error']}")
            
            elif action == "stats":
                # 顯示工具統計
                stats = tool_manager.get_usage_stats()
                
                if not stats:
                    console.print("📊 暫無工具使用統計", style="yellow")
                    return
                
                # 創建統計表格
                stats_table = Table(title="📊 工具使用統計")
                stats_table.add_column("工具名稱", style="cyan")
                stats_table.add_column("使用次數", justify="right", style="magenta")
                stats_table.add_column("成功次數", justify="right", style="green")
                stats_table.add_column("失敗次數", justify="right", style="red")
                stats_table.add_column("成功率", justify="right", style="yellow")
                
                for tool_name, stat in sorted(stats.items(), key=lambda x: x[1]["usage_count"], reverse=True):
                    usage = stat["usage_count"]
                    success = stat["success_count"]
                    failure = stat["failure_count"]
                    success_rate = (success / usage * 100) if usage > 0 else 0
                    
                    stats_table.add_row(
                        tool_name,
                        str(usage),
                        str(success),
                        str(failure),
                        f"{success_rate:.1f}%"
                    )
                
                console.print(stats_table)
                
                # 顯示兼容性矩陣
                compatibility = tool_manager.get_tool_compatibility_matrix()
                console.print(f"\n🔗 工具兼容性: {len(compatibility)} 個工具可協作")
                for tool, compatible_tools in compatibility.items():
                    if compatible_tools:
                        console.print(f"  • {tool} 可與 {len(compatible_tools)} 個工具協作")
            
            else:
                console.print(f"❌ 未知操作: {action}", style="red")
                console.print("可用操作: list, execute, workflow, demo, stats", style="yellow")
                
        except Exception as e:
            console.print(f"❌ 工具操作失敗: {e}", style="red")
    
    run_async(_handle_tools())


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="跳過確認")
):
    """清空知識庫"""
    
    if not confirm:
        if not Confirm.ask("⚠️  確定要清空整個知識庫嗎？此操作不可逆！"):
            console.print("操作已取消", style="yellow")
            return
    
    async def _clear():
        try:
            from core.storage.vector_store import get_vector_store_manager
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("清空知識庫...", total=None)
                
                vector_store_manager = get_vector_store_manager()
                store = vector_store_manager.get_store()
                success = await store.clear_collection()
                
                if success:
                    progress.update(task, description="✅ 清空完成")
                    console.print("✅ 知識庫已清空", style="green")
                else:
                    progress.update(task, description="❌ 清空失敗")
                    console.print("❌ 清空知識庫失敗", style="red")
                    
        except Exception as e:
            console.print(f"❌ 清空失敗: {e}", style="red")
    
    run_async(_clear())


def main():
    """主入口點"""
    app()


if __name__ == "__main__":
    main()
