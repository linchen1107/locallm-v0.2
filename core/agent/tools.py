"""
Agent工具系統
提供各種工具的統一接口和管理
"""

import asyncio
import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import tempfile

from config.settings import get_settings
from config.security import get_sandbox_executor, get_permission_manager
from core.rag.retrieval_system import get_rag_system


class ToolResult:
    """工具執行結果"""
    
    def __init__(self, success: bool, data: Any = None, error: str = None, metadata: Dict[str, Any] = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class BaseTool(ABC):
    """工具基類"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.settings = get_settings()
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """執行工具"""
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """驗證參數"""
        required_params = self.parameters.get("required", [])
        for param in required_params:
            if param not in kwargs:
                return False
        return True
    
    def get_tool_info(self) -> Dict[str, Any]:
        """獲取工具信息"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class FileOperationTool(BaseTool):
    """文件操作工具"""
    
    def __init__(self):
        super().__init__(
            name="file_operation",
            description="執行文件和目錄操作（讀取、寫入、創建、刪除等）",
            parameters={
                "required": ["operation"],
                "properties": {
                    "operation": {"type": "string", "enum": ["read", "write", "create_dir", "delete", "list", "copy", "move"]},
                    "path": {"type": "string", "description": "文件或目錄路徑"},
                    "content": {"type": "string", "description": "文件內容（用於寫入操作）"},
                    "destination": {"type": "string", "description": "目標路徑（用於複製或移動操作）"}
                }
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        if not self.validate_parameters(**kwargs):
            return ToolResult(False, error="Missing required parameters")
        
        if not get_permission_manager().check_permission("write_files"):
            return ToolResult(False, error="File operation permission denied")
        
        operation = kwargs.get("operation")
        path = kwargs.get("path")
        
        try:
            if operation == "read":
                return await self._read_file(path)
            elif operation == "write":
                content = kwargs.get("content", "")
                return await self._write_file(path, content)
            elif operation == "create_dir":
                return await self._create_directory(path)
            elif operation == "delete":
                return await self._delete_path(path)
            elif operation == "list":
                return await self._list_directory(path)
            elif operation == "copy":
                destination = kwargs.get("destination")
                return await self._copy_path(path, destination)
            elif operation == "move":
                destination = kwargs.get("destination")
                return await self._move_path(path, destination)
            else:
                return ToolResult(False, error=f"Unknown operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, error=str(e))
    
    async def _read_file(self, path: str) -> ToolResult:
        """讀取文件"""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return ToolResult(True, data={"content": content, "path": path})
        except Exception as e:
            return ToolResult(False, error=f"Failed to read file: {e}")
    
    async def _write_file(self, path: str, content: str) -> ToolResult:
        """寫入文件"""
        try:
            # 確保目錄存在
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return ToolResult(True, data={"path": path, "bytes_written": len(content.encode('utf-8'))})
        except Exception as e:
            return ToolResult(False, error=f"Failed to write file: {e}")
    
    async def _create_directory(self, path: str) -> ToolResult:
        """創建目錄"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return ToolResult(True, data={"path": path, "created": True})
        except Exception as e:
            return ToolResult(False, error=f"Failed to create directory: {e}")
    
    async def _delete_path(self, path: str) -> ToolResult:
        """刪除文件或目錄"""
        try:
            path_obj = Path(path)
            if path_obj.is_file():
                path_obj.unlink()
                return ToolResult(True, data={"path": path, "type": "file", "deleted": True})
            elif path_obj.is_dir():
                shutil.rmtree(path)
                return ToolResult(True, data={"path": path, "type": "directory", "deleted": True})
            else:
                return ToolResult(False, error="Path does not exist")
        except Exception as e:
            return ToolResult(False, error=f"Failed to delete: {e}")
    
    async def _list_directory(self, path: str) -> ToolResult:
        """列出目錄內容"""
        try:
            path_obj = Path(path)
            if not path_obj.is_dir():
                return ToolResult(False, error="Path is not a directory")
            
            items = []
            for item in path_obj.iterdir():
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            
            return ToolResult(True, data={"path": path, "items": items})
        except Exception as e:
            return ToolResult(False, error=f"Failed to list directory: {e}")
    
    async def _copy_path(self, source: str, destination: str) -> ToolResult:
        """複製文件或目錄"""
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            if source_path.is_file():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                return ToolResult(True, data={"source": source, "destination": destination, "type": "file"})
            elif source_path.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
                return ToolResult(True, data={"source": source, "destination": destination, "type": "directory"})
            else:
                return ToolResult(False, error="Source path does not exist")
        except Exception as e:
            return ToolResult(False, error=f"Failed to copy: {e}")
    
    async def _move_path(self, source: str, destination: str) -> ToolResult:
        """移動文件或目錄"""
        try:
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(source, destination)
            return ToolResult(True, data={"source": source, "destination": destination})
        except Exception as e:
            return ToolResult(False, error=f"Failed to move: {e}")


class CodeExecutionTool(BaseTool):
    """代碼執行工具"""
    
    def __init__(self):
        super().__init__(
            name="code_execution",
            description="在安全沙箱中執行代碼",
            parameters={
                "required": ["code", "language"],
                "properties": {
                    "code": {"type": "string", "description": "要執行的代碼"},
                    "language": {"type": "string", "enum": ["python", "bash", "javascript"], "description": "代碼語言"},
                    "timeout": {"type": "integer", "description": "執行超時時間（秒）", "default": 30}
                }
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        if not self.validate_parameters(**kwargs):
            return ToolResult(False, error="Missing required parameters")
        
        if not self.settings.agent.enable_code_execution:
            return ToolResult(False, error="Code execution is disabled")
        
        code = kwargs.get("code")
        language = kwargs.get("language")
        timeout = kwargs.get("timeout", 30)
        
        try:
            if language == "python":
                return await self._execute_python(code, timeout)
            elif language == "bash":
                return await self._execute_bash(code, timeout)
            elif language == "javascript":
                return await self._execute_javascript(code, timeout)
            else:
                return ToolResult(False, error=f"Unsupported language: {language}")
                
        except Exception as e:
            return ToolResult(False, error=str(e))
    
    async def _execute_python(self, code: str, timeout: int) -> ToolResult:
        """執行Python代碼"""
        sandbox = get_sandbox_executor()
        result = sandbox.execute_python_code(code, timeout)
        
        return ToolResult(
            success=result["success"],
            data={
                "output": result["output"],
                "error": result["error"],
                "return_code": result["return_code"]
            },
            error=result["error"] if not result["success"] else None
        )
    
    async def _execute_bash(self, code: str, timeout: int) -> ToolResult:
        """執行Bash代碼"""
        sandbox = get_sandbox_executor()
        result = sandbox.execute_command(code, timeout)
        
        return ToolResult(
            success=result["success"],
            data={
                "output": result["output"],
                "error": result["error"],
                "return_code": result["return_code"]
            },
            error=result["error"] if not result["success"] else None
        )
    
    async def _execute_javascript(self, code: str, timeout: int) -> ToolResult:
        """執行JavaScript代碼"""
        # 簡單的Node.js執行
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            sandbox = get_sandbox_executor()
            result = sandbox.execute_command(f"node {temp_file}", timeout)
            
            return ToolResult(
                success=result["success"],
                data={
                    "output": result["output"],
                    "error": result["error"],
                    "return_code": result["return_code"]
                },
                error=result["error"] if not result["success"] else None
            )
        finally:
            os.unlink(temp_file)


class CodeGenerationTool(BaseTool):
    """代碼生成工具"""
    
    def __init__(self):
        super().__init__(
            name="code_generation",
            description="根據需求生成代碼",
            parameters={
                "required": ["description", "language"],
                "properties": {
                    "description": {"type": "string", "description": "代碼功能描述"},
                    "language": {"type": "string", "description": "編程語言"},
                    "style": {"type": "string", "description": "代碼風格要求", "default": "clean and well-commented"}
                }
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        if not self.validate_parameters(**kwargs):
            return ToolResult(False, error="Missing required parameters")
        
        if not self.settings.agent.enable_code_generation:
            return ToolResult(False, error="Code generation is disabled")
        
        description = kwargs.get("description")
        language = kwargs.get("language")
        style = kwargs.get("style", "clean and well-commented")
        
        try:
            from core.models.ollama_client import get_ollama_client
            client = await get_ollama_client()
            
            prompt = f"""Generate {language} code for the following requirement:

Requirement: {description}

Style requirements: {style}

Please provide clean, well-documented code with appropriate comments.
Include error handling where appropriate.
Follow best practices for {language}.

Code:
"""
            
            generated_code = await client.generate_code(
                prompt=description,
                language=language
            )
            
            return ToolResult(
                success=True,
                data={
                    "code": generated_code,
                    "language": language,
                    "description": description
                }
            )
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to generate code: {e}")


class KnowledgeBaseTool(BaseTool):
    """知識庫查詢工具"""
    
    def __init__(self):
        super().__init__(
            name="knowledge_base",
            description="查詢知識庫獲取相關信息",
            parameters={
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "查詢問題"},
                    "max_results": {"type": "integer", "description": "最大結果數", "default": 5},
                    "file_type": {"type": "string", "description": "文件類型過濾"}
                }
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        if not self.validate_parameters(**kwargs):
            return ToolResult(False, error="Missing required parameters")
        
        query = kwargs.get("query")
        max_results = kwargs.get("max_results", 5)
        file_type = kwargs.get("file_type")
        
        try:
            rag_system = await get_rag_system()
            
            # 構建過濾條件
            filters = {}
            if file_type:
                filters["file_type"] = file_type
            
            # 執行查詢
            result = await rag_system.query(
                question=query,
                filters=filters if filters else None,
                max_results=max_results
            )
            
            return ToolResult(
                success=True,
                data={
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "sources": [
                        {
                            "file_name": source.metadata.file_name,
                            "file_path": str(source.metadata.file_path),
                            "content_preview": source.content[:200] + "..." if len(source.content) > 200 else source.content
                        }
                        for source in result.sources
                    ],
                    "retrieval_time": result.retrieval_time,
                    "generation_time": result.generation_time
                }
            )
            
        except Exception as e:
            return ToolResult(False, error=f"Knowledge base query failed: {e}")


class DocumentSummaryTool(BaseTool):
    """文檔總結工具"""
    
    def __init__(self):
        super().__init__(
            name="document_summary",
            description="生成文檔或目錄的總結",
            parameters={
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "文檔或目錄路徑"},
                    "output_format": {"type": "string", "enum": ["txt", "md", "pdf"], "default": "md"},
                    "summary_type": {"type": "string", "enum": ["brief", "detailed", "technical"], "default": "detailed"}
                }
            }
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        if not self.validate_parameters(**kwargs):
            return ToolResult(False, error="Missing required parameters")
        
        path = kwargs.get("path")
        output_format = kwargs.get("output_format", "md")
        summary_type = kwargs.get("summary_type", "detailed")
        
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return ToolResult(False, error="Path does not exist")
            
            # 處理文檔
            from core.rag.document_processing import DocumentProcessingPipeline
            processor = DocumentProcessingPipeline()
            
            if path_obj.is_file():
                chunks = await processor.process_file(path)
            else:
                chunks = await processor.process_directory(path, recursive=True)
            
            if not chunks:
                return ToolResult(False, error="No documents found or processed")
            
            # 生成總結
            summary = await self._generate_summary(chunks, summary_type, path)
            
            # 格式化輸出
            formatted_summary = self._format_summary(summary, output_format, path)
            
            # 保存總結文件
            output_path = self._save_summary(formatted_summary, path, output_format)
            
            return ToolResult(
                success=True,
                data={
                    "summary": summary,
                    "formatted_summary": formatted_summary,
                    "output_path": output_path,
                    "document_count": len(chunks),
                    "output_format": output_format
                }
            )
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to generate summary: {e}")
    
    async def _generate_summary(self, chunks: List, summary_type: str, source_path: str) -> str:
        """生成總結"""
        from core.models.ollama_client import get_ollama_client
        
        # 合併文檔內容
        combined_content = "\n\n".join([chunk.content for chunk in chunks[:20]])  # 限制長度
        
        # 根據總結類型選擇提示
        if summary_type == "brief":
            prompt = f"""請為以下文檔內容生成一個簡短總結（2-3段）：

來源路徑：{source_path}
文檔數量：{len(chunks)}

內容：
{combined_content[:3000]}

請提供：
1. 主要內容概述
2. 關鍵要點
3. 總體結論

總結："""
        
        elif summary_type == "technical":
            prompt = f"""請為以下技術文檔生成詳細的技術總結：

來源路徑：{source_path}
文檔數量：{len(chunks)}

內容：
{combined_content[:4000]}

請包括：
1. 技術概述
2. 主要技術點
3. 實現方法
4. 技術要求
5. 注意事項

技術總結："""
        
        else:  # detailed
            prompt = f"""請為以下文檔內容生成詳細總結：

來源路徑：{source_path}
文檔數量：{len(chunks)}

內容：
{combined_content[:4000]}

請提供：
1. 背景介紹
2. 主要內容分析
3. 關鍵概念和要點
4. 結論和建議
5. 相關資源或參考

詳細總結："""
        
        client = await get_ollama_client()
        summary = await client.generate_text(prompt, temperature=0.3)
        
        return summary
    
    def _format_summary(self, summary: str, output_format: str, source_path: str) -> str:
        """格式化總結"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if output_format == "md":
            formatted = f"""# 文檔總結報告

**來源路徑：** {source_path}  
**生成時間：** {timestamp}  

---

{summary}

---

*此總結由AI助理自動生成*
"""
        elif output_format == "txt":
            formatted = f"""文檔總結報告
===============

來源路徑：{source_path}
生成時間：{timestamp}

{summary}

此總結由AI助理自動生成
"""
        else:  # pdf會在後續處理
            formatted = summary
        
        return formatted
    
    def _save_summary(self, summary: str, source_path: str, output_format: str) -> str:
        """保存總結文件"""
        source_name = Path(source_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_filename = f"summary_{source_name}_{timestamp}.{output_format}"
        output_path = Path(self.settings.storage.documents_dir) / "summaries" / output_filename
        
        # 確保目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == "pdf":
            # 簡單的PDF生成（需要reportlab）
            try:
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter
                
                c = canvas.Canvas(str(output_path), pagesize=letter)
                
                # 簡單的文本布局
                y_position = 750
                for line in summary.split('\n'):
                    if y_position < 50:
                        c.showPage()
                        y_position = 750
                    
                    c.drawString(50, y_position, line[:80])  # 限制行長度
                    y_position -= 15
                
                c.save()
            except ImportError:
                # 如果reportlab不可用，保存為文本
                with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                    f.write(summary)
                output_path = output_path.with_suffix('.txt')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
        
        return str(output_path)


class ToolRegistry:
    """工具註冊表"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """註冊默認工具"""
        default_tools = [
            FileOperationTool(),
            CodeExecutionTool(),
            CodeGenerationTool(),
            KnowledgeBaseTool(),
            DocumentSummaryTool()
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool):
        """註冊工具"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """獲取工具"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        return [tool.get_tool_info() for tool in self.tools.values()]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """執行工具"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(False, error=f"Tool '{tool_name}' not found")
        
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(False, error=f"Tool execution failed: {e}")


# 全局工具註冊表實例
_tool_registry = None


def get_tool_registry() -> ToolRegistry:
    """獲取工具註冊表實例"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry

