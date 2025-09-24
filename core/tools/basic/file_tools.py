"""
基礎文件操作工具
支持文件讀寫、目錄管理、路徑操作等
"""

import os
import shutil
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..base import BaseTool, ToolResult, ToolMetadata, ToolCategory, tool


@tool(
    category=ToolCategory.BASIC,
    name="file_operation",
    description="基礎文件操作工具，支持文件讀寫、複製、移動、刪除等操作",
    tags=["file", "io", "basic"]
)
class FileOperationTool(BaseTool):
    """文件操作工具"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        
        # 定義輸入輸出結構
        self.metadata.input_schema = {
            "operation": {"type": "string", "enum": ["read", "write", "copy", "move", "delete", "list", "create_dir"]},
            "path": {"type": "string", "description": "文件或目錄路徑"},
            "content": {"type": "string", "description": "寫入內容（僅用於write操作）"},
            "destination": {"type": "string", "description": "目標路徑（用於copy/move操作）"},
            "recursive": {"type": "boolean", "default": False, "description": "是否遞歸操作"}
        }
        
        self.metadata.output_schema = {
            "success": {"type": "boolean"},
            "content": {"type": "string", "description": "文件內容（讀取操作）"},
            "file_list": {"type": "array", "description": "文件列表（列表操作）"},
            "file_info": {"type": "object", "description": "文件信息"}
        }
    
    async def execute(self, operation: str, path: str, content: str = None, 
                     destination: str = None, recursive: bool = False, **kwargs) -> ToolResult:
        """執行文件操作"""
        
        try:
            path_obj = Path(path)
            
            if operation == "read":
                return await self._read_file(path_obj)
            elif operation == "write":
                return await self._write_file(path_obj, content)
            elif operation == "copy":
                return await self._copy_file(path_obj, Path(destination), recursive)
            elif operation == "move":
                return await self._move_file(path_obj, Path(destination))
            elif operation == "delete":
                return await self._delete_file(path_obj, recursive)
            elif operation == "list":
                return await self._list_directory(path_obj, recursive)
            elif operation == "create_dir":
                return await self._create_directory(path_obj)
            else:
                return ToolResult(False, error=f"Unsupported operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, error=f"File operation failed: {str(e)}")
    
    async def _read_file(self, path: Path) -> ToolResult:
        """讀取文件"""
        try:
            if not path.exists():
                return ToolResult(False, error=f"File not found: {path}")
            
            if path.is_dir():
                return ToolResult(False, error=f"Path is a directory: {path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_info = {
                "size": path.stat().st_size,
                "modified_time": path.stat().st_mtime,
                "extension": path.suffix
            }
            
            result = ToolResult(True, data={
                "content": content,
                "file_info": file_info
            })
            
            # 添加可分享數據
            result.add_shareable_data("file_content", content)
            result.add_shareable_data("file_path", str(path))
            result.add_shareable_data("file_info", file_info)
            
            # 建議相關工具
            if path.suffix in ['.py', '.js', '.ts']:
                result.suggest_next_tool("code_analysis", "檢測到代碼文件，建議進行代碼分析")
            elif path.suffix in ['.txt', '.md']:
                result.suggest_next_tool("text_analysis", "檢測到文本文件，建議進行文本分析")
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to read file: {str(e)}")
    
    async def _write_file(self, path: Path, content: str) -> ToolResult:
        """寫入文件"""
        try:
            # 確保父目錄存在
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_info = {
                "size": path.stat().st_size,
                "created": True
            }
            
            result = ToolResult(True, data={
                "message": f"File written successfully: {path}",
                "file_info": file_info
            })
            
            result.add_shareable_data("created_file", str(path))
            result.add_shareable_data("file_size", file_info["size"])
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to write file: {str(e)}")
    
    async def _copy_file(self, source: Path, destination: Path, recursive: bool) -> ToolResult:
        """複製文件或目錄"""
        try:
            if source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                operation = "file_copied"
            elif source.is_dir() and recursive:
                shutil.copytree(source, destination, dirs_exist_ok=True)
                operation = "directory_copied"
            else:
                return ToolResult(False, error="Source is directory but recursive=False")
            
            result = ToolResult(True, data={
                "message": f"Successfully copied {source} to {destination}",
                "operation": operation
            })
            
            result.add_shareable_data("copied_to", str(destination))
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to copy: {str(e)}")
    
    async def _move_file(self, source: Path, destination: Path) -> ToolResult:
        """移動文件或目錄"""
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, destination)
            
            result = ToolResult(True, data={
                "message": f"Successfully moved {source} to {destination}"
            })
            
            result.add_shareable_data("moved_to", str(destination))
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to move: {str(e)}")
    
    async def _delete_file(self, path: Path, recursive: bool) -> ToolResult:
        """刪除文件或目錄"""
        try:
            if path.is_file():
                path.unlink()
                operation = "file_deleted"
            elif path.is_dir():
                if recursive:
                    shutil.rmtree(path)
                    operation = "directory_deleted"
                else:
                    path.rmdir()  # 只能刪除空目錄
                    operation = "empty_directory_deleted"
            else:
                return ToolResult(False, error=f"Path not found: {path}")
            
            result = ToolResult(True, data={
                "message": f"Successfully deleted {path}",
                "operation": operation
            })
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to delete: {str(e)}")
    
    async def _list_directory(self, path: Path, recursive: bool) -> ToolResult:
        """列出目錄內容"""
        try:
            if not path.exists():
                return ToolResult(False, error=f"Directory not found: {path}")
            
            if not path.is_dir():
                return ToolResult(False, error=f"Path is not a directory: {path}")
            
            file_list = []
            
            if recursive:
                for item in path.rglob("*"):
                    file_list.append({
                        "path": str(item.relative_to(path)),
                        "full_path": str(item),
                        "type": "file" if item.is_file() else "directory",
                        "size": item.stat().st_size if item.is_file() else None
                    })
            else:
                for item in path.iterdir():
                    file_list.append({
                        "path": item.name,
                        "full_path": str(item),
                        "type": "file" if item.is_file() else "directory",
                        "size": item.stat().st_size if item.is_file() else None
                    })
            
            result = ToolResult(True, data={
                "file_list": file_list,
                "total_items": len(file_list)
            })
            
            result.add_shareable_data("directory_listing", file_list)
            result.add_shareable_data("directory_path", str(path))
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to list directory: {str(e)}")
    
    async def _create_directory(self, path: Path) -> ToolResult:
        """創建目錄"""
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            result = ToolResult(True, data={
                "message": f"Directory created: {path}",
                "path": str(path)
            })
            
            result.add_shareable_data("created_directory", str(path))
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Failed to create directory: {str(e)}")


@tool(
    category=ToolCategory.BASIC,
    name="path_utility",
    description="路徑處理工具，提供路徑分析、轉換、驗證等功能",
    tags=["path", "utility", "filesystem"]
)
class PathUtilityTool(BaseTool):
    """路徑工具"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        
        self.metadata.input_schema = {
            "operation": {"type": "string", "enum": ["analyze", "join", "relative", "absolute", "exists", "validate"]},
            "path": {"type": "string", "description": "主要路徑"},
            "base_path": {"type": "string", "description": "基礎路徑（用於相對路徑計算）"},
            "paths": {"type": "array", "description": "路徑列表（用於join操作）"}
        }
    
    async def execute(self, operation: str, path: str = None, base_path: str = None, 
                     paths: List[str] = None, **kwargs) -> ToolResult:
        """執行路徑操作"""
        
        try:
            if operation == "analyze":
                return await self._analyze_path(Path(path))
            elif operation == "join":
                return await self._join_paths(paths or [])
            elif operation == "relative":
                return await self._get_relative_path(Path(path), Path(base_path))
            elif operation == "absolute":
                return await self._get_absolute_path(Path(path))
            elif operation == "exists":
                return await self._check_exists(Path(path))
            elif operation == "validate":
                return await self._validate_path(path)
            else:
                return ToolResult(False, error=f"Unsupported operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, error=f"Path operation failed: {str(e)}")
    
    async def _analyze_path(self, path: Path) -> ToolResult:
        """分析路徑信息"""
        analysis = {
            "original": str(path),
            "absolute": str(path.absolute()),
            "parent": str(path.parent),
            "name": path.name,
            "stem": path.stem,
            "suffix": path.suffix,
            "parts": list(path.parts),
            "exists": path.exists(),
            "is_file": path.is_file() if path.exists() else None,
            "is_dir": path.is_dir() if path.exists() else None
        }
        
        if path.exists() and path.is_file():
            stat = path.stat()
            analysis["size"] = stat.st_size
            analysis["modified_time"] = stat.st_mtime
        
        result = ToolResult(True, data=analysis)
        result.add_shareable_data("path_analysis", analysis)
        
        return result
    
    async def _join_paths(self, paths: List[str]) -> ToolResult:
        """連接路徑"""
        if not paths:
            return ToolResult(False, error="No paths provided")
        
        joined_path = Path(paths[0])
        for path_part in paths[1:]:
            joined_path = joined_path / path_part
        
        result = ToolResult(True, data={
            "joined_path": str(joined_path),
            "absolute_path": str(joined_path.absolute())
        })
        
        result.add_shareable_data("joined_path", str(joined_path))
        
        return result
    
    async def _get_relative_path(self, path: Path, base_path: Path) -> ToolResult:
        """獲取相對路徑"""
        try:
            relative_path = path.relative_to(base_path)
            result = ToolResult(True, data={
                "relative_path": str(relative_path)
            })
            result.add_shareable_data("relative_path", str(relative_path))
            return result
        except ValueError as e:
            return ToolResult(False, error=f"Cannot compute relative path: {str(e)}")
    
    async def _get_absolute_path(self, path: Path) -> ToolResult:
        """獲取絕對路徑"""
        absolute_path = path.absolute()
        result = ToolResult(True, data={
            "absolute_path": str(absolute_path)
        })
        result.add_shareable_data("absolute_path", str(absolute_path))
        return result
    
    async def _check_exists(self, path: Path) -> ToolResult:
        """檢查路徑是否存在"""
        exists = path.exists()
        result = ToolResult(True, data={
            "exists": exists,
            "path": str(path),
            "type": "file" if path.is_file() else "directory" if path.is_dir() else "not_found"
        })
        result.add_shareable_data("path_exists", exists)
        return result
    
    async def _validate_path(self, path: str) -> ToolResult:
        """驗證路徑格式"""
        issues = []
        
        # 檢查非法字符
        illegal_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in illegal_chars:
            if char in path:
                issues.append(f"Contains illegal character: {char}")
        
        # 檢查長度
        if len(path) > 260:  # Windows路徑長度限制
            issues.append("Path too long (>260 characters)")
        
        # 檢查保留名稱
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        path_obj = Path(path)
        if path_obj.stem.upper() in reserved_names:
            issues.append(f"Uses reserved name: {path_obj.stem}")
        
        is_valid = len(issues) == 0
        
        result = ToolResult(True, data={
            "valid": is_valid,
            "issues": issues,
            "path": path
        })
        
        result.add_shareable_data("path_valid", is_valid)
        
        return result

