"""
開發領域專用工具
支持Git操作、項目管理、測試生成、代碼分析等
"""

import subprocess
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import ast

from ..base import BaseTool, ToolResult, ToolMetadata, ToolCategory, tool


@tool(
    category=ToolCategory.DOMAIN,
    name="git_operation",
    description="Git操作工具，支持版本控制、分支管理、提交記錄等",
    tags=["git", "version-control", "development"]
)
class GitOperationTool(BaseTool):
    """Git操作工具"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        
        self.metadata.input_schema = {
            "operation": {"type": "string", "enum": ["status", "add", "commit", "push", "pull", "branch", "log", "diff"]},
            "repository_path": {"type": "string", "description": "Git倉庫路徑"},
            "files": {"type": "array", "description": "文件列表"},
            "message": {"type": "string", "description": "提交信息"},
            "branch_name": {"type": "string", "description": "分支名稱"},
            "remote": {"type": "string", "default": "origin", "description": "遠程倉庫名"},
            "limit": {"type": "number", "default": 10, "description": "日誌條目限制"}
        }
    
    async def execute(self, operation: str, repository_path: str = ".", 
                     files: List[str] = None, message: str = None,
                     branch_name: str = None, remote: str = "origin",
                     limit: int = 10, **kwargs) -> ToolResult:
        """執行Git操作"""
        
        try:
            repo_path = Path(repository_path).absolute()
            
            if operation == "status":
                return await self._git_status(repo_path)
            elif operation == "add":
                return await self._git_add(repo_path, files)
            elif operation == "commit":
                return await self._git_commit(repo_path, message)
            elif operation == "push":
                return await self._git_push(repo_path, remote, branch_name)
            elif operation == "pull":
                return await self._git_pull(repo_path, remote, branch_name)
            elif operation == "branch":
                return await self._git_branch(repo_path, branch_name)
            elif operation == "log":
                return await self._git_log(repo_path, limit)
            elif operation == "diff":
                return await self._git_diff(repo_path, files)
            else:
                return ToolResult(False, error=f"Unsupported operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, error=f"Git operation failed: {str(e)}")
    
    async def _run_git_command(self, repo_path: Path, command: List[str]) -> Dict[str, Any]:
        """運行Git命令"""
        try:
            result = subprocess.run(
                ["git"] + command,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    async def _git_status(self, repo_path: Path) -> ToolResult:
        """Git狀態"""
        result = await self._run_git_command(repo_path, ["status", "--porcelain"])
        
        if not result["success"]:
            return ToolResult(False, error=f"Git status failed: {result['stderr']}")
        
        # 解析狀態
        files = []
        for line in result["stdout"].strip().split('\n'):
            if line:
                status = line[:2]
                filename = line[3:]
                files.append({
                    "file": filename,
                    "status": status,
                    "staged": status[0] != ' ' and status[0] != '?',
                    "modified": status[1] != ' '
                })
        
        # 獲取分支信息
        branch_result = await self._run_git_command(repo_path, ["branch", "--show-current"])
        current_branch = branch_result["stdout"].strip() if branch_result["success"] else "unknown"
        
        result_data = {
            "current_branch": current_branch,
            "files": files,
            "total_files": len(files),
            "staged_files": len([f for f in files if f["staged"]]),
            "modified_files": len([f for f in files if f["modified"]])
        }
        
        result = ToolResult(True, data=result_data)
        result.add_shareable_data("git_status", result_data)
        result.add_shareable_data("current_branch", current_branch)
        
        if files:
            result.suggest_next_tool("git_operation", "檢測到修改的文件，建議添加並提交")
        
        return result
    
    async def _git_add(self, repo_path: Path, files: List[str] = None) -> ToolResult:
        """添加文件到暫存區"""
        if files:
            command = ["add"] + files
        else:
            command = ["add", "."]
        
        result = await self._run_git_command(repo_path, command)
        
        if not result["success"]:
            return ToolResult(False, error=f"Git add failed: {result['stderr']}")
        
        result_data = {
            "message": "Files added to staging area",
            "files_added": files or ["all files"]
        }
        
        result = ToolResult(True, data=result_data)
        result.add_shareable_data("files_staged", files or ["all"])
        result.suggest_next_tool("git_operation", "文件已添加到暫存區，建議提交更改")
        
        return result
    
    async def _git_commit(self, repo_path: Path, message: str = None) -> ToolResult:
        """提交更改"""
        if not message:
            return ToolResult(False, error="Commit message is required")
        
        result = await self._run_git_command(repo_path, ["commit", "-m", message])
        
        if not result["success"]:
            return ToolResult(False, error=f"Git commit failed: {result['stderr']}")
        
        # 獲取提交哈希
        hash_result = await self._run_git_command(repo_path, ["rev-parse", "HEAD"])
        commit_hash = hash_result["stdout"].strip()[:7] if hash_result["success"] else "unknown"
        
        result_data = {
            "message": "Changes committed successfully",
            "commit_hash": commit_hash,
            "commit_message": message
        }
        
        result = ToolResult(True, data=result_data)
        result.add_shareable_data("commit_hash", commit_hash)
        result.add_shareable_data("commit_message", message)
        result.suggest_next_tool("git_operation", "提交完成，建議推送到遠程倉庫")
        
        return result
    
    async def _git_log(self, repo_path: Path, limit: int) -> ToolResult:
        """查看提交日誌"""
        command = ["log", f"--max-count={limit}", "--pretty=format:%h|%an|%ad|%s", "--date=short"]
        result = await self._run_git_command(repo_path, command)
        
        if not result["success"]:
            return ToolResult(False, error=f"Git log failed: {result['stderr']}")
        
        commits = []
        for line in result["stdout"].strip().split('\n'):
            if line:
                parts = line.split('|')
                if len(parts) == 4:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "message": parts[3]
                    })
        
        result_data = {
            "commits": commits,
            "total_commits": len(commits)
        }
        
        result = ToolResult(True, data=result_data)
        result.add_shareable_data("commit_history", commits)
        
        return result
    
    async def _git_diff(self, repo_path: Path, files: List[str] = None) -> ToolResult:
        """查看文件差異"""
        command = ["diff"]
        if files:
            command.extend(files)
        
        result = await self._run_git_command(repo_path, command)
        
        if not result["success"]:
            return ToolResult(False, error=f"Git diff failed: {result['stderr']}")
        
        result_data = {
            "diff_output": result["stdout"],
            "has_changes": bool(result["stdout"].strip())
        }
        
        result = ToolResult(True, data=result_data)
        result.add_shareable_data("code_diff", result["stdout"])
        
        return result


@tool(
    category=ToolCategory.DOMAIN,
    name="project_management",
    description="項目管理工具，支持項目結構分析、依賴管理、構建任務等",
    tags=["project", "management", "structure", "dependencies"]
)
class ProjectManagementTool(BaseTool):
    """項目管理工具"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        
        self.metadata.input_schema = {
            "operation": {"type": "string", "enum": ["analyze", "dependencies", "structure", "build", "test", "lint"]},
            "project_path": {"type": "string", "description": "項目路徑"},
            "config_file": {"type": "string", "description": "配置文件路徑"},
            "language": {"type": "string", "enum": ["python", "javascript", "typescript", "auto"], "default": "auto"}
        }
    
    async def execute(self, operation: str, project_path: str = ".", 
                     config_file: str = None, language: str = "auto", **kwargs) -> ToolResult:
        """執行項目管理操作"""
        
        try:
            project_path_obj = Path(project_path).absolute()
            
            if operation == "analyze":
                return await self._analyze_project(project_path_obj)
            elif operation == "dependencies":
                return await self._analyze_dependencies(project_path_obj, language)
            elif operation == "structure":
                return await self._analyze_structure(project_path_obj)
            elif operation == "build":
                return await self._build_project(project_path_obj, language)
            elif operation == "test":
                return await self._run_tests(project_path_obj, language)
            elif operation == "lint":
                return await self._lint_code(project_path_obj, language)
            else:
                return ToolResult(False, error=f"Unsupported operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, error=f"Project management failed: {str(e)}")
    
    async def _detect_language(self, project_path: Path) -> str:
        """檢測項目語言"""
        if (project_path / "requirements.txt").exists() or (project_path / "setup.py").exists():
            return "python"
        elif (project_path / "package.json").exists():
            return "javascript"
        elif (project_path / "tsconfig.json").exists():
            return "typescript"
        else:
            return "unknown"
    
    async def _analyze_project(self, project_path: Path) -> ToolResult:
        """分析項目"""
        try:
            detected_language = await self._detect_language(project_path)
            
            # 統計文件類型
            file_stats = {}
            total_files = 0
            total_size = 0
            
            for file_path in project_path.rglob("*"):
                if file_path.is_file():
                    suffix = file_path.suffix or "no_extension"
                    file_stats[suffix] = file_stats.get(suffix, 0) + 1
                    total_files += 1
                    total_size += file_path.stat().st_size
            
            # 檢查配置文件
            config_files = []
            common_configs = [
                "requirements.txt", "setup.py", "pyproject.toml",  # Python
                "package.json", "yarn.lock", "package-lock.json",  # Node.js
                "Makefile", "Dockerfile", ".gitignore", "README.md"
            ]
            
            for config in common_configs:
                if (project_path / config).exists():
                    config_files.append(config)
            
            analysis = {
                "project_path": str(project_path),
                "detected_language": detected_language,
                "total_files": total_files,
                "total_size_bytes": total_size,
                "file_type_distribution": file_stats,
                "config_files": config_files
            }
            
            result = ToolResult(True, data=analysis)
            result.add_shareable_data("project_analysis", analysis)
            result.add_shareable_data("project_language", detected_language)
            
            result.suggest_next_tool("project_management", "項目分析完成，建議檢查依賴關係")
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Project analysis failed: {str(e)}")
    
    async def _analyze_dependencies(self, project_path: Path, language: str) -> ToolResult:
        """分析項目依賴"""
        if language == "auto":
            language = await self._detect_language(project_path)
        
        dependencies = {}
        
        try:
            if language == "python":
                # 檢查requirements.txt
                req_file = project_path / "requirements.txt"
                if req_file.exists():
                    with open(req_file, 'r') as f:
                        dependencies["requirements"] = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                # 檢查setup.py
                setup_file = project_path / "setup.py"
                if setup_file.exists():
                    dependencies["setup_py"] = "Found setup.py file"
            
            elif language in ["javascript", "typescript"]:
                # 檢查package.json
                package_file = project_path / "package.json"
                if package_file.exists():
                    with open(package_file, 'r') as f:
                        package_data = json.load(f)
                        dependencies["dependencies"] = package_data.get("dependencies", {})
                        dependencies["devDependencies"] = package_data.get("devDependencies", {})
            
            result_data = {
                "language": language,
                "dependencies": dependencies,
                "dependency_count": len(dependencies.get("dependencies", {})) + len(dependencies.get("requirements", []))
            }
            
            result = ToolResult(True, data=result_data)
            result.add_shareable_data("project_dependencies", dependencies)
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Dependency analysis failed: {str(e)}")
    
    async def _analyze_structure(self, project_path: Path) -> ToolResult:
        """分析項目結構"""
        try:
            structure = {}
            
            def build_tree(path: Path, max_depth: int = 3, current_depth: int = 0) -> Dict:
                if current_depth >= max_depth:
                    return {"...": "max depth reached"}
                
                tree = {}
                try:
                    for item in sorted(path.iterdir()):
                        if item.name.startswith('.') and item.name not in ['.gitignore', '.env']:
                            continue
                        
                        if item.is_dir():
                            tree[f"{item.name}/"] = build_tree(item, max_depth, current_depth + 1)
                        else:
                            tree[item.name] = {"type": "file", "size": item.stat().st_size}
                except PermissionError:
                    tree["<permission denied>"] = {}
                
                return tree
            
            structure = build_tree(project_path)
            
            result_data = {
                "project_structure": structure,
                "analysis_depth": 3
            }
            
            result = ToolResult(True, data=result_data)
            result.add_shareable_data("project_structure", structure)
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Structure analysis failed: {str(e)}")


@tool(
    category=ToolCategory.DOMAIN,
    name="test_generation",
    description="測試生成工具，支持單元測試、集成測試、測試報告生成",
    tags=["testing", "unit-test", "integration", "pytest"]
)
class TestGenerationTool(BaseTool):
    """測試生成工具"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        
        self.metadata.input_schema = {
            "operation": {"type": "string", "enum": ["generate", "run", "coverage", "report"]},
            "source_file": {"type": "string", "description": "源代碼文件路徑"},
            "test_framework": {"type": "string", "enum": ["pytest", "unittest", "jest"], "default": "pytest"},
            "test_type": {"type": "string", "enum": ["unit", "integration", "functional"], "default": "unit"}
        }
    
    async def execute(self, operation: str, source_file: str = None,
                     test_framework: str = "pytest", test_type: str = "unit", **kwargs) -> ToolResult:
        """執行測試操作"""
        
        try:
            if operation == "generate":
                return await self._generate_tests(source_file, test_framework, test_type)
            elif operation == "run":
                return await self._run_tests(test_framework)
            elif operation == "coverage":
                return await self._run_coverage(test_framework)
            elif operation == "report":
                return await self._generate_report(test_framework)
            else:
                return ToolResult(False, error=f"Unsupported operation: {operation}")
                
        except Exception as e:
            return ToolResult(False, error=f"Test operation failed: {str(e)}")
    
    async def _generate_tests(self, source_file: str, framework: str, test_type: str) -> ToolResult:
        """生成測試代碼"""
        try:
            source_path = Path(source_file)
            if not source_path.exists():
                return ToolResult(False, error=f"Source file not found: {source_file}")
            
            # 讀取源代碼
            with open(source_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # 解析Python代碼中的函數和類
            try:
                tree = ast.parse(source_code)
                functions = []
                classes = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                
            except SyntaxError:
                return ToolResult(False, error="Failed to parse source code")
            
            # 生成測試代碼
            if framework == "pytest":
                test_code = self._generate_pytest_code(source_path.stem, functions, classes)
            elif framework == "unittest":
                test_code = self._generate_unittest_code(source_path.stem, functions, classes)
            else:
                return ToolResult(False, error=f"Unsupported framework: {framework}")
            
            # 保存測試文件
            test_filename = f"test_{source_path.stem}.py"
            test_path = source_path.parent / test_filename
            
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            result_data = {
                "test_file": str(test_path),
                "functions_tested": functions,
                "classes_tested": classes,
                "framework": framework,
                "test_code": test_code
            }
            
            result = ToolResult(True, data=result_data)
            result.add_shareable_data("test_file", str(test_path))
            result.add_shareable_data("test_code", test_code)
            
            result.suggest_next_tool("test_generation", "測試代碼已生成，建議運行測試")
            
            return result
            
        except Exception as e:
            return ToolResult(False, error=f"Test generation failed: {str(e)}")
    
    def _generate_pytest_code(self, module_name: str, functions: List[str], classes: List[str]) -> str:
        """生成pytest測試代碼"""
        lines = [
            f'"""',
            f'Tests for {module_name} module',
            f'Generated automatically',
            f'"""',
            f'',
            f'import pytest',
            f'from {module_name} import *',
            f'',
            f''
        ]
        
        # 為每個函數生成測試
        for func in functions:
            if not func.startswith('_'):  # 跳過私有函數
                lines.extend([
                    f'def test_{func}():',
                    f'    """Test {func} function"""',
                    f'    # TODO: Implement test for {func}',
                    f'    assert True  # Placeholder',
                    f'',
                    f''
                ])
        
        # 為每個類生成測試
        for cls in classes:
            lines.extend([
                f'class Test{cls}:',
                f'    """Test class for {cls}"""',
                f'    ',
                f'    def test_init(self):',
                f'        """Test {cls} initialization"""',
                f'        # TODO: Implement initialization test',
                f'        assert True  # Placeholder',
                f'    ',
                f'    def test_methods(self):',
                f'        """Test {cls} methods"""',
                f'        # TODO: Implement method tests',
                f'        assert True  # Placeholder',
                f'',
                f''
            ])
        
        return '\n'.join(lines)
    
    def _generate_unittest_code(self, module_name: str, functions: List[str], classes: List[str]) -> str:
        """生成unittest測試代碼"""
        lines = [
            f'"""',
            f'Tests for {module_name} module',
            f'Generated automatically',
            f'"""',
            f'',
            f'import unittest',
            f'from {module_name} import *',
            f'',
            f'',
            f'class Test{module_name.capitalize()}(unittest.TestCase):',
            f'    """Test case for {module_name} module"""',
            f'    ',
            f'    def setUp(self):',
            f'        """Set up test fixtures"""',
            f'        pass',
            f'    ',
            f'    def tearDown(self):',
            f'        """Tear down test fixtures"""',
            f'        pass',
            f'    '
        ]
        
        # 為每個函數生成測試方法
        for func in functions:
            if not func.startswith('_'):
                lines.extend([
                    f'    def test_{func}(self):',
                    f'        """Test {func} function"""',
                    f'        # TODO: Implement test for {func}',
                    f'        self.assertTrue(True)  # Placeholder',
                    f'    '
                ])
        
        lines.extend([
            f'',
            f'',
            f'if __name__ == "__main__":',
            f'    unittest.main()'
        ])
        
        return '\n'.join(lines)

