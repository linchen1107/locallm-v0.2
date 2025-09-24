"""
安全執行框架
提供全面的安全執行和監控功能
"""

import asyncio
import os
import psutil
import signal
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None
from datetime import datetime, timedelta

from config.settings import get_settings
from config.security import (
    get_security_manager, get_sandbox_executor, 
    get_permission_manager, get_security_auditor
)


class SecurityViolationError(Exception):
    """安全違規錯誤"""
    pass


class ResourceLimitExceededError(Exception):
    """資源限制超出錯誤"""
    pass


class SecurityPolicy:
    """安全政策"""
    
    def __init__(self):
        self.settings = get_settings()
        self.max_memory = self.settings.security.max_memory_usage
        self.max_execution_time = self.settings.agent.max_execution_time
        self.allowed_file_extensions = ['.py', '.txt', '.md', '.json', '.csv', '.pdf', '.docx']
        self.blocked_imports = [
            'os', 'subprocess', 'sys', 'shutil', 'socket', 'urllib', 'requests',
            'http', 'ftplib', 'smtplib', 'telnetlib', 'paramiko'
        ]
        self.safe_builtins = {
            'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float',
            'int', 'len', 'list', 'map', 'max', 'min', 'range', 'set', 'str',
            'sum', 'tuple', 'type', 'zip', 'sorted', 'reversed'
        }
    
    def validate_code(self, code: str) -> List[str]:
        """驗證代碼安全性"""
        violations = []
        
        # 檢查危險導入
        for blocked in self.blocked_imports:
            if f"import {blocked}" in code or f"from {blocked}" in code:
                violations.append(f"Blocked import detected: {blocked}")
        
        # 檢查危險函數調用
        dangerous_patterns = [
            'exec(', 'eval(', 'compile(', '__import__(',
            'getattr(', 'setattr(', 'delattr(', 'hasattr(',
            'globals(', 'locals(', 'vars(', 'dir(',
            'open(', 'file(', 'input(', 'raw_input('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                violations.append(f"Dangerous function call detected: {pattern}")
        
        return violations
    
    def validate_file_access(self, file_path: Union[str, Path]) -> bool:
        """驗證文件訪問權限"""
        path = Path(file_path)
        
        # 檢查文件是否在允許的目錄內
        allowed_dirs = [
            self.settings.storage.data_dir,
            self.settings.storage.documents_dir,
            Path.cwd()
        ]
        
        try:
            for allowed_dir in allowed_dirs:
                if path.resolve().is_relative_to(allowed_dir.resolve()):
                    return True
        except (ValueError, OSError):
            pass
        
        return False
    
    def check_resource_usage(self, process_id: int) -> Dict[str, Any]:
        """檢查進程資源使用情況"""
        try:
            process = psutil.Process(process_id)
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            return {
                "memory_rss": memory_info.rss,
                "memory_vms": memory_info.vms,
                "cpu_percent": cpu_percent,
                "within_limits": memory_info.rss <= self.max_memory
            }
        except psutil.NoSuchProcess:
            return {"error": "Process not found"}


class SecureExecutionEnvironment:
    """安全執行環境"""
    
    def __init__(self):
        self.settings = get_settings()
        self.policy = SecurityPolicy()
        self.active_processes: Dict[int, Dict[str, Any]] = {}
        self.resource_monitor_active = False
    
    async def execute_with_monitoring(self, 
                                    func: Callable,
                                    *args, 
                                    timeout: Optional[int] = None,
                                    memory_limit: Optional[int] = None,
                                    **kwargs) -> Any:
        """在監控環境中執行函數"""
        timeout = timeout or self.settings.agent.max_execution_time
        memory_limit = memory_limit or self.settings.security.max_memory_usage
        
        # 創建執行任務
        task = asyncio.create_task(func(*args, **kwargs))
        
        # 創建監控任務
        monitor_task = asyncio.create_task(
            self._monitor_execution(task, timeout, memory_limit)
        )
        
        try:
            # 等待執行完成或監控中斷
            done, pending = await asyncio.wait(
                [task, monitor_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 取消未完成的任務
            for pending_task in pending:
                pending_task.cancel()
                try:
                    await pending_task
                except asyncio.CancelledError:
                    pass
            
            # 獲取結果
            for completed_task in done:
                if completed_task == task:
                    return await completed_task
                elif completed_task == monitor_task:
                    # 監控任務完成意味著發生了違規
                    raise SecurityViolationError("Security violation detected during execution")
            
        except asyncio.TimeoutError:
            task.cancel()
            raise ResourceLimitExceededError(f"Execution timeout after {timeout} seconds")
        except Exception as e:
            task.cancel()
            raise e
    
    async def _monitor_execution(self, task: asyncio.Task, timeout: int, memory_limit: int):
        """監控執行過程"""
        start_time = time.time()
        
        while not task.done():
            await asyncio.sleep(0.1)  # 檢查間隔
            
            # 檢查超時
            if time.time() - start_time > timeout:
                raise asyncio.TimeoutError()
            
            # 檢查內存使用
            current_process = psutil.Process()
            memory_usage = current_process.memory_info().rss
            
            if memory_usage > memory_limit:
                raise ResourceLimitExceededError(f"Memory limit exceeded: {memory_usage} > {memory_limit}")
    
    @contextmanager
    def secure_temporary_directory(self):
        """創建安全的臨時目錄"""
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="ai_assistant_secure_")
            
            # 設置目錄權限
            os.chmod(temp_dir, 0o700)
            
            # 記錄臨時目錄創建
            get_security_auditor().log_action(
                "temp_dir_created",
                details={"path": temp_dir}
            )
            
            yield Path(temp_dir)
            
        finally:
            if temp_dir and os.path.exists(temp_dir):
                # 清理臨時目錄
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                get_security_auditor().log_action(
                    "temp_dir_cleanup",
                    details={"path": temp_dir}
                )
    
    def validate_and_sanitize_input(self, input_data: Any, input_type: str) -> Any:
        """驗證和清理輸入數據"""
        if input_type == "code":
            if isinstance(input_data, str):
                violations = self.policy.validate_code(input_data)
                if violations:
                    raise SecurityViolationError(f"Code validation failed: {violations}")
                return input_data
            else:
                raise SecurityViolationError("Code input must be string")
        
        elif input_type == "file_path":
            if isinstance(input_data, (str, Path)):
                path = Path(input_data)
                if not self.policy.validate_file_access(path):
                    raise SecurityViolationError(f"File access denied: {path}")
                return path
            else:
                raise SecurityViolationError("File path must be string or Path")
        
        elif input_type == "json":
            if isinstance(input_data, str):
                try:
                    import json
                    return json.loads(input_data)
                except json.JSONDecodeError:
                    raise SecurityViolationError("Invalid JSON format")
            else:
                return input_data
        
        return input_data


class SecureCodeExecutor:
    """安全代碼執行器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.environment = SecureExecutionEnvironment()
        self.policy = SecurityPolicy()
    
    async def execute_python_code_secure(self, code: str, 
                                       timeout: Optional[int] = None,
                                       allowed_imports: Optional[List[str]] = None) -> Dict[str, Any]:
        """安全執行Python代碼"""
        
        # 驗證代碼
        violations = self.policy.validate_code(code)
        if violations:
            return {
                "success": False,
                "error": f"Security violations: {violations}",
                "output": "",
                "violations": violations
            }
        
        # 準備安全的執行環境
        safe_globals = {
            "__builtins__": {
                name: getattr(__builtins__, name) 
                for name in self.policy.safe_builtins
                if hasattr(__builtins__, name)
            }
        }
        
        # 添加允許的導入
        if allowed_imports:
            for module_name in allowed_imports:
                try:
                    safe_globals[module_name] = __import__(module_name)
                except ImportError:
                    pass
        
        # 在臨時目錄中執行
        with self.environment.secure_temporary_directory() as temp_dir:
            # 創建代碼文件
            code_file = temp_dir / "secure_code.py"
            code_file.write_text(code, encoding='utf-8')
            
            # 執行代碼
            try:
                result = await self._execute_in_subprocess(
                    str(code_file), 
                    timeout or self.settings.security.sandbox_timeout
                )
                
                return result
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "output": "",
                    "security_error": True
                }
    
    async def _execute_in_subprocess(self, code_file: str, timeout: int) -> Dict[str, Any]:
        """在子進程中執行代碼"""
        import subprocess
        
        try:
            # 構建執行命令
            cmd = [
                "python", "-c", 
                f"exec(open('{code_file}').read())"
            ]
            
            # 執行子進程
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024*1024  # 1MB輸出限制
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                return {
                    "success": process.returncode == 0,
                    "output": stdout.decode('utf-8', errors='ignore'),
                    "error": stderr.decode('utf-8', errors='ignore'),
                    "return_code": process.returncode
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "error": f"Execution timeout after {timeout} seconds",
                    "output": "",
                    "timeout": True
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Subprocess execution failed: {e}",
                "output": "",
                "subprocess_error": True
            }


class SecurityManager:
    """安全管理器主類"""
    
    def __init__(self):
        self.settings = get_settings()
        self.environment = SecureExecutionEnvironment()
        self.code_executor = SecureCodeExecutor()
        self.policy = SecurityPolicy()
        
    async def secure_function_execution(self, 
                                      func: Callable,
                                      *args,
                                      security_level: str = "medium",
                                      **kwargs) -> Any:
        """安全函數執行"""
        
        # 根據安全級別設置限制
        if security_level == "high":
            timeout = 30
            memory_limit = 256 * 1024 * 1024  # 256MB
        elif security_level == "medium":
            timeout = 60
            memory_limit = 512 * 1024 * 1024  # 512MB
        else:  # low
            timeout = 120
            memory_limit = 1024 * 1024 * 1024  # 1GB
        
        # 記錄執行開始
        get_security_auditor().log_action(
            "secure_execution_start",
            details={
                "function": func.__name__ if hasattr(func, '__name__') else str(func),
                "security_level": security_level,
                "timeout": timeout,
                "memory_limit": memory_limit
            }
        )
        
        try:
            result = await self.environment.execute_with_monitoring(
                func, *args,
                timeout=timeout,
                memory_limit=memory_limit,
                **kwargs
            )
            
            # 記錄執行成功
            get_security_auditor().log_action(
                "secure_execution_success",
                details={"function": func.__name__ if hasattr(func, '__name__') else str(func)}
            )
            
            return result
            
        except (SecurityViolationError, ResourceLimitExceededError) as e:
            # 記錄安全事件
            get_security_auditor().log_security_event(
                "execution_security_violation",
                severity="warning",
                details={
                    "function": func.__name__ if hasattr(func, '__name__') else str(func),
                    "error": str(e),
                    "security_level": security_level
                }
            )
            raise e
    
    def validate_user_input(self, input_data: Any, input_type: str) -> Any:
        """驗證用戶輸入"""
        return self.environment.validate_and_sanitize_input(input_data, input_type)
    
    async def safe_code_execution(self, code: str, language: str = "python") -> Dict[str, Any]:
        """安全代碼執行"""
        if language == "python":
            return await self.code_executor.execute_python_code_secure(code)
        else:
            return {
                "success": False,
                "error": f"Unsupported language: {language}",
                "output": ""
            }
    
    def get_security_status(self) -> Dict[str, Any]:
        """獲取安全狀態"""
        return {
            "sandbox_enabled": self.settings.agent.sandbox_enabled,
            "code_execution_enabled": self.settings.agent.enable_code_execution,
            "security_policies_active": True,
            "resource_monitoring": True,
            "audit_logging": True,
            "settings": {
                "max_memory": self.settings.security.max_memory_usage,
                "max_execution_time": self.settings.agent.max_execution_time,
                "sandbox_timeout": self.settings.security.sandbox_timeout
            }
        }


# 全局安全管理器實例
_security_manager = None


def get_security_framework() -> SecurityManager:
    """獲取安全管理器實例"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager
