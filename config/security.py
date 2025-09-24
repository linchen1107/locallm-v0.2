"""
安全管理模塊
提供加密、沙箱執行、權限控制等安全功能
"""

import hashlib
import hmac
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from cryptography.fernet import Fernet
from datetime import datetime, timedelta
import psutil
import signal


class SecurityManager:
    """安全管理器"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.environ.get("SECRET_KEY", "default-secret-key")
        self._fernet = Fernet(self._generate_key())
    
    def _generate_key(self) -> bytes:
        """生成加密密鑰"""
        return Fernet.generate_key()
    
    def encrypt_data(self, data: str) -> str:
        """加密數據"""
        return self._fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """解密數據"""
        return self._fernet.decrypt(encrypted_data.encode()).decode()
    
    def generate_hash(self, data: str) -> str:
        """生成數據哈希"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_hash(self, data: str, hash_value: str) -> bool:
        """驗證數據哈希"""
        return self.generate_hash(data) == hash_value


class SandboxExecutor:
    """沙箱執行器"""
    
    def __init__(self, timeout: int = 30, max_memory: int = 512 * 1024 * 1024):
        self.timeout = timeout
        self.max_memory = max_memory
        self.allowed_commands = {
            "ls", "cat", "grep", "find", "wc", "head", "tail", 
            "python", "pip", "git", "mkdir", "touch", "cp", "mv"
        }
    
    def is_command_allowed(self, command: str) -> bool:
        """檢查命令是否被允許"""
        cmd_name = command.split()[0] if command.strip() else ""
        return cmd_name in self.allowed_commands
    
    def execute_command(self, command: str, working_dir: Optional[Path] = None) -> Dict[str, Any]:
        """安全執行命令"""
        if not self.is_command_allowed(command):
            return {
                "success": False,
                "error": f"Command '{command.split()[0]}' is not allowed",
                "output": "",
                "return_code": -1
            }
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                work_dir = working_dir or temp_dir
                
                # 設置資源限制
                def preexec():
                    # 設置內存限制
                    import resource
                    resource.setrlimit(resource.RLIMIT_AS, (self.max_memory, self.max_memory))
                
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=work_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=preexec
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout)
                    return {
                        "success": process.returncode == 0,
                        "output": stdout,
                        "error": stderr,
                        "return_code": process.returncode
                    }
                except subprocess.TimeoutExpired:
                    process.kill()
                    return {
                        "success": False,
                        "error": f"Command timed out after {self.timeout} seconds",
                        "output": "",
                        "return_code": -1
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "return_code": -1
            }
    
    def execute_python_code(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """安全執行Python代碼"""
        exec_timeout = timeout or self.timeout
        
        # 檢查危險函數
        dangerous_patterns = [
            "import os", "import subprocess", "import sys", "__import__",
            "exec(", "eval(", "open(", "file(", "input(", "raw_input("
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return {
                    "success": False,
                    "error": f"Dangerous pattern detected: {pattern}",
                    "output": "",
                    "return_code": -1
                }
        
        # 在臨時文件中執行代碼
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = self.execute_command(f"python {temp_file}", timeout=exec_timeout)
            return result
        finally:
            # 清理臨時文件
            try:
                os.unlink(temp_file)
            except:
                pass


class PermissionManager:
    """權限管理器"""
    
    def __init__(self):
        self.permissions = {
            "read_files": True,
            "write_files": True,
            "execute_commands": False,
            "access_network": False,
            "modify_system": False
        }
    
    def check_permission(self, action: str) -> bool:
        """檢查權限"""
        return self.permissions.get(action, False)
    
    def grant_permission(self, action: str):
        """授予權限"""
        if action in self.permissions:
            self.permissions[action] = True
    
    def revoke_permission(self, action: str):
        """撤銷權限"""
        if action in self.permissions:
            self.permissions[action] = False
    
    def get_permissions(self) -> Dict[str, bool]:
        """獲取所有權限"""
        return self.permissions.copy()


class SecurityAuditor:
    """安全審計器"""
    
    def __init__(self):
        self.audit_log = []
    
    def log_action(self, action: str, user: str = "system", details: Dict[str, Any] = None):
        """記錄操作"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "details": details or {},
            "success": True
        }
        self.audit_log.append(log_entry)
    
    def log_security_event(self, event_type: str, severity: str = "info", details: Dict[str, Any] = None):
        """記錄安全事件"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "security_event",
            "event_type": event_type,
            "severity": severity,
            "details": details or {}
        }
        self.audit_log.append(log_entry)
    
    def get_audit_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """獲取審計日誌"""
        logs = sorted(self.audit_log, key=lambda x: x["timestamp"], reverse=True)
        return logs[:limit] if limit else logs
    
    def clear_old_logs(self, days: int = 30):
        """清理舊日誌"""
        cutoff_date = datetime.now() - timedelta(days=days)
        self.audit_log = [
            log for log in self.audit_log 
            if datetime.fromisoformat(log["timestamp"]) > cutoff_date
        ]


# 全局安全管理實例
security_manager = SecurityManager()
sandbox_executor = SandboxExecutor()
permission_manager = PermissionManager()
security_auditor = SecurityAuditor()


def get_security_manager() -> SecurityManager:
    """獲取安全管理器"""
    return security_manager


def get_sandbox_executor() -> SandboxExecutor:
    """獲取沙箱執行器"""
    return sandbox_executor


def get_permission_manager() -> PermissionManager:
    """獲取權限管理器"""
    return permission_manager


def get_security_auditor() -> SecurityAuditor:
    """獲取安全審計器"""
    return security_auditor

