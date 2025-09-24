"""
全局配置管理系統
管理LocalLM全局CLI的配置設置
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class GlobalConfig:
    """全局配置結構"""
    
    # 核心設置
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "qwen3:latest"
    embedding_model: str = "embeddinggemma:latest"
    
    # 路徑設置
    data_directory: str = "~/.locallm/data"
    cache_directory: str = "~/.locallm/cache"
    logs_directory: str = "~/.locallm/logs"
    
    # 行為設置
    auto_detect_project: bool = True
    remember_context: bool = True
    default_output_format: str = "rich"  # rich, json, plain
    
    # 工具設置
    enable_file_operations: bool = True
    enable_data_analysis: bool = True
    enable_web_scraping: bool = False  # 默認關閉網絡功能
    
    # 安全設置
    safe_mode: bool = True
    allowed_directories: Optional[List[str]] = None
    blocked_extensions: Optional[List[str]] = None
    
    # 性能設置
    max_file_size_mb: int = 100
    max_context_length: int = 8192
    parallel_processing: bool = True
    cache_enabled: bool = True
    
    # 調試設置
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # 版本信息
    config_version: str = "2.0.0"
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        """後處理：展開路徑中的~"""
        self.data_directory = os.path.expanduser(self.data_directory)
        self.cache_directory = os.path.expanduser(self.cache_directory)
        self.logs_directory = os.path.expanduser(self.logs_directory)
        
        if self.allowed_directories:
            self.allowed_directories = [os.path.expanduser(d) for d in self.allowed_directories]
        
        if self.blocked_extensions is None:
            self.blocked_extensions = ['.exe', '.dll', '.so', '.dylib']
        
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()


class ConfigManager:
    """全局配置管理器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_paths = self._get_config_paths(config_path)
        self.config = self._load_config()
        self._ensure_directories()
    
    def _get_config_paths(self, custom_path: Optional[Path] = None) -> List[Path]:
        """獲取配置文件可能的路徑"""
        paths = []
        
        # 1. 用戶指定的路徑
        if custom_path:
            paths.append(Path(custom_path))
        
        # 2. 環境變量指定的路徑
        env_config = os.environ.get('LOCALLM_CONFIG')
        if env_config:
            paths.append(Path(env_config))
        
        # 3. 標準配置路徑（按優先級排序）
        home = Path.home()
        paths.extend([
            home / '.locallm' / 'config.yaml',
            home / '.locallm' / 'config.yml',
            home / '.config' / 'locallm' / 'config.yaml',
            home / '.config' / 'locallm' / 'config.yml',
        ])
        
        # 4. 項目本地配置（如果存在）
        cwd = Path.cwd()
        paths.extend([
            cwd / '.locallm.yaml',
            cwd / '.locallm.yml',
            cwd / 'locallm.yaml',
            cwd / 'locallm.yml'
        ])
        
        return paths
    
    def _load_config(self) -> GlobalConfig:
        """加載配置文件"""
        # 嘗試從文件加載配置
        for config_path in self.config_paths:
            if config_path.exists() and config_path.is_file():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f) or {}
                    
                    # 過濾掉None值和無效字段
                    config_data = {k: v for k, v in config_data.items() 
                                 if v is not None and hasattr(GlobalConfig, k)}
                    
                    print(f"已加載配置文件: {config_path}")
                    return GlobalConfig(**config_data)
                    
                except Exception as e:
                    print(f"配置文件加載失敗 {config_path}: {e}")
                    continue
        
        # 如果沒有找到有效配置文件，創建默認配置
        print("未找到配置文件，創建默認配置...")
        return self._create_default_config()
    
    def _create_default_config(self) -> GlobalConfig:
        """創建默認配置"""
        config = GlobalConfig()
        
        # 自動保存默認配置到第一個路徑
        try:
            self.save_config(config)
        except Exception as e:
            print(f"[警告] 無法保存默認配置: {e}")
        
        return config
    
    def _ensure_directories(self):
        """確保必要的目錄存在"""
        directories = [
            self.config.data_directory,
            self.config.cache_directory,
            self.config.logs_directory
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"警告: 無法創建目錄 {directory}: {e}")
    
    def save_config(self, config: Optional[GlobalConfig] = None):
        """保存配置到文件"""
        config = config or self.config
        
        # 更新時間戳
        config.last_updated = datetime.now().isoformat()
        
        # 選擇保存路徑（優先使用用戶主目錄）
        save_path = self.config_paths[0]  # ~/.locallm/config.yaml
        
        # 確保目錄存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 轉換為字典並保存
            config_dict = asdict(config)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_dict, 
                    f, 
                    default_flow_style=False, 
                    allow_unicode=True,
                    sort_keys=True,
                    indent=2
                )
            
            print(f"[成功] 配置已保存到: {save_path}")
            
        except Exception as e:
            raise Exception(f"保存配置失敗: {e}")
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """更新配置項"""
        try:
            # 驗證更新項
            valid_updates = {}
            for key, value in updates.items():
                if hasattr(GlobalConfig, key):
                    valid_updates[key] = value
                else:
                    print(f"[警告] 忽略無效的配置項: {key}")
            
            # 應用更新
            if valid_updates:
                for key, value in valid_updates.items():
                    setattr(self.config, key, value)
                
                # 重新處理路徑
                self.config.__post_init__()
                
                # 保存更新後的配置
                self.save_config()
                
                # 確保目錄存在
                self._ensure_directories()
                
                print(f"[成功] 已更新 {len(valid_updates)} 個配置項")
                return True
            else:
                print("[警告] 沒有有效的配置更新")
                return False
                
        except Exception as e:
            print(f"[失敗] 配置更新失敗: {e}")
            return False
    
    def get_config_info(self) -> Dict[str, Any]:
        """獲取配置信息摘要"""
        return {
            'config_version': self.config.config_version,
            'last_updated': self.config.last_updated,
            'config_file': str(self.config_paths[0]),
            'data_directory': self.config.data_directory,
            'cache_directory': self.config.cache_directory,
            'logs_directory': self.config.logs_directory,
            'auto_detect_project': self.config.auto_detect_project,
            'default_output_format': self.config.default_output_format,
            'safe_mode': self.config.safe_mode,
            'debug_mode': self.config.debug_mode
        }
    
    def validate_config(self) -> List[str]:
        """驗證配置有效性"""
        issues = []
        
        # 檢查路徑是否可寫
        for path_name, path_value in [
            ('data_directory', self.config.data_directory),
            ('cache_directory', self.config.cache_directory),
            ('logs_directory', self.config.logs_directory)
        ]:
            try:
                path = Path(path_value)
                path.mkdir(parents=True, exist_ok=True)
                
                # 嘗試創建測試文件
                test_file = path / '.test_write'
                test_file.touch()
                test_file.unlink()
                
            except Exception as e:
                issues.append(f"{path_name} 不可寫: {e}")
        
        # 檢查模型名稱格式
        if not self.config.chat_model:
            issues.append("chat_model 不能為空")
        
        if not self.config.embedding_model:
            issues.append("embedding_model 不能為空")
        
        # 檢查輸出格式
        valid_formats = ['rich', 'json', 'plain']
        if self.config.default_output_format not in valid_formats:
            issues.append(f"default_output_format 必須是 {valid_formats} 之一")
        
        # 檢查數值範圍
        if self.config.max_file_size_mb <= 0:
            issues.append("max_file_size_mb 必須大於0")
        
        if self.config.max_context_length <= 0:
            issues.append("max_context_length 必須大於0")
        
        return issues
    
    def reset_to_defaults(self) -> bool:
        """重置為默認配置"""
        try:
            self.config = GlobalConfig()
            self.save_config()
            self._ensure_directories()
            print("[成功] 配置已重置為默認值")
            return True
        except Exception as e:
            print(f"[失敗] 配置重置失敗: {e}")
            return False
    
    def export_config(self, export_path: Path) -> bool:
        """導出配置到指定路徑"""
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = asdict(self.config)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=True,
                    indent=2
                )
            
            print(f"[成功] 配置已導出到: {export_path}")
            return True
            
        except Exception as e:
            print(f"[失敗] 配置導出失敗: {e}")
            return False
    
    def import_config(self, import_path: Path) -> bool:
        """從指定路徑導入配置"""
        try:
            if not import_path.exists():
                print(f"[失敗] 配置文件不存在: {import_path}")
                return False
            
            with open(import_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            # 過濾有效字段
            valid_data = {k: v for k, v in config_data.items() 
                         if hasattr(GlobalConfig, k) and v is not None}
            
            if valid_data:
                self.config = GlobalConfig(**valid_data)
                self.save_config()
                self._ensure_directories()
                print(f"[成功] 配置已從 {import_path} 導入")
                return True
            else:
                print("[失敗] 導入文件中沒有有效的配置項")
                return False
                
        except Exception as e:
            print(f"[失敗] 配置導入失敗: {e}")
            return False


# 全局配置管理器實例
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """獲取全局配置管理器實例"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager


def get_global_config() -> GlobalConfig:
    """獲取全局配置"""
    return get_config_manager().config
