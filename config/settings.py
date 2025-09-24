"""
配置管理模塊
支援環境變量和配置文件的統一管理
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class OllamaConfig(BaseSettings):
    """Ollama模型配置"""
    
    host: str = Field(default="http://localhost:11434", description="Ollama服務地址")
    
    # 聊天模型配置
    chat_model: str = Field(default="qwen3:latest", description="聊天模型名稱")
    chat_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="聊天模型溫度")
    chat_max_tokens: int = Field(default=2048, gt=0, description="聊天模型最大token數")
    
    # 嵌入模型配置
    embedding_model: str = Field(default="embeddinggemma:latest", description="嵌入模型名稱")
    embedding_batch_size: int = Field(default=32, gt=0, description="嵌入批處理大小")
    
    # 代碼模型配置
    code_model: str = Field(default="qwen3:latest", description="代碼生成模型名稱")
    code_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="代碼模型溫度")
    
    # 連接配置
    timeout: int = Field(default=120, gt=0, description="請求超時時間（秒）")
    max_retries: int = Field(default=3, ge=0, description="最大重試次數")
    
    class Config:
        env_prefix = "OLLAMA_"


class StorageConfig(BaseSettings):
    """存儲配置"""
    
    # 數據路徑
    data_dir: Path = Field(default=Path("./data"), description="數據存儲目錄")
    knowledge_base_dir: Path = Field(default=Path("./data/knowledge_base"), description="知識庫目錄")
    documents_dir: Path = Field(default=Path("./data/documents"), description="文檔存儲目錄")
    cache_dir: Path = Field(default=Path("./data/cache"), description="緩存目錄")
    
    # 向量數據庫配置
    vector_db_type: str = Field(default="chroma", description="向量數據庫類型")
    vector_db_path: Path = Field(default=Path("./data/vector_db"), description="向量數據庫路徑")
    vector_collection_name: str = Field(default="documents", description="向量集合名稱")
    
    # 文件處理配置
    max_file_size: int = Field(default=100 * 1024 * 1024, description="最大文件大小（bytes）")
    supported_formats: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md", ".py", ".js", ".html", ".json", ".xml", ".csv"],
        description="支援的文件格式"
    )
    
    @validator('data_dir', 'knowledge_base_dir', 'documents_dir', 'cache_dir', 'vector_db_path')
    def create_directories(cls, v):
        """自動創建目錄"""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_prefix = "STORAGE_"


class AgentConfig(BaseSettings):
    """Agent配置"""
    
    # 執行配置
    max_execution_time: int = Field(default=300, gt=0, description="最大執行時間（秒）")
    max_parallel_tasks: int = Field(default=5, gt=0, description="最大並行任務數")
    
    # 安全配置
    enable_code_execution: bool = Field(default=False, description="是否啟用代碼執行")
    sandbox_enabled: bool = Field(default=True, description="是否啟用沙箱模式")
    allowed_commands: List[str] = Field(
        default=["ls", "cat", "grep", "find", "wc"],
        description="允許執行的命令"
    )
    
    # 工具配置
    enable_file_operations: bool = Field(default=True, description="是否啟用文件操作")
    enable_web_search: bool = Field(default=False, description="是否啟用網絡搜索")
    enable_code_generation: bool = Field(default=True, description="是否啟用代碼生成")
    
    class Config:
        env_prefix = "AGENT_"


class RAGConfig(BaseSettings):
    """RAG配置"""
    
    # 檢索配置
    chunk_size: int = Field(default=1000, gt=0, description="文本塊大小")
    chunk_overlap: int = Field(default=200, ge=0, description="文本塊重疊")
    top_k: int = Field(default=5, gt=0, description="檢索結果數量")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度閾值")
    
    # 重新排序配置
    enable_reranking: bool = Field(default=True, description="是否啟用重新排序")
    rerank_top_k: int = Field(default=10, gt=0, description="重新排序候選數量")
    
    # 生成配置
    max_context_length: int = Field(default=4000, gt=0, description="最大上下文長度")
    include_source: bool = Field(default=True, description="是否包含來源信息")
    
    class Config:
        env_prefix = "RAG_"


class SecurityConfig(BaseSettings):
    """安全配置"""
    
    # 加密配置
    secret_key: str = Field(default="your-secret-key-change-in-production", description="密鑰")
    algorithm: str = Field(default="HS256", description="加密算法")
    
    # 沙箱配置
    sandbox_timeout: int = Field(default=30, gt=0, description="沙箱超時時間")
    max_memory_usage: int = Field(default=512 * 1024 * 1024, description="最大內存使用量（bytes）")
    
    # 日誌配置
    log_level: str = Field(default="INFO", description="日誌級別")
    log_file: Optional[str] = Field(default=None, description="日誌文件路徑")
    
    class Config:
        env_prefix = "SECURITY_"


class Settings(BaseSettings):
    """主配置類"""
    
    # 應用基本信息
    app_name: str = Field(default="Personal AI Knowledge Assistant", description="應用名稱")
    version: str = Field(default="0.1.0", description="版本號")
    debug: bool = Field(default=False, description="調試模式")
    
    # 子配置
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    @validator('ollama', 'storage', 'agent', 'rag', 'security', pre=True)
    def parse_nested_config(cls, v):
        """解析嵌套配置"""
        if isinstance(v, dict):
            return v
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 全局配置實例
settings = Settings()


def get_settings() -> Settings:
    """獲取配置實例"""
    return settings


def reload_settings():
    """重新加載配置"""
    global settings
    settings = Settings()
    return settings
