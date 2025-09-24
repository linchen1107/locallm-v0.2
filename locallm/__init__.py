"""
LocalLM - 本地大語言模型智能助理
全局CLI版本 v2.0
"""

__version__ = "2.0.0"
__author__ = "LocalLM Team"
__description__ = "本地大語言模型智能助理 - 全局CLI"

# 確保核心模塊能夠被正確導入
try:
    # 導入核心優化模塊
    from core.optimization.integrated_rag import get_optimized_rag_system
    from core.optimization.cache_manager import get_cache_manager
    from core.optimization.resource_monitor import get_resource_monitor
    
    RAG_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    RAG_OPTIMIZATIONS_AVAILABLE = False

# 全局CLI可用性檢查
CLI_AVAILABLE = True


