@echo off
REM LocalLM 全局CLI安裝腳本 (Windows版本)
setlocal enabledelayedexpansion

echo 🚀 正在安裝 LocalLM 全局CLI...
echo ==================================

REM 檢查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安裝或不在PATH中
    echo 請從 https://python.org 安裝Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python可用
python --version

REM 檢查pip
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip不可用
    pause
    exit /b 1
)

echo 📦 pip可用

REM 1. 安裝Python包
echo.
echo 📦 正在安裝LocalLM包...
python -m pip install -e .
if %errorlevel% neq 0 (
    echo ❌ 包安裝失敗
    pause
    exit /b 1
)

REM 2. 創建全局配置目錄
echo.
echo 📁 創建配置目錄...
if not exist "%USERPROFILE%\.locallm" mkdir "%USERPROFILE%\.locallm"
if not exist "%USERPROFILE%\.locallm\config" mkdir "%USERPROFILE%\.locallm\config"
if not exist "%USERPROFILE%\.locallm\data" mkdir "%USERPROFILE%\.locallm\data"
if not exist "%USERPROFILE%\.locallm\cache" mkdir "%USERPROFILE%\.locallm\cache"
if not exist "%USERPROFILE%\.locallm\logs" mkdir "%USERPROFILE%\.locallm\logs"

REM 3. 創建默認配置文件
echo.
echo ⚙️ 創建默認配置...
(
echo # LocalLM 全局配置文件
echo # 您可以根據需要修改這些設置
echo.
echo # 核心設置
echo ollama_host: "http://localhost:11434"
echo chat_model: "qwen3:latest"
echo embedding_model: "embeddinggemma:latest"
echo.
echo # 路徑設置
echo data_directory: "~/.locallm/data"
echo cache_directory: "~/.locallm/cache"
echo logs_directory: "~/.locallm/logs"
echo.
echo # 行為設置
echo auto_detect_project: true
echo remember_context: true
echo default_output_format: "rich"  # rich, json, plain
echo.
echo # 工具設置
echo enable_file_operations: true
echo enable_data_analysis: true
echo enable_web_scraping: false  # 默認關閉網絡功能
echo.
echo # 安全設置
echo safe_mode: true
echo max_file_size_mb: 100
echo max_context_length: 8192
echo.
echo # 性能設置
echo parallel_processing: true
echo cache_enabled: true
echo.
echo # 調試設置
echo debug_mode: false
echo log_level: "INFO"
echo.
echo config_version: "2.0.0"
) > "%USERPROFILE%\.locallm\config.yaml"

REM 4. 驗證安裝
echo.
echo 🔍 驗證安裝...

locallm --version >nul 2>&1
if %errorlevel% eq 0 (
    echo ✅ locallm命令可用
) else (
    echo ❌ locallm命令不可用，安裝可能失敗
    echo 請確保Python Scripts目錄在您的PATH中
    pause
    exit /b 1
)

REM 5. 檢查Ollama（可選）
echo.
echo 🤖 檢查Ollama依賴...
ollama --version >nul 2>&1
if %errorlevel% eq 0 (
    echo ✅ Ollama已安裝
    
    REM 檢查模型
    ollama list | findstr "qwen3" >nul 2>&1
    if %errorlevel% eq 0 (
        echo ✅ qwen3模型可用
    ) else (
        echo ⚠️ qwen3模型未安裝，運行: ollama pull qwen3:latest
    )
    
    ollama list | findstr "embeddinggemma" >nul 2>&1
    if %errorlevel% eq 0 (
        echo ✅ embeddinggemma模型可用
    ) else (
        echo ⚠️ embeddinggemma模型未安裝，運行: ollama pull embeddinggemma:latest
    )
) else (
    echo ⚠️ Ollama未安裝
    echo 請訪問 https://ollama.ai 安裝Ollama
    echo 然後運行:
    echo   ollama pull qwen3:latest
    echo   ollama pull embeddinggemma:latest
)

echo.
echo ==================================
echo ✅ LocalLM安裝完成!
echo.
echo 🎯 快速開始:
echo   locallm --help              # 查看幫助
echo   locallm analyze              # 分析當前項目
echo   locallm "請問今天天氣如何?"   # 自然語言查詢
echo   locallm config show          # 查看配置
echo.
echo 📁 配置文件位置: %USERPROFILE%\.locallm\config.yaml
echo 📖 數據目錄: %USERPROFILE%\.locallm\
echo.
echo 🎉 開始使用LocalLM吧！
echo.
pause


