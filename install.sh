#!/bin/bash
# LocalLM 全局CLI安裝腳本

set -e  # 遇到錯誤時退出

echo "🚀 正在安裝 LocalLM 全局CLI..."
echo "=================================="

# 檢查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

echo "🐍 檢查Python版本..."
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version (滿足要求 >= $required_version)"
else
    echo "❌ Python版本過低: $python_version (需要 >= $required_version)"
    exit 1
fi

# 檢查pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3未安裝，請先安裝pip"
    exit 1
fi

echo "📦 pip3可用"

# 1. 安裝Python包
echo ""
echo "📦 正在安裝LocalLM包..."
pip3 install -e .

# 2. 創建全局配置目錄
echo ""
echo "📁 創建配置目錄..."
mkdir -p ~/.locallm/{config,data,cache,logs}

# 3. 創建默認配置文件
echo ""
echo "⚙️ 創建默認配置..."
cat > ~/.locallm/config.yaml << 'EOF'
# LocalLM 全局配置文件
# 您可以根據需要修改這些設置

# 核心設置
ollama_host: "http://localhost:11434"
chat_model: "qwen3:latest"
embedding_model: "embeddinggemma:latest"

# 路徑設置
data_directory: "~/.locallm/data"
cache_directory: "~/.locallm/cache"
logs_directory: "~/.locallm/logs"

# 行為設置
auto_detect_project: true
remember_context: true
default_output_format: "rich"  # rich, json, plain

# 工具設置
enable_file_operations: true
enable_data_analysis: true
enable_web_scraping: false  # 默認關閉網絡功能

# 安全設置
safe_mode: true
max_file_size_mb: 100
max_context_length: 8192

# 性能設置
parallel_processing: true
cache_enabled: true

# 調試設置
debug_mode: false
log_level: "INFO"

config_version: "2.0.0"
EOF

# 4. 設置命令補全（如果支持）
echo ""
echo "🔧 設置命令補全..."

# 檢測Shell類型並設置補全
if [[ "$SHELL" == *"bash"* ]]; then
    echo "檢測到Bash Shell"
    if [[ -f ~/.bashrc ]]; then
        if ! grep -q "locallm" ~/.bashrc; then
            echo '# LocalLM命令補全' >> ~/.bashrc
            echo 'eval "$(_LOCALLM_COMPLETE=bash_source locallm)"' >> ~/.bashrc
            echo "✅ 已添加Bash命令補全到 ~/.bashrc"
        else
            echo "ℹ️ Bash補全已存在"
        fi
    fi
elif [[ "$SHELL" == *"zsh"* ]]; then
    echo "檢測到Zsh Shell"
    if [[ -f ~/.zshrc ]]; then
        if ! grep -q "locallm" ~/.zshrc; then
            echo '# LocalLM命令補全' >> ~/.zshrc
            echo 'eval "$(_LOCALLM_COMPLETE=zsh_source locallm)"' >> ~/.zshrc
            echo "✅ 已添加Zsh命令補全到 ~/.zshrc"
        else
            echo "ℹ️ Zsh補全已存在"
        fi
    fi
else
    echo "ℹ️ 未檢測到支持的Shell，跳過命令補全設置"
fi

# 5. 驗證安裝
echo ""
echo "🔍 驗證安裝..."

if command -v locallm &> /dev/null; then
    echo "✅ locallm命令可用"
    
    # 測試命令
    echo "🧪 測試基本功能..."
    if locallm --version &> /dev/null; then
        echo "✅ 版本檢查通過"
    else
        echo "⚠️ 版本檢查失敗，但命令已安裝"
    fi
else
    echo "❌ locallm命令不可用，安裝可能失敗"
    echo "請確保 ~/.local/bin 在您的 PATH 中，或重新啟動終端"
    exit 1
fi

# 6. 檢查Ollama（可選）
echo ""
echo "🤖 檢查Ollama依賴..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama已安裝"
    
    # 檢查模型
    if ollama list | grep -q "qwen3"; then
        echo "✅ qwen3模型可用"
    else
        echo "⚠️ qwen3模型未安裝，運行: ollama pull qwen3:latest"
    fi
    
    if ollama list | grep -q "embeddinggemma"; then
        echo "✅ embeddinggemma模型可用"
    else
        echo "⚠️ embeddinggemma模型未安裝，運行: ollama pull embeddinggemma:latest"
    fi
else
    echo "⚠️ Ollama未安裝"
    echo "請訪問 https://ollama.ai 安裝Ollama"
    echo "然後運行:"
    echo "  ollama pull qwen3:latest"
    echo "  ollama pull embeddinggemma:latest"
fi

echo ""
echo "=================================="
echo "✅ LocalLM安裝完成!"
echo ""
echo "🎯 快速開始:"
echo "  locallm --help              # 查看幫助"
echo "  locallm analyze              # 分析當前項目"
echo "  locallm \"請問今天天氣如何?\"   # 自然語言查詢"
echo "  locallm config show          # 查看配置"
echo ""
echo "📁 配置文件位置: ~/.locallm/config.yaml"
echo "📖 數據目錄: ~/.locallm/"
echo ""

# 如果添加了補全，提醒用戶重載shell
if [[ "$SHELL" == *"bash"* ]] || [[ "$SHELL" == *"zsh"* ]]; then
    echo "💡 提示: 重新啟動終端或運行 'source ~/.bashrc' (或 ~/.zshrc) 以啟用命令補全"
fi

echo ""
echo "🎉 開始使用LocalLM吧！"


