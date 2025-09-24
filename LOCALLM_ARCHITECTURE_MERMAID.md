# 🎨 LocalLM 系統架構圖 - Mermaid 圖表

## 🏗️ **整體系統架構**

```mermaid
graph TB
    %% 用戶界面層
    subgraph "🖥️ 用戶界面層"
        CLI[🖥️ CLI界面<br/>locallm命令]
        NLP[🧠 自然語言處理<br/>雙語支持]
        SC[⚡ 快捷命令<br/>45個快捷方式]
    end
    
    %% 智能化功能層
    subgraph "🧠 智能化功能層"
        CP[🧠 命令解析器<br/>意圖識別]
        WG[🔄 工作流生成器<br/>5個模板]
        AC[💡 命令補全<br/>智能建議]
        IM[🎯 智能管理器<br/>統一協調]
    end
    
    %% 工具協作層
    subgraph "🔧 工具協作層"
        TM[🛠️ 工具管理器<br/>7個工具]
        FO[📁 文件操作]
        DA[📊 數據分析]
        PM[📋 項目管理]
        GT[🌐 Git/Web工具]
    end
    
    %% 性能優化層
    subgraph "⚡ 性能優化層"
        CM[🗄️ 緩存管理<br/>93萬ops/sec]
        BP[⚡ 批處理器<br/>11.8萬items/sec]
        PE[🔀 並行執行器<br/>100%成功率]
        RM[📈 資源監控<br/>實時監控]
        RT[🔄 智能重試<br/>錯誤恢復]
        ER[🔍 增強檢索<br/>混合檢索]
    end
    
    %% 核心系統層
    subgraph "🎯 核心系統層"
        RAG[🔍 RAG引擎<br/>文檔檢索生成]
        AGT[🤖 Agent系統<br/>任務規劃執行]
        MEM[🧠 記憶系統<br/>會話/長期記憶]
        SEC[🛡️ 安全框架<br/>沙箱執行]
        CFG[⚙️ 配置管理<br/>全局配置]
    end
    
    %% 數據存儲層
    subgraph "💾 數據存儲層"
        VDB[🗂️ 向量數據庫<br/>Chroma]
        KG[🕸️ 知識圖譜<br/>實體關係]
        FS[📂 文件系統<br/>文檔存儲]
        CACHE[💾 緩存系統<br/>多級緩存]
    end
    
    %% 模型服務層
    subgraph "🤖 模型服務層"
        OLLAMA[🦙 Ollama服務<br/>本地LLM]
        CHAT[💬 聊天模型<br/>qwen3:latest]
        EMB[🔗 嵌入模型<br/>embeddinggemma]
        CODE[💻 代碼模型<br/>代碼生成]
    end
    
    %% 連接關係
    CLI --> CP
    NLP --> CP
    SC --> CP
    
    CP --> IM
    WG --> IM
    AC --> IM
    
    IM --> TM
    TM --> FO
    TM --> DA
    TM --> PM
    TM --> GT
    
    TM --> CM
    TM --> BP
    TM --> PE
    
    IM --> RAG
    IM --> AGT
    IM --> MEM
    
    RAG --> VDB
    MEM --> KG
    AGT --> SEC
    
    RAG --> OLLAMA
    AGT --> OLLAMA
    MEM --> OLLAMA
    
    OLLAMA --> CHAT
    OLLAMA --> EMB
    OLLAMA --> CODE
    
    CM --> CACHE
    RAG --> VDB
    MEM --> FS
    
    RM --> PE
    RT --> BP
    ER --> RAG
    
    CFG --> IM
    SEC --> TM
```

## 🔄 **智能化功能模塊架構**

```mermaid
graph LR
    subgraph "🧠 智能化功能核心"
        USER[👤 用戶輸入<br/>自然語言/命令]
        
        subgraph "🧠 自然語言處理"
            NLP[🧠 NLP解析器]
            INTENT[🎯 意圖識別<br/>analyze/chat/execute]
            TARGET[📍 目標提取<br/>文件/目錄/URL]
            PARAM[⚙️ 參數解析<br/>格式/深度/詳細度]
            CONF[📊 置信度評分<br/>解析準確度]
        end
        
        subgraph "🔄 工作流引擎"
            TEMPLATE[📋 模板匹配<br/>5個預定義模板]
            DYNAMIC[🎨 動態生成<br/>自定義工作流]
            CUSTOM[🛠️ 參數自定義<br/>上下文適配]
            EXEC[▶️ 執行引擎<br/>流程控制]
        end
        
        subgraph "💡 智能補全"
            CMD_COMP[🖥️ 命令補全<br/>主/子命令]
            PARAM_COMP[⚙️ 參數補全<br/>--output等]
            PATH_COMP[📂 路徑補全<br/>文件/目錄]
            SMART_SUG[🤖 智能建議<br/>基於上下文]
        end
        
        subgraph "⚡ 快捷系統"
            STATIC[📌 靜態快捷方式<br/>45個預定義]
            DYNAMIC_SC[🎯 動態快捷方式<br/>基於文件類型]
            CONTEXT[🌐 上下文感知<br/>環境適配]
            HISTORY[📚 歷史命令<br/>數字快捷方式]
        end
        
        subgraph "🎯 智能管理器"
            COORD[🎛️ 功能協調<br/>統一管理]
            MODE[🔍 模式檢測<br/>自動識別]
            ROUTE[🔀 結果路由<br/>智能分發]
            STATS[📊 統計分析<br/>使用追蹤]
        end
    end
    
    USER --> NLP
    NLP --> INTENT
    NLP --> TARGET
    NLP --> PARAM
    NLP --> CONF
    
    INTENT --> TEMPLATE
    TARGET --> CUSTOM
    PARAM --> DYNAMIC
    
    USER --> CMD_COMP
    CMD_COMP --> PARAM_COMP
    PARAM_COMP --> PATH_COMP
    PATH_COMP --> SMART_SUG
    
    USER --> STATIC
    STATIC --> DYNAMIC_SC
    DYNAMIC_SC --> CONTEXT
    CONTEXT --> HISTORY
    
    TEMPLATE --> COORD
    DYNAMIC --> COORD
    SMART_SUG --> COORD
    STATIC --> COORD
    
    COORD --> MODE
    MODE --> ROUTE
    ROUTE --> STATS
```

## 🔧 **工具系統架構**

```mermaid
graph TB
    subgraph "🔧 工具協作生態"
        TM[🛠️ 工具管理器<br/>EnhancedToolManager]
        
        subgraph "📁 文件工具組"
            FO[📁 文件操作工具<br/>FileOperationTool]
            PU[🔗 路徑處理工具<br/>PathUtilityTool]
        end
        
        subgraph "📊 數據工具組"
            DA[📊 數據分析工具<br/>DataAnalysisTool]
            VIS[📈 數據可視化<br/>圖表生成]
        end
        
        subgraph "📋 項目工具組"
            PM[📋 項目管理工具<br/>ProjectManagementTool]
            TG[🧪 測試生成工具<br/>TestGenerationTool]
        end
        
        subgraph "🌐 外部工具組"
            GO[🌐 Git操作工具<br/>GitOperationTool]
            WS[🕷️ Web爬取工具<br/>WebScrapingTool]
        end
        
        subgraph "🤝 協作機制"
            TR[📝 工具推薦<br/>智能推薦系統]
            TC[🔗 工具鏈<br/>工具協作鏈]
            WF[🔄 工作流<br/>自動化流程]
            STATS[📊 使用統計<br/>性能追蹤]
        end
    end
    
    TM --> FO
    TM --> PU
    TM --> DA
    TM --> PM
    TM --> TG
    TM --> GO
    TM --> WS
    
    FO --> DA
    DA --> VIS
    PM --> TG
    GO --> PM
    
    TM --> TR
    TR --> TC
    TC --> WF
    WF --> STATS
    
    FO -.->|協作| DA
    DA -.->|協作| PM
    PM -.->|協作| GO
```

## ⚡ **性能優化架構**

```mermaid
graph TD
    subgraph "⚡ 高性能處理引擎"
        subgraph "🗄️ 多級緩存系統"
            MEM_CACHE[💾 內存緩存<br/>LRU策略]
            DISK_CACHE[💿 磁盤緩存<br/>SQLite索引]
            CACHE_MGR[🎛️ 緩存管理器<br/>CacheManager]
            CACHE_STATS[📊 緩存統計<br/>命中率監控]
        end
        
        subgraph "⚡ 批處理引擎"
            ASYNC_BATCH[🔄 異步批處理<br/>11.8萬items/sec]
            THREAD_BATCH[🧵 線程批處理<br/>多線程並發]
            PROCESS_BATCH[🔀 進程批處理<br/>多進程並行]
            ADAPTIVE[🎯 自適應<br/>動態批次調整]
        end
        
        subgraph "🔀 並行執行器"
            CPU_TASK[🖥️ CPU密集任務<br/>多進程處理]
            IO_TASK[💾 IO密集任務<br/>異步IO處理]
            PRIORITY[⚖️ 任務優先級<br/>智能調度]
            RESOURCE[📊 資源管理<br/>CPU/內存控制]
        end
        
        subgraph "📈 資源監控"
            CPU_MON[🖥️ CPU監控<br/>實時使用率]
            MEM_MON[🧠 內存監控<br/>內存追蹤]
            DISK_MON[💿 磁盤監控<br/>存儲空間]
            GPU_MON[🎮 GPU監控<br/>可選支持]
            TREND[📈 趨勢分析<br/>預測性分析]
        end
        
        subgraph "🔄 智能重試"
            ERROR_CLASS[🏷️ 錯誤分類<br/>智能分類器]
            RETRY_STRAT[🎯 重試策略<br/>指數/線性/固定]
            CIRCUIT[🔌 熔斷器<br/>故障保護]
            RECOVERY[🔧 恢復動作<br/>自動恢復]
        end
        
        subgraph "🔍 增強檢索"
            VECTOR_RET[🎯 向量檢索<br/>語義相似度]
            KEYWORD_RET[🔍 關鍵詞檢索<br/>傳統匹配]
            HYBRID_RET[🔀 混合檢索<br/>組合策略]
            RERANK[🏆 重排序<br/>交叉編碼器]
        end
    end
    
    CACHE_MGR --> MEM_CACHE
    CACHE_MGR --> DISK_CACHE
    CACHE_MGR --> CACHE_STATS
    
    ADAPTIVE --> ASYNC_BATCH
    ADAPTIVE --> THREAD_BATCH
    ADAPTIVE --> PROCESS_BATCH
    
    PRIORITY --> CPU_TASK
    PRIORITY --> IO_TASK
    PRIORITY --> RESOURCE
    
    TREND --> CPU_MON
    TREND --> MEM_MON
    TREND --> DISK_MON
    TREND --> GPU_MON
    
    ERROR_CLASS --> RETRY_STRAT
    RETRY_STRAT --> CIRCUIT
    CIRCUIT --> RECOVERY
    
    HYBRID_RET --> VECTOR_RET
    HYBRID_RET --> KEYWORD_RET
    HYBRID_RET --> RERANK
    
    %% 性能指標
    MEM_CACHE -.->|930,000 ops/sec| CACHE_STATS
    ASYNC_BATCH -.->|118,583 items/sec| ADAPTIVE
    CPU_TASK -.->|100% 成功率| PRIORITY
```

## 🎯 **核心系統架構**

```mermaid
graph TB
    subgraph "🎯 核心系統引擎"
        subgraph "🔍 RAG引擎"
            DOC_ING[📥 文檔攝取<br/>多格式支持]
            SEMANTIC[🧠 語義分塊<br/>智能分割]
            VECTOR[🔗 向量化<br/>嵌入生成]
            RETRIEVAL[🔍 檢索生成<br/>智能檢索]
            QUALITY[✅ 質量驗證<br/>結果評估]
        end
        
        subgraph "🤖 Agent系統"
            TOOL_REG[🛠️ 工具註冊<br/>動態註冊]
            TASK_PLAN[📋 任務規劃<br/>智能分解]
            SAFE_EXEC[🛡️ 安全執行<br/>沙箱環境]
            WORK_MEM[🧠 工作記憶<br/>上下文管理]
        end
        
        subgraph "🧠 記憶系統"
            SESSION[💬 會話記憶<br/>對話歷史]
            LONGTERM[📚 長期記憶<br/>知識持久化]
            KNOWLEDGE[🕸️ 知識圖譜<br/>實體關係]
            PERSONAL[👤 個人記憶<br/>偏好模式]
            CONTEXT[🎯 上下文管理<br/>智能切換]
        end
        
        subgraph "🛡️ 安全框架"
            POLICY[📋 安全策略<br/>細粒度控制]
            SANDBOX[📦 沙箱執行<br/>隔離環境]
            PERM[🔐 權限控制<br/>操作權限]
            DOCKER[🐳 Docker支持<br/>容器化執行]
        end
        
        subgraph "⚙️ 配置管理"
            GLOBAL[🌐 全局配置<br/>統一管理]
            ENV_DET[🔍 環境檢測<br/>自動識別]
            PATH_MGR[📂 路徑管理<br/>智能配置]
            VALID[✅ 設置驗證<br/>配置檢查]
        end
        
        subgraph "🖥️ CLI界面"
            GLOBAL_CMD[🌐 全局命令<br/>locallm入口]
            RICH_UI[🎨 富終端界面<br/>Rich美化]
            CMD_ROUTE[🔀 命令路由<br/>智能分發]
            CTX_DET[🔍 上下文檢測<br/>環境感知]
        end
    end
    
    DOC_ING --> SEMANTIC
    SEMANTIC --> VECTOR
    VECTOR --> RETRIEVAL
    RETRIEVAL --> QUALITY
    
    TOOL_REG --> TASK_PLAN
    TASK_PLAN --> SAFE_EXEC
    SAFE_EXEC --> WORK_MEM
    
    SESSION --> LONGTERM
    LONGTERM --> KNOWLEDGE
    KNOWLEDGE --> PERSONAL
    PERSONAL --> CONTEXT
    
    POLICY --> SANDBOX
    SANDBOX --> PERM
    PERM --> DOCKER
    
    GLOBAL --> ENV_DET
    ENV_DET --> PATH_MGR
    PATH_MGR --> VALID
    
    GLOBAL_CMD --> RICH_UI
    RICH_UI --> CMD_ROUTE
    CMD_ROUTE --> CTX_DET
    
    %% 跨系統連接
    TASK_PLAN -.->|使用| TOOL_REG
    RETRIEVAL -.->|查詢| KNOWLEDGE
    SAFE_EXEC -.->|遵循| POLICY
    CMD_ROUTE -.->|配置| GLOBAL
```

## 🔄 **數據流架構**

```mermaid
sequenceDiagram
    participant U as 👤 用戶
    participant CLI as 🖥️ CLI界面
    participant NLP as 🧠 自然語言處理
    participant IM as 🎯 智能管理器
    participant WF as 🔄 工作流引擎
    participant TM as 🛠️ 工具管理器
    participant PERF as ⚡ 性能優化
    participant RAG as 🔍 RAG引擎
    participant MEM as 🧠 記憶系統
    participant OLLAMA as 🦙 Ollama服務
    
    U->>CLI: 輸入命令/自然語言
    CLI->>NLP: 解析用戶輸入
    NLP->>NLP: 意圖識別/目標提取
    NLP->>IM: 返回解析結果
    
    IM->>WF: 生成工作流
    WF->>WF: 模板匹配/動態生成
    WF->>IM: 返回執行計劃
    
    IM->>TM: 調用工具執行
    TM->>PERF: 性能優化處理
    PERF->>PERF: 緩存/批處理/並行
    
    alt 需要RAG檢索
        TM->>RAG: 文檔檢索請求
        RAG->>OLLAMA: 向量化/檢索
        OLLAMA->>RAG: 返回檢索結果
        RAG->>TM: 檢索結果
    end
    
    alt 需要記憶功能
        TM->>MEM: 記憶操作請求
        MEM->>MEM: 會話/長期/知識圖譜
        MEM->>TM: 記憶結果
    end
    
    TM->>PERF: 結果處理
    PERF->>IM: 優化後結果
    IM->>CLI: 格式化輸出
    CLI->>U: 展示結果
```

## 📊 **性能指標圖表**

```mermaid
xychart-beta
    title "LocalLM 系統性能指標"
    x-axis [緩存系統, 批處理器, 並行執行, 智能化功能, 工具協作, 整體穩定]
    y-axis "性能指標 (%)" 0 --> 100
    bar [100, 85.7, 100, 100, 100, 87]
```

## 🎯 **模塊依賴關係**

```mermaid
graph LR
    subgraph "🎯 依賴關係圖"
        CLI[🖥️ CLI界面] --> IM[🎯 智能管理器]
        NLP[🧠 自然語言處理] --> IM
        WF[🔄 工作流引擎] --> IM
        AC[💡 命令補全] --> IM
        SC[⚡ 快捷命令] --> IM
        
        IM --> TM[🛠️ 工具管理器]
        IM --> RAG[🔍 RAG引擎]
        IM --> MEM[🧠 記憶系統]
        IM --> AGT[🤖 Agent系統]
        
        TM --> PERF[⚡ 性能優化]
        RAG --> PERF
        MEM --> PERF
        AGT --> PERF
        
        PERF --> OLLAMA[🦙 Ollama服務]
        RAG --> OLLAMA
        MEM --> OLLAMA
        AGT --> OLLAMA
        
        TM --> SEC[🛡️ 安全框架]
        AGT --> SEC
        
        IM --> CFG[⚙️ 配置管理]
        CLI --> CFG
    end
    
    %% 樣式定義
    classDef intelligence fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef tools fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef performance fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef core fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class CLI,NLP,WF,AC,SC,IM intelligence
    class TM tools
    class PERF performance
    class RAG,MEM,AGT,SEC,CFG,OLLAMA core
```

---

## 🎯 **圖表說明**

### **1. 整體系統架構**
- 展示LocalLM的6層架構：用戶界面、智能化功能、工具協作、性能優化、核心系統、數據存儲
- 清晰顯示各模塊間的依賴關係和數據流

### **2. 智能化功能模塊架構**
- 詳細展示5個智能化模塊的內部結構和協作關係
- 突出自然語言處理、工作流生成、命令補全等核心功能

### **3. 工具系統架構**
- 展示7個工具的分組和協作機制
- 突出工具推薦、工具鏈、工作流等協作特性

### **4. 性能優化架構**
- 展示6個性能優化模塊的內部結構
- 突出緩存、批處理、並行執行等高性能特性

### **5. 核心系統架構**
- 展示RAG、Agent、記憶、安全等核心系統
- 突出各系統的內部組件和功能

### **6. 數據流架構**
- 展示從用戶輸入到結果輸出的完整數據流程
- 清晰顯示各模塊的交互時序

### **7. 性能指標圖表**
- 直觀展示各模塊的性能表現
- 突出系統的高性能特性

### **8. 模塊依賴關係**
- 展示各模塊間的依賴關係
- 使用不同顏色區分模塊類型

---

## 📝 **使用說明**

1. **在Markdown支持Mermaid的環境中查看**：
   - GitHub、GitLab
   - Notion、Obsidian
   - VS Code (with Mermaid extension)
   - 在線Mermaid編輯器

2. **複製Mermaid代碼**：
   - 複製```mermaid代碼塊中的內容
   - 在支持Mermaid的工具中渲染

3. **圖表特點**：
   - 📊 清晰的模塊分層
   - 🔄 完整的數據流程
   - ⚡ 突出性能指標
   - 🎨 美觀的視覺設計

**🎨 這些Mermaid圖表完整展示了LocalLM系統的架構設計，幫助您深入理解系統的組織結構和運行機制！**
