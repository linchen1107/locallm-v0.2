"""
自然語言命令解析器

將用戶的自然語言輸入解析為結構化的命令，支援中文和英文
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedCommand:
    """解析後的命令結構"""
    intent: str                    # 意圖：analyze, chat, execute等
    targets: List[str]             # 目標對象：文件、目錄等
    actions: List[str]             # 動作：read, analyze, generate等
    parameters: Dict[str, Any]     # 參數
    natural_language: str          # 原始自然語言
    confidence: float              # 解析置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            'intent': self.intent,
            'targets': self.targets,
            'actions': self.actions,
            'parameters': self.parameters,
            'natural_language': self.natural_language,
            'confidence': self.confidence
        }


class NaturalLanguageParser:
    """自然語言命令解析器"""
    
    # 意圖識別模式
    INTENT_PATTERNS = {
        'analyze': [
            r'分析', r'檢查', r'掃描', r'統計', r'檢測', r'調查', r'查看.*結構',
            r'analyze', r'check', r'scan', r'stats', r'examine', r'inspect', r'investigate'
        ],
        'chat': [
            r'問', r'詢問', r'解釋', r'說明', r'怎麼', r'如何', r'為什麼', r'什麼是', r'告訴我',
            r'ask', r'explain', r'how', r'why', r'what', r'help', r'tell\s+me', r'\?'
        ],
        'execute': [
            r'執行', r'運行', r'處理', r'生成', r'創建', r'製作', r'建立', r'產生',
            r'execute', r'run', r'process', r'generate', r'create', r'make', r'build', r'produce'
        ],
        'read': [
            r'讀取', r'打開', r'查看', r'顯示', r'載入', r'導入',
            r'read', r'open', r'view', r'show', r'display', r'load', r'import'
        ],
        'write': [
            r'寫入', r'保存', r'創建文件', r'輸出', r'導出', r'存檔', r'儲存',
            r'write', r'save', r'create\s+file', r'output', r'export', r'store'
        ],
        'manage': [
            r'管理', r'配置', r'設定', r'更新', r'修改', r'編輯',
            r'manage', r'configure', r'setup', r'update', r'modify', r'edit'
        ]
    }
    
    # 目標對象模式
    TARGET_PATTERNS = {
        'file': r'[\w\-\.]+\.[a-zA-Z]{2,4}',  # 文件名模式
        'directory': r'[\w\-/\\\.]+(?<!\.[\w]+)',  # 目錄模式
        'url': r'https?://[^\s]+',  # URL模式
        'current': r'這裡|當前|此|current|here|this|\.(?:\s|$)',  # 當前位置
        'project': r'項目|專案|工程|project',
        'data': r'數據|資料|data',
        'code': r'代碼|程式碼|code',
        'document': r'文檔|文件|檔案|document'
    }
    
    # 動作關鍵詞
    ACTION_KEYWORDS = {
        'read': ['讀取', '打開', '載入', 'read', 'open', 'load'],
        'analyze': ['分析', '統計', '檢查', 'analyze', 'stats', 'examine'],
        'summarize': ['總結', '摘要', '概括', 'summarize', 'summary', 'overview'],
        'visualize': ['可視化', '圖表', '繪圖', 'visualize', 'chart', 'plot', 'graph'],
        'report': ['報告', '生成報告', '輸出報告', 'report', 'generate report'],
        'export': ['導出', '輸出', '匯出', 'export', 'output'],
        'compare': ['比較', '對比', '對照', 'compare', 'contrast'],
        'filter': ['過濾', '篩選', '選擇', 'filter', 'select', 'choose'],
        'transform': ['轉換', '變換', '處理', 'transform', 'convert', 'process']
    }
    
    def __init__(self):
        self.command_history = []
        self.context_cache = {}
    
    def parse(self, command: str, context: Optional[Dict[str, Any]] = None) -> ParsedCommand:
        """解析自然語言命令"""
        if context is None:
            context = {}
        
        # 預處理命令
        normalized_command = self._normalize_command(command)
        
        # 提取各個組件
        intent = self._extract_intent(normalized_command)
        targets = self._extract_targets(normalized_command, context)
        actions = self._extract_actions(normalized_command, intent)
        parameters = self._extract_parameters(normalized_command)
        
        # 計算置信度
        confidence = self._calculate_confidence(normalized_command, intent, targets, actions)
        
        # 創建解析結果
        parsed_command = ParsedCommand(
            intent=intent,
            targets=targets,
            actions=actions,
            parameters=parameters,
            natural_language=command,
            confidence=confidence
        )
        
        # 記錄到歷史
        self.command_history.append(parsed_command)
        if len(self.command_history) > 50:  # 保持最近50條記錄
            self.command_history.pop(0)
        
        return parsed_command
    
    def _normalize_command(self, command: str) -> str:
        """標準化命令文本"""
        # 移除多餘空格
        command = re.sub(r'\s+', ' ', command.strip())
        
        # 處理標點符號
        command = re.sub(r'[，。！？]', ', ', command)
        command = re.sub(r'[,!?]+', ', ', command)
        
        return command
    
    def _extract_intent(self, command: str) -> str:
        """提取命令意圖"""
        command_lower = command.lower()
        
        # 特殊優先規則
        if '?' in command or '？' in command:
            return 'chat'
        
        # 檢測關鍵詞組合
        has_file = any(ext in command_lower for ext in ['.csv', '.json', '.txt', '.pdf', '.py'])
        has_read = any(word in command_lower for word in ['讀取', '打開', '載入', 'read', 'open', 'load'])
        has_analyze = any(word in command_lower for word in ['分析', '統計', '檢查', 'analyze', 'stats', 'examine'])
        has_generate = any(word in command_lower for word in ['生成', '創建', '製作', 'generate', 'create', 'make'])
        has_report = any(word in command_lower for word in ['報告', 'report'])
        
        # 複合意圖推斷邏輯
        if has_file and has_read and has_generate and has_report:
            # 讀取文件並生成報告 -> analyze (分析為主，生成為輔)
            return 'analyze'
        elif has_file and (has_read or has_analyze):
            # 讀取/分析文件 -> analyze
            return 'analyze'
        elif has_generate and (has_report or has_analyze) and not has_file:
            # 生成報告或創建分析（無具體文件） -> execute
            return 'execute'
        elif has_file and has_analyze:
            # 包含文件和分析 -> analyze
            return 'analyze'
        elif has_generate:
            # 包含生成/創建 -> execute
            return 'execute'
        
        # 計算每個意圖的匹配分數（作為備用）
        intent_scores = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, command_lower))
                score += matches
            intent_scores[intent] = score
        
        # 選擇最高分的意圖
        if intent_scores and max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        # 最終默認邏輯
        if has_file:
            return 'analyze'
        else:
            return 'execute'
    
    def _extract_targets(self, command: str, context: Dict[str, Any]) -> List[str]:
        """提取目標對象"""
        targets = []
        
        # 1. 精確的文件名提取（避免動詞前綴）
        # 使用更精確的模式來匹配文件名
        file_pattern = r'(?<!\w)([a-zA-Z_][\w\-]*\.[a-zA-Z]{2,4})(?!\w)'
        file_matches = re.findall(file_pattern, command)
        
        # 過濾掉可能的誤匹配（如包含動詞的）
        clean_file_matches = []
        for match in file_matches:
            # 檢查文件名前是否有動詞
            match_pos = command.find(match)
            if match_pos > 0:
                prefix = command[:match_pos].strip()
                # 如果前面有動詞，跳過
                if not any(verb in prefix.lower().split()[-1:] for verb in 
                         ['讀取', '打開', '載入', 'read', 'open', 'load'] if prefix.split()):
                    clean_file_matches.append(match)
            else:
                clean_file_matches.append(match)
        
        targets.extend(clean_file_matches)
        
        # 2. URL
        url_matches = re.findall(self.TARGET_PATTERNS['url'], command)
        targets.extend(url_matches)
        
        # 3. 當前目錄引用
        if re.search(self.TARGET_PATTERNS['current'], command, re.IGNORECASE):
            targets.append('.')
        
        # 4. 項目級別的引用
        if re.search(self.TARGET_PATTERNS['project'], command, re.IGNORECASE):
            targets.append('.')
        
        # 5. 基於上下文的智能推斷
        if not targets:
            targets.extend(self._infer_targets_from_context(command, context))
        
        # 6. 如果仍然沒有目標，使用當前目錄
        if not targets:
            targets.append('.')
        
        # 去重並保持順序
        unique_targets = []
        for target in targets:
            if target not in unique_targets:
                unique_targets.append(target)
        
        return unique_targets[:5]  # 最多5個目標
    
    def _infer_targets_from_context(self, command: str, context: Dict[str, Any]) -> List[str]:
        """基於上下文推斷目標"""
        targets = []
        command_lower = command.lower()
        
        # 基於項目類型推斷
        project_info = context.get('project_info', {})
        project_type = project_info.get('project_type', 'unknown')
        
        if 'data' in command_lower or 'csv' in command_lower:
            # 查找數據文件
            file_summary = context.get('file_summary', {})
            notable_files = file_summary.get('notable_files', [])
            data_files = [f for f in notable_files 
                         if any(f.lower().endswith(ext) for ext in ['.csv', '.json', '.xlsx', '.parquet'])]
            targets.extend(data_files[:3])
        
        elif project_type == 'python' and ('code' in command_lower or 'py' in command_lower):
            # Python項目，查找主要Python文件
            file_summary = context.get('file_summary', {})
            file_types = file_summary.get('file_types', {})
            if '.py' in file_types:
                # 查找主要Python文件
                working_dir = Path(context.get('working_directory', '.'))
                main_files = ['main.py', 'app.py', '__init__.py', 'run.py']
                for main_file in main_files:
                    if (working_dir / main_file).exists():
                        targets.append(main_file)
                        break
        
        elif 'document' in command_lower or 'doc' in command_lower:
            # 查找文檔文件
            file_summary = context.get('file_summary', {})
            notable_files = file_summary.get('notable_files', [])
            doc_files = [f for f in notable_files 
                        if any(f.lower().endswith(ext) for ext in ['.md', '.txt', '.pdf', '.docx'])]
            targets.extend(doc_files[:3])
        
        return targets
    
    def _extract_actions(self, command: str, intent: str) -> List[str]:
        """提取動作列表"""
        actions = []
        command_lower = command.lower()
        
        # 基於關鍵詞提取動作
        for action, keywords in self.ACTION_KEYWORDS.items():
            if any(keyword in command_lower for keyword in keywords):
                actions.append(action)
        
        # 基於意圖補充默認動作
        if not actions:
            intent_default_actions = {
                'analyze': ['analyze'],
                'chat': ['explain'],
                'execute': ['execute'],
                'read': ['read'],
                'write': ['write'],
                'manage': ['configure']
            }
            actions = intent_default_actions.get(intent, ['analyze'])
        
        return actions
    
    def _extract_parameters(self, command: str) -> Dict[str, Any]:
        """提取命令參數"""
        parameters = {}
        command_lower = command.lower()
        
        # 輸出格式檢測
        format_patterns = {
            'json': r'json|JSON',
            'csv': r'csv|CSV',
            'yaml': r'yaml|YAML|yml',
            'markdown': r'markdown|md|報告',
            'html': r'html|網頁',
            'pdf': r'pdf|PDF'
        }
        
        for fmt, pattern in format_patterns.items():
            if re.search(pattern, command, re.IGNORECASE):
                parameters['output_format'] = fmt
                break
        
        # 詳細程度檢測
        if any(word in command_lower for word in ['詳細', 'detail', 'verbose', '完整']):
            parameters['verbose'] = True
        elif any(word in command_lower for word in ['簡潔', 'brief', 'summary', '概要']):
            parameters['verbose'] = False
        
        # 深度檢測
        depth_patterns = [
            r'深度\s*(\d+)',
            r'depth\s*(\d+)',
            r'level\s*(\d+)',
            r'層級\s*(\d+)'
        ]
        
        for pattern in depth_patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                parameters['max_depth'] = int(match.group(1))
                break
        
        # 數量限制檢測
        count_patterns = [
            r'前\s*(\d+)',
            r'最多\s*(\d+)',
            r'top\s*(\d+)',
            r'first\s*(\d+)',
            r'limit\s*(\d+)'
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                parameters['limit'] = int(match.group(1))
                break
        
        # 排序檢測
        if any(word in command_lower for word in ['排序', '按.*排列', 'sort', 'order']):
            parameters['sort'] = True
            
            # 排序方式
            if any(word in command_lower for word in ['大小', 'size']):
                parameters['sort_by'] = 'size'
            elif any(word in command_lower for word in ['時間', 'time', 'date']):
                parameters['sort_by'] = 'time'
            elif any(word in command_lower for word in ['名稱', 'name']):
                parameters['sort_by'] = 'name'
        
        # 過濾條件檢測
        if any(word in command_lower for word in ['包含', 'contain', '只要', 'only']):
            # 簡單的包含檢測，後續可以擴展
            parameters['filter'] = True
        
        return parameters
    
    def _calculate_confidence(self, command: str, intent: str, targets: List[str], actions: List[str]) -> float:
        """計算解析置信度"""
        confidence = 0.0
        
        # 基礎分數
        confidence += 0.3
        
        # 意圖匹配度
        intent_keywords = self.INTENT_PATTERNS.get(intent, [])
        command_lower = command.lower()
        intent_matches = sum(1 for pattern in intent_keywords if re.search(pattern, command_lower))
        if intent_matches > 0:
            confidence += min(0.3, intent_matches * 0.1)
        
        # 目標識別度
        if targets:
            # 有具體文件名的置信度更高
            specific_targets = [t for t in targets if t != '.' and '/' not in t and '\\' not in t]
            if specific_targets:
                confidence += 0.2
            else:
                confidence += 0.1
        
        # 動作清晰度
        if actions and actions != ['analyze']:
            confidence += 0.2
        
        # 命令長度合理性
        word_count = len(command.split())
        if 3 <= word_count <= 15:
            confidence += 0.1
        elif word_count > 15:
            confidence -= 0.1
        
        # 標點符號的存在
        if '?' in command or '？' in command:
            if intent == 'chat':
                confidence += 0.1
            else:
                confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def get_parsing_suggestions(self, command: str) -> List[str]:
        """獲取解析建議"""
        suggestions = []
        
        if len(command.strip()) < 3:
            suggestions.append("命令太短，請提供更多細節")
        
        # 檢查是否包含常見文件類型
        common_extensions = ['.csv', '.json', '.py', '.md', '.txt']
        found_files = []
        for ext in common_extensions:
            if ext in command.lower():
                found_files.append(ext)
        
        if found_files:
            suggestions.append(f"檢測到文件類型: {', '.join(found_files)}")
        
        # 檢查是否是問句
        if '?' in command or '？' in command:
            suggestions.append("檢測到問句，建議使用chat命令")
        
        return suggestions
    
    def get_command_history(self, limit: int = 10) -> List[ParsedCommand]:
        """獲取命令歷史"""
        return self.command_history[-limit:]
    
    def clear_history(self):
        """清空命令歷史"""
        self.command_history.clear()


def get_natural_language_parser() -> NaturalLanguageParser:
    """獲取自然語言解析器實例"""
    if not hasattr(get_natural_language_parser, '_instance'):
        get_natural_language_parser._instance = NaturalLanguageParser()
    return get_natural_language_parser._instance

