"""
環境上下文檢測系統
智能檢測當前工作環境的項目類型、文件結構和相關信息
"""

import os
import platform
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ProjectFeature:
    """項目特徵"""
    name: str
    confidence: float
    evidence: List[str]
    category: str  # 'file', 'directory', 'content', 'pattern'


class ProjectDetector:
    """項目類型檢測器"""
    
    def __init__(self):
        # 定義項目類型檢測規則
        self.project_patterns = {
            'python': {
                'files': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile', '__init__.py'],
                'extensions': ['.py', '.pyx', '.pyi'],
                'directories': ['__pycache__', '.venv', 'venv', 'env'],
                'weight': 1.0
            },
            'javascript': {
                'files': ['package.json', 'yarn.lock', 'package-lock.json', '.nvmrc'],
                'extensions': ['.js', '.jsx', '.ts', '.tsx'],
                'directories': ['node_modules', 'dist', 'build'],
                'weight': 1.0
            },
            'java': {
                'files': ['pom.xml', 'build.gradle', 'gradle.properties'],
                'extensions': ['.java', '.class', '.jar'],
                'directories': ['src/main/java', 'target', 'build'],
                'weight': 1.0
            },
            'rust': {
                'files': ['Cargo.toml', 'Cargo.lock'],
                'extensions': ['.rs'],
                'directories': ['target', 'src'],
                'weight': 1.0
            },
            'go': {
                'files': ['go.mod', 'go.sum'],
                'extensions': ['.go'],
                'directories': ['vendor'],
                'weight': 1.0
            },
            'web': {
                'files': ['index.html', 'index.htm', 'webpack.config.js'],
                'extensions': ['.html', '.htm', '.css', '.scss', '.sass'],
                'directories': ['assets', 'static', 'public'],
                'weight': 0.8
            },
            'data_science': {
                'files': ['requirements.txt', 'environment.yml', 'Pipfile'],
                'extensions': ['.ipynb', '.py', '.r', '.csv', '.json'],
                'directories': ['data', 'notebooks', 'models'],
                'weight': 0.9
            },
            'docker': {
                'files': ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml', '.dockerignore'],
                'extensions': [],
                'directories': [],
                'weight': 0.7
            }
        }
    
    def detect(self, directory: Path) -> Dict[str, Any]:
        """檢測項目類型"""
        if not directory.exists() or not directory.is_dir():
            return {'primary_type': 'unknown', 'confidence_scores': {}, 'detected_features': []}
        
        # 計算每種類型的置信度
        confidence_scores = {}
        all_features = []
        
        for project_type, patterns in self.project_patterns.items():
            features, confidence = self._calculate_type_confidence(directory, project_type, patterns)
            confidence_scores[project_type] = confidence
            all_features.extend(features)
        
        # 確定主要類型
        primary_type = max(confidence_scores.items(), key=lambda x: x[1])[0]
        if confidence_scores[primary_type] < 0.3:
            primary_type = 'unknown'
        
        return {
            'primary_type': primary_type,
            'confidence_scores': confidence_scores,
            'detected_features': [f.name for f in all_features]
        }
    
    def _calculate_type_confidence(self, directory: Path, project_type: str, 
                                 patterns: Dict[str, List[str]]) -> Tuple[List[ProjectFeature], float]:
        """計算特定類型的置信度"""
        features = []
        confidence = 0.0
        
        # 檢查特徵文件
        for file_name in patterns['files']:
            if (directory / file_name).exists():
                features.append(ProjectFeature(
                    name=f"Found {file_name}",
                    confidence=0.8,
                    evidence=[file_name],
                    category='file'
                ))
                confidence += 0.2
        
        # 檢查目錄結構
        for dir_name in patterns['directories']:
            if (directory / dir_name).exists():
                features.append(ProjectFeature(
                    name=f"Found directory {dir_name}",
                    confidence=0.6,
                    evidence=[dir_name],
                    category='directory'
                ))
                confidence += 0.1
        
        # 檢查文件擴展名分布
        if patterns['extensions']:
            ext_count = self._count_extensions(directory, patterns['extensions'])
            if ext_count > 0:
                features.append(ProjectFeature(
                    name=f"Found {ext_count} files with relevant extensions",
                    confidence=min(0.8, ext_count * 0.1),
                    evidence=[f"{ext_count} relevant files"],
                    category='content'
                ))
                confidence += min(0.3, ext_count * 0.05)
        
        # 應用權重
        confidence *= patterns['weight']
        
        return features, min(1.0, confidence)
    
    def _count_extensions(self, directory: Path, extensions: List[str], max_depth: int = 2) -> int:
        """統計特定擴展名的文件數量"""
        count = 0
        try:
            for item in directory.rglob('*'):
                if item.is_file() and item.suffix.lower() in extensions:
                    count += 1
                    if count >= 20:  # 避免遍歷過多文件
                        break
        except (PermissionError, OSError):
            pass
        
        return count


class FileScanner:
    """智能文件掃描器"""
    
    def __init__(self):
        self.max_scan_depth = 3
        self.max_files_per_scan = 1000
        self.notable_patterns = [
            'README', 'LICENSE', 'CHANGELOG', 'CONTRIBUTING',
            'requirements.txt', 'package.json', 'Dockerfile',
            'docker-compose.yml', '.gitignore', 'Makefile'
        ]
    
    def scan(self, directory: Path, max_depth: int = None, max_files: int = None) -> Dict[str, Any]:
        """掃描目錄並生成摘要"""
        max_depth = max_depth or self.max_scan_depth
        max_files = max_files or self.max_files_per_scan
        
        summary = {
            'total_files': 0,
            'total_directories': 0,
            'file_types': {},
            'size_distribution': {'small': 0, 'medium': 0, 'large': 0, 'very_large': 0},
            'notable_files': [],
            'large_files': [],
            'recent_files': []
        }
        
        try:
            current_depth = 0
            files_scanned = 0
            
            for item in self._safe_walk(directory, max_depth):
                if files_scanned >= max_files:
                    break
                
                if item.is_file():
                    self._process_file(item, summary, directory)
                    files_scanned += 1
                elif item.is_dir() and item != directory:
                    summary['total_directories'] += 1
                    
        except Exception as e:
            summary['scan_error'] = str(e)
        
        return summary
    
    def _safe_walk(self, directory: Path, max_depth: int):
        """安全遍歷目錄"""
        try:
            def _walk_recursive(path: Path, current_depth: int):
                if current_depth > max_depth:
                    return
                
                try:
                    for item in path.iterdir():
                        if item.name.startswith('.') and item.name not in ['.gitignore', '.env']:
                            continue  # 跳過隱藏文件
                        
                        yield item
                        
                        if item.is_dir() and current_depth < max_depth:
                            yield from _walk_recursive(item, current_depth + 1)
                            
                except (PermissionError, OSError):
                    pass  # 跳過無法訪問的目錄
            
            yield from _walk_recursive(directory, 0)
            
        except Exception:
            pass
    
    def _process_file(self, file_path: Path, summary: Dict, base_dir: Path):
        """處理單個文件"""
        summary['total_files'] += 1
        
        # 文件類型統計
        extension = file_path.suffix.lower()
        summary['file_types'][extension] = summary['file_types'].get(extension, 0) + 1
        
        try:
            # 文件大小統計
            size = file_path.stat().st_size
            size_category = self._categorize_size(size)
            summary['size_distribution'][size_category] += 1
            
            # 大文件記錄 (> 10MB)
            if size > 10 * 1024 * 1024:
                relative_path = str(file_path.relative_to(base_dir))
                summary['large_files'].append({
                    'path': relative_path,
                    'size_mb': round(size / (1024 * 1024), 2)
                })
            
            # 注目文件檢測
            if self._is_notable_file(file_path):
                relative_path = str(file_path.relative_to(base_dir))
                summary['notable_files'].append(relative_path)
            
            # 最近修改的文件
            mtime = file_path.stat().st_mtime
            if len(summary['recent_files']) < 10:
                relative_path = str(file_path.relative_to(base_dir))
                summary['recent_files'].append({
                    'path': relative_path,
                    'modified': datetime.fromtimestamp(mtime).isoformat()
                })
            
        except (OSError, PermissionError):
            pass  # 忽略無法訪問的文件
    
    def _categorize_size(self, size_bytes: int) -> str:
        """文件大小分類"""
        if size_bytes < 1024:  # < 1KB
            return 'small'
        elif size_bytes < 1024 * 1024:  # < 1MB
            return 'medium'
        elif size_bytes < 10 * 1024 * 1024:  # < 10MB
            return 'large'
        else:
            return 'very_large'
    
    def _is_notable_file(self, file_path: Path) -> bool:
        """檢測是否為重要文件"""
        file_name = file_path.name.lower()
        return any(pattern.lower() in file_name for pattern in self.notable_patterns)


class GitDetector:
    """Git倉庫檢測器"""
    
    def detect(self, directory: Path) -> Dict[str, Any]:
        """檢測Git信息"""
        git_info = {
            'is_git_repo': False,
            'current_branch': None,
            'has_uncommitted_changes': False,
            'remote_url': None,
            'last_commit': None
        }
        
        git_dir = directory / '.git'
        if not git_dir.exists():
            # 檢查父目錄是否為Git倉庫
            parent = directory.parent
            while parent != parent.parent:
                if (parent / '.git').exists():
                    git_dir = parent / '.git'
                    break
                parent = parent.parent
        
        if git_dir.exists():
            git_info['is_git_repo'] = True
            
            try:
                # 獲取當前分支
                result = subprocess.run(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    git_info['current_branch'] = result.stdout.strip()
                
                # 檢查是否有未提交的更改
                result = subprocess.run(
                    ['git', 'status', '--porcelain'],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    git_info['has_uncommitted_changes'] = bool(result.stdout.strip())
                
                # 獲取遠程URL
                result = subprocess.run(
                    ['git', 'remote', 'get-url', 'origin'],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    git_info['remote_url'] = result.stdout.strip()
                
                # 獲取最後一次提交
                result = subprocess.run(
                    ['git', 'log', '-1', '--format=%h %s'],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    git_info['last_commit'] = result.stdout.strip()
                
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                pass  # Git命令不可用或超時
        
        return git_info


class ContextDetector:
    """環境上下文檢測器"""
    
    def __init__(self, working_directory: Path):
        self.working_dir = working_directory
        self.project_detector = ProjectDetector()
        self.file_scanner = FileScanner()
        self.git_detector = GitDetector()
    
    def detect_full_context(self) -> Dict[str, Any]:
        """檢測完整上下文信息"""
        context = {
            'timestamp': self._get_timestamp(),
            'working_directory': str(self.working_dir),
            'directory_info': self._get_directory_info(),
            'project_info': self.project_detector.detect(self.working_dir),
            'file_summary': self.file_scanner.scan(self.working_dir),
            'git_info': self.git_detector.detect(self.working_dir),
            'environment_info': self._detect_environment(),
            'recent_activity': self._get_recent_activity()
        }
        
        return context
    
    def _get_timestamp(self) -> str:
        """獲取當前時間戳"""
        return datetime.now().isoformat()
    
    def _get_directory_info(self) -> Dict[str, Any]:
        """獲取目錄基本信息"""
        try:
            if not self.working_dir.exists():
                return {'exists': False, 'error': 'Directory does not exist'}
            
            stat = self.working_dir.stat()
            
            # 計算目錄大小（概估）
            total_size = 0
            item_count = 0
            
            try:
                for item in self.working_dir.iterdir():
                    item_count += 1
                    if item.is_file():
                        total_size += item.stat().st_size
                    if item_count >= 100:  # 限制掃描數量
                        break
            except (PermissionError, OSError):
                pass
            
            return {
                'exists': True,
                'is_directory': self.working_dir.is_dir(),
                'permissions': oct(stat.st_mode)[-3:] if hasattr(stat, 'st_mode') else 'unknown',
                'estimated_size_bytes': total_size,
                'item_count': item_count,
                'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            
        except Exception as e:
            return {'exists': False, 'error': str(e)}
    
    def _detect_environment(self) -> Dict[str, Any]:
        """檢測運行環境"""
        import sys
        
        env_info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'python_version': sys.version,
            'python_executable': sys.executable,
            'hostname': platform.node(),
            'user': os.getenv('USER', os.getenv('USERNAME', 'unknown')),
            'home_directory': str(Path.home()),
            'current_directory': str(Path.cwd())
        }
        
        # 環境變量
        interesting_env_vars = ['PATH', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV']
        env_info['environment_variables'] = {
            var: os.getenv(var, '') for var in interesting_env_vars if os.getenv(var)
        }
        
        # Shell信息
        shell = os.getenv('SHELL', '')
        if shell:
            env_info['shell'] = shell
        
        terminal = os.getenv('TERM', '')
        if terminal:
            env_info['terminal'] = terminal
        
        return env_info
    
    def _get_recent_activity(self) -> Dict[str, Any]:
        """獲取最近活動信息"""
        activity = {
            'last_detection': datetime.now().isoformat(),
            'working_directory_changed': False,
            'detection_count': 1
        }
        
        # 這裡可以添加更多活動追蹤邏輯
        # 比如檢測最近修改的文件、命令歷史等
        
        return activity


