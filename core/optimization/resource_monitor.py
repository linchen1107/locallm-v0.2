"""
資源監控器
實時監控系統資源使用情況，提供智能調度建議
"""

import asyncio
import time
import json
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None
from collections import deque, defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


@dataclass
class ResourceSnapshot:
    """資源快照"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_usage_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    active_processes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceMetrics:
    """性能指標"""
    avg_cpu_usage: float
    max_cpu_usage: float
    avg_memory_usage: float
    max_memory_usage: float
    avg_response_time: float
    throughput_per_second: float
    error_rate: float
    uptime_hours: float
    
    def __post_init__(self):
        # 確保百分比在合理範圍內
        self.avg_cpu_usage = max(0, min(100, self.avg_cpu_usage))
        self.max_cpu_usage = max(0, min(100, self.max_cpu_usage))
        self.avg_memory_usage = max(0, min(100, self.avg_memory_usage))
        self.max_memory_usage = max(0, min(100, self.max_memory_usage))
        self.error_rate = max(0, min(100, self.error_rate))


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alert_handlers: List[Callable] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_rules = {
            'high_cpu': {'threshold': 80, 'duration': 60, 'enabled': True},
            'high_memory': {'threshold': 85, 'duration': 30, 'enabled': True},
            'low_disk': {'threshold': 90, 'duration': 0, 'enabled': True},
            'high_error_rate': {'threshold': 5, 'duration': 60, 'enabled': True}
        }
    
    def add_alert_handler(self, handler: Callable):
        """添加告警處理器"""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, snapshot: ResourceSnapshot, metrics: PerformanceMetrics):
        """檢查告警條件"""
        alerts = []
        
        # CPU告警
        if (self.alert_rules['high_cpu']['enabled'] and 
            snapshot.cpu_percent > self.alert_rules['high_cpu']['threshold']):
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f"CPU使用率過高: {snapshot.cpu_percent:.1f}%",
                'value': snapshot.cpu_percent,
                'threshold': self.alert_rules['high_cpu']['threshold']
            })
        
        # 內存告警
        if (self.alert_rules['high_memory']['enabled'] and 
            snapshot.memory_percent > self.alert_rules['high_memory']['threshold']):
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning',
                'message': f"內存使用率過高: {snapshot.memory_percent:.1f}%",
                'value': snapshot.memory_percent,
                'threshold': self.alert_rules['high_memory']['threshold']
            })
        
        # 磁盤告警
        if (self.alert_rules['low_disk']['enabled'] and 
            snapshot.disk_usage_percent > self.alert_rules['low_disk']['threshold']):
            alerts.append({
                'type': 'low_disk',
                'severity': 'critical',
                'message': f"磁盤空間不足: {snapshot.disk_usage_percent:.1f}%已使用",
                'value': snapshot.disk_usage_percent,
                'threshold': self.alert_rules['low_disk']['threshold']
            })
        
        # 錯誤率告警
        if (self.alert_rules['high_error_rate']['enabled'] and 
            metrics.error_rate > self.alert_rules['high_error_rate']['threshold']):
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'warning',
                'message': f"錯誤率過高: {metrics.error_rate:.1f}%",
                'value': metrics.error_rate,
                'threshold': self.alert_rules['high_error_rate']['threshold']
            })
        
        # 發送告警
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """發送告警"""
        alert['timestamp'] = datetime.now().isoformat()
        self.alert_history.append(alert)
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Alert handler error: {e}")


class ResourceMonitor:
    """資源監控器"""
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 history_size: int = 3600,
                 enable_gpu_monitoring: bool = True):
        
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # 歷史數據
        self.snapshots: deque = deque(maxlen=history_size)
        self.performance_history: deque = deque(maxlen=100)
        
        # 監控狀態
        self._monitoring = False
        self._monitor_thread = None
        
        # 統計數據
        self.start_time = datetime.now()
        self.task_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'errors': 0})
        
        # 告警管理
        self.alert_manager = AlertManager()
        
        # 網絡統計基準
        self._network_baseline = None
        
        # GPU監控
        self._gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """檢查GPU可用性"""
        if not self.enable_gpu_monitoring or not GPU_AVAILABLE:
            return False
        
        try:
            GPUtil.getGPUs()
            return True
        except:
            return False
    
    def start_monitoring(self):
        """開始監控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        print("📊 資源監控已啟動")
    
    def stop_monitoring(self):
        """停止監控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        print("📊 資源監控已停止")
    
    def _monitoring_loop(self):
        """監控循環"""
        while self._monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                
                # 計算性能指標
                if len(self.snapshots) >= 60:  # 至少有1分鐘數據
                    metrics = self._calculate_metrics()
                    self.performance_history.append(metrics)
                    
                    # 檢查告警
                    self.alert_manager.check_alerts(snapshot, metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """拍攝資源快照"""
        # 基本系統信息
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 網絡信息
        net_io = psutil.net_io_counters()
        if self._network_baseline is None:
            self._network_baseline = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'timestamp': time.time()
            }
            network_sent_mb = 0.0
            network_recv_mb = 0.0
        else:
            time_diff = time.time() - self._network_baseline['timestamp']
            if time_diff > 0:
                network_sent_mb = (net_io.bytes_sent - self._network_baseline['bytes_sent']) / (1024 * 1024) / time_diff
                network_recv_mb = (net_io.bytes_recv - self._network_baseline['bytes_recv']) / (1024 * 1024) / time_diff
            else:
                network_sent_mb = 0.0
                network_recv_mb = 0.0
            
            # 更新基準
            self._network_baseline = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'timestamp': time.time()
            }
        
        # GPU信息
        gpu_usage = 0.0
        gpu_memory = 0.0
        if self._gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
                    gpu_memory = gpus[0].memoryUsed
            except:
                pass
        
        # 進程數量
        active_processes = len(psutil.pids())
        
        return ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024 * 1024 * 1024),
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            gpu_usage_percent=gpu_usage,
            gpu_memory_used_mb=gpu_memory,
            active_processes=active_processes
        )
    
    def _calculate_metrics(self) -> PerformanceMetrics:
        """計算性能指標"""
        if len(self.snapshots) < 2:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        recent_snapshots = list(self.snapshots)[-60:]  # 最近60個快照（約1分鐘）
        
        # CPU指標
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        avg_cpu = np.mean(cpu_values)
        max_cpu = np.max(cpu_values)
        
        # 內存指標
        memory_values = [s.memory_percent for s in recent_snapshots]
        avg_memory = np.mean(memory_values)
        max_memory = np.max(memory_values)
        
        # 計算吞吐量（基於任務統計）
        total_tasks = sum(stats['count'] for stats in self.task_stats.values())
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        throughput = total_tasks / uptime_hours if uptime_hours > 0 else 0
        
        # 計算錯誤率
        total_errors = sum(stats['errors'] for stats in self.task_stats.values())
        error_rate = (total_errors / total_tasks * 100) if total_tasks > 0 else 0
        
        # 計算平均響應時間
        total_time = sum(stats['total_time'] for stats in self.task_stats.values())
        avg_response_time = total_time / total_tasks if total_tasks > 0 else 0
        
        return PerformanceMetrics(
            avg_cpu_usage=avg_cpu,
            max_cpu_usage=max_cpu,
            avg_memory_usage=avg_memory,
            max_memory_usage=max_memory,
            avg_response_time=avg_response_time,
            throughput_per_second=throughput / 3600,  # 轉換為每秒
            error_rate=error_rate,
            uptime_hours=uptime_hours
        )
    
    def record_task_execution(self, task_name: str, execution_time: float, success: bool):
        """記錄任務執行"""
        stats = self.task_stats[task_name]
        stats['count'] += 1
        stats['total_time'] += execution_time
        if not success:
            stats['errors'] += 1
    
    def get_current_status(self) -> Dict[str, Any]:
        """獲取當前狀態"""
        if not self.snapshots:
            return {'status': 'no_data'}
        
        latest_snapshot = self.snapshots[-1]
        metrics = self._calculate_metrics() if len(self.snapshots) >= 60 else None
        
        status = {
            'timestamp': latest_snapshot.timestamp.isoformat(),
            'resources': {
                'cpu_percent': latest_snapshot.cpu_percent,
                'memory_percent': latest_snapshot.memory_percent,
                'memory_used_gb': latest_snapshot.memory_used_mb / 1024,
                'disk_usage_percent': latest_snapshot.disk_usage_percent,
                'disk_free_gb': latest_snapshot.disk_free_gb,
                'active_processes': latest_snapshot.active_processes
            }
        }
        
        if self._gpu_available:
            status['resources']['gpu_usage_percent'] = latest_snapshot.gpu_usage_percent
            status['resources']['gpu_memory_used_mb'] = latest_snapshot.gpu_memory_used_mb
        
        if metrics:
            status['performance'] = asdict(metrics)
        
        # 系統健康評分
        status['health_score'] = self._calculate_health_score(latest_snapshot)
        
        return status
    
    def _calculate_health_score(self, snapshot: ResourceSnapshot) -> float:
        """計算系統健康評分 (0-100)"""
        score = 100.0
        
        # CPU負載影響 (權重: 25%)
        if snapshot.cpu_percent > 80:
            score -= (snapshot.cpu_percent - 80) * 0.5
        
        # 內存使用影響 (權重: 30%)
        if snapshot.memory_percent > 75:
            score -= (snapshot.memory_percent - 75) * 0.6
        
        # 磁盤空間影響 (權重: 25%)
        if snapshot.disk_usage_percent > 80:
            score -= (snapshot.disk_usage_percent - 80) * 0.5
        
        # GPU負載影響 (權重: 20%)
        if self._gpu_available and snapshot.gpu_usage_percent > 90:
            score -= (snapshot.gpu_usage_percent - 90) * 0.4
        
        return max(0, min(100, score))
    
    def get_resource_recommendations(self) -> List[str]:
        """獲取資源優化建議"""
        if not self.snapshots:
            return []
        
        latest = self.snapshots[-1]
        recommendations = []
        
        # CPU建議
        if latest.cpu_percent > 80:
            recommendations.append("💡 CPU使用率過高，建議減少並行任務數量或優化CPU密集型操作")
        elif latest.cpu_percent < 20:
            recommendations.append("💡 CPU利用率較低，可以增加並行處理來提升效率")
        
        # 內存建議
        if latest.memory_percent > 85:
            recommendations.append("⚠️ 內存使用率過高，建議啟用緩存清理或增加批處理大小限制")
        elif latest.memory_percent < 30:
            recommendations.append("💡 內存充足，可以增加緩存大小或提高批處理大小")
        
        # 磁盤建議
        if latest.disk_usage_percent > 90:
            recommendations.append("🚨 磁盤空間嚴重不足，請清理臨時文件和舊緩存")
        elif latest.disk_free_gb < 1:
            recommendations.append("⚠️ 可用磁盤空間不足1GB，建議清理不必要的文件")
        
        # GPU建議
        if self._gpu_available:
            if latest.gpu_usage_percent > 95:
                recommendations.append("🔥 GPU使用率接近滿載，建議優化GPU任務調度")
            elif latest.gpu_usage_percent < 10:
                recommendations.append("💡 GPU利用率較低，可以啟用GPU加速的處理模式")
        
        return recommendations
    
    def get_trend_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """獲取趨勢分析"""
        if len(self.snapshots) < 2:
            return {'error': 'insufficient_data'}
        
        # 計算時間範圍
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {'error': 'insufficient_recent_data'}
        
        # 計算趨勢
        timestamps = [s.timestamp for s in recent_snapshots]
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]
        
        # 線性回歸計算趨勢
        def calculate_trend(values):
            if not NUMPY_AVAILABLE or len(values) <= 1:
                return 0.0
            
            try:
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                return slope
            except:
                return 0.0
        
        cpu_trend = calculate_trend(cpu_values)
        memory_trend = calculate_trend(memory_values)
        
        # 趨勢分析
        analysis = {
            'time_range_hours': hours,
            'data_points': len(recent_snapshots),
            'trends': {
                'cpu_trend_per_hour': cpu_trend * (3600 / self.monitoring_interval),
                'memory_trend_per_hour': memory_trend * (3600 / self.monitoring_interval),
                'cpu_direction': 'increasing' if cpu_trend > 0.1 else 'decreasing' if cpu_trend < -0.1 else 'stable',
                'memory_direction': 'increasing' if memory_trend > 0.1 else 'decreasing' if memory_trend < -0.1 else 'stable'
            },
            'statistics': {
                'avg_cpu': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max_cpu': max(cpu_values) if cpu_values else 0,
                'min_cpu': min(cpu_values) if cpu_values else 0,
                'avg_memory': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max_memory': max(memory_values) if memory_values else 0,
                'min_memory': min(memory_values) if memory_values else 0
            }
        }
        
        return analysis
    
    def export_metrics(self, file_path: str, format: str = 'json'):
        """導出監控數據"""
        data = {
            'export_time': datetime.now().isoformat(),
            'monitoring_period': {
                'start': self.start_time.isoformat(),
                'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600
            },
            'snapshots': [s.to_dict() for s in self.snapshots],
            'task_statistics': dict(self.task_stats),
            'alert_history': list(self.alert_manager.alert_history)
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📊 監控數據已導出到: {file_path}")
    
    def get_resource_usage_prediction(self, hours_ahead: int = 1) -> Dict[str, Any]:
        """預測未來資源使用"""
        if len(self.snapshots) < 10:
            return {'error': 'insufficient_data_for_prediction'}
        
        recent_snapshots = list(self.snapshots)[-100:]  # 使用最近100個數據點
        
        # 提取時間序列數據
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]
        
        # 簡單的線性外推預測
        def predict_values(values, steps_ahead):
            if not NUMPY_AVAILABLE or len(values) < 2:
                return values[-1] if values else 0
            
            try:
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                future_x = len(values) + steps_ahead
                predicted = slope * future_x + intercept
                return max(0, min(100, predicted))  # 限制在合理範圍內
            except:
                return values[-1] if values else 0
        
        # 計算預測步數
        steps_ahead = int(hours_ahead * 3600 / self.monitoring_interval)
        
        predicted_cpu = predict_values(cpu_values, steps_ahead)
        predicted_memory = predict_values(memory_values, steps_ahead)
        
        # 風險評估
        risk_level = 'low'
        if predicted_cpu > 90 or predicted_memory > 90:
            risk_level = 'high'
        elif predicted_cpu > 75 or predicted_memory > 75:
            risk_level = 'medium'
        
        return {
            'prediction_time': (datetime.now() + timedelta(hours=hours_ahead)).isoformat(),
            'predicted_cpu_percent': predicted_cpu,
            'predicted_memory_percent': predicted_memory,
            'confidence': min(100, len(recent_snapshots) * 5),  # 基於數據點數量的置信度
            'risk_level': risk_level,
            'recommendations': self._get_prediction_recommendations(predicted_cpu, predicted_memory, risk_level)
        }
    
    def _get_prediction_recommendations(self, cpu: float, memory: float, risk: str) -> List[str]:
        """基於預測的建議"""
        recommendations = []
        
        if risk == 'high':
            recommendations.append("🚨 預測資源使用率將達到危險水平，建議立即採取優化措施")
            if cpu > 90:
                recommendations.append("💻 預計CPU負載過高，考慮減少並行任務或增加處理間隔")
            if memory > 90:
                recommendations.append("🧠 預計內存不足，建議啟用緩存清理或降低批處理大小")
        elif risk == 'medium':
            recommendations.append("⚠️ 資源使用率可能升高，建議監控並準備優化措施")
        else:
            recommendations.append("✅ 預計資源使用正常，可以考慮提升處理效率")
        
        return recommendations


# 全局監控實例
_global_monitor = None


def get_resource_monitor() -> ResourceMonitor:
    """獲取全局資源監控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResourceMonitor()
    return _global_monitor


# 監控裝飾器
def monitor_performance(task_name: str = None):
    """性能監控裝飾器"""
    def decorator(func):
        actual_task_name = task_name or func.__name__
        
        async def async_wrapper(*args, **kwargs):
            monitor = get_resource_monitor()
            start_time = time.time()
            success = False
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                raise e
            finally:
                execution_time = time.time() - start_time
                monitor.record_task_execution(actual_task_name, execution_time, success)
        
        def sync_wrapper(*args, **kwargs):
            monitor = get_resource_monitor()
            start_time = time.time()
            success = False
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                raise e
            finally:
                execution_time = time.time() - start_time
                monitor.record_task_execution(actual_task_name, execution_time, success)
        
        # 根據函數類型返回相應的包裝器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
