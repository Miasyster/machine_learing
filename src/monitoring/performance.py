"""
性能监控模块

提供模型性能、系统资源、推理延迟等监控功能
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
import numpy as np
from dataclasses import dataclass

from .base import BaseMonitor, MonitoringConfig, MetricData, MetricType

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据"""
    latency_ms: float
    throughput_rps: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    error_rate: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ModelPerformanceTracker(BaseMonitor):
    """
    模型性能跟踪器
    """
    
    def __init__(self, config: MonitoringConfig, model_name: str = "default"):
        """
        初始化模型性能跟踪器
        
        Args:
            config: 监控配置
            model_name: 模型名称
        """
        super().__init__(config)
        self.model_name = model_name
        self.prediction_times = deque(maxlen=1000)
        self.prediction_counts = deque(maxlen=100)
        self.error_counts = deque(maxlen=100)
        self.accuracy_scores = deque(maxlen=100)
        self.drift_scores = deque(maxlen=100)
        
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._stop_event = threading.Event()
    
    def record_prediction(self, 
                         latency: float,
                         success: bool = True,
                         accuracy: Optional[float] = None,
                         drift_score: Optional[float] = None) -> None:
        """
        记录预测性能
        
        Args:
            latency: 预测延迟（毫秒）
            success: 是否成功
            accuracy: 准确率
            drift_score: 漂移分数
        """
        with self._lock:
            self.prediction_times.append(latency)
            
            if not success:
                self.error_counts.append(1)
            else:
                self.error_counts.append(0)
            
            if accuracy is not None:
                self.accuracy_scores.append(accuracy)
            
            if drift_score is not None:
                self.drift_scores.append(drift_score)
    
    def collect_metrics(self) -> List[MetricData]:
        """收集模型性能指标"""
        metrics = []
        
        with self._lock:
            if not self.prediction_times:
                return metrics
            
            # 延迟指标
            latencies = list(self.prediction_times)
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            metrics.extend([
                MetricData(
                    name=f"model.{self.model_name}.latency.avg",
                    value=avg_latency,
                    metric_type=MetricType.GAUGE,
                    unit="ms",
                    tags={"model": self.model_name}
                ),
                MetricData(
                    name=f"model.{self.model_name}.latency.p95",
                    value=p95_latency,
                    metric_type=MetricType.GAUGE,
                    unit="ms",
                    tags={"model": self.model_name}
                ),
                MetricData(
                    name=f"model.{self.model_name}.latency.p99",
                    value=p99_latency,
                    metric_type=MetricType.GAUGE,
                    unit="ms",
                    tags={"model": self.model_name}
                )
            ])
            
            # 错误率
            if self.error_counts:
                error_rate = np.mean(list(self.error_counts))
                metrics.append(MetricData(
                    name=f"model.{self.model_name}.error_rate",
                    value=error_rate,
                    metric_type=MetricType.GAUGE,
                    unit="ratio",
                    tags={"model": self.model_name}
                ))
            
            # 准确率
            if self.accuracy_scores:
                avg_accuracy = np.mean(list(self.accuracy_scores))
                metrics.append(MetricData(
                    name=f"model.{self.model_name}.accuracy",
                    value=avg_accuracy,
                    metric_type=MetricType.GAUGE,
                    unit="ratio",
                    tags={"model": self.model_name}
                ))
            
            # 数据漂移
            if self.drift_scores:
                avg_drift = np.mean(list(self.drift_scores))
                metrics.append(MetricData(
                    name=f"model.{self.model_name}.drift_score",
                    value=avg_drift,
                    metric_type=MetricType.GAUGE,
                    unit="score",
                    tags={"model": self.model_name}
                ))
        
        return metrics
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        logger.info(f"Model performance monitoring started for {self.model_name}")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info(f"Model performance monitoring stopped for {self.model_name}")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while not self._stop_event.is_set():
            try:
                if self.should_collect():
                    metrics = self.collect_metrics()
                    for metric in metrics:
                        self.add_metric(metric)
                    self.update_collection_time()
                
                time.sleep(1)  # 每秒检查一次
            
            except Exception as e:
                logger.error(f"Error in model performance monitoring: {e}")
                time.sleep(5)


class SystemResourceMonitor(BaseMonitor):
    """
    系统资源监控器
    """
    
    def __init__(self, config: MonitoringConfig):
        """
        初始化系统资源监控器
        
        Args:
            config: 监控配置
        """
        super().__init__(config)
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        
        # GPU监控（如果可用）
        self.gpu_available = self._check_gpu_available()
    
    def _check_gpu_available(self) -> bool:
        """检查GPU是否可用"""
        try:
            import GPUtil
            return len(GPUtil.getGPUs()) > 0
        except ImportError:
            return False
    
    def collect_metrics(self) -> List[MetricData]:
        """收集系统资源指标"""
        metrics = []
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricData(
                name="system.cpu.usage",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                unit="percent"
            ))
            
            # 内存使用率
            memory = psutil.virtual_memory()
            metrics.extend([
                MetricData(
                    name="system.memory.usage",
                    value=memory.percent,
                    metric_type=MetricType.GAUGE,
                    unit="percent"
                ),
                MetricData(
                    name="system.memory.available",
                    value=memory.available / (1024**3),  # GB
                    metric_type=MetricType.GAUGE,
                    unit="GB"
                ),
                MetricData(
                    name="system.memory.used",
                    value=memory.used / (1024**3),  # GB
                    metric_type=MetricType.GAUGE,
                    unit="GB"
                )
            ])
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            metrics.extend([
                MetricData(
                    name="system.disk.usage",
                    value=(disk.used / disk.total) * 100,
                    metric_type=MetricType.GAUGE,
                    unit="percent"
                ),
                MetricData(
                    name="system.disk.free",
                    value=disk.free / (1024**3),  # GB
                    metric_type=MetricType.GAUGE,
                    unit="GB"
                )
            ])
            
            # 网络IO
            net_io = psutil.net_io_counters()
            metrics.extend([
                MetricData(
                    name="system.network.bytes_sent",
                    value=net_io.bytes_sent,
                    metric_type=MetricType.COUNTER,
                    unit="bytes"
                ),
                MetricData(
                    name="system.network.bytes_recv",
                    value=net_io.bytes_recv,
                    metric_type=MetricType.COUNTER,
                    unit="bytes"
                )
            ])
            
            # GPU使用率（如果可用）
            if self.gpu_available:
                gpu_metrics = self._collect_gpu_metrics()
                metrics.extend(gpu_metrics)
        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _collect_gpu_metrics(self) -> List[MetricData]:
        """收集GPU指标"""
        metrics = []
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            for i, gpu in enumerate(gpus):
                metrics.extend([
                    MetricData(
                        name=f"system.gpu.{i}.usage",
                        value=gpu.load * 100,
                        metric_type=MetricType.GAUGE,
                        unit="percent",
                        tags={"gpu_id": str(i), "gpu_name": gpu.name}
                    ),
                    MetricData(
                        name=f"system.gpu.{i}.memory_usage",
                        value=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                        metric_type=MetricType.GAUGE,
                        unit="percent",
                        tags={"gpu_id": str(i), "gpu_name": gpu.name}
                    ),
                    MetricData(
                        name=f"system.gpu.{i}.temperature",
                        value=gpu.temperature,
                        metric_type=MetricType.GAUGE,
                        unit="celsius",
                        tags={"gpu_id": str(i), "gpu_name": gpu.name}
                    )
                ])
        
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
        
        return metrics
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        logger.info("System resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("System resource monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while not self._stop_event.is_set():
            try:
                if self.should_collect():
                    metrics = self.collect_metrics()
                    for metric in metrics:
                        self.add_metric(metric)
                    self.update_collection_time()
                
                time.sleep(self.config.collection_interval)
            
            except Exception as e:
                logger.error(f"Error in system resource monitoring: {e}")
                time.sleep(5)


class InferenceLatencyMonitor(BaseMonitor):
    """
    推理延迟监控器
    """
    
    def __init__(self, config: MonitoringConfig):
        """
        初始化推理延迟监控器
        
        Args:
            config: 监控配置
        """
        super().__init__(config)
        self.latency_data = defaultdict(deque)
        self._lock = threading.Lock()
    
    def record_latency(self, 
                      operation: str,
                      latency: float,
                      tags: Optional[Dict[str, str]] = None) -> None:
        """
        记录延迟数据
        
        Args:
            operation: 操作名称
            latency: 延迟时间（毫秒）
            tags: 标签
        """
        with self._lock:
            key = f"{operation}_{tags or {}}"
            self.latency_data[key].append({
                'latency': latency,
                'timestamp': datetime.now(),
                'tags': tags or {}
            })
            
            # 保持最近1000条记录
            if len(self.latency_data[key]) > 1000:
                self.latency_data[key].popleft()
    
    def collect_metrics(self) -> List[MetricData]:
        """收集延迟指标"""
        metrics = []
        
        with self._lock:
            for key, data_points in self.latency_data.items():
                if not data_points:
                    continue
                
                # 提取最近的延迟数据
                recent_data = [
                    dp for dp in data_points 
                    if (datetime.now() - dp['timestamp']).total_seconds() <= 300
                ]
                
                if not recent_data:
                    continue
                
                latencies = [dp['latency'] for dp in recent_data]
                tags = recent_data[0]['tags']
                operation = key.split('_')[0]
                
                # 计算统计指标
                avg_latency = np.mean(latencies)
                p50_latency = np.percentile(latencies, 50)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)
                max_latency = np.max(latencies)
                
                metrics.extend([
                    MetricData(
                        name=f"inference.{operation}.latency.avg",
                        value=avg_latency,
                        metric_type=MetricType.GAUGE,
                        unit="ms",
                        tags=tags
                    ),
                    MetricData(
                        name=f"inference.{operation}.latency.p50",
                        value=p50_latency,
                        metric_type=MetricType.GAUGE,
                        unit="ms",
                        tags=tags
                    ),
                    MetricData(
                        name=f"inference.{operation}.latency.p95",
                        value=p95_latency,
                        metric_type=MetricType.GAUGE,
                        unit="ms",
                        tags=tags
                    ),
                    MetricData(
                        name=f"inference.{operation}.latency.p99",
                        value=p99_latency,
                        metric_type=MetricType.GAUGE,
                        unit="ms",
                        tags=tags
                    ),
                    MetricData(
                        name=f"inference.{operation}.latency.max",
                        value=max_latency,
                        metric_type=MetricType.GAUGE,
                        unit="ms",
                        tags=tags
                    )
                ])
        
        return metrics
    
    def start_monitoring(self) -> None:
        """开始监控"""
        logger.info("Inference latency monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        logger.info("Inference latency monitoring stopped")


class ThroughputMonitor(BaseMonitor):
    """
    吞吐量监控器
    """
    
    def __init__(self, config: MonitoringConfig):
        """
        初始化吞吐量监控器
        
        Args:
            config: 监控配置
        """
        super().__init__(config)
        self.request_counts = defaultdict(deque)
        self._lock = threading.Lock()
    
    def record_request(self, 
                      operation: str,
                      count: int = 1,
                      tags: Optional[Dict[str, str]] = None) -> None:
        """
        记录请求数量
        
        Args:
            operation: 操作名称
            count: 请求数量
            tags: 标签
        """
        with self._lock:
            key = f"{operation}_{tags or {}}"
            self.request_counts[key].append({
                'count': count,
                'timestamp': datetime.now(),
                'tags': tags or {}
            })
            
            # 保持最近1小时的记录
            cutoff_time = datetime.now() - timedelta(hours=1)
            while (self.request_counts[key] and 
                   self.request_counts[key][0]['timestamp'] < cutoff_time):
                self.request_counts[key].popleft()
    
    def collect_metrics(self) -> List[MetricData]:
        """收集吞吐量指标"""
        metrics = []
        
        with self._lock:
            for key, data_points in self.request_counts.items():
                if not data_points:
                    continue
                
                # 计算不同时间窗口的吞吐量
                now = datetime.now()
                windows = [
                    ('1m', timedelta(minutes=1)),
                    ('5m', timedelta(minutes=5)),
                    ('15m', timedelta(minutes=15)),
                    ('1h', timedelta(hours=1))
                ]
                
                operation = key.split('_')[0]
                tags = data_points[0]['tags']
                
                for window_name, window_duration in windows:
                    cutoff_time = now - window_duration
                    window_data = [
                        dp for dp in data_points 
                        if dp['timestamp'] >= cutoff_time
                    ]
                    
                    if window_data:
                        total_requests = sum(dp['count'] for dp in window_data)
                        rps = total_requests / window_duration.total_seconds()
                        
                        metrics.append(MetricData(
                            name=f"throughput.{operation}.rps_{window_name}",
                            value=rps,
                            metric_type=MetricType.GAUGE,
                            unit="rps",
                            tags=tags
                        ))
        
        return metrics
    
    def start_monitoring(self) -> None:
        """开始监控"""
        logger.info("Throughput monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        logger.info("Throughput monitoring stopped")


class PerformanceMonitor:
    """
    综合性能监控器
    """
    
    def __init__(self, config: MonitoringConfig):
        """
        初始化性能监控器
        
        Args:
            config: 监控配置
        """
        self.config = config
        self.monitors = {}
        
        # 初始化各种监控器
        if config.monitor_system_resources:
            self.monitors['system'] = SystemResourceMonitor(config)
        
        if config.monitor_inference_latency:
            self.monitors['latency'] = InferenceLatencyMonitor(config)
        
        if config.monitor_throughput:
            self.monitors['throughput'] = ThroughputMonitor(config)
    
    def add_model_monitor(self, model_name: str) -> ModelPerformanceTracker:
        """
        添加模型监控器
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型性能跟踪器
        """
        monitor = ModelPerformanceTracker(self.config, model_name)
        self.monitors[f'model_{model_name}'] = monitor
        return monitor
    
    def start_all(self) -> None:
        """启动所有监控器"""
        for name, monitor in self.monitors.items():
            try:
                monitor.start_monitoring()
                logger.info(f"Started {name} monitor")
            except Exception as e:
                logger.error(f"Failed to start {name} monitor: {e}")
    
    def stop_all(self) -> None:
        """停止所有监控器"""
        for name, monitor in self.monitors.items():
            try:
                monitor.stop_monitoring()
                logger.info(f"Stopped {name} monitor")
            except Exception as e:
                logger.error(f"Failed to stop {name} monitor: {e}")
    
    def get_all_metrics(self) -> List[MetricData]:
        """获取所有指标"""
        all_metrics = []
        
        for monitor in self.monitors.values():
            try:
                metrics = monitor.get_metrics()
                all_metrics.extend(metrics)
            except Exception as e:
                logger.error(f"Failed to get metrics from monitor: {e}")
        
        return all_metrics
    
    def get_monitor(self, name: str) -> Optional[BaseMonitor]:
        """
        获取指定监控器
        
        Args:
            name: 监控器名称
            
        Returns:
            监控器实例
        """
        return self.monitors.get(name)


def create_performance_monitor(config: MonitoringConfig) -> PerformanceMonitor:
    """
    创建性能监控器
    
    Args:
        config: 监控配置
        
    Returns:
        性能监控器实例
    """
    return PerformanceMonitor(config)