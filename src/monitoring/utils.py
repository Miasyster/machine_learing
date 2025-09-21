"""
监控工具函数模块

提供监控系统的辅助工具函数。
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import json
import yaml
import logging
from contextlib import contextmanager
from functools import wraps
import inspect

from .base import MetricType, MetricData, MonitoringConfig, MonitoringError
from .metrics import MetricCollector, MetricConfig


def get_system_metrics() -> Dict[str, float]:
    """获取系统指标"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 网络统计
        net_io = psutil.net_io_counters()
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available': memory.available / (1024**3),  # GB
            'memory_total': memory.total / (1024**3),  # GB
            'disk_usage': disk.percent,
            'disk_free': disk.free / (1024**3),  # GB
            'disk_total': disk.total / (1024**3),  # GB
            'network_bytes_sent': net_io.bytes_sent,
            'network_bytes_recv': net_io.bytes_recv,
            'network_packets_sent': net_io.packets_sent,
            'network_packets_recv': net_io.packets_recv
        }
    except Exception as e:
        raise MonitoringError(f"获取系统指标失败: {e}")


def get_process_metrics(pid: Optional[int] = None) -> Dict[str, float]:
    """获取进程指标"""
    try:
        if pid is None:
            process = psutil.Process()
        else:
            process = psutil.Process(pid)
        
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        return {
            'process_cpu_usage': cpu_percent,
            'process_memory_rss': memory_info.rss / (1024**2),  # MB
            'process_memory_vms': memory_info.vms / (1024**2),  # MB
            'process_num_threads': process.num_threads(),
            'process_num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
            'process_create_time': process.create_time()
        }
    except Exception as e:
        raise MonitoringError(f"获取进程指标失败: {e}")


def monitor_function_performance(metric_collector: MetricCollector,
                               metric_prefix: str = "function"):
    """函数性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                
                # 记录成功执行
                metric_collector.record_metric(
                    f"{metric_prefix}.{function_name}.success",
                    1.0,
                    tags={'function': function_name, 'status': 'success'}
                )
                
                return result
                
            except Exception as e:
                # 记录异常
                metric_collector.record_metric(
                    f"{metric_prefix}.{function_name}.error",
                    1.0,
                    tags={'function': function_name, 'status': 'error', 'error_type': type(e).__name__}
                )
                raise
                
            finally:
                # 记录执行时间
                execution_time = (time.time() - start_time) * 1000  # 毫秒
                metric_collector.record_metric(
                    f"{metric_prefix}.{function_name}.duration",
                    execution_time,
                    tags={'function': function_name}
                )
        
        return wrapper
    return decorator


@contextmanager
def monitor_operation(metric_collector: MetricCollector, 
                     operation_name: str,
                     tags: Optional[Dict[str, str]] = None):
    """操作监控上下文管理器"""
    start_time = time.time()
    operation_tags = tags or {}
    operation_tags['operation'] = operation_name
    
    try:
        yield
        
        # 记录成功
        metric_collector.record_metric(
            f"operation.{operation_name}.success",
            1.0,
            tags={**operation_tags, 'status': 'success'}
        )
        
    except Exception as e:
        # 记录失败
        metric_collector.record_metric(
            f"operation.{operation_name}.error",
            1.0,
            tags={**operation_tags, 'status': 'error', 'error_type': type(e).__name__}
        )
        raise
        
    finally:
        # 记录执行时间
        duration = (time.time() - start_time) * 1000  # 毫秒
        metric_collector.record_metric(
            f"operation.{operation_name}.duration",
            duration,
            tags=operation_tags
        )


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, metric_collector: MetricCollector, 
                 interval: int = 60):
        self.metric_collector = metric_collector
        self.interval = interval
        self.is_running = False
        self.monitor_thread = None
        
        # 注册系统指标
        self._register_system_metrics()
    
    def _register_system_metrics(self):
        """注册系统指标"""
        system_metrics = [
            MetricConfig(
                name="system.cpu.usage",
                metric_type=MetricType.GAUGE,
                description="CPU使用率",
                unit="percentage"
            ),
            MetricConfig(
                name="system.memory.usage",
                metric_type=MetricType.GAUGE,
                description="内存使用率",
                unit="percentage"
            ),
            MetricConfig(
                name="system.disk.usage",
                metric_type=MetricType.GAUGE,
                description="磁盘使用率",
                unit="percentage"
            ),
            MetricConfig(
                name="system.network.bytes_sent",
                metric_type=MetricType.COUNTER,
                description="网络发送字节数",
                unit="bytes"
            ),
            MetricConfig(
                name="system.network.bytes_recv",
                metric_type=MetricType.COUNTER,
                description="网络接收字节数",
                unit="bytes"
            )
        ]
        
        for config in system_metrics:
            self.metric_collector.register_metric(config)
    
    def start(self):
        """启动系统监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """停止系统监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 获取系统指标
                system_metrics = get_system_metrics()
                
                # 记录指标
                for metric_name, value in system_metrics.items():
                    full_metric_name = f"system.{metric_name}"
                    if full_metric_name in self.metric_collector.configs:
                        self.metric_collector.record_metric(full_metric_name, value)
                
                # 获取进程指标
                process_metrics = get_process_metrics()
                
                # 记录进程指标
                for metric_name, value in process_metrics.items():
                    self.metric_collector.record_metric(
                        f"process.{metric_name}",
                        value
                    )
                
            except Exception as e:
                logging.error(f"系统监控出错: {e}")
            
            time.sleep(self.interval)


def create_monitoring_config(config_path: Path) -> MonitoringConfig:
    """从文件创建监控配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        return MonitoringConfig(**config_data)
        
    except Exception as e:
        raise MonitoringError(f"加载监控配置失败: {e}")


def save_monitoring_config(config: MonitoringConfig, config_path: Path):
    """保存监控配置到文件"""
    try:
        config_data = {
            'enabled': config.enabled,
            'collection_interval': config.collection_interval,
            'retention_days': config.retention_days,
            'alert_enabled': config.alert_enabled,
            'dashboard_enabled': config.dashboard_enabled,
            'export_enabled': config.export_enabled
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
    except Exception as e:
        raise MonitoringError(f"保存监控配置失败: {e}")


def validate_metric_config(config: MetricConfig) -> List[str]:
    """验证指标配置"""
    errors = []
    
    if not config.name:
        errors.append("指标名称不能为空")
    
    if not config.name.replace('.', '').replace('_', '').isalnum():
        errors.append("指标名称只能包含字母、数字、点和下划线")
    
    if config.retention_days <= 0:
        errors.append("保留天数必须大于0")
    
    if config.aggregation_interval <= 0:
        errors.append("聚合间隔必须大于0")
    
    return errors


def calculate_metric_statistics(values: List[float]) -> Dict[str, float]:
    """计算指标统计信息"""
    if not values:
        return {}
    
    import statistics
    
    stats = {
        'count': len(values),
        'sum': sum(values),
        'mean': statistics.mean(values),
        'min': min(values),
        'max': max(values)
    }
    
    if len(values) >= 2:
        stats['median'] = statistics.median(values)
        stats['stdev'] = statistics.stdev(values)
        
        # 计算百分位数
        sorted_values = sorted(values)
        stats['p25'] = sorted_values[int(0.25 * len(sorted_values))]
        stats['p75'] = sorted_values[int(0.75 * len(sorted_values))]
        stats['p95'] = sorted_values[int(0.95 * len(sorted_values))]
        stats['p99'] = sorted_values[int(0.99 * len(sorted_values))]
    
    return stats


def format_metric_value(value: float, unit: str = "") -> str:
    """格式化指标值"""
    if unit == "bytes":
        # 格式化字节数
        for unit_name in ['B', 'KB', 'MB', 'GB', 'TB']:
            if value < 1024.0:
                return f"{value:.2f} {unit_name}"
            value /= 1024.0
        return f"{value:.2f} PB"
    
    elif unit == "percentage":
        return f"{value:.2f}%"
    
    elif unit == "ms":
        if value >= 1000:
            return f"{value/1000:.2f}s"
        return f"{value:.2f}ms"
    
    elif unit == "count":
        if value >= 1000000:
            return f"{value/1000000:.2f}M"
        elif value >= 1000:
            return f"{value/1000:.2f}K"
        return f"{int(value)}"
    
    else:
        return f"{value:.2f} {unit}".strip()


def export_metrics_to_csv(metric_collector: MetricCollector,
                         metric_names: List[str],
                         start_time: datetime,
                         end_time: datetime,
                         output_path: Path):
    """导出指标到CSV文件"""
    import csv
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入标题行
            headers = ['timestamp', 'metric_name', 'value', 'tags']
            writer.writerow(headers)
            
            # 写入数据
            for metric_name in metric_names:
                metrics = metric_collector.get_metrics(metric_name, start_time, end_time)
                
                for metric in metrics:
                    writer.writerow([
                        metric.timestamp.isoformat(),
                        metric.name,
                        metric.value,
                        json.dumps(metric.tags)
                    ])
                    
    except Exception as e:
        raise MonitoringError(f"导出指标到CSV失败: {e}")


def export_metrics_to_json(metric_collector: MetricCollector,
                          metric_names: List[str],
                          start_time: datetime,
                          end_time: datetime,
                          output_path: Path):
    """导出指标到JSON文件"""
    try:
        all_metrics = []
        
        for metric_name in metric_names:
            metrics = metric_collector.get_metrics(metric_name, start_time, end_time)
            
            for metric in metrics:
                all_metrics.append({
                    'timestamp': metric.timestamp.isoformat(),
                    'name': metric.name,
                    'value': metric.value,
                    'tags': metric.tags,
                    'type': metric.metric_type.value
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        raise MonitoringError(f"导出指标到JSON失败: {e}")


def create_health_check(metric_collector: MetricCollector,
                       checks: Dict[str, Callable[[], bool]]) -> Callable[[], Dict[str, Any]]:
    """创建健康检查函数"""
    def health_check() -> Dict[str, Any]:
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'checks': {}
        }
        
        overall_healthy = True
        
        for check_name, check_func in checks.items():
            try:
                is_healthy = check_func()
                results['checks'][check_name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'timestamp': datetime.now().isoformat()
                }
                
                if not is_healthy:
                    overall_healthy = False
                
                # 记录健康检查指标
                metric_collector.record_metric(
                    f"health.{check_name}",
                    1.0 if is_healthy else 0.0,
                    tags={'check': check_name}
                )
                
            except Exception as e:
                results['checks'][check_name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                overall_healthy = False
                
                # 记录健康检查错误
                metric_collector.record_metric(
                    f"health.{check_name}",
                    0.0,
                    tags={'check': check_name, 'error': 'true'}
                )
        
        results['status'] = 'healthy' if overall_healthy else 'unhealthy'
        return results
    
    return health_check


def setup_monitoring_logging(log_level: str = "INFO") -> logging.Logger:
    """设置监控日志"""
    logger = logging.getLogger('monitoring')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger