"""
监控系统基础类

定义监控配置、指标数据、告警规则等基础类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MonitoringError(Exception):
    """监控系统基础异常"""
    pass


class AlertError(MonitoringError):
    """告警系统异常"""
    pass


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """告警严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MonitoringConfig:
    """
    监控配置
    """
    # 基本配置
    enabled: bool = True
    collection_interval: int = 60  # 秒
    retention_days: int = 30
    storage_path: str = "./monitoring_data"
    
    # 性能监控配置
    monitor_model_performance: bool = True
    monitor_system_resources: bool = True
    monitor_inference_latency: bool = True
    monitor_throughput: bool = True
    
    # 告警配置
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["email"])
    alert_cooldown: int = 300  # 秒
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "json"
    log_rotation: str = "daily"
    max_log_files: int = 30
    
    # 仪表板配置
    enable_dashboard: bool = True
    dashboard_port: int = 8080
    dashboard_host: str = "localhost"
    
    # 自定义配置
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricData:
    """
    指标数据
    """
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'unit': self.unit,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricData':
        """从字典创建"""
        return cls(
            name=data['name'],
            value=data['value'],
            metric_type=MetricType(data['type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            tags=data.get('tags', {}),
            unit=data.get('unit'),
            description=data.get('description')
        )


@dataclass
class AlertRule:
    """
    告警规则
    """
    name: str
    metric_name: str
    condition: str  # 条件表达式，如 "> 0.8", "< 100", "== 0"
    threshold: Union[int, float]
    severity: AlertSeverity
    enabled: bool = True
    cooldown: int = 300  # 秒
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    # 高级配置
    evaluation_window: int = 300  # 评估窗口（秒）
    min_samples: int = 1  # 最小样本数
    aggregation: str = "avg"  # 聚合方式：avg, max, min, sum
    
    def evaluate(self, values: List[float]) -> bool:
        """
        评估告警条件
        
        Args:
            values: 指标值列表
            
        Returns:
            是否触发告警
        """
        if not values or len(values) < self.min_samples:
            return False
        
        # 聚合值
        if self.aggregation == "avg":
            aggregated_value = sum(values) / len(values)
        elif self.aggregation == "max":
            aggregated_value = max(values)
        elif self.aggregation == "min":
            aggregated_value = min(values)
        elif self.aggregation == "sum":
            aggregated_value = sum(values)
        else:
            aggregated_value = values[-1]  # 最新值
        
        # 评估条件
        if self.condition.startswith(">"):
            return aggregated_value > self.threshold
        elif self.condition.startswith("<"):
            return aggregated_value < self.threshold
        elif self.condition.startswith(">="):
            return aggregated_value >= self.threshold
        elif self.condition.startswith("<="):
            return aggregated_value <= self.threshold
        elif self.condition.startswith("=="):
            return aggregated_value == self.threshold
        elif self.condition.startswith("!="):
            return aggregated_value != self.threshold
        else:
            logger.warning(f"Unknown condition: {self.condition}")
            return False


@dataclass
class AlertEvent:
    """
    告警事件
    """
    rule_name: str
    metric_name: str
    current_value: Union[int, float]
    threshold: Union[int, float]
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'rule_name': self.rule_name,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'tags': self.tags
        }
    
    def resolve(self) -> None:
        """解决告警"""
        self.resolved = True
        self.resolved_at = datetime.now()


class BaseMonitor(ABC):
    """
    基础监控器抽象类
    """
    
    def __init__(self, config: MonitoringConfig):
        """
        初始化监控器
        
        Args:
            config: 监控配置
        """
        self.config = config
        self.enabled = config.enabled
        self.metrics_buffer: List[MetricData] = []
        self.last_collection_time = datetime.now()
        
    @abstractmethod
    def collect_metrics(self) -> List[MetricData]:
        """
        收集指标数据
        
        Returns:
            指标数据列表
        """
        pass
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """开始监控"""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> None:
        """停止监控"""
        pass
    
    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self.enabled
    
    def enable(self) -> None:
        """启用监控"""
        self.enabled = True
        logger.info(f"{self.__class__.__name__} enabled")
    
    def disable(self) -> None:
        """禁用监控"""
        self.enabled = False
        logger.info(f"{self.__class__.__name__} disabled")
    
    def add_metric(self, metric: MetricData) -> None:
        """
        添加指标数据
        
        Args:
            metric: 指标数据
        """
        if self.enabled:
            self.metrics_buffer.append(metric)
    
    def get_metrics(self, clear_buffer: bool = True) -> List[MetricData]:
        """
        获取指标数据
        
        Args:
            clear_buffer: 是否清空缓冲区
            
        Returns:
            指标数据列表
        """
        metrics = self.metrics_buffer.copy()
        if clear_buffer:
            self.metrics_buffer.clear()
        return metrics
    
    def should_collect(self) -> bool:
        """
        检查是否应该收集指标
        
        Returns:
            是否应该收集
        """
        if not self.enabled:
            return False
        
        now = datetime.now()
        elapsed = (now - self.last_collection_time).total_seconds()
        return elapsed >= self.config.collection_interval
    
    def update_collection_time(self) -> None:
        """更新收集时间"""
        self.last_collection_time = datetime.now()


class BaseAlertHandler(ABC):
    """
    基础告警处理器抽象类
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化告警处理器
        
        Args:
            config: 处理器配置
        """
        self.config = config
        self.enabled = config.get('enabled', True)
    
    @abstractmethod
    def send_alert(self, alert: AlertEvent) -> bool:
        """
        发送告警
        
        Args:
            alert: 告警事件
            
        Returns:
            是否发送成功
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        测试连接
        
        Returns:
            连接是否正常
        """
        pass
    
    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self.enabled
    
    def enable(self) -> None:
        """启用处理器"""
        self.enabled = True
    
    def disable(self) -> None:
        """禁用处理器"""
        self.enabled = False


class MetricFilter:
    """
    指标过滤器
    """
    
    def __init__(self, 
                 metric_names: Optional[List[str]] = None,
                 tags: Optional[Dict[str, str]] = None,
                 time_range: Optional[Tuple[datetime, datetime]] = None):
        """
        初始化过滤器
        
        Args:
            metric_names: 指标名称列表
            tags: 标签过滤条件
            time_range: 时间范围
        """
        self.metric_names = metric_names
        self.tags = tags or {}
        self.time_range = time_range
    
    def matches(self, metric: MetricData) -> bool:
        """
        检查指标是否匹配过滤条件
        
        Args:
            metric: 指标数据
            
        Returns:
            是否匹配
        """
        # 检查指标名称
        if self.metric_names and metric.name not in self.metric_names:
            return False
        
        # 检查标签
        for key, value in self.tags.items():
            if key not in metric.tags or metric.tags[key] != value:
                return False
        
        # 检查时间范围
        if self.time_range:
            start_time, end_time = self.time_range
            if not (start_time <= metric.timestamp <= end_time):
                return False
        
        return True


class MetricAggregator:
    """
    指标聚合器
    """
    
    @staticmethod
    def aggregate_metrics(metrics: List[MetricData], 
                         aggregation: str = "avg",
                         group_by: Optional[List[str]] = None) -> List[MetricData]:
        """
        聚合指标数据
        
        Args:
            metrics: 指标数据列表
            aggregation: 聚合方式
            group_by: 分组字段
            
        Returns:
            聚合后的指标数据
        """
        if not metrics:
            return []
        
        # 按分组字段分组
        groups = {}
        for metric in metrics:
            if group_by:
                group_key = tuple(metric.tags.get(field, '') for field in group_by)
            else:
                group_key = metric.name
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(metric)
        
        # 聚合每个分组
        aggregated_metrics = []
        for group_key, group_metrics in groups.items():
            if not group_metrics:
                continue
            
            values = [m.value for m in group_metrics]
            
            if aggregation == "avg":
                aggregated_value = sum(values) / len(values)
            elif aggregation == "sum":
                aggregated_value = sum(values)
            elif aggregation == "max":
                aggregated_value = max(values)
            elif aggregation == "min":
                aggregated_value = min(values)
            elif aggregation == "count":
                aggregated_value = len(values)
            else:
                aggregated_value = values[-1]  # 最新值
            
            # 创建聚合指标
            base_metric = group_metrics[0]
            aggregated_metric = MetricData(
                name=f"{base_metric.name}_{aggregation}",
                value=aggregated_value,
                metric_type=base_metric.metric_type,
                timestamp=max(m.timestamp for m in group_metrics),
                tags=base_metric.tags.copy(),
                unit=base_metric.unit,
                description=f"{aggregation.upper()} of {base_metric.name}"
            )
            
            aggregated_metrics.append(aggregated_metric)
        
        return aggregated_metrics