"""
指标收集模块

提供指标收集、存储、聚合和查询功能。
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import sqlite3
from pathlib import Path
import statistics

from .base import MetricType, MetricData, MonitoringError


class AggregationType(Enum):
    """聚合类型枚举"""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


@dataclass
class MetricConfig:
    """指标配置"""
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    retention_days: int = 30
    aggregation_interval: int = 60  # 秒
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


class MetricCollector:
    """指标收集器"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("metrics.db")
        self.metrics_buffer = defaultdict(deque)
        self.configs = {}
        self.lock = threading.RLock()
        
        # 初始化数据库
        self._init_database()
        
        # 启动后台聚合任务
        self.aggregation_thread = None
        self.is_running = False
        self.start_aggregation()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    tags TEXT,
                    metric_type TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aggregated_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    aggregation_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    tags TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_aggregated_name_time 
                ON aggregated_metrics(name, start_time, end_time)
            """)
    
    def register_metric(self, config: MetricConfig):
        """注册指标"""
        with self.lock:
            self.configs[config.name] = config
    
    def record_metric(self, name: str, value: float, 
                     tags: Optional[Dict[str, str]] = None,
                     timestamp: Optional[datetime] = None):
        """记录指标"""
        if name not in self.configs:
            raise MonitoringError(f"指标 {name} 未注册")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_data = MetricData(
            name=name,
            value=value,
            timestamp=timestamp,
            tags=tags or {},
            metric_type=self.configs[name].metric_type
        )
        
        with self.lock:
            # 添加到缓冲区
            self.metrics_buffer[name].append(metric_data)
            
            # 限制缓冲区大小
            if len(self.metrics_buffer[name]) > 1000:
                self.metrics_buffer[name].popleft()
        
        # 立即存储到数据库
        self._store_metric(metric_data)
    
    def _store_metric(self, metric: MetricData):
        """存储指标到数据库"""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT INTO metrics (name, value, timestamp, tags, metric_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.value,
                    metric.timestamp.timestamp(),
                    json.dumps(metric.tags),
                    metric.metric_type.value
                ))
        except Exception as e:
            print(f"存储指标时出错: {e}")
    
    def get_metrics(self, name: str, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   tags: Optional[Dict[str, str]] = None) -> List[MetricData]:
        """获取指标数据"""
        query = "SELECT name, value, timestamp, tags, metric_type FROM metrics WHERE name = ?"
        params = [name]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.timestamp())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.timestamp())
        
        query += " ORDER BY timestamp"
        
        metrics = []
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    metric_tags = json.loads(row[3]) if row[3] else {}
                    
                    # 过滤标签
                    if tags and not all(metric_tags.get(k) == v for k, v in tags.items()):
                        continue
                    
                    metrics.append(MetricData(
                        name=row[0],
                        value=row[1],
                        timestamp=datetime.fromtimestamp(row[2]),
                        tags=metric_tags,
                        metric_type=MetricType(row[4])
                    ))
        except Exception as e:
            print(f"获取指标时出错: {e}")
        
        return metrics
    
    def start_aggregation(self):
        """启动聚合任务"""
        if self.is_running:
            return
        
        self.is_running = True
        self.aggregation_thread = threading.Thread(target=self._aggregation_worker)
        self.aggregation_thread.daemon = True
        self.aggregation_thread.start()
    
    def stop_aggregation(self):
        """停止聚合任务"""
        self.is_running = False
        if self.aggregation_thread:
            self.aggregation_thread.join()
    
    def _aggregation_worker(self):
        """聚合工作线程"""
        while self.is_running:
            try:
                self._perform_aggregation()
                time.sleep(60)  # 每分钟聚合一次
            except Exception as e:
                print(f"聚合任务出错: {e}")
    
    def _perform_aggregation(self):
        """执行聚合"""
        current_time = datetime.now()
        
        for name, config in self.configs.items():
            # 计算聚合时间窗口
            window_start = current_time - timedelta(seconds=config.aggregation_interval)
            
            # 获取时间窗口内的数据
            metrics = self.get_metrics(name, window_start, current_time)
            
            if not metrics:
                continue
            
            # 执行各种聚合
            values = [m.value for m in metrics]
            aggregations = self._calculate_aggregations(values)
            
            # 存储聚合结果
            for agg_type, agg_value in aggregations.items():
                self._store_aggregated_metric(
                    name, agg_type, agg_value, window_start, current_time
                )
    
    def _calculate_aggregations(self, values: List[float]) -> Dict[str, float]:
        """计算聚合值"""
        if not values:
            return {}
        
        aggregations = {
            AggregationType.SUM.value: sum(values),
            AggregationType.AVERAGE.value: statistics.mean(values),
            AggregationType.MIN.value: min(values),
            AggregationType.MAX.value: max(values),
            AggregationType.COUNT.value: len(values)
        }
        
        # 计算百分位数
        if len(values) >= 2:
            sorted_values = sorted(values)
            aggregations[AggregationType.PERCENTILE_50.value] = statistics.median(sorted_values)
            
            if len(values) >= 20:  # 足够的数据点才计算高百分位数
                p95_idx = int(0.95 * len(sorted_values))
                p99_idx = int(0.99 * len(sorted_values))
                
                aggregations[AggregationType.PERCENTILE_95.value] = sorted_values[p95_idx]
                aggregations[AggregationType.PERCENTILE_99.value] = sorted_values[p99_idx]
        
        return aggregations
    
    def _store_aggregated_metric(self, name: str, agg_type: str, value: float,
                                start_time: datetime, end_time: datetime):
        """存储聚合指标"""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT INTO aggregated_metrics 
                    (name, aggregation_type, value, start_time, end_time, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    name,
                    agg_type,
                    value,
                    start_time.timestamp(),
                    end_time.timestamp(),
                    json.dumps({})
                ))
        except Exception as e:
            print(f"存储聚合指标时出错: {e}")
    
    def get_aggregated_metrics(self, name: str, aggregation_type: str,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """获取聚合指标"""
        query = """
            SELECT aggregation_type, value, start_time, end_time 
            FROM aggregated_metrics 
            WHERE name = ? AND aggregation_type = ?
        """
        params = [name, aggregation_type]
        
        if start_time:
            query += " AND end_time >= ?"
            params.append(start_time.timestamp())
        
        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time.timestamp())
        
        query += " ORDER BY start_time"
        
        results = []
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    results.append({
                        'aggregation_type': row[0],
                        'value': row[1],
                        'start_time': datetime.fromtimestamp(row[2]),
                        'end_time': datetime.fromtimestamp(row[3])
                    })
        except Exception as e:
            print(f"获取聚合指标时出错: {e}")
        
        return results
    
    def cleanup_old_metrics(self):
        """清理过期指标"""
        current_time = datetime.now()
        
        for name, config in self.configs.items():
            cutoff_time = current_time - timedelta(days=config.retention_days)
            
            try:
                with sqlite3.connect(self.storage_path) as conn:
                    # 清理原始指标
                    conn.execute("""
                        DELETE FROM metrics 
                        WHERE name = ? AND timestamp < ?
                    """, (name, cutoff_time.timestamp()))
                    
                    # 清理聚合指标
                    conn.execute("""
                        DELETE FROM aggregated_metrics 
                        WHERE name = ? AND end_time < ?
                    """, (name, cutoff_time.timestamp()))
                    
            except Exception as e:
                print(f"清理指标 {name} 时出错: {e}")


class MetricQuery:
    """指标查询器"""
    
    def __init__(self, collector: MetricCollector):
        self.collector = collector
    
    def query(self, name: str, 
             start_time: Optional[datetime] = None,
             end_time: Optional[datetime] = None,
             aggregation: Optional[str] = None,
             tags: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """查询指标"""
        if aggregation:
            return self.collector.get_aggregated_metrics(
                name, aggregation, start_time, end_time
            )
        else:
            metrics = self.collector.get_metrics(name, start_time, end_time, tags)
            return [
                {
                    'value': m.value,
                    'timestamp': m.timestamp,
                    'tags': m.tags
                }
                for m in metrics
            ]
    
    def get_latest_value(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """获取最新值"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)  # 最近5分钟
        
        metrics = self.collector.get_metrics(name, start_time, end_time, tags)
        
        if metrics:
            return metrics[-1].value
        return None
    
    def calculate_rate(self, name: str, window_minutes: int = 5) -> Optional[float]:
        """计算变化率"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=window_minutes)
        
        metrics = self.collector.get_metrics(name, start_time, end_time)
        
        if len(metrics) < 2:
            return None
        
        first_metric = metrics[0]
        last_metric = metrics[-1]
        
        time_diff = (last_metric.timestamp - first_metric.timestamp).total_seconds()
        value_diff = last_metric.value - first_metric.value
        
        if time_diff > 0:
            return value_diff / time_diff
        return None
    
    def get_statistics(self, name: str, window_hours: int = 1) -> Dict[str, float]:
        """获取统计信息"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=window_hours)
        
        metrics = self.collector.get_metrics(name, start_time, end_time)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
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
        
        return stats


def create_metric_collector(storage_path: Optional[Path] = None) -> MetricCollector:
    """创建指标收集器"""
    return MetricCollector(storage_path)


def create_standard_metrics() -> List[MetricConfig]:
    """创建标准指标配置"""
    return [
        MetricConfig(
            name="model.prediction.latency",
            metric_type=MetricType.HISTOGRAM,
            description="模型预测延迟",
            unit="ms",
            alert_thresholds={"p95": 1000.0}
        ),
        MetricConfig(
            name="model.prediction.count",
            metric_type=MetricType.COUNTER,
            description="预测请求数量",
            unit="count"
        ),
        MetricConfig(
            name="model.accuracy",
            metric_type=MetricType.GAUGE,
            description="模型准确率",
            unit="percentage",
            alert_thresholds={"min": 0.8}
        ),
        MetricConfig(
            name="system.cpu.usage",
            metric_type=MetricType.GAUGE,
            description="CPU使用率",
            unit="percentage",
            alert_thresholds={"max": 80.0}
        ),
        MetricConfig(
            name="system.memory.usage",
            metric_type=MetricType.GAUGE,
            description="内存使用率",
            unit="percentage",
            alert_thresholds={"max": 85.0}
        ),
        MetricConfig(
            name="model.error.count",
            metric_type=MetricType.COUNTER,
            description="模型错误数量",
            unit="count",
            alert_thresholds={"rate": 0.05}
        )
    ]