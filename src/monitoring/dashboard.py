"""
监控仪表板模块

提供Web仪表板、实时监控、图表生成和报告功能。
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from flask import Flask, render_template_string, jsonify, request
import plotly.graph_objs as go
import plotly.utils
from plotly.subplots import make_subplots

from .base import MetricType, MonitoringError
from .metrics import MetricCollector, MetricQuery
from .performance import PerformanceMonitor
from .alerts import AlertManager


@dataclass
class DashboardConfig:
    """仪表板配置"""
    title: str = "ML监控仪表板"
    refresh_interval: int = 30  # 秒
    port: int = 8080
    host: str = "localhost"
    theme: str = "light"
    auto_refresh: bool = True


class ChartGenerator:
    """图表生成器"""
    
    def __init__(self, metric_query: MetricQuery):
        self.metric_query = metric_query
    
    def create_time_series_chart(self, metric_name: str, 
                                hours: int = 1,
                                aggregation: Optional[str] = None) -> Dict[str, Any]:
        """创建时间序列图表"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        data = self.metric_query.query(
            metric_name, start_time, end_time, aggregation
        )
        
        if not data:
            return self._empty_chart(f"No data for {metric_name}")
        
        timestamps = []
        values = []
        
        for point in data:
            if aggregation:
                timestamps.append(point['end_time'])
            else:
                timestamps.append(point['timestamp'])
            values.append(point['value'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name=metric_name,
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title=f"{metric_name} - Last {hours}h",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified'
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    def create_histogram_chart(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """创建直方图"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        data = self.metric_query.query(metric_name, start_time, end_time)
        
        if not data:
            return self._empty_chart(f"No data for {metric_name}")
        
        values = [point['value'] for point in data]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=30,
            name=metric_name
        ))
        
        fig.update_layout(
            title=f"{metric_name} Distribution - Last {hours}h",
            xaxis_title="Value",
            yaxis_title="Frequency"
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    def create_gauge_chart(self, metric_name: str, 
                          min_val: float = 0, max_val: float = 100,
                          thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """创建仪表盘图表"""
        current_value = self.metric_query.get_latest_value(metric_name)
        
        if current_value is None:
            return self._empty_chart(f"No current data for {metric_name}")
        
        # 设置颜色阈值
        if thresholds is None:
            thresholds = {'yellow': max_val * 0.7, 'red': max_val * 0.9}
        
        color = "green"
        if current_value >= thresholds.get('red', max_val):
            color = "red"
        elif current_value >= thresholds.get('yellow', max_val * 0.7):
            color = "yellow"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': metric_name},
            delta={'reference': thresholds.get('yellow', max_val * 0.7)},
            gauge={
                'axis': {'range': [None, max_val]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, thresholds.get('yellow', max_val * 0.7)], 'color': "lightgray"},
                    {'range': [thresholds.get('yellow', max_val * 0.7), thresholds.get('red', max_val * 0.9)], 'color': "yellow"},
                    {'range': [thresholds.get('red', max_val * 0.9), max_val], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': thresholds.get('red', max_val * 0.9)
                }
            }
        ))
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    def create_comparison_chart(self, metric_names: List[str], hours: int = 1) -> Dict[str, Any]:
        """创建对比图表"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        fig = go.Figure()
        
        for metric_name in metric_names:
            data = self.metric_query.query(metric_name, start_time, end_time)
            
            if data:
                timestamps = [point['timestamp'] for point in data]
                values = [point['value'] for point in data]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines',
                    name=metric_name
                ))
        
        fig.update_layout(
            title=f"Metrics Comparison - Last {hours}h",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified'
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    def create_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """创建性能摘要图表"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Latency', 'Throughput', 'Error Rate', 'Resource Usage'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 预测延迟
        latency_data = self.metric_query.query(
            "model.prediction.latency", start_time, end_time, "average"
        )
        if latency_data:
            timestamps = [point['end_time'] for point in latency_data]
            values = [point['value'] for point in latency_data]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name="Latency (ms)"),
                row=1, col=1
            )
        
        # 吞吐量
        throughput_data = self.metric_query.query(
            "model.prediction.count", start_time, end_time, "sum"
        )
        if throughput_data:
            timestamps = [point['end_time'] for point in throughput_data]
            values = [point['value'] for point in throughput_data]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name="Requests/min"),
                row=1, col=2
            )
        
        # 错误率
        error_data = self.metric_query.query(
            "model.error.count", start_time, end_time, "sum"
        )
        if error_data:
            timestamps = [point['end_time'] for point in error_data]
            values = [point['value'] for point in error_data]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name="Errors/min"),
                row=2, col=1
            )
        
        # 资源使用
        cpu_data = self.metric_query.query(
            "system.cpu.usage", start_time, end_time, "average"
        )
        if cpu_data:
            timestamps = [point['end_time'] for point in cpu_data]
            values = [point['value'] for point in cpu_data]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name="CPU %"),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Performance Summary - Last {hours}h",
            height=600
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    def _empty_chart(self, message: str) -> Dict[str, Any]:
        """创建空图表"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))


class DashboardServer:
    """仪表板服务器"""
    
    def __init__(self, config: DashboardConfig,
                 metric_collector: MetricCollector,
                 performance_monitor: Optional[PerformanceMonitor] = None,
                 alert_manager: Optional[AlertManager] = None):
        self.config = config
        self.metric_collector = metric_collector
        self.performance_monitor = performance_monitor
        self.alert_manager = alert_manager
        
        self.metric_query = MetricQuery(metric_collector)
        self.chart_generator = ChartGenerator(self.metric_query)
        
        self.app = Flask(__name__)
        self._setup_routes()
        
        self.server_thread = None
        self.is_running = False
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self._get_dashboard_template())
        
        @self.app.route('/api/metrics/<metric_name>')
        def get_metric_data(metric_name):
            hours = request.args.get('hours', 1, type=int)
            aggregation = request.args.get('aggregation')
            
            chart_data = self.chart_generator.create_time_series_chart(
                metric_name, hours, aggregation
            )
            
            return jsonify(chart_data)
        
        @self.app.route('/api/metrics/<metric_name>/histogram')
        def get_metric_histogram(metric_name):
            hours = request.args.get('hours', 1, type=int)
            chart_data = self.chart_generator.create_histogram_chart(metric_name, hours)
            return jsonify(chart_data)
        
        @self.app.route('/api/metrics/<metric_name>/gauge')
        def get_metric_gauge(metric_name):
            min_val = request.args.get('min', 0, type=float)
            max_val = request.args.get('max', 100, type=float)
            
            chart_data = self.chart_generator.create_gauge_chart(
                metric_name, min_val, max_val
            )
            return jsonify(chart_data)
        
        @self.app.route('/api/performance/summary')
        def get_performance_summary():
            hours = request.args.get('hours', 24, type=int)
            chart_data = self.chart_generator.create_performance_summary(hours)
            return jsonify(chart_data)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            if self.alert_manager:
                # 获取最近的告警
                alerts = []  # 这里应该从alert_manager获取告警数据
                return jsonify(alerts)
            return jsonify([])
        
        @self.app.route('/api/status')
        def get_system_status():
            status = {
                'timestamp': datetime.now().isoformat(),
                'metrics_count': len(self.metric_collector.configs),
                'uptime': time.time() - getattr(self, 'start_time', time.time())
            }
            
            if self.performance_monitor:
                # 添加性能状态
                pass
            
            return jsonify(status)
    
    def _get_dashboard_template(self) -> str:
        """获取仪表板HTML模板"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>{{ config.title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            margin: -20px -20px 20px -20px;
            text-align: center;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-bar {
            background: #34495e;
            color: white;
            padding: 10px;
            margin: -20px -20px 20px -20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .alert {
            background: #e74c3c;
            color: white;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>ML监控仪表板</h1>
    </div>
    
    <div class="status-bar">
        <div>状态: <span id="system-status">正常</span></div>
        <div>最后更新: <span id="last-update">--</span></div>
        <div>自动刷新: <span id="auto-refresh">开启</span></div>
    </div>
    
    <div id="alerts-container"></div>
    
    <div class="grid">
        <div class="chart-container">
            <div id="performance-summary"></div>
        </div>
        
        <div class="chart-container">
            <div id="latency-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="throughput-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="cpu-gauge"></div>
        </div>
        
        <div class="chart-container">
            <div id="memory-gauge"></div>
        </div>
        
        <div class="chart-container">
            <div id="error-chart"></div>
        </div>
    </div>

    <script>
        // 更新图表
        function updateCharts() {
            // 性能摘要
            fetch('/api/performance/summary?hours=24')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('performance-summary', data.data, data.layout);
                });
            
            // 延迟图表
            fetch('/api/metrics/model.prediction.latency?hours=1&aggregation=average')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('latency-chart', data.data, data.layout);
                });
            
            // 吞吐量图表
            fetch('/api/metrics/model.prediction.count?hours=1&aggregation=sum')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('throughput-chart', data.data, data.layout);
                });
            
            // CPU仪表盘
            fetch('/api/metrics/system.cpu.usage/gauge?max=100')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('cpu-gauge', data.data, data.layout);
                });
            
            // 内存仪表盘
            fetch('/api/metrics/system.memory.usage/gauge?max=100')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('memory-gauge', data.data, data.layout);
                });
            
            // 错误图表
            fetch('/api/metrics/model.error.count?hours=1&aggregation=sum')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('error-chart', data.data, data.layout);
                });
            
            // 更新状态
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('last-update').textContent = 
                        new Date(data.timestamp).toLocaleTimeString();
                });
            
            // 更新告警
            fetch('/api/alerts')
                .then(response => response.json())
                .then(alerts => {
                    const container = document.getElementById('alerts-container');
                    container.innerHTML = '';
                    
                    alerts.forEach(alert => {
                        const div = document.createElement('div');
                        div.className = 'alert';
                        div.textContent = `${alert.severity}: ${alert.message}`;
                        container.appendChild(div);
                    });
                });
        }
        
        // 初始加载
        updateCharts();
        
        // 定期刷新
        setInterval(updateCharts, 30000); // 30秒刷新一次
    </script>
</body>
</html>
        """
    
    def start(self):
        """启动仪表板服务器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        
        print(f"仪表板启动在 http://{self.config.host}:{self.config.port}")
    
    def stop(self):
        """停止仪表板服务器"""
        self.is_running = False
        # Flask服务器停止需要特殊处理
    
    def _run_server(self):
        """运行服务器"""
        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=False,
            use_reloader=False
        )


def create_dashboard(metric_collector: MetricCollector,
                    config: Optional[DashboardConfig] = None,
                    performance_monitor: Optional[PerformanceMonitor] = None,
                    alert_manager: Optional[AlertManager] = None) -> DashboardServer:
    """创建仪表板"""
    if config is None:
        config = DashboardConfig()
    
    return DashboardServer(
        config, metric_collector, performance_monitor, alert_manager
    )