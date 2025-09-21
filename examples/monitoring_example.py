"""
监控系统示例

演示性能监控、异常告警、日志记录、指标收集和仪表板等功能。
"""

import numpy as np
import pandas as pd
import time
import threading
from datetime import datetime, timedelta

# 导入我们的框架
from src.data import DataLoader, DataPreprocessor
from src.training import ModelTrainer
from src.deployment import ModelSerializer, ModelInferenceEngine
from src.monitoring import (
    create_log_manager, PerformanceMonitor, AlertManager,
    MetricCollector, DashboardServer, create_dashboard,
    EmailAlertHandler, get_system_metrics, monitor_function_performance
)


def logging_system_example():
    """日志系统示例"""
    print("=== 日志系统示例 ===")
    
    # 创建日志管理器
    log_manager = create_log_manager(
        log_level='DEBUG',
        log_file='logs/monitoring_example.log',
        max_file_size=10*1024*1024,  # 10MB
        backup_count=5
    )
    
    # 获取不同类型的日志器
    app_logger = log_manager.get_logger('application')
    model_logger = log_manager.get_logger('model')
    system_logger = log_manager.get_logger('system')
    
    print("1. 测试不同级别的日志...")
    
    app_logger.debug("这是一个调试信息")
    app_logger.info("应用程序启动")
    app_logger.warning("这是一个警告信息")
    app_logger.error("这是一个错误信息")
    
    # 结构化日志
    model_logger.info("模型训练开始", extra={
        'model_type': 'random_forest',
        'dataset_size': 1000,
        'features': 20
    })
    
    # 系统日志
    system_logger.info("系统资源监控", extra={
        'cpu_usage': 45.2,
        'memory_usage': 2048,
        'disk_usage': 75.5
    })
    
    print("2. 日志聚合和分析...")
    
    # 获取日志聚合器
    log_aggregator = log_manager.get_aggregator()
    
    # 模拟一些日志事件
    for i in range(10):
        if i % 3 == 0:
            app_logger.error(f"模拟错误 {i}")
        elif i % 2 == 0:
            app_logger.warning(f"模拟警告 {i}")
        else:
            app_logger.info(f"模拟信息 {i}")
    
    # 分析日志
    log_stats = log_aggregator.get_log_statistics(
        start_time=datetime.now() - timedelta(minutes=5)
    )
    
    print(f"  日志统计: {log_stats}")
    
    # 搜索特定日志
    error_logs = log_aggregator.search_logs(
        level='ERROR',
        start_time=datetime.now() - timedelta(minutes=5)
    )
    
    print(f"  错误日志数量: {len(error_logs)}")


def performance_monitoring_example():
    """性能监控示例"""
    print("\n=== 性能监控示例 ===")
    
    # 创建性能监控器
    performance_monitor = PerformanceMonitor(
        monitoring_interval=1.0,
        enable_system_monitoring=True,
        enable_model_monitoring=True
    )
    
    print("1. 启动性能监控...")
    performance_monitor.start_monitoring()
    
    # 模拟模型训练过程
    print("2. 模拟模型训练监控...")
    
    @monitor_function_performance
    def simulate_model_training():
        """模拟模型训练"""
        time.sleep(2)  # 模拟训练时间
        return "训练完成"
    
    @monitor_function_performance
    def simulate_data_preprocessing():
        """模拟数据预处理"""
        time.sleep(1)  # 模拟预处理时间
        return "预处理完成"
    
    # 执行监控的函数
    preprocessing_result = simulate_data_preprocessing()
    training_result = simulate_model_training()
    
    print(f"  {preprocessing_result}")
    print(f"  {training_result}")
    
    # 模拟推理过程监控
    print("3. 模拟推理过程监控...")
    
    for i in range(20):
        # 模拟推理指标
        inference_time = np.random.exponential(0.05)
        memory_usage = np.random.normal(100, 20)
        cpu_usage = np.random.normal(50, 15)
        
        performance_monitor.record_inference_metrics(
            inference_time=inference_time,
            memory_usage=max(0, memory_usage),
            cpu_usage=max(0, min(100, cpu_usage))
        )
        
        time.sleep(0.1)
    
    # 获取性能统计
    print("4. 获取性能统计...")
    stats = performance_monitor.get_statistics()
    
    print(f"  平均推理时间: {stats.get('avg_inference_time', 0):.4f}s")
    print(f"  最大内存使用: {stats.get('max_memory_usage', 0):.2f}MB")
    print(f"  平均CPU使用率: {stats.get('avg_cpu_usage', 0):.2f}%")
    print(f"  总推理次数: {stats.get('total_inferences', 0)}")
    
    # 获取系统指标
    system_metrics = get_system_metrics()
    print(f"  当前系统指标: {system_metrics}")
    
    performance_monitor.stop_monitoring()
    print("5. 性能监控已停止")


def alert_system_example():
    """告警系统示例"""
    print("\n=== 告警系统示例 ===")
    
    # 创建告警管理器
    alert_manager = AlertManager()
    
    # 添加邮件告警处理器（示例配置）
    email_handler = EmailAlertHandler(
        smtp_server="smtp.example.com",
        smtp_port=587,
        username="alerts@example.com",
        password="password",
        from_email="alerts@example.com"
    )
    
    alert_manager.add_handler("email", email_handler)
    
    print("1. 配置告警规则...")
    
    # 添加告警规则
    alert_manager.add_rule(
        rule_id="high_cpu_usage",
        metric_name="cpu_usage",
        threshold=80.0,
        comparison="greater_than",
        severity="warning",
        description="CPU使用率过高"
    )
    
    alert_manager.add_rule(
        rule_id="high_memory_usage",
        metric_name="memory_usage",
        threshold=90.0,
        comparison="greater_than",
        severity="critical",
        description="内存使用率过高"
    )
    
    alert_manager.add_rule(
        rule_id="slow_inference",
        metric_name="inference_time",
        threshold=1.0,
        comparison="greater_than",
        severity="warning",
        description="推理时间过长"
    )
    
    print("2. 启动告警监控...")
    alert_manager.start_monitoring()
    
    # 模拟指标数据触发告警
    print("3. 模拟告警触发...")
    
    # 模拟正常指标
    alert_manager.check_metric("cpu_usage", 45.0)
    alert_manager.check_metric("memory_usage", 60.0)
    alert_manager.check_metric("inference_time", 0.05)
    
    # 模拟异常指标
    alert_manager.check_metric("cpu_usage", 85.0)  # 触发CPU告警
    alert_manager.check_metric("memory_usage", 95.0)  # 触发内存告警
    alert_manager.check_metric("inference_time", 1.5)  # 触发推理时间告警
    
    time.sleep(1)  # 等待告警处理
    
    # 获取告警历史
    print("4. 获取告警历史...")
    alert_history = alert_manager.get_alert_history()
    
    for alert in alert_history:
        print(f"  告警: {alert.rule_id} - {alert.description} ({alert.severity})")
        print(f"    时间: {alert.timestamp}")
        print(f"    值: {alert.value}")
    
    # 解决告警
    print("5. 解决告警...")
    for alert in alert_history:
        if not alert.resolved:
            alert_manager.resolve_alert(alert.alert_id, "问题已修复")
    
    alert_manager.stop_monitoring()
    print("6. 告警监控已停止")


def metrics_collection_example():
    """指标收集示例"""
    print("\n=== 指标收集示例 ===")
    
    # 创建指标收集器
    metric_collector = MetricCollector(
        storage_backend='memory',  # 使用内存存储
        retention_days=7
    )
    
    print("1. 注册指标...")
    
    # 注册不同类型的指标
    metric_collector.register_metric(
        name="model_accuracy",
        metric_type="gauge",
        description="模型准确率"
    )
    
    metric_collector.register_metric(
        name="prediction_count",
        metric_type="counter",
        description="预测次数计数器"
    )
    
    metric_collector.register_metric(
        name="inference_latency",
        metric_type="histogram",
        description="推理延迟分布"
    )
    
    print("2. 记录指标数据...")
    
    # 模拟指标数据
    for i in range(100):
        # 记录模型准确率
        accuracy = 0.85 + np.random.normal(0, 0.05)
        metric_collector.record_metric("model_accuracy", accuracy)
        
        # 增加预测计数
        metric_collector.increment_counter("prediction_count")
        
        # 记录推理延迟
        latency = np.random.exponential(0.05)
        metric_collector.record_metric("inference_latency", latency)
        
        time.sleep(0.01)
    
    print("3. 查询指标数据...")
    
    # 查询最新值
    latest_accuracy = metric_collector.get_latest_value("model_accuracy")
    print(f"  最新准确率: {latest_accuracy:.4f}")
    
    total_predictions = metric_collector.get_latest_value("prediction_count")
    print(f"  总预测次数: {total_predictions}")
    
    # 查询时间范围内的数据
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=1)
    
    accuracy_data = metric_collector.query_metrics(
        metric_name="model_accuracy",
        start_time=start_time,
        end_time=end_time
    )
    
    print(f"  准确率数据点数: {len(accuracy_data)}")
    
    # 聚合查询
    avg_accuracy = metric_collector.aggregate_metrics(
        metric_name="model_accuracy",
        aggregation_type="mean",
        start_time=start_time,
        end_time=end_time
    )
    
    print(f"  平均准确率: {avg_accuracy:.4f}")
    
    # 获取指标统计
    print("4. 指标统计...")
    stats = metric_collector.get_statistics()
    
    for metric_name, metric_stats in stats.items():
        print(f"  {metric_name}: {metric_stats}")


def dashboard_example():
    """仪表板示例"""
    print("\n=== 仪表板示例 ===")
    
    # 创建仪表板
    dashboard = create_dashboard(
        title="机器学习监控仪表板",
        host="localhost",
        port=8050,
        debug=False
    )
    
    print("1. 配置仪表板...")
    
    # 生成示例数据
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=1),
        end=datetime.now(),
        freq='1min'
    )
    
    # 模型性能数据
    accuracy_data = pd.DataFrame({
        'timestamp': timestamps,
        'accuracy': 0.85 + np.random.normal(0, 0.05, len(timestamps)),
        'loss': 0.3 + np.random.normal(0, 0.1, len(timestamps))
    })
    
    # 系统资源数据
    resource_data = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': 50 + np.random.normal(0, 15, len(timestamps)),
        'memory_usage': 60 + np.random.normal(0, 20, len(timestamps)),
        'disk_usage': 70 + np.random.normal(0, 5, len(timestamps))
    })
    
    # 推理性能数据
    inference_data = pd.DataFrame({
        'timestamp': timestamps,
        'inference_time': np.random.exponential(0.05, len(timestamps)),
        'throughput': 100 + np.random.normal(0, 20, len(timestamps))
    })
    
    print("2. 添加图表...")
    
    # 添加模型性能图表
    dashboard.add_chart(
        chart_id="model_performance",
        chart_type="line",
        title="模型性能",
        data=accuracy_data,
        x_column="timestamp",
        y_columns=["accuracy", "loss"]
    )
    
    # 添加系统资源图表
    dashboard.add_chart(
        chart_id="system_resources",
        chart_type="line",
        title="系统资源使用",
        data=resource_data,
        x_column="timestamp",
        y_columns=["cpu_usage", "memory_usage", "disk_usage"]
    )
    
    # 添加推理性能图表
    dashboard.add_chart(
        chart_id="inference_performance",
        chart_type="scatter",
        title="推理性能",
        data=inference_data,
        x_column="timestamp",
        y_columns=["inference_time", "throughput"]
    )
    
    # 添加仪表盘图表
    current_accuracy = accuracy_data['accuracy'].iloc[-1]
    dashboard.add_gauge(
        gauge_id="current_accuracy",
        title="当前准确率",
        value=current_accuracy,
        min_value=0,
        max_value=1,
        threshold=0.8
    )
    
    print("3. 启动仪表板服务器...")
    
    # 在后台线程中启动仪表板
    def start_dashboard():
        try:
            dashboard.run(host="localhost", port=8050, debug=False)
        except Exception as e:
            print(f"仪表板启动失败: {str(e)}")
    
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    
    print("  仪表板已启动，访问 http://localhost:8050")
    print("  (注意：这是一个示例，实际运行时需要安装相应的Web框架)")
    
    # 模拟实时数据更新
    print("4. 模拟实时数据更新...")
    
    for i in range(5):
        time.sleep(1)
        
        # 生成新数据点
        new_timestamp = datetime.now()
        new_accuracy = 0.85 + np.random.normal(0, 0.05)
        new_cpu = 50 + np.random.normal(0, 15)
        
        # 更新图表数据（在实际实现中）
        print(f"  更新数据点 {i+1}: 准确率={new_accuracy:.4f}, CPU={new_cpu:.1f}%")
    
    print("5. 仪表板示例完成")


def integrated_monitoring_example():
    """集成监控示例"""
    print("\n=== 集成监控示例 ===")
    
    # 创建完整的监控系统
    log_manager = create_log_manager(log_level='INFO')
    logger = log_manager.get_logger('integrated_monitoring')
    
    performance_monitor = PerformanceMonitor()
    alert_manager = AlertManager()
    metric_collector = MetricCollector()
    
    print("1. 启动集成监控系统...")
    
    performance_monitor.start_monitoring()
    alert_manager.start_monitoring()
    
    # 配置告警规则
    alert_manager.add_rule(
        rule_id="model_accuracy_drop",
        metric_name="model_accuracy",
        threshold=0.8,
        comparison="less_than",
        severity="critical",
        description="模型准确率下降"
    )
    
    logger.info("集成监控系统启动完成")
    
    print("2. 模拟完整的ML工作流监控...")
    
    # 模拟数据加载
    logger.info("开始数据加载")
    time.sleep(0.5)
    logger.info("数据加载完成", extra={'data_size': 10000})
    
    # 模拟模型训练
    logger.info("开始模型训练")
    
    for epoch in range(5):
        # 模拟训练指标
        train_loss = 1.0 - epoch * 0.2 + np.random.normal(0, 0.1)
        val_accuracy = 0.6 + epoch * 0.08 + np.random.normal(0, 0.05)
        
        # 记录指标
        metric_collector.record_metric("train_loss", train_loss)
        metric_collector.record_metric("model_accuracy", val_accuracy)
        
        # 检查告警
        alert_manager.check_metric("model_accuracy", val_accuracy)
        
        logger.info(f"训练进度", extra={
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy
        })
        
        time.sleep(0.2)
    
    logger.info("模型训练完成")
    
    # 模拟模型部署和推理
    logger.info("开始模型部署")
    time.sleep(0.5)
    logger.info("模型部署完成")
    
    print("3. 模拟推理监控...")
    
    for i in range(20):
        # 模拟推理
        inference_time = np.random.exponential(0.05)
        prediction_accuracy = 0.85 + np.random.normal(0, 0.1)
        
        # 记录性能指标
        performance_monitor.record_inference_metrics(
            inference_time=inference_time,
            memory_usage=np.random.normal(100, 20),
            cpu_usage=np.random.normal(50, 15)
        )
        
        # 记录业务指标
        metric_collector.record_metric("inference_time", inference_time)
        metric_collector.record_metric("prediction_accuracy", prediction_accuracy)
        
        # 检查告警
        alert_manager.check_metric("prediction_accuracy", prediction_accuracy)
        
        if i % 5 == 0:
            logger.info(f"推理批次 {i//5 + 1} 完成", extra={
                'batch_size': 5,
                'avg_inference_time': inference_time,
                'avg_accuracy': prediction_accuracy
            })
        
        time.sleep(0.1)
    
    print("4. 生成监控报告...")
    
    # 获取各种统计信息
    performance_stats = performance_monitor.get_statistics()
    alert_history = alert_manager.get_alert_history()
    metric_stats = metric_collector.get_statistics()
    
    print(f"  性能统计: {performance_stats}")
    print(f"  告警数量: {len(alert_history)}")
    print(f"  指标统计: {metric_stats}")
    
    # 停止监控
    performance_monitor.stop_monitoring()
    alert_manager.stop_monitoring()
    
    logger.info("集成监控示例完成")
    print("5. 集成监控示例完成")


if __name__ == "__main__":
    try:
        logging_system_example()
        performance_monitoring_example()
        alert_system_example()
        metrics_collection_example()
        dashboard_example()
        integrated_monitoring_example()
        
        print("\n所有监控示例运行完成！")
        
    except Exception as e:
        print(f"监控示例运行出错: {str(e)}")
        raise