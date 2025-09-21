"""
性能监控测试模块

测试PerformanceMonitor类的各种监控功能
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os

from src.monitoring.performance import PerformanceMonitor, PerformanceMetrics


class TestPerformanceMonitor(unittest.TestCase):
    """PerformanceMonitor类的测试用例"""
    
    def setUp(self):
        """测试前的设置"""
        self.monitor = PerformanceMonitor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_start_stop_timer(self):
        """测试计时器启动和停止"""
        # 启动计时器
        self.monitor.start_timer('test_operation')
        
        # 模拟一些操作
        time.sleep(0.1)
        
        # 停止计时器
        elapsed_time = self.monitor.stop_timer('test_operation')
        
        # 验证结果
        self.assertIsInstance(elapsed_time, float)
        self.assertGreater(elapsed_time, 0.05)  # 至少50ms
        self.assertLess(elapsed_time, 0.5)      # 不超过500ms
    
    def test_context_manager_timer(self):
        """测试上下文管理器计时"""
        with self.monitor.timer('context_operation') as timer:
            time.sleep(0.05)
        
        # 验证结果
        self.assertIsInstance(timer.elapsed_time, float)
        self.assertGreater(timer.elapsed_time, 0.02)
    
    def test_decorator_timer(self):
        """测试装饰器计时"""
        @self.monitor.time_it('decorated_function')
        def test_function():
            time.sleep(0.05)
            return "result"
        
        result = test_function()
        
        # 验证结果
        self.assertEqual(result, "result")
        
        # 检查是否记录了性能指标
        metrics = self.monitor.get_metrics('decorated_function')
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.total_time, 0)
    
    def test_record_metric(self):
        """测试记录指标"""
        # 记录一些指标
        self.monitor.record_metric('cpu_usage', 75.5)
        self.monitor.record_metric('memory_usage', 1024)
        self.monitor.record_metric('cpu_usage', 80.2)  # 再次记录CPU使用率
        
        # 获取指标
        cpu_metrics = self.monitor.get_metrics('cpu_usage')
        memory_metrics = self.monitor.get_metrics('memory_usage')
        
        # 验证结果
        self.assertIsInstance(cpu_metrics, PerformanceMetrics)
        self.assertEqual(cpu_metrics.count, 2)
        self.assertAlmostEqual(cpu_metrics.average, 77.85, places=2)
        self.assertEqual(cpu_metrics.min_value, 75.5)
        self.assertEqual(cpu_metrics.max_value, 80.2)
        
        self.assertEqual(memory_metrics.count, 1)
        self.assertEqual(memory_metrics.average, 1024)
    
    def test_increment_counter(self):
        """测试计数器递增"""
        # 递增计数器
        self.monitor.increment_counter('requests')
        self.monitor.increment_counter('requests', 5)
        self.monitor.increment_counter('errors', 2)
        
        # 获取计数器值
        requests_count = self.monitor.get_counter('requests')
        errors_count = self.monitor.get_counter('errors')
        
        # 验证结果
        self.assertEqual(requests_count, 6)  # 1 + 5
        self.assertEqual(errors_count, 2)
    
    def test_gauge_metrics(self):
        """测试仪表盘指标"""
        # 设置仪表盘值
        self.monitor.set_gauge('active_connections', 10)
        self.monitor.set_gauge('queue_size', 25)
        self.monitor.set_gauge('active_connections', 15)  # 更新值
        
        # 获取仪表盘值
        connections = self.monitor.get_gauge('active_connections')
        queue_size = self.monitor.get_gauge('queue_size')
        
        # 验证结果
        self.assertEqual(connections, 15)
        self.assertEqual(queue_size, 25)
    
    def test_histogram_metrics(self):
        """测试直方图指标"""
        # 记录一些响应时间
        response_times = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22]
        
        for rt in response_times:
            self.monitor.record_histogram('response_time', rt)
        
        # 获取直方图统计
        histogram = self.monitor.get_histogram('response_time')
        
        # 验证结果
        self.assertEqual(histogram['count'], 7)
        self.assertAlmostEqual(histogram['mean'], sum(response_times) / len(response_times), places=3)
        self.assertEqual(histogram['min'], min(response_times))
        self.assertEqual(histogram['max'], max(response_times))
        
        # 检查百分位数
        self.assertIn('p50', histogram)
        self.assertIn('p95', histogram)
        self.assertIn('p99', histogram)
    
    def test_system_metrics_collection(self):
        """测试系统指标收集"""
        # 收集系统指标
        system_metrics = self.monitor.collect_system_metrics()
        
        # 验证结果
        self.assertIsInstance(system_metrics, dict)
        self.assertIn('cpu_percent', system_metrics)
        self.assertIn('memory_percent', system_metrics)
        self.assertIn('disk_usage', system_metrics)
        
        # 检查指标值的合理性
        self.assertGreaterEqual(system_metrics['cpu_percent'], 0)
        self.assertLessEqual(system_metrics['cpu_percent'], 100)
        self.assertGreaterEqual(system_metrics['memory_percent'], 0)
        self.assertLessEqual(system_metrics['memory_percent'], 100)
    
    def test_concurrent_metrics_recording(self):
        """测试并发指标记录"""
        def record_worker(worker_id):
            for i in range(100):
                self.monitor.record_metric(f'worker_{worker_id}', i)
                self.monitor.increment_counter('total_operations')
        
        # 启动多个线程
        threads = []
        num_workers = 5
        
        for worker_id in range(num_workers):
            thread = threading.Thread(target=record_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        total_operations = self.monitor.get_counter('total_operations')
        self.assertEqual(total_operations, num_workers * 100)
        
        # 检查每个工作线程的指标
        for worker_id in range(num_workers):
            metrics = self.monitor.get_metrics(f'worker_{worker_id}')
            self.assertEqual(metrics.count, 100)
    
    def test_metrics_export(self):
        """测试指标导出"""
        # 记录一些指标
        self.monitor.record_metric('test_metric', 10)
        self.monitor.record_metric('test_metric', 20)
        self.monitor.increment_counter('test_counter', 5)
        self.monitor.set_gauge('test_gauge', 100)
        
        # 导出指标
        exported_metrics = self.monitor.export_metrics()
        
        # 验证结果
        self.assertIsInstance(exported_metrics, dict)
        self.assertIn('metrics', exported_metrics)
        self.assertIn('counters', exported_metrics)
        self.assertIn('gauges', exported_metrics)
        
        # 检查具体指标
        self.assertIn('test_metric', exported_metrics['metrics'])
        self.assertIn('test_counter', exported_metrics['counters'])
        self.assertIn('test_gauge', exported_metrics['gauges'])
    
    def test_metrics_reset(self):
        """测试指标重置"""
        # 记录一些指标
        self.monitor.record_metric('test_metric', 10)
        self.monitor.increment_counter('test_counter', 5)
        self.monitor.set_gauge('test_gauge', 100)
        
        # 验证指标存在
        self.assertIsNotNone(self.monitor.get_metrics('test_metric'))
        self.assertEqual(self.monitor.get_counter('test_counter'), 5)
        self.assertEqual(self.monitor.get_gauge('test_gauge'), 100)
        
        # 重置指标
        self.monitor.reset_metrics()
        
        # 验证指标已重置
        self.assertIsNone(self.monitor.get_metrics('test_metric'))
        self.assertEqual(self.monitor.get_counter('test_counter'), 0)
        self.assertEqual(self.monitor.get_gauge('test_gauge'), 0)
    
    def test_metrics_persistence(self):
        """测试指标持久化"""
        # 记录一些指标
        self.monitor.record_metric('persistent_metric', 42)
        self.monitor.increment_counter('persistent_counter', 10)
        
        # 保存指标到文件
        metrics_file = os.path.join(self.temp_dir, 'metrics.json')
        self.monitor.save_metrics(metrics_file)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(metrics_file))
        
        # 创建新的监控器并加载指标
        new_monitor = PerformanceMonitor()
        new_monitor.load_metrics(metrics_file)
        
        # 验证指标已加载
        loaded_metric = new_monitor.get_metrics('persistent_metric')
        loaded_counter = new_monitor.get_counter('persistent_counter')
        
        self.assertIsNotNone(loaded_metric)
        self.assertEqual(loaded_metric.average, 42)
        self.assertEqual(loaded_counter, 10)
    
    def test_alert_thresholds(self):
        """测试告警阈值"""
        # 设置告警阈值
        self.monitor.set_alert_threshold('cpu_usage', max_value=80)
        self.monitor.set_alert_threshold('response_time', max_value=1.0)
        
        # 记录正常值（不应触发告警）
        alerts1 = self.monitor.record_metric('cpu_usage', 70)
        self.assertEqual(len(alerts1), 0)
        
        # 记录超过阈值的值（应触发告警）
        alerts2 = self.monitor.record_metric('cpu_usage', 90)
        self.assertEqual(len(alerts2), 1)
        self.assertEqual(alerts2[0]['metric'], 'cpu_usage')
        self.assertEqual(alerts2[0]['value'], 90)
        self.assertEqual(alerts2[0]['threshold'], 80)
    
    def test_performance_summary(self):
        """测试性能摘要"""
        # 记录一些性能数据
        for i in range(10):
            with self.monitor.timer('operation'):
                time.sleep(0.01)  # 10ms
            
            self.monitor.record_metric('throughput', 100 + i)
            self.monitor.increment_counter('requests')
        
        # 获取性能摘要
        summary = self.monitor.get_performance_summary()
        
        # 验证结果
        self.assertIsInstance(summary, dict)
        self.assertIn('total_operations', summary)
        self.assertIn('average_response_time', summary)
        self.assertIn('throughput', summary)
        self.assertIn('error_rate', summary)
        
        # 检查具体值
        self.assertEqual(summary['total_operations'], 10)
        self.assertGreater(summary['average_response_time'], 0)
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = {
            'enable_system_metrics': True,
            'collection_interval': 5,
            'max_metrics_history': 1000,
            'alert_enabled': True
        }
        
        monitor_with_config = PerformanceMonitor(config)
        
        # 验证配置
        self.assertTrue(monitor_with_config.config['enable_system_metrics'])
        self.assertEqual(monitor_with_config.config['collection_interval'], 5)
        self.assertEqual(monitor_with_config.config['max_metrics_history'], 1000)
        self.assertTrue(monitor_with_config.config['alert_enabled'])


class TestPerformanceMetrics(unittest.TestCase):
    """PerformanceMetrics类的测试用例"""
    
    def test_metrics_initialization(self):
        """测试指标初始化"""
        metrics = PerformanceMetrics()
        
        # 验证初始状态
        self.assertEqual(metrics.count, 0)
        self.assertEqual(metrics.total_time, 0)
        self.assertEqual(metrics.average, 0)
        self.assertIsNone(metrics.min_value)
        self.assertIsNone(metrics.max_value)
    
    def test_add_measurement(self):
        """测试添加测量值"""
        metrics = PerformanceMetrics()
        
        # 添加一些测量值
        metrics.add_measurement(10)
        metrics.add_measurement(20)
        metrics.add_measurement(15)
        
        # 验证结果
        self.assertEqual(metrics.count, 3)
        self.assertEqual(metrics.total_time, 45)
        self.assertEqual(metrics.average, 15)
        self.assertEqual(metrics.min_value, 10)
        self.assertEqual(metrics.max_value, 20)
    
    def test_metrics_statistics(self):
        """测试指标统计"""
        metrics = PerformanceMetrics()
        
        # 添加一系列值
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for value in values:
            metrics.add_measurement(value)
        
        # 计算统计信息
        stats = metrics.get_statistics()
        
        # 验证结果
        self.assertEqual(stats['count'], 10)
        self.assertEqual(stats['mean'], 5.5)
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 10)
        self.assertAlmostEqual(stats['std'], 3.0277, places=3)
    
    def test_percentiles(self):
        """测试百分位数计算"""
        metrics = PerformanceMetrics()
        
        # 添加100个值
        for i in range(1, 101):
            metrics.add_measurement(i)
        
        # 计算百分位数
        percentiles = metrics.get_percentiles([50, 90, 95, 99])
        
        # 验证结果
        self.assertAlmostEqual(percentiles[50], 50.5, places=1)
        self.assertAlmostEqual(percentiles[90], 90.1, places=1)
        self.assertAlmostEqual(percentiles[95], 95.05, places=1)
        self.assertAlmostEqual(percentiles[99], 99.01, places=1)
    
    def test_metrics_serialization(self):
        """测试指标序列化"""
        metrics = PerformanceMetrics()
        
        # 添加一些数据
        for i in range(5):
            metrics.add_measurement(i * 10)
        
        # 序列化
        serialized = metrics.to_dict()
        
        # 验证结果
        self.assertIsInstance(serialized, dict)
        self.assertEqual(serialized['count'], 5)
        self.assertEqual(serialized['total_time'], 100)
        self.assertEqual(serialized['average'], 20)
        self.assertEqual(serialized['min_value'], 0)
        self.assertEqual(serialized['max_value'], 40)
        
        # 反序列化
        new_metrics = PerformanceMetrics.from_dict(serialized)
        
        # 验证反序列化结果
        self.assertEqual(new_metrics.count, metrics.count)
        self.assertEqual(new_metrics.total_time, metrics.total_time)
        self.assertEqual(new_metrics.average, metrics.average)
        self.assertEqual(new_metrics.min_value, metrics.min_value)
        self.assertEqual(new_metrics.max_value, metrics.max_value)


class TestPerformanceMonitorIntegration(unittest.TestCase):
    """PerformanceMonitor集成测试"""
    
    def setUp(self):
        """测试前的设置"""
        self.monitor = PerformanceMonitor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_real_world_monitoring_scenario(self):
        """测试真实世界监控场景"""
        # 模拟一个Web服务的监控
        
        # 模拟处理请求
        for i in range(50):
            with self.monitor.timer('request_processing'):
                # 模拟不同的处理时间
                processing_time = 0.01 + (i % 10) * 0.005
                time.sleep(processing_time)
            
            # 记录其他指标
            self.monitor.increment_counter('total_requests')
            self.monitor.record_metric('request_size', 1000 + i * 10)
            
            # 模拟偶尔的错误
            if i % 10 == 9:
                self.monitor.increment_counter('errors')
        
        # 获取性能摘要
        summary = self.monitor.get_performance_summary()
        
        # 验证结果
        self.assertEqual(summary['total_operations'], 50)
        self.assertEqual(self.monitor.get_counter('total_requests'), 50)
        self.assertEqual(self.monitor.get_counter('errors'), 5)
        
        # 检查请求大小指标
        request_size_metrics = self.monitor.get_metrics('request_size')
        self.assertEqual(request_size_metrics.count, 50)
        self.assertEqual(request_size_metrics.min_value, 1000)
        self.assertEqual(request_size_metrics.max_value, 1490)
    
    def test_monitoring_with_alerts(self):
        """测试带告警的监控"""
        # 设置告警阈值
        self.monitor.set_alert_threshold('error_rate', max_value=0.1)  # 10%错误率
        self.monitor.set_alert_threshold('response_time', max_value=0.5)  # 500ms响应时间
        
        alerts_triggered = []
        
        # 模拟正常操作
        for i in range(20):
            with self.monitor.timer('operation'):
                time.sleep(0.01)  # 10ms，正常响应时间
            
            self.monitor.increment_counter('total_operations')
            
            # 偶尔的错误（5%错误率，不应触发告警）
            if i % 20 == 19:
                self.monitor.increment_counter('errors')
        
        # 模拟异常情况
        for i in range(10):
            with self.monitor.timer('operation'):
                time.sleep(0.6)  # 600ms，超过阈值
            
            self.monitor.increment_counter('total_operations')
            self.monitor.increment_counter('errors')  # 高错误率
        
        # 检查是否触发了告警
        # 注意：这里需要实现告警检查逻辑
        
        # 验证最终状态
        total_ops = self.monitor.get_counter('total_operations')
        total_errors = self.monitor.get_counter('errors')
        error_rate = total_errors / total_ops if total_ops > 0 else 0
        
        self.assertEqual(total_ops, 30)
        self.assertEqual(total_errors, 11)
        self.assertGreater(error_rate, 0.1)  # 错误率超过阈值
    
    def test_long_running_monitoring(self):
        """测试长时间运行的监控"""
        # 模拟长时间运行的服务监控
        
        start_time = time.time()
        
        # 运行一段时间的监控
        while time.time() - start_time < 1.0:  # 运行1秒
            with self.monitor.timer('background_task'):
                time.sleep(0.01)
            
            # 收集系统指标
            system_metrics = self.monitor.collect_system_metrics()
            for metric_name, value in system_metrics.items():
                self.monitor.record_metric(f'system_{metric_name}', value)
            
            self.monitor.increment_counter('background_tasks')
        
        # 验证收集到的数据
        background_tasks = self.monitor.get_counter('background_tasks')
        self.assertGreater(background_tasks, 50)  # 至少执行了50次
        
        # 检查系统指标
        cpu_metrics = self.monitor.get_metrics('system_cpu_percent')
        memory_metrics = self.monitor.get_metrics('system_memory_percent')
        
        self.assertIsNotNone(cpu_metrics)
        self.assertIsNotNone(memory_metrics)
        self.assertGreater(cpu_metrics.count, 0)
        self.assertGreater(memory_metrics.count, 0)


if __name__ == '__main__':
    unittest.main()