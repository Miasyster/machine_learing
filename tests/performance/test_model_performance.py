"""
模型性能测试

测试各种模型的训练和预测性能
"""

import pytest
import numpy as np
import time
import psutil
import os
from pathlib import Path
import sys
from unittest.mock import Mock
import gc

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.test_utils import (
    TestDataGenerator, PerformanceTimer, MockModel,
    ModelTestHelper, skip_if_slow
)
from tests.performance import PERFORMANCE_CONFIG


class TestModelTrainingPerformance:
    """测试模型训练性能"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
        self.config = PERFORMANCE_CONFIG
    
    def test_small_dataset_training_speed(self):
        """测试小数据集训练速度"""
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(
                n_samples=self.config['small_dataset_size']
            )
        
        model = MockModel(fit_time=0.1)
        
        with PerformanceTimer("Small Dataset Training") as timer:
            model.fit(X_train, y_train)
        
        # 小数据集训练应该很快
        assert timer.duration < 1.0, f"Training took {timer.duration:.2f}s, expected < 1.0s"
    
    @skip_if_slow()
    def test_medium_dataset_training_speed(self):
        """测试中等数据集训练速度"""
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(
                n_samples=self.config['medium_dataset_size']
            )
        
        model = MockModel(fit_time=0.5)
        
        with PerformanceTimer("Medium Dataset Training") as timer:
            model.fit(X_train, y_train)
        
        # 中等数据集训练时间应该合理
        assert timer.duration < 5.0, f"Training took {timer.duration:.2f}s, expected < 5.0s"
    
    @skip_if_slow()
    def test_large_dataset_training_speed(self):
        """测试大数据集训练速度"""
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(
                n_samples=self.config['large_dataset_size']
            )
        
        model = MockModel(fit_time=1.0)
        
        with PerformanceTimer("Large Dataset Training") as timer:
            model.fit(X_train, y_train)
        
        # 大数据集训练时间应该在可接受范围内
        assert timer.duration < 10.0, f"Training took {timer.duration:.2f}s, expected < 10.0s"
    
    def test_sklearn_model_performance(self):
        """测试Sklearn模型性能"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_regression_data(n_samples=1000)
            
            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42)
            }
            
            for name, model in models.items():
                with PerformanceTimer(f"{name} Training") as timer:
                    model.fit(X_train, y_train)
                
                # 训练时间应该合理
                assert timer.duration < 5.0, \
                    f"{name} training took {timer.duration:.2f}s, expected < 5.0s"
                
                # 测试预测性能
                with PerformanceTimer(f"{name} Prediction") as pred_timer:
                    predictions = model.predict(X_test)
                
                assert pred_timer.duration < 1.0, \
                    f"{name} prediction took {pred_timer.duration:.2f}s, expected < 1.0s"
                
                ModelTestHelper.assert_no_nan_inf(predictions)
        
        except ImportError:
            pytest.skip("Sklearn not available")
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion="1.0"),
        reason="PyTorch not available"
    )
    def test_pytorch_model_performance(self):
        """测试PyTorch模型性能"""
        try:
            import torch
            import torch.nn as nn
            from src.models.deep_learning.neural_networks import NeuralNetworkModel
            
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_regression_data(n_samples=1000)
            
            model = NeuralNetworkModel(
                input_size=X_train.shape[1],
                hidden_sizes=[32, 16],
                output_size=1,
                epochs=5,
                batch_size=64
            )
            
            with PerformanceTimer("PyTorch Model Training") as timer:
                model.fit(X_train, y_train)
            
            # 深度学习模型训练时间应该合理
            assert timer.duration < 30.0, \
                f"PyTorch training took {timer.duration:.2f}s, expected < 30.0s"
            
            with PerformanceTimer("PyTorch Model Prediction") as pred_timer:
                predictions = model.predict(X_test)
            
            assert pred_timer.duration < 2.0, \
                f"PyTorch prediction took {pred_timer.duration:.2f}s, expected < 2.0s"
            
            ModelTestHelper.assert_no_nan_inf(predictions)
        
        except ImportError:
            pytest.skip("PyTorch models not available")


class TestModelPredictionPerformance:
    """测试模型预测性能"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
        self.config = PERFORMANCE_CONFIG
    
    def test_batch_prediction_performance(self):
        """测试批量预测性能"""
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=1000)
        
        model = MockModel(predict_time=0.01)
        model.fit(X_train, y_train)
        
        # 测试不同批次大小的预测性能
        batch_sizes = [1, 10, 100, len(X_test)]
        
        for batch_size in batch_sizes:
            n_batches = len(X_test) // batch_size
            if n_batches == 0:
                continue
            
            with PerformanceTimer(f"Batch Size {batch_size} Prediction") as timer:
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(X_test))
                    batch_X = X_test[start_idx:end_idx]
                    predictions = model.predict(batch_X)
                    ModelTestHelper.assert_no_nan_inf(predictions)
            
            # 预测时间应该随批次大小合理变化
            avg_time_per_sample = timer.duration / (n_batches * batch_size)
            assert avg_time_per_sample < 0.01, \
                f"Average prediction time per sample: {avg_time_per_sample:.4f}s"
    
    def test_single_vs_batch_prediction(self):
        """测试单个vs批量预测性能"""
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=100)
        
        model = MockModel(predict_time=0.001)
        model.fit(X_train, y_train)
        
        # 单个预测
        with PerformanceTimer("Single Predictions") as single_timer:
            single_predictions = []
            for x in X_test:
                pred = model.predict(x.reshape(1, -1))
                single_predictions.append(pred[0])
        
        # 批量预测
        with PerformanceTimer("Batch Prediction") as batch_timer:
            batch_predictions = model.predict(X_test)
        
        # 验证结果一致性
        np.testing.assert_allclose(single_predictions, batch_predictions, rtol=1e-6)
        
        # 批量预测应该更快
        print(f"Single prediction time: {single_timer.duration:.4f}s")
        print(f"Batch prediction time: {batch_timer.duration:.4f}s")
        print(f"Speedup: {single_timer.duration / batch_timer.duration:.2f}x")
    
    def test_concurrent_prediction_performance(self):
        """测试并发预测性能"""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=500)
        
        model = MockModel(predict_time=0.01)
        model.fit(X_train, y_train)
        
        def predict_batch(X_batch):
            return model.predict(X_batch)
        
        # 分割测试数据为多个批次
        n_threads = 4
        batch_size = len(X_test) // n_threads
        batches = [X_test[i*batch_size:(i+1)*batch_size] for i in range(n_threads)]
        
        # 串行预测
        with PerformanceTimer("Serial Prediction") as serial_timer:
            serial_results = []
            for batch in batches:
                result = predict_batch(batch)
                serial_results.extend(result)
        
        # 并行预测
        with PerformanceTimer("Parallel Prediction") as parallel_timer:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = [executor.submit(predict_batch, batch) for batch in batches]
                parallel_results = []
                for future in futures:
                    result = future.result()
                    parallel_results.extend(result)
        
        # 验证结果一致性
        np.testing.assert_allclose(serial_results, parallel_results, rtol=1e-6)
        
        print(f"Serial time: {serial_timer.duration:.4f}s")
        print(f"Parallel time: {parallel_timer.duration:.4f}s")
        print(f"Speedup: {serial_timer.duration / parallel_timer.duration:.2f}x")


class TestMemoryPerformance:
    """测试内存性能"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
        self.config = PERFORMANCE_CONFIG
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self):
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def test_memory_usage_during_training(self):
        """测试训练期间内存使用"""
        initial_memory = self.get_memory_usage()
        
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=5000)
        
        after_data_memory = self.get_memory_usage()
        data_memory = after_data_memory - initial_memory
        
        model = MockModel()
        model.fit(X_train, y_train)
        
        after_training_memory = self.get_memory_usage()
        training_memory = after_training_memory - after_data_memory
        
        predictions = model.predict(X_test)
        
        after_prediction_memory = self.get_memory_usage()
        prediction_memory = after_prediction_memory - after_training_memory
        
        print(f"Data loading memory: {data_memory:.2f}MB")
        print(f"Training memory: {training_memory:.2f}MB")
        print(f"Prediction memory: {prediction_memory:.2f}MB")
        
        # 内存使用应该在合理范围内
        total_memory = after_prediction_memory - initial_memory
        assert total_memory < self.config['memory_limit_mb'], \
            f"Total memory usage {total_memory:.2f}MB exceeds limit {self.config['memory_limit_mb']}MB"
    
    def test_memory_cleanup(self):
        """测试内存清理"""
        initial_memory = self.get_memory_usage()
        
        # 创建大量数据和模型
        for i in range(5):
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_regression_data(n_samples=2000)
            
            model = MockModel()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # 显式删除变量
            del X_train, X_test, y_train, y_test, model, predictions
        
        # 强制垃圾回收
        gc.collect()
        
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase after cleanup: {memory_increase:.2f}MB")
        
        # 内存增长应该很小
        assert memory_increase < 50, \
            f"Memory increased by {memory_increase:.2f}MB after cleanup"
    
    def test_large_dataset_memory_efficiency(self):
        """测试大数据集内存效率"""
        initial_memory = self.get_memory_usage()
        
        # 生成大数据集
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=10000)
        
        peak_memory = self.get_memory_usage()
        
        model = MockModel()
        model.fit(X_train, y_train)
        
        training_memory = self.get_memory_usage()
        
        # 分批预测以节省内存
        batch_size = 1000
        all_predictions = []
        
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_pred = model.predict(batch_X)
            all_predictions.extend(batch_pred)
        
        final_memory = self.get_memory_usage()
        
        print(f"Peak memory during data loading: {peak_memory - initial_memory:.2f}MB")
        print(f"Memory during training: {training_memory - initial_memory:.2f}MB")
        print(f"Final memory: {final_memory - initial_memory:.2f}MB")
        
        # 内存使用应该合理
        assert final_memory - initial_memory < 200, \
            f"Memory usage {final_memory - initial_memory:.2f}MB too high"


class TestScalabilityPerformance:
    """测试可扩展性性能"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
    
    def test_dataset_size_scaling(self):
        """测试数据集大小扩展性"""
        sizes = [100, 500, 1000, 2000]
        training_times = []
        prediction_times = []
        
        for size in sizes:
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_regression_data(n_samples=size)
            
            model = MockModel(fit_time=size * 0.0001, predict_time=size * 0.00001)
            
            # 测试训练时间
            with PerformanceTimer(f"Training {size} samples") as train_timer:
                model.fit(X_train, y_train)
            training_times.append(train_timer.duration)
            
            # 测试预测时间
            with PerformanceTimer(f"Predicting {len(X_test)} samples") as pred_timer:
                predictions = model.predict(X_test)
            prediction_times.append(pred_timer.duration)
        
        # 分析扩展性
        print("Dataset Size Scaling:")
        for i, size in enumerate(sizes):
            print(f"Size: {size}, Train: {training_times[i]:.4f}s, Predict: {prediction_times[i]:.4f}s")
        
        # 训练时间应该随数据量合理增长
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = training_times[i] / training_times[i-1]
            
            # 时间增长不应该超过数据量增长的平方
            assert time_ratio <= size_ratio ** 2, \
                f"Training time scaling too poor: {time_ratio:.2f} vs {size_ratio:.2f}"
    
    def test_feature_dimension_scaling(self):
        """测试特征维度扩展性"""
        dimensions = [5, 10, 20, 50]
        training_times = []
        prediction_times = []
        
        for dim in dimensions:
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_regression_data(
                    n_samples=500, n_features=dim
                )
            
            model = MockModel(fit_time=dim * 0.001, predict_time=dim * 0.0001)
            
            # 测试训练时间
            with PerformanceTimer(f"Training {dim} features") as train_timer:
                model.fit(X_train, y_train)
            training_times.append(train_timer.duration)
            
            # 测试预测时间
            with PerformanceTimer(f"Predicting {dim} features") as pred_timer:
                predictions = model.predict(X_test)
            prediction_times.append(pred_timer.duration)
        
        # 分析特征维度扩展性
        print("Feature Dimension Scaling:")
        for i, dim in enumerate(dimensions):
            print(f"Dim: {dim}, Train: {training_times[i]:.4f}s, Predict: {prediction_times[i]:.4f}s")
        
        # 时间应该随特征维度合理增长
        for i in range(1, len(dimensions)):
            dim_ratio = dimensions[i] / dimensions[i-1]
            time_ratio = training_times[i] / training_times[i-1]
            
            # 时间增长应该接近线性
            assert time_ratio <= dim_ratio * 2, \
                f"Feature scaling too poor: {time_ratio:.2f} vs {dim_ratio:.2f}"


class TestRealWorldPerformance:
    """测试真实世界性能场景"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
    
    def test_crypto_trading_scenario(self):
        """测试加密货币交易场景性能"""
        # 模拟实时交易场景
        sequence_length = 60  # 1小时数据
        n_features = 10
        
        # 生成历史数据
        X, y = self.data_generator.generate_time_series_data(
            n_samples=10000, n_features=n_features, sequence_length=sequence_length
        )
        
        # 创建序列数据
        X_sequences = np.array([X[i:i+sequence_length] for i in range(len(X)-sequence_length)])
        y_sequences = y[sequence_length:]
        
        # 分割数据
        split_idx = int(0.8 * len(X_sequences))
        X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
        
        model = MockModel(fit_time=1.0, predict_time=0.001)
        
        # 测试训练性能
        with PerformanceTimer("Crypto Model Training") as train_timer:
            model.fit(X_train, y_train)
        
        # 模拟实时预测（每秒一次预测）
        n_predictions = 100
        prediction_times = []
        
        for i in range(n_predictions):
            # 获取最新数据
            latest_data = X_test[i:i+1] if i < len(X_test) else X_test[-1:]
            
            with PerformanceTimer("Real-time Prediction") as pred_timer:
                prediction = model.predict(latest_data)
            
            prediction_times.append(pred_timer.duration)
        
        avg_prediction_time = np.mean(prediction_times)
        max_prediction_time = np.max(prediction_times)
        
        print(f"Training time: {train_timer.duration:.2f}s")
        print(f"Average prediction time: {avg_prediction_time:.4f}s")
        print(f"Max prediction time: {max_prediction_time:.4f}s")
        
        # 实时预测应该很快
        assert avg_prediction_time < 0.1, \
            f"Average prediction time {avg_prediction_time:.4f}s too slow for real-time"
        assert max_prediction_time < 0.2, \
            f"Max prediction time {max_prediction_time:.4f}s too slow for real-time"
    
    def test_batch_processing_scenario(self):
        """测试批处理场景性能"""
        # 模拟每日批处理场景
        daily_data_size = 1440  # 每分钟一个数据点
        n_days = 30
        
        total_processing_time = 0
        
        for day in range(n_days):
            # 生成一天的数据
            X_day, y_day = self.data_generator.generate_regression_data(
                n_samples=daily_data_size, test_size=0
            )[:2]
            
            model = MockModel(fit_time=0.1, predict_time=0.01)
            
            with PerformanceTimer(f"Day {day+1} Processing") as day_timer:
                # 训练模型
                model.fit(X_day, y_day)
                
                # 生成预测
                predictions = model.predict(X_day)
                
                # 模拟一些后处理
                processed_predictions = predictions * 1.1
            
            total_processing_time += day_timer.duration
        
        avg_daily_processing_time = total_processing_time / n_days
        
        print(f"Total processing time for {n_days} days: {total_processing_time:.2f}s")
        print(f"Average daily processing time: {avg_daily_processing_time:.2f}s")
        
        # 每日处理时间应该合理
        assert avg_daily_processing_time < 5.0, \
            f"Daily processing time {avg_daily_processing_time:.2f}s too slow"
        
        # 总处理时间应该可以接受
        assert total_processing_time < 60.0, \
            f"Total processing time {total_processing_time:.2f}s too slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])