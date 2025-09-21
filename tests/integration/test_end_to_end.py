"""
端到端集成测试

测试完整的机器学习工作流程
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.test_utils import (
    TestDataGenerator, TestEnvironment, PerformanceTimer,
    ModelTestHelper, MockModel
)


class TestEndToEndWorkflow:
    """测试端到端工作流程"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
        self.test_env = TestEnvironment()
    
    def teardown_method(self):
        """测试后清理"""
        self.test_env.cleanup()
    
    def test_complete_ml_pipeline(self):
        """测试完整的机器学习管道"""
        # 1. 数据生成
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=500)
        
        # 2. 数据预处理（模拟）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 3. 模型训练
        model = MockModel()
        model.fit(X_train_scaled, y_train)
        
        # 4. 模型预测
        predictions = model.predict(X_test_scaled)
        
        # 5. 模型评估
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # 验证结果
        assert len(predictions) == len(y_test)
        assert isinstance(mse, float)
        assert isinstance(r2, float)
        ModelTestHelper.assert_no_nan_inf(predictions)
    
    def test_model_registry_integration(self):
        """测试模型注册表集成"""
        try:
            from src.models.registry import ModelRegistry
            
            # 清空注册表
            ModelRegistry._models.clear()
            
            # 注册测试模型
            @ModelRegistry.register('integration_test_model')
            class IntegrationTestModel:
                def __init__(self, param1=1):
                    self.param1 = param1
                    self.is_fitted = False
                
                def fit(self, X, y):
                    self.is_fitted = True
                    return self
                
                def predict(self, X):
                    if not self.is_fitted:
                        raise ValueError("Model not fitted")
                    return np.ones(len(X))
            
            # 通过注册表创建模型
            model = ModelRegistry.create_model('integration_test_model', param1=10)
            assert model.param1 == 10
            
            # 测试完整流程
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_regression_data(n_samples=100)
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            assert len(predictions) == len(X_test)
            assert all(pred == 1.0 for pred in predictions)
            
        except ImportError:
            pytest.skip("ModelRegistry not available")
    
    def test_training_evaluation_integration(self):
        """测试训练和评估集成"""
        try:
            from src.models.training.trainer import SklearnTrainer
            from src.models.evaluation.metrics import RegressionMetrics
            from sklearn.linear_model import LinearRegression
            
            # 生成数据
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_regression_data(n_samples=200)
            
            # 创建训练器
            model = LinearRegression()
            trainer = SklearnTrainer(model)
            
            # 训练模型
            trainer.fit(X_train, y_train)
            
            # 预测
            predictions = trainer.predict(X_test)
            
            # 评估
            metrics_calculator = RegressionMetrics()
            metrics = metrics_calculator.calculate_metrics(y_test, predictions)
            
            # 验证结果
            assert isinstance(metrics, dict)
            assert 'mse' in metrics
            assert 'rmse' in metrics
            assert 'mae' in metrics
            assert 'r2' in metrics
            
            for metric_name, metric_value in metrics.items():
                assert isinstance(metric_value, (int, float))
                assert not np.isnan(metric_value)
                assert not np.isinf(metric_value)
            
        except ImportError:
            pytest.skip("Training/Evaluation modules not available")
    
    def test_data_pipeline_integration(self):
        """测试数据管道集成"""
        try:
            from src.models.training.data_loader import CryptoDataLoader, DataAugmentation
            
            # 生成模拟加密货币数据
            crypto_data = self.data_generator.generate_crypto_like_data(n_samples=500)
            
            # 保存到临时文件
            temp_dir = self.test_env.create_temp_dir()
            data_file = temp_dir / "crypto_data.csv"
            crypto_data.to_csv(data_file, index=False)
            
            # 创建数据加载器
            loader = CryptoDataLoader(
                data_path=str(data_file),
                sequence_length=24,
                target_column='close'
            )
            
            # 获取数据加载器
            train_loader, val_loader, test_loader = loader.get_data_loaders(
                batch_size=32,
                train_ratio=0.7,
                val_ratio=0.15
            )
            
            # 测试数据增强
            augmenter = DataAugmentation()
            
            # 从训练加载器获取一个批次
            for batch_x, batch_y in train_loader:
                # 应用数据增强
                augmented_x = augmenter.add_noise(batch_x.numpy(), noise_level=0.1)
                
                # 验证增强后的数据
                assert augmented_x.shape == batch_x.shape
                ModelTestHelper.assert_no_nan_inf(augmented_x)
                break
            
        except ImportError:
            pytest.skip("Data pipeline modules not available")
    
    def test_callback_integration(self):
        """测试回调函数集成"""
        try:
            from src.models.training.callbacks import (
                CallbackManager, EarlyStopping, ModelCheckpoint, MetricsLogger
            )
            
            # 创建临时目录
            temp_dir = self.test_env.create_temp_dir()
            checkpoint_path = temp_dir / "model.pkl"
            
            # 创建回调函数
            early_stopping = EarlyStopping(monitor='val_loss', patience=3)
            checkpoint = ModelCheckpoint(str(checkpoint_path), monitor='val_loss')
            metrics_logger = MetricsLogger()
            
            # 创建回调管理器
            callback_manager = CallbackManager([
                early_stopping, checkpoint, metrics_logger
            ])
            
            # 模拟训练过程
            callback_manager.on_train_begin({})
            
            # 模拟多个epoch
            for epoch in range(1, 6):
                callback_manager.on_epoch_begin(epoch, {})
                
                # 模拟损失下降然后上升
                val_loss = 1.0 - 0.1 * epoch if epoch <= 3 else 0.7 + 0.1 * (epoch - 3)
                logs = {'loss': val_loss * 0.9, 'val_loss': val_loss}
                
                callback_manager.on_epoch_end(epoch, logs)
                
                # 检查是否应该早停
                if early_stopping.should_stop:
                    break
            
            callback_manager.on_train_end({})
            
            # 验证回调效果
            assert len(metrics_logger.history['loss']) > 0
            assert len(metrics_logger.history['val_loss']) > 0
            assert early_stopping.best_score is not None
            
        except ImportError:
            pytest.skip("Callback modules not available")


class TestModelComparison:
    """测试模型比较"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
    
    def test_multiple_model_comparison(self):
        """测试多模型比较"""
        try:
            from src.models.training.validator import ModelComparator
            
            # 生成数据
            X, y = self.data_generator.generate_regression_data(
                n_samples=200, test_size=0
            )[:2]
            
            # 创建多个模型
            models = {
                'model_1': MockModel(prediction_value=1.0),
                'model_2': MockModel(prediction_value=2.0),
                'model_3': MockModel(prediction_value=3.0)
            }
            
            # 创建比较器
            comparator = ModelComparator()
            
            # 执行比较
            with PerformanceTimer("Model Comparison"):
                results = comparator.compare_models(models, X, y)
            
            # 验证结果
            assert len(results) == 3
            
            for model_name, result in results.items():
                assert isinstance(result, dict)
                assert 'scores' in result
                assert 'mean_score' in result
                assert 'std_score' in result
                assert isinstance(result['mean_score'], (int, float))
                assert isinstance(result['std_score'], (int, float))
            
        except ImportError:
            pytest.skip("ModelComparator not available")
    
    def test_benchmark_integration(self):
        """测试基准测试集成"""
        try:
            from src.models.evaluation.benchmark import PerformanceBenchmark
            
            # 生成数据
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_regression_data(n_samples=300)
            
            # 创建模型
            model = MockModel(fit_time=0.1, predict_time=0.01)
            
            # 创建基准测试
            benchmark = PerformanceBenchmark()
            
            # 执行基准测试
            with PerformanceTimer("Benchmark Test"):
                results = benchmark.run_benchmark(
                    model, X_train, y_train, X_test, y_test
                )
            
            # 验证结果
            assert isinstance(results, dict)
            assert 'prediction_performance' in results
            assert 'training_performance' in results
            assert 'memory_usage' in results
            
            # 检查性能指标
            pred_perf = results['prediction_performance']
            assert 'accuracy_metrics' in pred_perf
            assert 'prediction_time' in pred_perf
            
            train_perf = results['training_performance']
            assert 'training_time' in train_perf
            
        except ImportError:
            pytest.skip("PerformanceBenchmark not available")


class TestDataIntegrity:
    """测试数据完整性"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
        self.test_env = TestEnvironment()
    
    def teardown_method(self):
        """测试后清理"""
        self.test_env.cleanup()
    
    def test_data_consistency(self):
        """测试数据一致性"""
        # 生成相同种子的数据应该一致
        generator1 = TestDataGenerator(random_state=42)
        generator2 = TestDataGenerator(random_state=42)
        
        X1, y1 = generator1.generate_regression_data(n_samples=100, test_size=0)[:2]
        X2, y2 = generator2.generate_regression_data(n_samples=100, test_size=0)[:2]
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_data_quality(self):
        """测试数据质量"""
        # 生成各种类型的数据
        X_reg, y_reg = self.data_generator.generate_regression_data(
            n_samples=200, test_size=0
        )[:2]
        
        X_cls, y_cls = self.data_generator.generate_classification_data(
            n_samples=200, test_size=0
        )[:2]
        
        X_ts, y_ts = self.data_generator.generate_time_series_data(n_samples=200)
        
        # 检查数据质量
        for X, y, name in [(X_reg, y_reg, "regression"), 
                          (X_cls, y_cls, "classification"),
                          (X_ts, y_ts, "time_series")]:
            
            # 检查无NaN和无穷值
            ModelTestHelper.assert_no_nan_inf(X, f"{name}_X")
            ModelTestHelper.assert_no_nan_inf(y, f"{name}_y")
            
            # 检查形状
            assert X.shape[0] == len(y), f"{name} X and y length mismatch"
            assert X.shape[0] > 0, f"{name} empty data"
            assert X.shape[1] > 0, f"{name} no features"
    
    def test_file_operations(self):
        """测试文件操作"""
        # 生成数据
        crypto_data = self.data_generator.generate_crypto_like_data(n_samples=100)
        
        # 保存到临时文件
        temp_dir = self.test_env.create_temp_dir()
        data_file = temp_dir / "test_data.csv"
        
        # 写入文件
        crypto_data.to_csv(data_file, index=False)
        assert data_file.exists()
        
        # 读取文件
        loaded_data = pd.read_csv(data_file)
        
        # 验证数据一致性
        pd.testing.assert_frame_equal(crypto_data, loaded_data)


class TestErrorHandling:
    """测试错误处理"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
    
    def test_invalid_model_operations(self):
        """测试无效模型操作"""
        model = MockModel()
        
        # 未训练时预测应该失败
        X_test = np.random.randn(10, 5)
        with pytest.raises(ValueError):
            model.predict(X_test)
    
    def test_data_validation(self):
        """测试数据验证"""
        model = MockModel()
        
        # 空数据
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])
        
        # 应该处理空数据（或抛出明确错误）
        try:
            model.fit(X_empty, y_empty)
        except (ValueError, IndexError):
            pass  # 预期的错误
    
    def test_resource_cleanup(self):
        """测试资源清理"""
        with TestEnvironment() as env:
            # 创建临时资源
            temp_dir = env.create_temp_dir()
            test_file = temp_dir / "test.txt"
            test_file.write_text("test content")
            
            assert test_file.exists()
        
        # 环境退出后资源应该被清理
        assert not test_file.exists()


class TestPerformanceIntegration:
    """测试性能集成"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
    
    def test_large_dataset_handling(self):
        """测试大数据集处理"""
        # 生成较大的数据集
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=2000)
        
        model = MockModel(fit_time=0.1, predict_time=0.05)
        
        # 测试训练性能
        with PerformanceTimer("Large Dataset Training") as train_timer:
            model.fit(X_train, y_train)
        
        # 测试预测性能
        with PerformanceTimer("Large Dataset Prediction") as pred_timer:
            predictions = model.predict(X_test)
        
        # 验证性能在合理范围内
        assert train_timer.duration < 1.0  # 训练时间应该小于1秒
        assert pred_timer.duration < 0.5   # 预测时间应该小于0.5秒
        assert len(predictions) == len(X_test)
    
    def test_memory_efficiency(self):
        """测试内存效率"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建和处理多个数据集
        for i in range(5):
            X, y = self.data_generator.generate_regression_data(
                n_samples=1000, test_size=0
            )[:2]
            
            model = MockModel()
            model.fit(X, y)
            predictions = model.predict(X)
            
            # 显式删除大对象
            del X, y, predictions
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内
        assert memory_increase < 50, f"Memory increased by {memory_increase:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])