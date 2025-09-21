"""
训练模块单元测试

测试训练器、数据加载器、回调函数等
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.test_utils import (
    TestDataGenerator, TestEnvironment, MockModel,
    PerformanceTimer, ModelTestHelper
)


class TestDataLoader:
    """测试数据加载器"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
        self.test_env = TestEnvironment()
    
    def teardown_method(self):
        """测试后清理"""
        self.test_env.cleanup()
    
    def test_time_series_dataset(self):
        """测试时间序列数据集"""
        try:
            from src.models.training.data_loader import TimeSeriesDataset
            
            # 生成时间序列数据
            X, y = self.data_generator.generate_time_series_data(
                n_samples=100, sequence_length=10
            )
            
            # 创建数据集
            dataset = TimeSeriesDataset(X, y, sequence_length=10)
            
            # 测试数据集长度
            assert len(dataset) == len(X) - 10 + 1
            
            # 测试数据获取
            sample_x, sample_y = dataset[0]
            assert sample_x.shape == (10, X.shape[1])
            assert isinstance(sample_y, (int, float, np.number))
            
            # 测试边界情况
            last_x, last_y = dataset[len(dataset) - 1]
            assert last_x.shape == (10, X.shape[1])
            
        except ImportError:
            pytest.skip("TimeSeriesDataset not available")
    
    def test_crypto_data_loader(self):
        """测试加密货币数据加载器"""
        try:
            from src.models.training.data_loader import CryptoDataLoader
            
            # 生成模拟加密货币数据
            crypto_data = self.data_generator.generate_crypto_like_data(n_samples=1000)
            
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
            
            # 测试数据加载
            train_loader, val_loader, test_loader = loader.get_data_loaders(
                batch_size=32,
                train_ratio=0.7,
                val_ratio=0.15
            )
            
            # 检查数据加载器
            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None
            
            # 测试批次数据
            for batch_x, batch_y in train_loader:
                assert batch_x.shape[0] <= 32  # 批次大小
                assert batch_x.shape[1] == 24  # 序列长度
                assert len(batch_y) == batch_x.shape[0]
                break  # 只测试第一个批次
            
        except ImportError:
            pytest.skip("CryptoDataLoader not available")
    
    def test_data_augmentation(self):
        """测试数据增强"""
        try:
            from src.models.training.data_loader import DataAugmentation
            
            # 生成测试数据
            X = np.random.randn(100, 10)
            
            # 创建数据增强器
            augmenter = DataAugmentation()
            
            # 测试添加噪声
            X_noise = augmenter.add_noise(X, noise_level=0.1)
            assert X_noise.shape == X.shape
            assert not np.array_equal(X, X_noise)  # 应该有差异
            
            # 测试时间扭曲
            X_warp = augmenter.time_warp(X, sigma=0.2)
            assert X_warp.shape == X.shape
            
            # 测试幅度扭曲
            X_magnitude = augmenter.magnitude_warp(X, sigma=0.2)
            assert X_magnitude.shape == X.shape
            
            # 测试窗口切片
            X_slice = augmenter.window_slice(X, reduce_ratio=0.8)
            assert X_slice.shape[0] == X.shape[0]
            assert X_slice.shape[1] <= X.shape[1]
            
        except ImportError:
            pytest.skip("DataAugmentation not available")


class TestTrainer:
    """测试训练器"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.data_generator.generate_regression_data(n_samples=100)
    
    def test_base_trainer(self):
        """测试基础训练器"""
        try:
            from src.models.training.trainer import BaseTrainer
            
            # 基础训练器是抽象类，不能直接实例化
            with pytest.raises(TypeError):
                BaseTrainer()
            
        except ImportError:
            pytest.skip("BaseTrainer not available")
    
    def test_sklearn_trainer(self):
        """测试Sklearn训练器"""
        try:
            from src.models.training.trainer import SklearnTrainer
            from sklearn.linear_model import LinearRegression
            
            # 创建模型和训练器
            model = LinearRegression()
            trainer = SklearnTrainer(model)
            
            # 测试训练
            trainer.fit(self.X_train, self.y_train)
            
            # 测试预测
            predictions = trainer.predict(self.X_test)
            assert len(predictions) == len(self.X_test)
            ModelTestHelper.assert_no_nan_inf(predictions)
            
            # 测试评估
            score = trainer.evaluate(self.X_test, self.y_test)
            assert isinstance(score, dict)
            assert 'score' in score
            
        except ImportError:
            pytest.skip("SklearnTrainer not available")
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion="1.0"),
        reason="PyTorch not available"
    )
    def test_pytorch_trainer(self):
        """测试PyTorch训练器"""
        try:
            from src.models.training.trainer import PyTorchTrainer
            import torch
            import torch.nn as nn
            
            # 创建简单的PyTorch模型
            class SimpleNet(nn.Module):
                def __init__(self, input_size):
                    super().__init__()
                    self.linear = nn.Linear(input_size, 1)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = SimpleNet(self.X_train.shape[1])
            trainer = PyTorchTrainer(
                model=model,
                criterion=nn.MSELoss(),
                optimizer=torch.optim.Adam(model.parameters()),
                epochs=5
            )
            
            # 测试训练
            history = trainer.fit(self.X_train, self.y_train)
            assert isinstance(history, dict)
            assert 'train_loss' in history
            
            # 测试预测
            predictions = trainer.predict(self.X_test)
            assert len(predictions) == len(self.X_test)
            ModelTestHelper.assert_no_nan_inf(predictions)
            
        except ImportError:
            pytest.skip("PyTorchTrainer not available")
    
    def test_trainer_factory(self):
        """测试训练器工厂"""
        try:
            from src.models.training.trainer import TrainerFactory
            from sklearn.linear_model import LinearRegression
            
            # 测试创建Sklearn训练器
            model = LinearRegression()
            trainer = TrainerFactory.create_trainer('sklearn', model=model)
            
            assert trainer is not None
            assert hasattr(trainer, 'fit')
            assert hasattr(trainer, 'predict')
            
            # 测试未知训练器类型
            with pytest.raises(ValueError):
                TrainerFactory.create_trainer('unknown_trainer')
            
        except ImportError:
            pytest.skip("TrainerFactory not available")
    
    def test_training_pipeline(self):
        """测试训练管道"""
        try:
            from src.models.training.trainer import ModelTrainingPipeline
            
            # 创建模拟模型
            model = MockModel()
            
            # 创建训练管道
            pipeline = ModelTrainingPipeline(
                model=model,
                trainer_type='mock'
            )
            
            # 测试训练
            with patch.object(pipeline, '_create_trainer') as mock_create:
                mock_trainer = Mock()
                mock_trainer.fit.return_value = {'train_loss': [1.0, 0.5]}
                mock_trainer.predict.return_value = np.ones(len(self.X_test))
                mock_trainer.evaluate.return_value = {'score': 0.8}
                mock_create.return_value = mock_trainer
                
                results = pipeline.train(
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_val=self.X_test,
                    y_val=self.y_test
                )
                
                assert isinstance(results, dict)
                assert 'history' in results
                assert 'validation_score' in results
            
        except ImportError:
            pytest.skip("ModelTrainingPipeline not available")


class TestCallbacks:
    """测试回调函数"""
    
    def setup_method(self):
        """测试前设置"""
        self.test_env = TestEnvironment()
    
    def teardown_method(self):
        """测试后清理"""
        self.test_env.cleanup()
    
    def test_callback_manager(self):
        """测试回调管理器"""
        try:
            from src.models.training.callbacks import CallbackManager, Callback
            
            # 创建模拟回调
            callback1 = Mock(spec=Callback)
            callback2 = Mock(spec=Callback)
            
            # 创建回调管理器
            manager = CallbackManager([callback1, callback2])
            
            # 测试训练开始回调
            manager.on_train_begin({})
            callback1.on_train_begin.assert_called_once()
            callback2.on_train_begin.assert_called_once()
            
            # 测试epoch开始回调
            manager.on_epoch_begin(1, {})
            callback1.on_epoch_begin.assert_called_once_with(1, {})
            callback2.on_epoch_begin.assert_called_once_with(1, {})
            
            # 测试epoch结束回调
            manager.on_epoch_end(1, {'loss': 0.5})
            callback1.on_epoch_end.assert_called_once_with(1, {'loss': 0.5})
            callback2.on_epoch_end.assert_called_once_with(1, {'loss': 0.5})
            
        except ImportError:
            pytest.skip("CallbackManager not available")
    
    def test_early_stopping(self):
        """测试早停回调"""
        try:
            from src.models.training.callbacks import EarlyStopping
            
            # 创建早停回调
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                min_delta=0.01
            )
            
            # 模拟训练过程
            early_stopping.on_train_begin({})
            
            # 第1个epoch - 损失下降
            early_stopping.on_epoch_end(1, {'val_loss': 1.0})
            assert not early_stopping.should_stop
            
            # 第2个epoch - 损失继续下降
            early_stopping.on_epoch_end(2, {'val_loss': 0.8})
            assert not early_stopping.should_stop
            
            # 第3个epoch - 损失略微上升但在容差内
            early_stopping.on_epoch_end(3, {'val_loss': 0.81})
            assert not early_stopping.should_stop
            
            # 第4-6个epoch - 损失持续上升
            early_stopping.on_epoch_end(4, {'val_loss': 0.85})
            early_stopping.on_epoch_end(5, {'val_loss': 0.9})
            early_stopping.on_epoch_end(6, {'val_loss': 0.95})
            
            # 应该触发早停
            assert early_stopping.should_stop
            
        except ImportError:
            pytest.skip("EarlyStopping not available")
    
    def test_model_checkpoint(self):
        """测试模型检查点回调"""
        try:
            from src.models.training.callbacks import ModelCheckpoint
            
            # 创建临时目录
            temp_dir = self.test_env.create_temp_dir()
            checkpoint_path = temp_dir / "model_checkpoint.pkl"
            
            # 创建检查点回调
            checkpoint = ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True
            )
            
            # 模拟模型
            mock_model = Mock()
            checkpoint.model = mock_model
            
            # 模拟训练过程
            checkpoint.on_train_begin({})
            
            # 第1个epoch
            checkpoint.on_epoch_end(1, {'val_loss': 1.0})
            
            # 第2个epoch - 更好的损失
            checkpoint.on_epoch_end(2, {'val_loss': 0.8})
            
            # 第3个epoch - 更差的损失
            checkpoint.on_epoch_end(3, {'val_loss': 0.9})
            
            # 检查是否保存了最佳模型
            assert checkpoint.best_score == 0.8
            
        except ImportError:
            pytest.skip("ModelCheckpoint not available")
    
    def test_learning_rate_scheduler(self):
        """测试学习率调度器回调"""
        try:
            from src.models.training.callbacks import LearningRateScheduler
            
            # 创建学习率调度器
            def lr_schedule(epoch):
                return 0.1 * (0.9 ** epoch)
            
            scheduler = LearningRateScheduler(lr_schedule)
            
            # 模拟优化器
            mock_optimizer = Mock()
            mock_optimizer.param_groups = [{'lr': 0.1}]
            scheduler.optimizer = mock_optimizer
            
            # 测试学习率调度
            scheduler.on_epoch_begin(1, {})
            expected_lr = lr_schedule(1)
            assert mock_optimizer.param_groups[0]['lr'] == expected_lr
            
            scheduler.on_epoch_begin(2, {})
            expected_lr = lr_schedule(2)
            assert mock_optimizer.param_groups[0]['lr'] == expected_lr
            
        except ImportError:
            pytest.skip("LearningRateScheduler not available")
    
    def test_metrics_logger(self):
        """测试指标记录器回调"""
        try:
            from src.models.training.callbacks import MetricsLogger
            
            # 创建指标记录器
            logger = MetricsLogger()
            
            # 模拟训练过程
            logger.on_train_begin({})
            
            logger.on_epoch_end(1, {'loss': 1.0, 'val_loss': 1.2})
            logger.on_epoch_end(2, {'loss': 0.8, 'val_loss': 0.9})
            logger.on_epoch_end(3, {'loss': 0.6, 'val_loss': 0.7})
            
            # 检查记录的指标
            assert len(logger.history['loss']) == 3
            assert len(logger.history['val_loss']) == 3
            assert logger.history['loss'] == [1.0, 0.8, 0.6]
            assert logger.history['val_loss'] == [1.2, 0.9, 0.7]
            
        except ImportError:
            pytest.skip("MetricsLogger not available")


class TestValidator:
    """测试验证器"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
        self.X, self.y = self.data_generator.generate_regression_data(
            n_samples=200, test_size=0
        )[:2]  # 只取X和y
    
    def test_cross_validator(self):
        """测试交叉验证器"""
        try:
            from src.models.training.validator import CrossValidator
            
            # 创建交叉验证器
            validator = CrossValidator(cv=5, random_state=42)
            
            # 创建模拟模型
            model = MockModel()
            
            # 执行交叉验证
            scores = validator.validate(model, self.X, self.y)
            
            assert isinstance(scores, dict)
            assert 'scores' in scores
            assert 'mean_score' in scores
            assert 'std_score' in scores
            assert len(scores['scores']) == 5  # 5折交叉验证
            
        except ImportError:
            pytest.skip("CrossValidator not available")
    
    def test_time_series_validator(self):
        """测试时间序列验证器"""
        try:
            from src.models.training.validator import TimeSeriesValidator
            
            # 生成时间序列数据
            X, y = self.data_generator.generate_time_series_data(n_samples=200)
            
            # 创建时间序列验证器
            validator = TimeSeriesValidator(n_splits=5, test_size=0.2)
            
            # 创建模拟模型
            model = MockModel()
            
            # 执行时间序列验证
            scores = validator.validate(model, X, y)
            
            assert isinstance(scores, dict)
            assert 'scores' in scores
            assert 'mean_score' in scores
            assert len(scores['scores']) == 5
            
        except ImportError:
            pytest.skip("TimeSeriesValidator not available")
    
    def test_walk_forward_validator(self):
        """测试滚动窗口验证器"""
        try:
            from src.models.training.validator import WalkForwardValidator
            
            # 生成时间序列数据
            X, y = self.data_generator.generate_time_series_data(n_samples=200)
            
            # 创建滚动窗口验证器
            validator = WalkForwardValidator(
                initial_train_size=100,
                step_size=20,
                test_size=20
            )
            
            # 创建模拟模型
            model = MockModel()
            
            # 执行滚动窗口验证
            scores = validator.validate(model, X, y)
            
            assert isinstance(scores, dict)
            assert 'scores' in scores
            assert 'mean_score' in scores
            assert len(scores['scores']) > 0
            
        except ImportError:
            pytest.skip("WalkForwardValidator not available")
    
    def test_model_comparator(self):
        """测试模型比较器"""
        try:
            from src.models.training.validator import ModelComparator
            
            # 创建模型比较器
            comparator = ModelComparator()
            
            # 创建多个模拟模型
            models = {
                'model1': MockModel(prediction_value=1.0),
                'model2': MockModel(prediction_value=2.0),
                'model3': MockModel(prediction_value=3.0)
            }
            
            # 执行模型比较
            results = comparator.compare_models(models, self.X, self.y)
            
            assert isinstance(results, dict)
            assert len(results) == 3
            
            for model_name, result in results.items():
                assert 'scores' in result
                assert 'mean_score' in result
                assert 'std_score' in result
            
        except ImportError:
            pytest.skip("ModelComparator not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])