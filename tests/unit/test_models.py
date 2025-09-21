"""
模型单元测试

测试各种机器学习模型的基本功能
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.test_utils import (
    TestDataGenerator, ModelTestHelper, MockModel, 
    PerformanceTimer, compare_arrays
)
from src.models.registry import ModelRegistry
from src.models.base import BaseModel


class TestBaseModel:
    """测试基础模型类"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.data_generator.generate_regression_data(n_samples=100)
    
    def test_base_model_interface(self):
        """测试基础模型接口"""
        # 创建一个简单的模型实现
        class SimpleModel(BaseModel):
            def __init__(self):
                super().__init__()
                self.model = None
            
            def fit(self, X, y):
                self.model = "fitted"
                return self
            
            def predict(self, X):
                if self.model is None:
                    raise ValueError("Model not fitted")
                return np.zeros(len(X))
        
        model = SimpleModel()
        
        # 测试接口存在
        ModelTestHelper.assert_model_interface(model)
        
        # 测试训练
        model.fit(self.X_train, self.y_train)
        
        # 测试预测
        predictions = model.predict(self.X_test)
        ModelTestHelper.assert_prediction_shape(predictions, (len(self.X_test),))
        ModelTestHelper.assert_no_nan_inf(predictions, "predictions")
    
    def test_model_validation(self):
        """测试模型验证"""
        model = MockModel()
        
        # 测试未训练时预测应该失败
        with pytest.raises(ValueError):
            model.predict(self.X_test)
        
        # 训练后应该可以预测
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        ModelTestHelper.assert_no_nan_inf(predictions)


class TestModelRegistry:
    """测试模型注册表"""
    
    def setup_method(self):
        """测试前设置"""
        # 清空注册表
        ModelRegistry._models.clear()
    
    def test_model_registration(self):
        """测试模型注册"""
        @ModelRegistry.register('test_model')
        class TestModel:
            pass
        
        # 检查模型是否注册成功
        assert 'test_model' in ModelRegistry._models
        assert ModelRegistry._models['test_model'] == TestModel
    
    def test_model_creation(self):
        """测试模型创建"""
        @ModelRegistry.register('test_model')
        class TestModel:
            def __init__(self, param1=1, param2=2):
                self.param1 = param1
                self.param2 = param2
        
        # 测试无参数创建
        model = ModelRegistry.create_model('test_model')
        assert isinstance(model, TestModel)
        assert model.param1 == 1
        assert model.param2 == 2
        
        # 测试带参数创建
        model = ModelRegistry.create_model('test_model', param1=10, param2=20)
        assert model.param1 == 10
        assert model.param2 == 20
    
    def test_unknown_model(self):
        """测试未知模型"""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.create_model('unknown_model')
    
    def test_list_models(self):
        """测试列出模型"""
        @ModelRegistry.register('model1')
        class Model1:
            pass
        
        @ModelRegistry.register('model2')
        class Model2:
            pass
        
        models = ModelRegistry.list_models()
        assert 'model1' in models
        assert 'model2' in models


class TestTraditionalModels:
    """测试传统机器学习模型"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
    
    def test_linear_regression(self):
        """测试线性回归模型"""
        try:
            from src.models.traditional.linear_models import LinearRegressionModel
            
            # 生成回归数据
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_regression_data(n_samples=100)
            
            # 创建和训练模型
            model = LinearRegressionModel()
            ModelTestHelper.assert_model_interface(model)
            
            model.fit(X_train, y_train)
            
            # 测试预测
            predictions = model.predict(X_test)
            ModelTestHelper.assert_prediction_shape(predictions, (len(X_test),))
            ModelTestHelper.assert_no_nan_inf(predictions)
            
            # 测试评估
            score = model.score(X_test, y_test)
            assert isinstance(score, float)
            assert 0 <= score <= 1  # R²分数应该在合理范围内
            
        except ImportError:
            pytest.skip("LinearRegressionModel not available")
    
    def test_random_forest(self):
        """测试随机森林模型"""
        try:
            from src.models.traditional.ensemble_models import RandomForestModel
            
            # 生成分类数据
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_classification_data(n_samples=100)
            
            # 创建和训练模型
            model = RandomForestModel(n_estimators=10, random_state=42)
            ModelTestHelper.assert_model_interface(model)
            
            model.fit(X_train, y_train)
            
            # 测试预测
            predictions = model.predict(X_test)
            ModelTestHelper.assert_prediction_shape(predictions, (len(X_test),))
            ModelTestHelper.assert_no_nan_inf(predictions)
            
            # 测试概率预测
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)
                assert probabilities.shape == (len(X_test), 2)
                ModelTestHelper.assert_no_nan_inf(probabilities)
                
                # 概率应该在[0,1]范围内且每行和为1
                ModelTestHelper.assert_prediction_range(probabilities, 0.0, 1.0)
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
            
        except ImportError:
            pytest.skip("RandomForestModel not available")
    
    def test_svm_model(self):
        """测试SVM模型"""
        try:
            from src.models.traditional.svm_models import SVMModel
            
            # 生成小规模分类数据（SVM训练较慢）
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_classification_data(n_samples=50)
            
            # 创建和训练模型
            model = SVMModel(kernel='linear', random_state=42)
            ModelTestHelper.assert_model_interface(model)
            
            with PerformanceTimer("SVM Training"):
                model.fit(X_train, y_train)
            
            # 测试预测
            predictions = model.predict(X_test)
            ModelTestHelper.assert_prediction_shape(predictions, (len(X_test),))
            ModelTestHelper.assert_no_nan_inf(predictions)
            
        except ImportError:
            pytest.skip("SVMModel not available")


class TestDeepLearningModels:
    """测试深度学习模型"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion="1.0"),
        reason="PyTorch not available"
    )
    def test_neural_network(self):
        """测试神经网络模型"""
        try:
            from src.models.deep_learning.neural_networks import NeuralNetworkModel
            
            # 生成回归数据
            X_train, X_test, y_train, y_test = \
                self.data_generator.generate_regression_data(n_samples=100)
            
            # 创建和训练模型
            model = NeuralNetworkModel(
                input_size=X_train.shape[1],
                hidden_sizes=[32, 16],
                output_size=1,
                epochs=5  # 少量epoch用于测试
            )
            ModelTestHelper.assert_model_interface(model)
            
            with PerformanceTimer("Neural Network Training"):
                model.fit(X_train, y_train)
            
            # 测试预测
            predictions = model.predict(X_test)
            ModelTestHelper.assert_prediction_shape(predictions, (len(X_test),))
            ModelTestHelper.assert_no_nan_inf(predictions)
            
        except ImportError:
            pytest.skip("NeuralNetworkModel not available")
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion="1.0"),
        reason="PyTorch not available"
    )
    def test_transformer_model(self):
        """测试Transformer模型"""
        try:
            from src.models.deep_learning.transformer_models import TransformerModel
            
            # 生成时间序列数据
            X, y = self.data_generator.generate_time_series_data(
                n_samples=200, sequence_length=20
            )
            
            # 重塑数据为序列格式
            X_seq = np.array([X[i:i+20] for i in range(len(X)-20)])
            y_seq = y[20:]
            
            # 分割数据
            split_idx = int(0.8 * len(X_seq))
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            # 创建和训练模型
            model = TransformerModel(
                input_size=X_train.shape[-1],
                d_model=32,
                nhead=4,
                num_layers=2,
                epochs=3  # 少量epoch用于测试
            )
            ModelTestHelper.assert_model_interface(model)
            
            with PerformanceTimer("Transformer Training"):
                model.fit(X_train, y_train)
            
            # 测试预测
            predictions = model.predict(X_test)
            ModelTestHelper.assert_prediction_shape(predictions, (len(X_test),))
            ModelTestHelper.assert_no_nan_inf(predictions)
            
        except ImportError:
            pytest.skip("TransformerModel not available")


class TestModelPerformance:
    """测试模型性能"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
    
    def test_training_speed(self):
        """测试训练速度"""
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=1000)
        
        model = MockModel(fit_time=0.1)
        
        with PerformanceTimer("Mock Model Training") as timer:
            model.fit(X_train, y_train)
        
        # 训练时间应该接近设定值
        assert 0.05 <= timer.duration <= 0.2
    
    def test_prediction_speed(self):
        """测试预测速度"""
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=1000)
        
        model = MockModel(predict_time=0.01)
        model.fit(X_train, y_train)
        
        with PerformanceTimer("Mock Model Prediction") as timer:
            predictions = model.predict(X_test)
        
        # 预测时间应该接近设定值
        assert timer.duration <= 0.05
        assert len(predictions) == len(X_test)
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建大量数据
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=10000)
        
        model = MockModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内（小于100MB）
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB"


class TestModelDeterminism:
    """测试模型确定性"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
    
    def test_reproducible_results(self):
        """测试结果可重现性"""
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=100)
        
        # 使用相同随机种子训练两个模型
        model1 = MockModel(prediction_value=1.0)
        model2 = MockModel(prediction_value=1.0)
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        
        # 预测结果应该相同
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_deterministic_prediction(self):
        """测试预测确定性"""
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=100)
        
        model = MockModel()
        model.fit(X_train, y_train)
        
        # 多次预测应该得到相同结果
        ModelTestHelper.assert_deterministic(model, X_test, n_runs=3)


class TestModelEdgeCases:
    """测试模型边界情况"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_generator = TestDataGenerator()
    
    def test_empty_data(self):
        """测试空数据"""
        model = MockModel()
        
        # 空训练数据
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            model.fit(X_empty, y_empty)
    
    def test_single_sample(self):
        """测试单样本数据"""
        model = MockModel()
        
        # 单样本训练数据
        X_single = np.array([[1, 2, 3, 4, 5]])
        y_single = np.array([1])
        
        model.fit(X_single, y_single)
        
        # 单样本预测
        prediction = model.predict(X_single)
        assert len(prediction) == 1
    
    def test_mismatched_dimensions(self):
        """测试维度不匹配"""
        X_train, X_test, y_train, y_test = \
            self.data_generator.generate_regression_data(n_samples=100)
        
        model = MockModel()
        model.fit(X_train, y_train)
        
        # 特征数量不匹配的测试数据
        X_wrong = np.random.randn(10, X_train.shape[1] + 1)
        
        # 应该处理维度不匹配（或抛出明确错误）
        try:
            predictions = model.predict(X_wrong)
            # 如果没有抛出错误，检查预测形状
            assert len(predictions) == len(X_wrong)
        except (ValueError, IndexError):
            # 抛出错误也是可接受的
            pass
    
    def test_extreme_values(self):
        """测试极值数据"""
        model = MockModel()
        
        # 极大值
        X_large = np.full((10, 5), 1e10)
        y_large = np.full(10, 1e10)
        
        # 极小值
        X_small = np.full((10, 5), 1e-10)
        y_small = np.full(10, 1e-10)
        
        # 模型应该能处理极值（或给出明确错误）
        try:
            model.fit(X_large, y_large)
            pred_large = model.predict(X_large)
            ModelTestHelper.assert_no_nan_inf(pred_large)
            
            model.fit(X_small, y_small)
            pred_small = model.predict(X_small)
            ModelTestHelper.assert_no_nan_inf(pred_small)
        except (ValueError, OverflowError, UnderflowError):
            # 抛出数值错误也是可接受的
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])