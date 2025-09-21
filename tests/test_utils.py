"""
测试工具模块

提供测试所需的通用工具和数据生成器
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
import tempfile
import shutil
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import warnings

# 忽略测试中的警告
warnings.filterwarnings('ignore')


class TestDataGenerator:
    """测试数据生成器"""
    
    def __init__(self, random_state: int = 42):
        """
        初始化测试数据生成器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_classification_data(self, n_samples: int = 1000, 
                                   n_features: int = 20,
                                   n_classes: int = 2,
                                   n_informative: Optional[int] = None,
                                   test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成分类测试数据
        
        Args:
            n_samples: 样本数量
            n_features: 特征数量
            n_classes: 类别数量
            n_informative: 有用特征数量
            test_size: 测试集比例
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if n_informative is None:
            n_informative = min(n_features, max(2, n_features // 2))
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=n_classes,
            random_state=self.random_state
        )
        
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
    
    def generate_regression_data(self, n_samples: int = 1000,
                               n_features: int = 20,
                               noise: float = 0.1,
                               test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成回归测试数据
        
        Args:
            n_samples: 样本数量
            n_features: 特征数量
            noise: 噪声水平
            test_size: 测试集比例
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=self.random_state
        )
        
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
    
    def generate_time_series_data(self, n_samples: int = 1000,
                                n_features: int = 5,
                                sequence_length: int = 50,
                                trend: bool = True,
                                seasonality: bool = True,
                                noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成时间序列测试数据
        
        Args:
            n_samples: 样本数量
            n_features: 特征数量
            sequence_length: 序列长度
            trend: 是否包含趋势
            seasonality: 是否包含季节性
            noise: 噪声水平
            
        Returns:
            X, y (时间序列数据)
        """
        # 生成时间索引
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # 基础信号
        t = np.arange(n_samples)
        signal = np.zeros(n_samples)
        
        # 添加趋势
        if trend:
            signal += 0.01 * t
        
        # 添加季节性
        if seasonality:
            signal += 2 * np.sin(2 * np.pi * t / 365.25)  # 年度季节性
            signal += 0.5 * np.sin(2 * np.pi * t / 7)     # 周度季节性
        
        # 添加噪声
        signal += np.random.normal(0, noise, n_samples)
        
        # 生成多个特征
        X = np.zeros((n_samples, n_features))
        for i in range(n_features):
            # 每个特征都是基础信号的变体
            feature_signal = signal + np.random.normal(0, noise * 0.5, n_samples)
            X[:, i] = feature_signal
        
        # 目标变量是所有特征的加权和加上一些噪声
        weights = np.random.uniform(0.5, 2.0, n_features)
        y = np.dot(X, weights) + np.random.normal(0, noise, n_samples)
        
        return X, y
    
    def generate_crypto_like_data(self, n_samples: int = 1000,
                                start_price: float = 50000.0,
                                volatility: float = 0.02) -> pd.DataFrame:
        """
        生成类似加密货币的价格数据
        
        Args:
            n_samples: 样本数量
            start_price: 起始价格
            volatility: 波动率
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
        
        # 生成价格随机游走
        returns = np.random.normal(0, volatility, n_samples)
        prices = [start_price]
        
        for i in range(1, n_samples):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.01))  # 确保价格为正
        
        prices = np.array(prices)
        
        # 生成OHLC数据
        data = []
        for i in range(n_samples):
            # 当前价格作为收盘价
            close = prices[i]
            
            # 生成开盘价（基于前一个收盘价）
            if i == 0:
                open_price = close
            else:
                open_price = prices[i-1]
            
            # 生成高低价
            high_low_range = abs(close - open_price) * np.random.uniform(1.0, 3.0)
            high = max(open_price, close) + high_low_range * np.random.uniform(0, 0.5)
            low = min(open_price, close) - high_low_range * np.random.uniform(0, 0.5)
            
            # 确保价格关系正确
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            low = max(low, 0.01)  # 确保价格为正
            
            # 生成成交量
            volume = np.random.lognormal(10, 1)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)


class TestEnvironment:
    """测试环境管理器"""
    
    def __init__(self):
        """初始化测试环境"""
        self.temp_dirs = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置测试日志"""
        logger = logging.getLogger('test_logger')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_temp_dir(self) -> Path:
        """创建临时目录"""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup(self):
        """清理测试环境"""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class ModelTestHelper:
    """模型测试辅助工具"""
    
    @staticmethod
    def assert_model_interface(model: Any, required_methods: List[str] = None):
        """
        断言模型具有必要的接口
        
        Args:
            model: 模型对象
            required_methods: 必需的方法列表
        """
        if required_methods is None:
            required_methods = ['fit', 'predict']
        
        for method in required_methods:
            assert hasattr(model, method), f"Model missing required method: {method}"
            assert callable(getattr(model, method)), f"Model {method} is not callable"
    
    @staticmethod
    def assert_prediction_shape(predictions: np.ndarray, expected_shape: Tuple[int, ...]):
        """
        断言预测结果的形状
        
        Args:
            predictions: 预测结果
            expected_shape: 期望的形状
        """
        assert predictions.shape == expected_shape, \
            f"Prediction shape {predictions.shape} != expected {expected_shape}"
    
    @staticmethod
    def assert_prediction_range(predictions: np.ndarray, min_val: float = None, 
                              max_val: float = None):
        """
        断言预测结果的范围
        
        Args:
            predictions: 预测结果
            min_val: 最小值
            max_val: 最大值
        """
        if min_val is not None:
            assert np.all(predictions >= min_val), \
                f"Predictions contain values below {min_val}"
        
        if max_val is not None:
            assert np.all(predictions <= max_val), \
                f"Predictions contain values above {max_val}"
    
    @staticmethod
    def assert_no_nan_inf(array: np.ndarray, name: str = "array"):
        """
        断言数组中没有NaN或无穷值
        
        Args:
            array: 数组
            name: 数组名称
        """
        assert not np.any(np.isnan(array)), f"{name} contains NaN values"
        assert not np.any(np.isinf(array)), f"{name} contains infinite values"
    
    @staticmethod
    def assert_deterministic(model: Any, X: np.ndarray, n_runs: int = 3):
        """
        断言模型预测的确定性
        
        Args:
            model: 模型对象
            X: 输入数据
            n_runs: 运行次数
        """
        predictions = []
        for _ in range(n_runs):
            pred = model.predict(X)
            predictions.append(pred)
        
        # 检查所有预测是否相同
        for i in range(1, n_runs):
            np.testing.assert_array_equal(
                predictions[0], predictions[i],
                err_msg="Model predictions are not deterministic"
            )


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self, name: str = "Operation"):
        """
        初始化计时器
        
        Args:
            name: 操作名称
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time
        print(f"{self.name} took {duration.total_seconds():.4f} seconds")
    
    @property
    def duration(self) -> float:
        """获取持续时间（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class MockModel:
    """模拟模型（用于测试）"""
    
    def __init__(self, prediction_type: str = 'regression', 
                 prediction_value: float = 1.0,
                 fit_time: float = 0.1,
                 predict_time: float = 0.01):
        """
        初始化模拟模型
        
        Args:
            prediction_type: 预测类型 ('regression' 或 'classification')
            prediction_value: 预测值
            fit_time: 训练时间（秒）
            predict_time: 预测时间（秒）
        """
        self.prediction_type = prediction_type
        self.prediction_value = prediction_value
        self.fit_time = fit_time
        self.predict_time = predict_time
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """模拟训练"""
        import time
        time.sleep(self.fit_time)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """模拟预测"""
        import time
        time.sleep(self.predict_time)
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = len(X)
        
        if self.prediction_type == 'regression':
            return np.full(n_samples, self.prediction_value)
        else:  # classification
            return np.full(n_samples, int(self.prediction_value))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """模拟概率预测"""
        if self.prediction_type != 'classification':
            raise AttributeError("predict_proba only available for classification")
        
        n_samples = len(X)
        # 返回二分类概率
        proba = np.zeros((n_samples, 2))
        proba[:, 1] = 0.7  # 正类概率
        proba[:, 0] = 0.3  # 负类概率
        return proba


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, 
                  tolerance: float = 1e-6, name: str = "arrays") -> bool:
    """
    比较两个数组是否在容差范围内相等
    
    Args:
        arr1: 数组1
        arr2: 数组2
        tolerance: 容差
        name: 数组名称
        
    Returns:
        是否相等
    """
    try:
        np.testing.assert_allclose(arr1, arr2, atol=tolerance, rtol=tolerance)
        return True
    except AssertionError as e:
        print(f"Arrays {name} are not equal within tolerance {tolerance}: {e}")
        return False


def create_test_config() -> Dict[str, Any]:
    """创建测试配置"""
    return {
        'random_seed': 42,
        'test_data_size': 100,
        'tolerance': 1e-6,
        'timeout': 30,
        'n_jobs': 1,
        'verbose': False
    }


def skip_if_no_gpu():
    """如果没有GPU则跳过测试的装饰器"""
    import pytest
    
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    
    return pytest.mark.skipif(not has_gpu, reason="GPU not available")


def skip_if_slow():
    """跳过慢速测试的装饰器"""
    import pytest
    import os
    
    return pytest.mark.skipif(
        os.environ.get('SKIP_SLOW_TESTS', '').lower() in ('1', 'true', 'yes'),
        reason="Slow tests skipped"
    )