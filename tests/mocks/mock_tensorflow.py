"""
TensorFlow模拟模块
用于在没有安装TensorFlow的环境中运行测试
"""

from unittest.mock import MagicMock
import numpy as np


class MockTensor:
    """模拟TensorFlow张量"""
    
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
    
    def numpy(self):
        return self.data
    
    def __getitem__(self, key):
        return MockTensor(self.data[key])
    
    def __repr__(self):
        return f"MockTensor({self.data})"


class MockModel:
    """模拟TensorFlow模型"""
    
    def __init__(self):
        self.layers = []
        self.compiled = False
    
    def add(self, layer):
        self.layers.append(layer)
    
    def compile(self, optimizer=None, loss=None, metrics=None):
        self.compiled = True
    
    def fit(self, x, y, epochs=1, batch_size=32, validation_data=None, **kwargs):
        return MagicMock()
    
    def predict(self, x):
        return np.random.random((len(x), 1))
    
    def evaluate(self, x, y):
        return [0.5, 0.8]  # loss, accuracy
    
    def save(self, filepath):
        pass
    
    def load_weights(self, filepath):
        pass


class MockLayer:
    """模拟TensorFlow层"""
    
    def __init__(self, *args, **kwargs):
        pass


# 创建模拟的tensorflow模块
tensorflow = MagicMock()
tensorflow.constant = lambda x: MockTensor(x)
tensorflow.zeros = lambda shape: MockTensor(np.zeros(shape))
tensorflow.ones = lambda shape: MockTensor(np.ones(shape))
tensorflow.random.normal = lambda shape: MockTensor(np.random.randn(*shape))

# Keras模块
tensorflow.keras = MagicMock()
tensorflow.keras.Model = MockModel
tensorflow.keras.Sequential = MockModel

# 层
tensorflow.keras.layers = MagicMock()
tensorflow.keras.layers.Dense = MockLayer
tensorflow.keras.layers.LSTM = MockLayer
tensorflow.keras.layers.Dropout = MockLayer
tensorflow.keras.layers.Input = MockLayer

# 优化器
tensorflow.keras.optimizers = MagicMock()
tensorflow.keras.optimizers.Adam = lambda **kwargs: MagicMock()
tensorflow.keras.optimizers.SGD = lambda **kwargs: MagicMock()

# 损失函数
tensorflow.keras.losses = MagicMock()
tensorflow.keras.losses.MeanSquaredError = lambda: MagicMock()
tensorflow.keras.losses.SparseCategoricalCrossentropy = lambda: MagicMock()

# 指标
tensorflow.keras.metrics = MagicMock()
tensorflow.keras.metrics.Accuracy = lambda: MagicMock()
tensorflow.keras.metrics.MeanAbsoluteError = lambda: MagicMock()

# 数据处理
tensorflow.data = MagicMock()
tensorflow.data.Dataset = MagicMock()

# 版本信息
tensorflow.__version__ = "2.0.0"

# 别名
tf = tensorflow