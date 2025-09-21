"""
PyTorch模拟模块
用于在没有安装PyTorch的环境中运行测试
"""

from unittest.mock import MagicMock
import numpy as np


class MockTensor:
    """模拟PyTorch张量"""
    
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
    
    def numpy(self):
        return self.data
    
    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]
    
    def __getitem__(self, key):
        return MockTensor(self.data[key])
    
    def __repr__(self):
        return f"MockTensor({self.data})"


class MockModule:
    """模拟PyTorch模块"""
    
    def __init__(self):
        self.training = True
        self.parameters_list = []
    
    def train(self, mode=True):
        self.training = mode
        return self
    
    def eval(self):
        self.training = False
        return self
    
    def parameters(self):
        return iter(self.parameters_list)
    
    def forward(self, x):
        return x


class MockOptimizer:
    """模拟PyTorch优化器"""
    
    def __init__(self, parameters, lr=0.001):
        self.param_groups = [{'params': list(parameters), 'lr': lr}]
    
    def step(self):
        pass
    
    def zero_grad(self):
        pass


class MockLoss:
    """模拟PyTorch损失函数"""
    
    def __call__(self, pred, target):
        return MockTensor(np.random.random())


# 创建模拟的torch模块
torch = MagicMock()
torch.Tensor = MockTensor
torch.tensor = lambda x: MockTensor(x)
torch.zeros = lambda *args: MockTensor(np.zeros(args))
torch.ones = lambda *args: MockTensor(np.ones(args))
torch.randn = lambda *args: MockTensor(np.random.randn(*args))
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

# 神经网络模块
torch.nn = MagicMock()
torch.nn.Module = MockModule
torch.nn.Linear = lambda in_features, out_features: MockModule()
torch.nn.ReLU = lambda: MockModule()
torch.nn.Dropout = lambda p=0.5: MockModule()
torch.nn.LSTM = lambda *args, **kwargs: MockModule()
torch.nn.MSELoss = MockLoss
torch.nn.CrossEntropyLoss = MockLoss

# 优化器
torch.optim = MagicMock()
torch.optim.Adam = MockOptimizer
torch.optim.SGD = MockOptimizer

# 功能函数
torch.nn.functional = MagicMock()
torch.nn.functional.relu = lambda x: x
torch.nn.functional.softmax = lambda x, dim=-1: x

# 数据加载
torch.utils = MagicMock()
torch.utils.data = MagicMock()
torch.utils.data.DataLoader = lambda dataset, **kwargs: iter([])
torch.utils.data.Dataset = object

# 分布模块
torch.distributions = MagicMock()
torch.distributions.Categorical = lambda logits: MagicMock()
torch.distributions.Normal = lambda loc, scale: MagicMock()