"""
超参数空间定义

定义超参数的类型、范围和分布
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """参数类型枚举"""
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class Distribution(Enum):
    """分布类型枚举"""
    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"


class Parameter:
    """超参数定义类"""
    
    def __init__(self, 
                 name: str,
                 param_type: ParameterType,
                 low: Optional[Union[float, int]] = None,
                 high: Optional[Union[float, int]] = None,
                 choices: Optional[List[Any]] = None,
                 distribution: Distribution = Distribution.UNIFORM,
                 log: bool = False,
                 step: Optional[Union[float, int]] = None):
        """
        初始化参数
        
        Args:
            name: 参数名称
            param_type: 参数类型
            low: 最小值（数值类型）
            high: 最大值（数值类型）
            choices: 选择列表（分类类型）
            distribution: 分布类型
            log: 是否使用对数尺度
            step: 步长（整数类型）
        """
        self.name = name
        self.param_type = param_type
        self.low = low
        self.high = high
        self.choices = choices
        self.distribution = distribution
        self.log = log
        self.step = step
        
        self._validate()
    
    def _validate(self):
        """验证参数定义"""
        if self.param_type in [ParameterType.FLOAT, ParameterType.INT]:
            if self.low is None or self.high is None:
                raise ValueError(f"Parameter {self.name}: low and high must be specified for {self.param_type}")
            if self.low >= self.high:
                raise ValueError(f"Parameter {self.name}: low must be less than high")
            
            if self.log and self.low <= 0:
                raise ValueError(f"Parameter {self.name}: low must be positive for log scale")
        
        elif self.param_type == ParameterType.CATEGORICAL:
            if not self.choices or len(self.choices) == 0:
                raise ValueError(f"Parameter {self.name}: choices must be specified for categorical type")
        
        elif self.param_type == ParameterType.BOOLEAN:
            self.choices = [True, False]
    
    def sample(self, random_state: Optional[np.random.RandomState] = None) -> Any:
        """
        从参数空间中采样
        
        Args:
            random_state: 随机状态
            
        Returns:
            采样值
        """
        if random_state is None:
            random_state = np.random
        
        if self.param_type == ParameterType.FLOAT:
            return self._sample_float(random_state)
        elif self.param_type == ParameterType.INT:
            return self._sample_int(random_state)
        elif self.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
            return self._sample_categorical(random_state)
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")
    
    def _sample_float(self, random_state: np.random.RandomState) -> float:
        """采样浮点数"""
        if self.log or self.distribution == Distribution.LOG_UNIFORM:
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            log_value = random_state.uniform(log_low, log_high)
            return np.exp(log_value)
        
        elif self.distribution == Distribution.NORMAL:
            mean = (self.low + self.high) / 2
            std = (self.high - self.low) / 6  # 3-sigma rule
            value = random_state.normal(mean, std)
            return np.clip(value, self.low, self.high)
        
        elif self.distribution == Distribution.LOG_NORMAL:
            log_mean = np.log((self.low + self.high) / 2)
            log_std = (np.log(self.high) - np.log(self.low)) / 6
            log_value = random_state.normal(log_mean, log_std)
            value = np.exp(log_value)
            return np.clip(value, self.low, self.high)
        
        else:  # UNIFORM
            return random_state.uniform(self.low, self.high)
    
    def _sample_int(self, random_state: np.random.RandomState) -> int:
        """采样整数"""
        if self.log or self.distribution == Distribution.LOG_UNIFORM:
            log_low = np.log(max(1, self.low))
            log_high = np.log(self.high)
            log_value = random_state.uniform(log_low, log_high)
            value = int(np.exp(log_value))
        else:
            value = random_state.randint(self.low, self.high + 1)
        
        # 应用步长
        if self.step is not None:
            value = self.low + ((value - self.low) // self.step) * self.step
        
        return int(np.clip(value, self.low, self.high))
    
    def _sample_categorical(self, random_state: np.random.RandomState) -> Any:
        """采样分类值"""
        return random_state.choice(self.choices)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.param_type.value,
            'low': self.low,
            'high': self.high,
            'choices': self.choices,
            'distribution': self.distribution.value,
            'log': self.log,
            'step': self.step
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Parameter':
        """从字典创建参数"""
        return cls(
            name=data['name'],
            param_type=ParameterType(data['type']),
            low=data.get('low'),
            high=data.get('high'),
            choices=data.get('choices'),
            distribution=Distribution(data.get('distribution', 'uniform')),
            log=data.get('log', False),
            step=data.get('step')
        )


class HyperparameterSpace:
    """超参数空间类"""
    
    def __init__(self, parameters: Optional[List[Parameter]] = None):
        """
        初始化超参数空间
        
        Args:
            parameters: 参数列表
        """
        self.parameters = {}
        
        if parameters:
            for param in parameters:
                self.add_parameter(param)
    
    def add_parameter(self, parameter: Parameter):
        """
        添加参数
        
        Args:
            parameter: 参数对象
        """
        self.parameters[parameter.name] = parameter
    
    def add_float(self, name: str, low: float, high: float, 
                  distribution: Distribution = Distribution.UNIFORM,
                  log: bool = False) -> 'HyperparameterSpace':
        """
        添加浮点参数
        
        Args:
            name: 参数名称
            low: 最小值
            high: 最大值
            distribution: 分布类型
            log: 是否使用对数尺度
            
        Returns:
            自身（支持链式调用）
        """
        param = Parameter(name, ParameterType.FLOAT, low, high, 
                         distribution=distribution, log=log)
        self.add_parameter(param)
        return self
    
    def add_int(self, name: str, low: int, high: int,
                distribution: Distribution = Distribution.UNIFORM,
                log: bool = False, step: Optional[int] = None) -> 'HyperparameterSpace':
        """
        添加整数参数
        
        Args:
            name: 参数名称
            low: 最小值
            high: 最大值
            distribution: 分布类型
            log: 是否使用对数尺度
            step: 步长
            
        Returns:
            自身（支持链式调用）
        """
        param = Parameter(name, ParameterType.INT, low, high,
                         distribution=distribution, log=log, step=step)
        self.add_parameter(param)
        return self
    
    def add_categorical(self, name: str, choices: List[Any]) -> 'HyperparameterSpace':
        """
        添加分类参数
        
        Args:
            name: 参数名称
            choices: 选择列表
            
        Returns:
            自身（支持链式调用）
        """
        param = Parameter(name, ParameterType.CATEGORICAL, choices=choices)
        self.add_parameter(param)
        return self
    
    def add_boolean(self, name: str) -> 'HyperparameterSpace':
        """
        添加布尔参数
        
        Args:
            name: 参数名称
            
        Returns:
            自身（支持链式调用）
        """
        param = Parameter(name, ParameterType.BOOLEAN)
        self.add_parameter(param)
        return self
    
    def sample(self, random_state: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
        """
        从超参数空间中采样
        
        Args:
            random_state: 随机状态
            
        Returns:
            超参数字典
        """
        if random_state is None:
            random_state = np.random
        
        params = {}
        for name, parameter in self.parameters.items():
            params[name] = parameter.sample(random_state)
        
        return params
    
    def get_parameter_names(self) -> List[str]:
        """获取参数名称列表"""
        return list(self.parameters.keys())
    
    def get_parameter(self, name: str) -> Parameter:
        """
        获取参数
        
        Args:
            name: 参数名称
            
        Returns:
            参数对象
        """
        if name not in self.parameters:
            raise KeyError(f"Parameter {name} not found")
        return self.parameters[name]
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        验证参数是否在空间内
        
        Args:
            params: 参数字典
            
        Returns:
            是否有效
        """
        for name, value in params.items():
            if name not in self.parameters:
                logger.warning(f"Unknown parameter: {name}")
                continue
            
            param = self.parameters[name]
            
            if param.param_type in [ParameterType.FLOAT, ParameterType.INT]:
                if not (param.low <= value <= param.high):
                    return False
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                if value not in param.choices:
                    return False
        
        return True
    
    def get_bounds(self) -> Dict[str, Tuple[Any, Any]]:
        """
        获取参数边界
        
        Returns:
            参数边界字典
        """
        bounds = {}
        for name, param in self.parameters.items():
            if param.param_type in [ParameterType.FLOAT, ParameterType.INT]:
                bounds[name] = (param.low, param.high)
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                bounds[name] = param.choices
        
        return bounds
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'parameters': {name: param.to_dict() for name, param in self.parameters.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperparameterSpace':
        """从字典创建超参数空间"""
        space = cls()
        
        for name, param_data in data['parameters'].items():
            param = Parameter.from_dict(param_data)
            space.add_parameter(param)
        
        return space
    
    def __len__(self) -> int:
        """返回参数数量"""
        return len(self.parameters)
    
    def __contains__(self, name: str) -> bool:
        """检查是否包含参数"""
        return name in self.parameters
    
    def __iter__(self):
        """迭代参数"""
        return iter(self.parameters.items())


def create_sklearn_space(model_type: str) -> HyperparameterSpace:
    """
    创建sklearn模型的超参数空间
    
    Args:
        model_type: 模型类型
        
    Returns:
        超参数空间
    """
    space = HyperparameterSpace()
    
    if model_type == 'random_forest':
        space.add_int('n_estimators', 10, 1000, log=True)
        space.add_int('max_depth', 1, 50)
        space.add_float('min_samples_split', 0.01, 1.0)
        space.add_float('min_samples_leaf', 0.01, 0.5)
        space.add_categorical('max_features', ['sqrt', 'log2', None])
        space.add_boolean('bootstrap')
    
    elif model_type == 'gradient_boosting':
        space.add_int('n_estimators', 50, 1000, log=True)
        space.add_float('learning_rate', 0.01, 1.0, log=True)
        space.add_int('max_depth', 1, 20)
        space.add_float('subsample', 0.5, 1.0)
        space.add_float('min_samples_split', 0.01, 1.0)
        space.add_float('min_samples_leaf', 0.01, 0.5)
    
    elif model_type == 'svm':
        space.add_float('C', 0.001, 1000, log=True)
        space.add_float('gamma', 0.001, 10, log=True)
        space.add_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
        space.add_int('degree', 2, 5)  # for poly kernel
    
    elif model_type == 'logistic_regression':
        space.add_float('C', 0.001, 1000, log=True)
        space.add_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        space.add_float('l1_ratio', 0.0, 1.0)  # for elasticnet
        space.add_categorical('solver', ['liblinear', 'saga', 'lbfgs'])
    
    elif model_type == 'xgboost':
        space.add_int('n_estimators', 50, 1000, log=True)
        space.add_float('learning_rate', 0.01, 1.0, log=True)
        space.add_int('max_depth', 1, 20)
        space.add_float('subsample', 0.5, 1.0)
        space.add_float('colsample_bytree', 0.5, 1.0)
        space.add_float('reg_alpha', 0.0, 10.0, log=True)
        space.add_float('reg_lambda', 0.0, 10.0, log=True)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return space


def create_neural_network_space() -> HyperparameterSpace:
    """
    创建神经网络的超参数空间
    
    Returns:
        超参数空间
    """
    space = HyperparameterSpace()
    
    # 网络结构
    space.add_int('n_layers', 1, 10)
    space.add_int('hidden_size', 16, 1024, log=True)
    space.add_float('dropout_rate', 0.0, 0.8)
    
    # 训练参数
    space.add_float('learning_rate', 1e-5, 1e-1, log=True)
    space.add_int('batch_size', 16, 512, log=True)
    space.add_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    space.add_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # 激活函数
    space.add_categorical('activation', ['relu', 'tanh', 'sigmoid', 'leaky_relu'])
    
    # 批归一化
    space.add_boolean('batch_norm')
    
    return space