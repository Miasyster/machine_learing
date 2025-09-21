"""
超参数优化模块

提供各种超参数优化算法和工具
"""

# 基础类
from .base import BaseOptimizer, OptimizationResult

# 优化器
from .grid_search import GridSearchOptimizer, AdaptiveGridSearchOptimizer
from .random_search import RandomSearchOptimizer, AdaptiveRandomSearchOptimizer, QuasiRandomSearchOptimizer
from .bayesian_optimization import BayesianOptimizer, TPEOptimizer

# Optuna优化器
try:
    from .optuna_optimizer import OptunaOptimizer, MultiObjectiveOptunaOptimizer
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

# 超参数空间
from .hyperparameter_space import (
    ParameterType, Distribution, Parameter, HyperparameterSpace,
    create_sklearn_space, create_neural_network_space
)

# 工具函数
from .utils import (
    create_objective_function, create_multi_objective_function,
    compare_optimizers, plot_optimization_comparison, plot_convergence_curves,
    plot_parameter_importance, save_optimization_results, load_optimization_results,
    calculate_optimization_efficiency, suggest_optimization_strategy
)

__all__ = [
    # 基础类
    'BaseOptimizer', 'OptimizationResult',
    
    # 优化器
    'GridSearchOptimizer', 'AdaptiveGridSearchOptimizer',
    'RandomSearchOptimizer', 'AdaptiveRandomSearchOptimizer', 'QuasiRandomSearchOptimizer',
    'BayesianOptimizer', 'TPEOptimizer',
    
    # 超参数空间
    'ParameterType', 'Distribution', 'Parameter', 'HyperparameterSpace',
    'create_sklearn_space', 'create_neural_network_space',
    
    # 工具函数
    'create_objective_function', 'create_multi_objective_function',
    'compare_optimizers', 'plot_optimization_comparison', 'plot_convergence_curves',
    'plot_parameter_importance', 'save_optimization_results', 'load_optimization_results',
    'calculate_optimization_efficiency', 'suggest_optimization_strategy'
]

# 添加Optuna优化器到__all__（如果可用）
if _OPTUNA_AVAILABLE:
    __all__.extend(['OptunaOptimizer', 'MultiObjectiveOptunaOptimizer'])

__version__ = '1.0.0'