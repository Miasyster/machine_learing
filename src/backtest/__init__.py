"""
回测引擎模块

提供完整的量化交易回测功能，包括：
- 基础数据结构和抽象接口
- 按bar回测引擎
- 事务管理系统
- 性能分析和可视化
- vectorbt集成优化
"""

from .base import (
    # 枚举类型
    OrderType, OrderSide, OrderStatus, PositionSide,
    
    # 数据结构
    Order, Trade, Position, BacktestResult, BacktestConfig,
    
    # 抽象基类
    BacktestEngine, SlippageModel, CommissionModel,
    
    # 具体实现
    FixedSlippageModel, VolumeSlippageModel,
    FixedCommissionModel, PercentageCommissionModel
)

# 尝试导入其他模块（如果存在）
try:
    from .engine import BarBacktestEngine
except ImportError:
    BarBacktestEngine = None

try:
    from .transaction import TransactionManager, PositionManager
except ImportError:
    TransactionManager = None
    PositionManager = None

try:
    from .analysis import BacktestAnalyzer, PerformanceMetrics
except ImportError:
    BacktestAnalyzer = None
    PerformanceMetrics = None

__all__ = [
    # 枚举
    'OrderType', 'OrderSide', 'OrderStatus', 'PositionSide',
    
    # 数据结构
    'Order', 'Trade', 'Position', 'BacktestResult', 'BacktestConfig',
    
    # 抽象基类
    'BacktestEngine', 'SlippageModel', 'CommissionModel',
    
    # 滑点模型
    'FixedSlippageModel', 'VolumeSlippageModel',
    
    # 手续费模型
    'FixedCommissionModel', 'PercentageCommissionModel',
]

# 添加可选模块到__all__
if BarBacktestEngine is not None:
    __all__.append('BarBacktestEngine')
if TransactionManager is not None:
    __all__.extend(['TransactionManager', 'PositionManager'])
if BacktestAnalyzer is not None:
    __all__.extend(['BacktestAnalyzer', 'PerformanceMetrics'])

__version__ = '1.0.0'