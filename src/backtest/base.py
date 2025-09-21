"""
回测引擎基础模块

定义回测引擎的核心数据结构和抽象接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"    # 限价单
    STOP = "stop"      # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"      # 待处理
    FILLED = "filled"        # 已成交
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"    # 已拒绝


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Order:
    """订单数据结构"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """成交记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: PositionSide
    quantity: float
    avg_price: float
    market_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    entry_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        """市值"""
        return self.quantity * self.market_price
    
    @property
    def cost_basis(self) -> float:
        """成本基础"""
        return self.quantity * self.avg_price
    
    def update_market_price(self, price: float, timestamp: datetime):
        """更新市场价格和未实现盈亏"""
        self.market_price = price
        self.last_update_time = timestamp
        
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.avg_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.avg_price - price) * self.quantity
        else:
            self.unrealized_pnl = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # 详细数据
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: List[Trade] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine(ABC):
    """回测引擎抽象基类"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        
    @abstractmethod
    def add_data(self, data: pd.DataFrame, symbol: str = "default"):
        """添加回测数据"""
        pass
    
    @abstractmethod
    def add_strategy(self, strategy):
        """添加交易策略"""
        pass
    
    @abstractmethod
    def run(self) -> BacktestResult:
        """运行回测"""
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """下单"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        pass
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)
    
    def get_portfolio_value(self) -> float:
        """获取组合总价值"""
        total_value = self.current_capital
        for position in self.positions.values():
            total_value += position.market_value
        return total_value
    
    def get_equity_curve(self) -> pd.Series:
        """获取权益曲线"""
        if self.timestamps and self.equity_curve:
            return pd.Series(self.equity_curve, index=self.timestamps)
        return pd.Series()


class SlippageModel(ABC):
    """滑点模型抽象基类"""
    
    @abstractmethod
    def calculate_slippage(self, order: Order, market_price: float, volume: float) -> float:
        """计算滑点"""
        pass


class FixedSlippageModel(SlippageModel):
    """固定滑点模型"""
    
    def __init__(self, slippage_bps: float = 5.0):
        """
        Args:
            slippage_bps: 滑点基点数 (1 bps = 0.01%)
        """
        self.slippage_bps = slippage_bps
    
    def calculate_slippage(self, order: Order, market_price: float, volume: float) -> float:
        """计算固定滑点"""
        slippage_rate = self.slippage_bps / 10000.0
        if order.side == OrderSide.BUY:
            return market_price * slippage_rate
        else:
            return -market_price * slippage_rate


class VolumeSlippageModel(SlippageModel):
    """基于成交量的滑点模型"""
    
    def __init__(self, base_slippage_bps: float = 2.0, volume_impact_factor: float = 0.1):
        """
        Args:
            base_slippage_bps: 基础滑点基点数
            volume_impact_factor: 成交量影响因子
        """
        self.base_slippage_bps = base_slippage_bps
        self.volume_impact_factor = volume_impact_factor
    
    def calculate_slippage(self, order: Order, market_price: float, volume: float) -> float:
        """计算基于成交量的滑点"""
        # 简化的成交量影响模型
        volume_impact = self.volume_impact_factor * np.sqrt(order.quantity / max(volume, 1))
        total_slippage_bps = self.base_slippage_bps + volume_impact
        slippage_rate = total_slippage_bps / 10000.0
        
        if order.side == OrderSide.BUY:
            return market_price * slippage_rate
        else:
            return -market_price * slippage_rate


class CommissionModel(ABC):
    """手续费模型抽象基类"""
    
    @abstractmethod
    def calculate_commission(self, order: Order, fill_price: float, fill_quantity: float) -> float:
        """计算手续费"""
        pass


class FixedCommissionModel(CommissionModel):
    """固定手续费模型"""
    
    def __init__(self, commission_per_trade: float = 5.0):
        """
        Args:
            commission_per_trade: 每笔交易的固定手续费
        """
        self.commission_per_trade = commission_per_trade
    
    def calculate_commission(self, order: Order, fill_price: float, fill_quantity: float) -> float:
        """计算固定手续费"""
        return self.commission_per_trade


class PercentageCommissionModel(CommissionModel):
    """百分比手续费模型"""
    
    def __init__(self, commission_rate: float = 0.001):
        """
        Args:
            commission_rate: 手续费率 (例如 0.001 = 0.1%)
        """
        self.commission_rate = commission_rate
    
    def calculate_commission(self, order: Order, fill_price: float, fill_quantity: float) -> float:
        """计算百分比手续费"""
        return fill_price * fill_quantity * self.commission_rate


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    slippage_model: SlippageModel = field(default_factory=FixedSlippageModel)
    commission_model: CommissionModel = field(default_factory=FixedCommissionModel)
    
    # 风险管理参数
    max_position_size: float = 0.1  # 最大单个持仓占比
    max_total_exposure: float = 1.0  # 最大总敞口
    margin_requirement: float = 0.1  # 保证金要求
    
    # 执行参数
    fill_delay: int = 0  # 成交延迟(bar数)
    partial_fill_enabled: bool = False  # 是否允许部分成交
    
    # 其他配置
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02  # 无风险利率
    trading_calendar: Optional[str] = None
    
    def __post_init__(self):
        """后处理初始化"""
        if isinstance(self.slippage_model, dict):
            # 从字典创建滑点模型
            model_type = self.slippage_model.get('type', 'fixed')
            if model_type == 'fixed':
                self.slippage_model = FixedSlippageModel(**self.slippage_model.get('params', {}))
            elif model_type == 'volume':
                self.slippage_model = VolumeSlippageModel(**self.slippage_model.get('params', {}))
        
        if isinstance(self.commission_model, dict):
            # 从字典创建手续费模型
            model_type = self.commission_model.get('type', 'fixed')
            if model_type == 'fixed':
                self.commission_model = FixedCommissionModel(**self.commission_model.get('params', {}))
            elif model_type == 'percentage':
                self.commission_model = PercentageCommissionModel(**self.commission_model.get('params', {}))