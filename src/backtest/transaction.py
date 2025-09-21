"""
事务管理系统

实现交易事务管理功能，包括：
- 开平仓管理
- 仓位限制
- 止损止盈
- 风险控制
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import pandas as pd
import numpy as np

from .base import (
    Order, Trade, Position, OrderType, OrderSide, OrderStatus, PositionSide
)


class StopLossType(Enum):
    """止损类型"""
    FIXED_AMOUNT = "fixed_amount"      # 固定金额止损
    PERCENTAGE = "percentage"          # 百分比止损
    ATR = "atr"                       # ATR止损
    TRAILING = "trailing"             # 移动止损


class TakeProfitType(Enum):
    """止盈类型"""
    FIXED_AMOUNT = "fixed_amount"      # 固定金额止盈
    PERCENTAGE = "percentage"          # 百分比止盈
    RISK_REWARD = "risk_reward"       # 风险收益比止盈
    TRAILING = "trailing"             # 移动止盈


@dataclass
class StopLossRule:
    """止损规则"""
    type: StopLossType
    value: float
    enabled: bool = True
    
    # 移动止损参数
    trailing_distance: Optional[float] = None
    trailing_step: Optional[float] = None
    
    # ATR止损参数
    atr_period: int = 14
    atr_multiplier: float = 2.0
    
    # 状态跟踪
    current_stop_price: Optional[float] = None
    highest_price: Optional[float] = None  # 用于移动止损
    lowest_price: Optional[float] = None   # 用于移动止损


@dataclass
class TakeProfitRule:
    """止盈规则"""
    type: TakeProfitType
    value: float
    enabled: bool = True
    
    # 风险收益比止盈参数
    risk_amount: Optional[float] = None
    
    # 移动止盈参数
    trailing_distance: Optional[float] = None
    trailing_step: Optional[float] = None
    
    # 状态跟踪
    current_target_price: Optional[float] = None
    highest_price: Optional[float] = None  # 用于移动止盈
    lowest_price: Optional[float] = None   # 用于移动止盈


@dataclass
class PositionLimit:
    """仓位限制"""
    max_position_size: Optional[float] = None      # 最大单个持仓大小
    max_total_exposure: Optional[float] = None     # 最大总敞口
    max_leverage: Optional[float] = None           # 最大杠杆
    max_correlation: Optional[float] = None        # 最大相关性
    
    # 按标的限制
    symbol_limits: Dict[str, float] = field(default_factory=dict)
    
    # 按行业/板块限制
    sector_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskRule:
    """风险规则"""
    max_daily_loss: Optional[float] = None         # 最大日损失
    max_drawdown: Optional[float] = None           # 最大回撤
    max_consecutive_losses: Optional[int] = None   # 最大连续亏损次数
    
    # 强制平仓条件
    force_close_on_margin_call: bool = True
    margin_call_threshold: float = 0.3
    
    # 交易时间限制
    trading_hours: Optional[Tuple[str, str]] = None
    trading_days: Optional[List[str]] = None


class TransactionManager:
    """事务管理器"""
    
    def __init__(self, engine=None):
        """
        初始化事务管理器
        
        Args:
            engine: 回测引擎实例
        """
        self.engine = engine
        
        # 止损止盈规则
        self.stop_loss_rules: Dict[str, StopLossRule] = {}
        self.take_profit_rules: Dict[str, TakeProfitRule] = {}
        
        # 仓位限制
        self.position_limits = PositionLimit()
        
        # 风险规则
        self.risk_rules = RiskRule()
        
        # 状态跟踪
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_pnl = 0.0
        
        # 历史记录
        self.transaction_history: List[Dict] = []
        
    def set_stop_loss(self, symbol: str, stop_loss_rule: StopLossRule):
        """设置止损规则"""
        self.stop_loss_rules[symbol] = stop_loss_rule
        
    def set_take_profit(self, symbol: str, take_profit_rule: TakeProfitRule):
        """设置止盈规则"""
        self.take_profit_rules[symbol] = take_profit_rule
        
    def set_position_limits(self, position_limits: PositionLimit):
        """设置仓位限制"""
        self.position_limits = position_limits
        
    def set_risk_rules(self, risk_rules: RiskRule):
        """设置风险规则"""
        self.risk_rules = risk_rules
        
    def check_position_limits(self, order: Order) -> Tuple[bool, str]:
        """
        检查仓位限制
        
        Args:
            order: 待检查的订单
            
        Returns:
            (是否通过检查, 错误信息)
        """
        if not self.engine:
            return True, ""
        
        # 检查单个持仓限制
        if self.position_limits.max_position_size is not None:
            portfolio_value = self.engine.get_portfolio_value()
            if portfolio_value > 0:
                estimated_value = self._estimate_order_value(order)
                current_position = self.engine.get_position(order.symbol)
                
                if current_position:
                    current_exposure = abs(current_position.market_value) / portfolio_value
                    new_exposure = (abs(current_position.market_value) + estimated_value) / portfolio_value
                    
                    if new_exposure > self.position_limits.max_position_size:
                        return False, f"超过最大单个持仓限制: {new_exposure:.2%} > {self.position_limits.max_position_size:.2%}"
        
        # 检查总敞口限制
        if self.position_limits.max_total_exposure is not None:
            portfolio_value = self.engine.get_portfolio_value()
            if portfolio_value > 0:
                total_exposure = self._calculate_total_exposure()
                estimated_value = self._estimate_order_value(order)
                new_total_exposure = (total_exposure + estimated_value) / portfolio_value
                
                if new_total_exposure > self.position_limits.max_total_exposure:
                    return False, f"超过最大总敞口限制: {new_total_exposure:.2%} > {self.position_limits.max_total_exposure:.2%}"
        
        # 检查标的特定限制
        if order.symbol in self.position_limits.symbol_limits:
            symbol_limit = self.position_limits.symbol_limits[order.symbol]
            portfolio_value = self.engine.get_portfolio_value()
            
            if portfolio_value > 0:
                estimated_value = self._estimate_order_value(order)
                current_position = self.engine.get_position(order.symbol)
                current_value = abs(current_position.market_value) if current_position else 0
                new_exposure = (current_value + estimated_value) / portfolio_value
                
                if new_exposure > symbol_limit:
                    return False, f"超过标的 {order.symbol} 持仓限制: {new_exposure:.2%} > {symbol_limit:.2%}"
        
        return True, ""
    
    def check_risk_rules(self) -> Tuple[bool, str]:
        """
        检查风险规则
        
        Returns:
            (是否通过检查, 错误信息)
        """
        if not self.engine:
            return True, ""
        
        # 检查最大日损失
        if self.risk_rules.max_daily_loss is not None:
            if self.daily_pnl < -abs(self.risk_rules.max_daily_loss):
                return False, f"超过最大日损失限制: {self.daily_pnl:.2f} < -{self.risk_rules.max_daily_loss:.2f}"
        
        # 检查最大回撤
        if self.risk_rules.max_drawdown is not None:
            equity_curve = self.engine.get_equity_curve()
            if len(equity_curve) > 1:
                rolling_max = equity_curve.expanding().max()
                current_drawdown = (equity_curve.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]
                
                if current_drawdown < -abs(self.risk_rules.max_drawdown):
                    return False, f"超过最大回撤限制: {current_drawdown:.2%} < -{self.risk_rules.max_drawdown:.2%}"
        
        # 检查连续亏损次数
        if self.risk_rules.max_consecutive_losses is not None:
            if self.consecutive_losses >= self.risk_rules.max_consecutive_losses:
                return False, f"超过最大连续亏损次数: {self.consecutive_losses} >= {self.risk_rules.max_consecutive_losses}"
        
        return True, ""
    
    def update_stop_loss_take_profit(self, symbol: str, current_price: float, timestamp: datetime) -> List[Order]:
        """
        更新止损止盈
        
        Args:
            symbol: 标的代码
            current_price: 当前价格
            timestamp: 当前时间
            
        Returns:
            生成的止损止盈订单列表
        """
        orders = []
        
        if not self.engine:
            return orders
        
        position = self.engine.get_position(symbol)
        if not position or position.side == PositionSide.FLAT:
            return orders
        
        # 更新止损
        if symbol in self.stop_loss_rules:
            stop_order = self._update_stop_loss(symbol, position, current_price, timestamp)
            if stop_order:
                orders.append(stop_order)
        
        # 更新止盈
        if symbol in self.take_profit_rules:
            profit_order = self._update_take_profit(symbol, position, current_price, timestamp)
            if profit_order:
                orders.append(profit_order)
        
        return orders
    
    def _update_stop_loss(self, symbol: str, position: Position, current_price: float, 
                         timestamp: datetime) -> Optional[Order]:
        """更新止损"""
        rule = self.stop_loss_rules[symbol]
        
        if not rule.enabled:
            return None
        
        stop_price = None
        
        if rule.type == StopLossType.FIXED_AMOUNT:
            if position.side == PositionSide.LONG:
                stop_price = position.avg_price - rule.value
            else:
                stop_price = position.avg_price + rule.value
                
        elif rule.type == StopLossType.PERCENTAGE:
            if position.side == PositionSide.LONG:
                stop_price = position.avg_price * (1 - rule.value)
            else:
                stop_price = position.avg_price * (1 + rule.value)
                
        elif rule.type == StopLossType.TRAILING:
            if rule.trailing_distance is None:
                return None
            
            if position.side == PositionSide.LONG:
                # 多头移动止损
                if rule.highest_price is None or current_price > rule.highest_price:
                    rule.highest_price = current_price
                
                new_stop_price = rule.highest_price - rule.trailing_distance
                
                if rule.current_stop_price is None or new_stop_price > rule.current_stop_price:
                    rule.current_stop_price = new_stop_price
                    stop_price = new_stop_price
            else:
                # 空头移动止损
                if rule.lowest_price is None or current_price < rule.lowest_price:
                    rule.lowest_price = current_price
                
                new_stop_price = rule.lowest_price + rule.trailing_distance
                
                if rule.current_stop_price is None or new_stop_price < rule.current_stop_price:
                    rule.current_stop_price = new_stop_price
                    stop_price = new_stop_price
        
        elif rule.type == StopLossType.ATR:
            # ATR止损需要历史数据计算ATR
            if self.engine:
                historical_data = self.engine.get_historical_data(symbol, rule.atr_period + 1)
                if historical_data is not None and len(historical_data) >= rule.atr_period:
                    atr = self._calculate_atr(historical_data, rule.atr_period)
                    
                    if position.side == PositionSide.LONG:
                        stop_price = current_price - atr * rule.atr_multiplier
                    else:
                        stop_price = current_price + atr * rule.atr_multiplier
        
        # 检查是否需要触发止损
        if stop_price is not None:
            should_trigger = False
            
            if position.side == PositionSide.LONG and current_price <= stop_price:
                should_trigger = True
            elif position.side == PositionSide.SHORT and current_price >= stop_price:
                should_trigger = True
            
            if should_trigger:
                # 创建止损订单
                order_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
                
                return Order(
                    symbol=symbol,
                    side=order_side,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    timestamp=timestamp,
                    metadata={'reason': 'stop_loss', 'stop_price': stop_price}
                )
        
        return None
    
    def _update_take_profit(self, symbol: str, position: Position, current_price: float,
                           timestamp: datetime) -> Optional[Order]:
        """更新止盈"""
        rule = self.take_profit_rules[symbol]
        
        if not rule.enabled:
            return None
        
        target_price = None
        
        if rule.type == TakeProfitType.FIXED_AMOUNT:
            if position.side == PositionSide.LONG:
                target_price = position.avg_price + rule.value
            else:
                target_price = position.avg_price - rule.value
                
        elif rule.type == TakeProfitType.PERCENTAGE:
            if position.side == PositionSide.LONG:
                target_price = position.avg_price * (1 + rule.value)
            else:
                target_price = position.avg_price * (1 - rule.value)
                
        elif rule.type == TakeProfitType.RISK_REWARD:
            if rule.risk_amount is not None:
                profit_target = rule.risk_amount * rule.value
                
                if position.side == PositionSide.LONG:
                    target_price = position.avg_price + profit_target
                else:
                    target_price = position.avg_price - profit_target
                    
        elif rule.type == TakeProfitType.TRAILING:
            if rule.trailing_distance is None:
                return None
            
            if position.side == PositionSide.LONG:
                # 多头移动止盈
                if rule.highest_price is None or current_price > rule.highest_price:
                    rule.highest_price = current_price
                
                new_target_price = rule.highest_price - rule.trailing_distance
                
                if rule.current_target_price is None or new_target_price > rule.current_target_price:
                    rule.current_target_price = new_target_price
                    target_price = new_target_price
            else:
                # 空头移动止盈
                if rule.lowest_price is None or current_price < rule.lowest_price:
                    rule.lowest_price = current_price
                
                new_target_price = rule.lowest_price + rule.trailing_distance
                
                if rule.current_target_price is None or new_target_price < rule.current_target_price:
                    rule.current_target_price = new_target_price
                    target_price = new_target_price
        
        # 检查是否需要触发止盈
        if target_price is not None:
            should_trigger = False
            
            if position.side == PositionSide.LONG and current_price >= target_price:
                should_trigger = True
            elif position.side == PositionSide.SHORT and current_price <= target_price:
                should_trigger = True
            
            if should_trigger:
                # 创建止盈订单
                order_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
                
                return Order(
                    symbol=symbol,
                    side=order_side,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    timestamp=timestamp,
                    metadata={'reason': 'take_profit', 'target_price': target_price}
                )
        
        return None
    
    def _estimate_order_value(self, order: Order) -> float:
        """估算订单价值"""
        if not self.engine:
            return 0.0
        
        if order.order_type == OrderType.MARKET:
            current_price = self.engine.get_current_price(order.symbol)
            if current_price:
                return current_price * order.quantity
        elif order.price is not None:
            return order.price * order.quantity
        
        return 0.0
    
    def _calculate_total_exposure(self) -> float:
        """计算总敞口"""
        if not self.engine:
            return 0.0
        
        total_exposure = 0.0
        for position in self.engine.positions.values():
            total_exposure += abs(position.market_value)
        
        return total_exposure
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> float:
        """计算ATR (Average True Range)"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0
    
    def on_trade_executed(self, trade: Trade):
        """交易执行后的回调"""
        # 更新日盈亏
        # 这里简化处理，实际应该根据开平仓计算
        
        # 更新连续亏损计数
        # 这里需要更复杂的逻辑来判断交易盈亏
        
        # 记录交易历史
        self.transaction_history.append({
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'side': trade.side,
            'quantity': trade.quantity,
            'price': trade.price,
            'commission': trade.commission,
            'type': 'trade'
        })
    
    def on_position_updated(self, symbol: str, position: Position):
        """持仓更新后的回调"""
        # 清理已平仓标的的止损止盈规则
        if position.side == PositionSide.FLAT:
            if symbol in self.stop_loss_rules:
                self.stop_loss_rules[symbol].current_stop_price = None
                self.stop_loss_rules[symbol].highest_price = None
                self.stop_loss_rules[symbol].lowest_price = None
            
            if symbol in self.take_profit_rules:
                self.take_profit_rules[symbol].current_target_price = None
                self.take_profit_rules[symbol].highest_price = None
                self.take_profit_rules[symbol].lowest_price = None


class PositionManager:
    """仓位管理器"""
    
    def __init__(self, transaction_manager: TransactionManager = None):
        """
        初始化仓位管理器
        
        Args:
            transaction_manager: 事务管理器
        """
        self.transaction_manager = transaction_manager
        self.position_sizing_rules: Dict[str, Callable] = {}
        
    def set_position_sizing_rule(self, symbol: str, rule: Callable[[float, float], float]):
        """
        设置仓位大小规则
        
        Args:
            symbol: 标的代码
            rule: 仓位大小计算函数，接受(当前价格, 组合价值)，返回仓位大小
        """
        self.position_sizing_rules[symbol] = rule
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                              portfolio_value: float) -> float:
        """
        计算仓位大小
        
        Args:
            symbol: 标的代码
            current_price: 当前价格
            portfolio_value: 组合价值
            
        Returns:
            建议的仓位大小
        """
        if symbol in self.position_sizing_rules:
            return self.position_sizing_rules[symbol](current_price, portfolio_value)
        
        # 默认固定比例仓位
        default_allocation = 0.1  # 10%
        return (portfolio_value * default_allocation) / current_price
    
    def kelly_criterion_sizing(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        凯利公式仓位计算
        
        Args:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
            
        Returns:
            凯利比例
        """
        if avg_loss <= 0:
            return 0.0
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
        
        # 限制最大仓位为25%
        return max(0, min(kelly_fraction, 0.25))
    
    def volatility_based_sizing(self, volatility: float, target_volatility: float = 0.15) -> float:
        """
        基于波动率的仓位计算
        
        Args:
            volatility: 标的波动率
            target_volatility: 目标组合波动率
            
        Returns:
            仓位比例
        """
        if volatility <= 0:
            return 0.0
        
        return min(target_volatility / volatility, 1.0)