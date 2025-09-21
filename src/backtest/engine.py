"""
按bar回测引擎

实现基于历史数据的逐bar回测功能，包括：
- 订单撮合逻辑
- 滑点和手续费处理
- 仓位管理
- 风险控制
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from collections import defaultdict, deque

from .base import (
    BacktestEngine, Order, Trade, Position, BacktestResult, BacktestConfig,
    OrderType, OrderSide, OrderStatus, PositionSide,
    SlippageModel, CommissionModel, FixedSlippageModel, FixedCommissionModel
)


class BarBacktestEngine(BacktestEngine):
    """按bar回测引擎"""
    
    def __init__(self, config: BacktestConfig = None):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
        """
        self.config = config or BacktestConfig()
        super().__init__(self.config.initial_capital)
        
        # 数据存储
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_bar: Dict[str, int] = {}
        self.current_time: Optional[datetime] = None
        
        # 策略相关
        self.strategies: List[Any] = []
        self.strategy_context: Dict[str, Any] = {}
        
        # 订单管理
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        self.next_order_id = 1
        
        # 执行相关
        self.slippage_model = self.config.slippage_model
        self.commission_model = self.config.commission_model
        
        # 性能跟踪
        self.portfolio_values: List[float] = []
        self.cash_history: List[float] = []
        self.position_history: List[Dict[str, Position]] = []
        
        # 风险管理
        self.max_position_size = self.config.max_position_size
        self.max_total_exposure = self.config.max_total_exposure
        
        # 统计信息
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        
    def add_data(self, data: pd.DataFrame, symbol: str = "default"):
        """
        添加回测数据
        
        Args:
            data: OHLCV数据，必须包含datetime索引
            symbol: 标的代码
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据必须有datetime索引")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必要列: {missing_columns}")
        
        # 确保数据按时间排序
        data = data.sort_index()
        
        self.data[symbol] = data
        self.current_bar[symbol] = 0
        
        # 初始化持仓
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                side=PositionSide.FLAT,
                quantity=0.0,
                avg_price=0.0
            )
    
    def add_strategy(self, strategy):
        """
        添加交易策略
        
        Args:
            strategy: 策略对象，需要实现on_bar方法
        """
        if not hasattr(strategy, 'on_bar'):
            raise ValueError("策略必须实现on_bar方法")
        
        self.strategies.append(strategy)
        
        # 为策略提供引擎引用
        if hasattr(strategy, 'set_engine'):
            strategy.set_engine(self)
    
    def run(self) -> BacktestResult:
        """
        运行回测
        
        Returns:
            回测结果
        """
        if not self.data:
            raise ValueError("没有添加回测数据")
        
        if not self.strategies:
            raise ValueError("没有添加交易策略")
        
        # 获取所有数据的时间范围
        all_dates = set()
        for symbol_data in self.data.values():
            all_dates.update(symbol_data.index)
        
        all_dates = sorted(all_dates)
        
        if self.config.start_date:
            all_dates = [d for d in all_dates if d >= self.config.start_date]
        if self.config.end_date:
            all_dates = [d for d in all_dates if d <= self.config.end_date]
        
        if not all_dates:
            raise ValueError("没有有效的回测日期范围")
        
        print(f"开始回测: {all_dates[0]} 到 {all_dates[-1]}")
        print(f"初始资金: {self.initial_capital:,.2f}")
        
        # 逐bar执行回测
        for i, current_time in enumerate(all_dates):
            self.current_time = current_time
            self._process_bar(current_time)
            
            # 记录组合价值
            portfolio_value = self.get_portfolio_value()
            self.portfolio_values.append(portfolio_value)
            self.cash_history.append(self.current_capital)
            self.timestamps.append(current_time)
            self.equity_curve.append(portfolio_value)
            
            # 记录持仓快照
            position_snapshot = {symbol: Position(
                symbol=pos.symbol,
                side=pos.side,
                quantity=pos.quantity,
                avg_price=pos.avg_price,
                market_price=pos.market_price,
                unrealized_pnl=pos.unrealized_pnl,
                realized_pnl=pos.realized_pnl,
                total_commission=pos.total_commission,
                entry_time=pos.entry_time,
                last_update_time=pos.last_update_time
            ) for symbol, pos in self.positions.items()}
            self.position_history.append(position_snapshot)
            
            if i % 1000 == 0 and i > 0:
                print(f"已处理 {i} 个交易日, 当前组合价值: {portfolio_value:,.2f}")
        
        print(f"回测完成! 最终组合价值: {self.portfolio_values[-1]:,.2f}")
        
        # 生成回测结果
        return self._generate_result()
    
    def _process_bar(self, current_time: datetime):
        """处理单个bar"""
        # 更新当前bar索引
        for symbol in self.data:
            symbol_data = self.data[symbol]
            if current_time in symbol_data.index:
                self.current_bar[symbol] = symbol_data.index.get_loc(current_time)
            else:
                # 如果当前时间不在数据中，找到最近的前一个时间点
                valid_dates = symbol_data.index[symbol_data.index <= current_time]
                if len(valid_dates) > 0:
                    last_valid_date = valid_dates[-1]
                    self.current_bar[symbol] = symbol_data.index.get_loc(last_valid_date)
        
        # 更新市场价格
        self._update_market_prices(current_time)
        
        # 执行策略（先执行策略生成订单）
        for strategy in self.strategies:
            try:
                strategy.on_bar(current_time, self._get_current_data())
            except Exception as e:
                print(f"策略执行错误: {e}")
                continue
        
        # 处理待成交订单（策略执行后再处理订单）
        self._process_pending_orders(current_time)
    
    def _update_market_prices(self, current_time: datetime):
        """更新市场价格"""
        for symbol in self.data:
            if symbol in self.current_bar:
                bar_idx = self.current_bar[symbol]
                symbol_data = self.data[symbol]
                
                if bar_idx < len(symbol_data):
                    current_price = symbol_data.iloc[bar_idx]['close']
                    
                    # 更新持仓的市场价格
                    if symbol in self.positions:
                        self.positions[symbol].update_market_price(current_price, current_time)
    
    def _get_current_data(self) -> Dict[str, pd.Series]:
        """获取当前bar的数据"""
        current_data = {}
        for symbol in self.data:
            if symbol in self.current_bar:
                bar_idx = self.current_bar[symbol]
                symbol_data = self.data[symbol]
                
                if bar_idx < len(symbol_data):
                    current_data[symbol] = symbol_data.iloc[bar_idx]
        
        return current_data
    
    def place_order(self, order: Order) -> str:
        """
        下单
        
        Args:
            order: 订单对象
            
        Returns:
            订单ID
        """
        # 生成订单ID
        order.order_id = f"order_{self.next_order_id:06d}"
        self.next_order_id += 1
        order.timestamp = self.current_time
        
        # 风险检查
        if not self._risk_check(order):
            return None
        
        # 添加到待处理订单列表
        self.pending_orders.append(order)
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        撤单
        
        Args:
            order_id: 订单ID
            
        Returns:
            是否成功撤单
        """
        for order in self.pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self.pending_orders.remove(order)
                self.order_history.append(order)
                return True
        
        return False
    
    def _risk_check(self, order: Order) -> bool:
        """风险检查"""
        # 检查资金充足性
        if order.side == OrderSide.BUY:
            current_data = self._get_current_data()
            
            if order.symbol in current_data:
                if order.order_type == OrderType.MARKET:
                    estimated_price = current_data[order.symbol]['close']
                elif order.order_type == OrderType.LIMIT:
                    estimated_price = order.price
                else:
                    estimated_price = current_data[order.symbol]['close']
                
                required_capital = estimated_price * order.quantity
                
                if required_capital > self.current_capital:
                    return False
            else:
                return False
        
        # 检查持仓限制
        if order.symbol in self.positions:
            current_position = self.positions[order.symbol]
            portfolio_value = self.get_portfolio_value()
            
            if portfolio_value > 0:
                current_exposure = abs(current_position.market_value) / portfolio_value
                
                # 估算新订单后的敞口
                if order.order_type == OrderType.MARKET:
                    current_data = self._get_current_data()
                    if order.symbol in current_data:
                        estimated_price = current_data[order.symbol]['close']
                        order_value = estimated_price * order.quantity
                        
                        if order.side == OrderSide.BUY:
                            new_exposure = (abs(current_position.market_value) + order_value) / portfolio_value
                        else:
                            new_exposure = abs(abs(current_position.market_value) - order_value) / portfolio_value
                        
                        if new_exposure > self.max_position_size:
                            return False
        else:
            # 新持仓的敞口检查
            if order.order_type == OrderType.MARKET:
                current_data = self._get_current_data()
                if order.symbol in current_data:
                    estimated_price = current_data[order.symbol]['close']
                    order_value = estimated_price * order.quantity
                    portfolio_value = self.get_portfolio_value()
                    
                    if portfolio_value > 0:
                        new_exposure = order_value / portfolio_value
                        
                        if new_exposure > self.max_position_size:
                            return False
        
        return True
    
    def _process_pending_orders(self, current_time: datetime):
        """处理待成交订单"""
        filled_orders = []
        
        for order in self.pending_orders[:]:
            if self._try_fill_order(order, current_time):
                filled_orders.append(order)
                self.pending_orders.remove(order)
        
        # 将已成交订单添加到历史记录
        self.order_history.extend(filled_orders)
    
    def _try_fill_order(self, order: Order, current_time: datetime) -> bool:
        """尝试成交订单"""
        current_data = self._get_current_data()
        
        if order.symbol not in current_data:
            return False
        
        bar_data = current_data[order.symbol]
        
        # 根据订单类型确定成交价格
        fill_price = None
        
        if order.order_type == OrderType.MARKET:
            # 市价单使用开盘价成交（假设在bar开始时成交）
            fill_price = bar_data['open']
        
        elif order.order_type == OrderType.LIMIT:
            # 限价单检查是否能够成交
            if order.side == OrderSide.BUY:
                # 买入限价单：当最低价 <= 限价时成交
                if bar_data['low'] <= order.price:
                    fill_price = min(order.price, bar_data['open'])
            else:
                # 卖出限价单：当最高价 >= 限价时成交
                if bar_data['high'] >= order.price:
                    fill_price = max(order.price, bar_data['open'])
        
        elif order.order_type == OrderType.STOP:
            # 止损单转为市价单
            if order.side == OrderSide.BUY:
                if bar_data['high'] >= order.stop_price:
                    fill_price = max(order.stop_price, bar_data['open'])
            else:
                if bar_data['low'] <= order.stop_price:
                    fill_price = min(order.stop_price, bar_data['open'])
        
        if fill_price is None:
            return False
        
        # 计算滑点
        slippage = self.slippage_model.calculate_slippage(
            order, fill_price, bar_data['volume']
        )
        
        # 应用滑点
        if order.side == OrderSide.BUY:
            fill_price += abs(slippage)
        else:
            fill_price -= abs(slippage)
        
        # 确保价格为正
        fill_price = max(fill_price, 0.01)
        
        # 计算手续费
        commission = self.commission_model.calculate_commission(
            order, fill_price, order.quantity
        )
        
        # 执行成交
        self._execute_fill(order, fill_price, order.quantity, commission, slippage, current_time)
        
        return True
    
    def _execute_fill(self, order: Order, fill_price: float, fill_quantity: float, 
                     commission: float, slippage: float, timestamp: datetime):
        """执行订单成交"""
        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_quantity = fill_quantity
        order.avg_fill_price = fill_price
        order.commission = commission
        order.slippage = slippage
        
        # 创建成交记录
        trade = Trade(
            trade_id=f"trade_{len(self.trades) + 1:06d}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            timestamp=timestamp,
            commission=commission,
            slippage=slippage
        )
        
        self.trades.append(trade)
        
        # 更新持仓
        self._update_position(order.symbol, order.side, fill_quantity, fill_price, commission, timestamp)
        
        # 更新资金
        if order.side == OrderSide.BUY:
            self.current_capital -= (fill_price * fill_quantity + commission)
        else:
            self.current_capital += (fill_price * fill_quantity - commission)
        
        # 更新统计
        self.total_commission_paid += commission
        self.total_slippage_cost += abs(slippage) * fill_quantity
    
    def _update_position(self, symbol: str, side: OrderSide, quantity: float, 
                        price: float, commission: float, timestamp: datetime):
        """更新持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                side=PositionSide.FLAT,
                quantity=0.0,
                avg_price=0.0,
                entry_time=timestamp
            )
        
        position = self.positions[symbol]
        
        if side == OrderSide.BUY:
            if position.side == PositionSide.FLAT:
                # 开多仓
                position.side = PositionSide.LONG
                position.quantity = quantity
                position.avg_price = price
                position.entry_time = timestamp
            elif position.side == PositionSide.LONG:
                # 加多仓
                total_cost = position.quantity * position.avg_price + quantity * price
                position.quantity += quantity
                position.avg_price = total_cost / position.quantity
            elif position.side == PositionSide.SHORT:
                # 平空仓
                if quantity >= position.quantity:
                    # 完全平仓或反向开仓
                    realized_pnl = (position.avg_price - price) * position.quantity
                    position.realized_pnl += realized_pnl
                    
                    remaining_quantity = quantity - position.quantity
                    if remaining_quantity > 0:
                        # 反向开多仓
                        position.side = PositionSide.LONG
                        position.quantity = remaining_quantity
                        position.avg_price = price
                        position.entry_time = timestamp
                    else:
                        # 完全平仓
                        position.side = PositionSide.FLAT
                        position.quantity = 0.0
                        position.avg_price = 0.0
                else:
                    # 部分平仓
                    realized_pnl = (position.avg_price - price) * quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= quantity
        
        else:  # OrderSide.SELL
            if position.side == PositionSide.FLAT:
                # 开空仓
                position.side = PositionSide.SHORT
                position.quantity = quantity
                position.avg_price = price
                position.entry_time = timestamp
            elif position.side == PositionSide.SHORT:
                # 加空仓
                total_cost = position.quantity * position.avg_price + quantity * price
                position.quantity += quantity
                position.avg_price = total_cost / position.quantity
            elif position.side == PositionSide.LONG:
                # 平多仓
                if quantity >= position.quantity:
                    # 完全平仓或反向开仓
                    realized_pnl = (price - position.avg_price) * position.quantity
                    position.realized_pnl += realized_pnl
                    
                    remaining_quantity = quantity - position.quantity
                    if remaining_quantity > 0:
                        # 反向开空仓
                        position.side = PositionSide.SHORT
                        position.quantity = remaining_quantity
                        position.avg_price = price
                        position.entry_time = timestamp
                    else:
                        # 完全平仓
                        position.side = PositionSide.FLAT
                        position.quantity = 0.0
                        position.avg_price = 0.0
                else:
                    # 部分平仓
                    realized_pnl = (price - position.avg_price) * quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= quantity
        
        # 更新手续费
        position.total_commission += commission
        position.last_update_time = timestamp
    
    def _generate_result(self) -> BacktestResult:
        """生成回测结果"""
        if not self.portfolio_values:
            raise ValueError("没有回测数据")
        
        # 基本统计
        start_date = self.timestamps[0]
        end_date = self.timestamps[-1]
        initial_capital = self.initial_capital
        final_capital = self.portfolio_values[-1]
        
        # 计算收益率
        total_return = (final_capital - initial_capital) / initial_capital
        
        # 计算年化收益率
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (final_capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # 创建权益曲线
        equity_curve = pd.Series(self.portfolio_values, index=self.timestamps)
        
        # 计算日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 计算最大回撤
        rolling_max = equity_curve.expanding().max()
        drawdown_series = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown_series.min()
        
        # 计算夏普比率
        if len(daily_returns) > 1:
            excess_returns = daily_returns - self.config.risk_free_rate / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 计算索提诺比率
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 1:
            downside_deviation = np.sqrt(252) * negative_returns.std()
            sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_deviation if downside_deviation > 0 else 0
        else:
            sortino_ratio = 0
        
        # 交易统计
        winning_trades = [t for t in self.trades if self._calculate_trade_pnl(t) > 0]
        losing_trades = [t for t in self.trades if self._calculate_trade_pnl(t) < 0]
        
        total_trades = len(self.trades)
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)
        
        win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
        
        # 平均交易收益
        trade_pnls = [self._calculate_trade_pnl(t) for t in self.trades]
        avg_trade_return = np.mean(trade_pnls) if trade_pnls else 0
        avg_winning_trade = np.mean([self._calculate_trade_pnl(t) for t in winning_trades]) if winning_trades else 0
        avg_losing_trade = np.mean([self._calculate_trade_pnl(t) for t in losing_trades]) if losing_trades else 0
        
        # 盈利因子
        total_profit = sum([self._calculate_trade_pnl(t) for t in winning_trades])
        total_loss = abs(sum([self._calculate_trade_pnl(t) for t in losing_trades]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        # 连续盈亏统计
        consecutive_wins, consecutive_losses = self._calculate_consecutive_stats()
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades_count,
            losing_trades=losing_trades_count,
            avg_trade_return=avg_trade_return,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            max_consecutive_wins=consecutive_wins,
            max_consecutive_losses=consecutive_losses,
            equity_curve=equity_curve,
            trades=self.trades,
            positions=list(self.positions.values()),
            orders=self.order_history + self.pending_orders,
            daily_returns=daily_returns,
            drawdown_series=drawdown_series,
            metadata={
                'total_commission_paid': self.total_commission_paid,
                'total_slippage_cost': self.total_slippage_cost,
                'config': self.config
            }
        )
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """计算单笔交易的盈亏（简化版本）"""
        # 这里简化处理，实际应该根据开平仓配对计算
        # 暂时返回0，在实际使用中需要更复杂的逻辑
        return 0.0
    
    def _calculate_consecutive_stats(self) -> tuple:
        """计算连续盈亏统计"""
        if not self.trades:
            return 0, 0
        
        consecutive_wins = 0
        consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            pnl = self._calculate_trade_pnl(trade)
            
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                consecutive_wins = max(consecutive_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                consecutive_losses = max(consecutive_losses, current_losses)
        
        return consecutive_wins, consecutive_losses
    
    # 便利方法
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        current_data = self._get_current_data()
        if symbol in current_data:
            return current_data[symbol]['close']
        return None
    
    def get_current_bar_data(self, symbol: str) -> Optional[pd.Series]:
        """获取当前bar数据"""
        current_data = self._get_current_data()
        return current_data.get(symbol)
    
    def get_historical_data(self, symbol: str, lookback: int = 1) -> Optional[pd.DataFrame]:
        """获取历史数据"""
        if symbol not in self.data or symbol not in self.current_bar:
            return None
        
        current_idx = self.current_bar[symbol]
        start_idx = max(0, current_idx - lookback + 1)
        
        return self.data[symbol].iloc[start_idx:current_idx + 1]