"""
VectorBT集成模块

提供基于vectorbt的高性能向量化回测功能，包括：
- 向量化回测引擎
- 性能优化
- 并行计算
- 大规模数据处理
"""

import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("vectorbt未安装，将使用基础回测引擎")

from .base import (
    BacktestEngine, BacktestResult, BacktestConfig, Order, Trade, Position,
    OrderType, OrderSide, OrderStatus, PositionSide
)


class VectorBTBacktestEngine(BacktestEngine):
    """基于VectorBT的向量化回测引擎"""
    
    def __init__(self, config: BacktestConfig = None):
        """
        初始化VectorBT回测引擎
        
        Args:
            config: 回测配置
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("需要安装vectorbt: pip install vectorbt")
        
        super().__init__(config.initial_capital if config else 100000.0)
        self.config = config or BacktestConfig()
        
        # VectorBT相关
        self.vbt_data: Optional[pd.DataFrame] = None
        self.vbt_portfolio: Optional[Any] = None
        
        # 策略信号
        self.buy_signals: Optional[pd.DataFrame] = None
        self.sell_signals: Optional[pd.DataFrame] = None
        self.size_signals: Optional[pd.DataFrame] = None
        
        # 性能设置
        self.use_numba = True
        self.parallel = True
        self.chunk_size = 10000
        
    def add_data(self, data: pd.DataFrame, symbol: str = "default"):
        """
        添加回测数据
        
        Args:
            data: OHLCV数据
            symbol: 标的代码
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据必须有datetime索引")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必要列: {missing_columns}")
        
        # 存储原始数据
        self.data[symbol] = data.sort_index()
        
        # 为VectorBT准备数据
        if self.vbt_data is None:
            self.vbt_data = pd.DataFrame(index=data.index)
        
        # 添加价格数据
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.vbt_data[f"{symbol}_{col}"] = data[col]
    
    def add_strategy(self, strategy):
        """
        添加交易策略
        
        Args:
            strategy: 策略对象，需要实现generate_signals方法
        """
        if not hasattr(strategy, 'generate_signals'):
            raise ValueError("策略必须实现generate_signals方法")
        
        self.strategies.append(strategy)
        
        # 为策略提供引擎引用
        if hasattr(strategy, 'set_engine'):
            strategy.set_engine(self)
    
    def run(self) -> BacktestResult:
        """
        运行向量化回测
        
        Returns:
            回测结果
        """
        if self.vbt_data is None or self.vbt_data.empty:
            raise ValueError("没有添加回测数据")
        
        if not self.strategies:
            raise ValueError("没有添加交易策略")
        
        print("开始向量化回测...")
        
        # 生成交易信号
        self._generate_signals()
        
        # 运行VectorBT回测
        self._run_vectorbt_backtest()
        
        # 生成回测结果
        result = self._generate_vectorbt_result()
        
        print(f"向量化回测完成! 最终组合价值: {result.final_capital:,.2f}")
        
        return result
    
    def _generate_signals(self):
        """生成交易信号"""
        symbols = list(self.data.keys())
        
        # 初始化信号DataFrame
        self.buy_signals = pd.DataFrame(
            False, 
            index=self.vbt_data.index, 
            columns=symbols
        )
        self.sell_signals = pd.DataFrame(
            False, 
            index=self.vbt_data.index, 
            columns=symbols
        )
        self.size_signals = pd.DataFrame(
            np.nan, 
            index=self.vbt_data.index, 
            columns=symbols
        )
        
        # 为每个策略生成信号
        for strategy in self.strategies:
            try:
                strategy_signals = strategy.generate_signals(self.vbt_data, symbols)
                
                if 'buy' in strategy_signals:
                    self.buy_signals |= strategy_signals['buy']
                if 'sell' in strategy_signals:
                    self.sell_signals |= strategy_signals['sell']
                if 'size' in strategy_signals:
                    self.size_signals = strategy_signals['size'].fillna(self.size_signals)
                    
            except Exception as e:
                print(f"策略信号生成错误: {e}")
                continue
        
        # 填充默认仓位大小
        self.size_signals = self.size_signals.fillna(1.0)
    
    def _run_vectorbt_backtest(self):
        """运行VectorBT回测"""
        symbols = list(self.data.keys())
        
        # 准备价格数据
        close_prices = pd.DataFrame({
            symbol: self.vbt_data[f"{symbol}_close"] 
            for symbol in symbols
        })
        
        # 设置VectorBT参数
        vbt_kwargs = {
            'init_cash': self.initial_capital,
            'fees': self._get_commission_rate(),
            'slippage': self._get_slippage_rate(),
            'freq': pd.infer_freq(close_prices.index) or 'D'
        }
        
        # 添加高级参数
        if hasattr(self.config, 'max_position_size') and self.config.max_position_size:
            vbt_kwargs['size_type'] = 'percent'
            vbt_kwargs['max_size'] = self.config.max_position_size
        
        # 运行回测
        try:
            if self.parallel and len(symbols) > 1:
                # 并行回测多个标的
                self.vbt_portfolio = vbt.Portfolio.from_signals(
                    close_prices,
                    entries=self.buy_signals,
                    exits=self.sell_signals,
                    size=self.size_signals,
                    **vbt_kwargs
                )
            else:
                # 单线程回测
                self.vbt_portfolio = vbt.Portfolio.from_signals(
                    close_prices,
                    entries=self.buy_signals,
                    exits=self.sell_signals,
                    size=self.size_signals,
                    **vbt_kwargs
                )
                
        except Exception as e:
            print(f"VectorBT回测执行错误: {e}")
            # 降级到基础参数重试
            basic_kwargs = {
                'init_cash': self.initial_capital,
                'fees': 0.001,
                'freq': 'D'
            }
            
            self.vbt_portfolio = vbt.Portfolio.from_signals(
                close_prices,
                entries=self.buy_signals,
                exits=self.sell_signals,
                **basic_kwargs
            )
    
    def _get_commission_rate(self) -> float:
        """获取手续费率"""
        from .base import PercentageCommissionModel
        
        if isinstance(self.config.commission_model, PercentageCommissionModel):
            return self.config.commission_model.commission_rate
        else:
            # 默认手续费率
            return 0.001
    
    def _get_slippage_rate(self) -> float:
        """获取滑点率"""
        from .base import FixedSlippageModel
        
        if isinstance(self.config.slippage_model, FixedSlippageModel):
            return self.config.slippage_model.slippage_bps / 10000.0
        else:
            # 默认滑点率
            return 0.0005
    
    def _generate_vectorbt_result(self) -> BacktestResult:
        """生成VectorBT回测结果"""
        if self.vbt_portfolio is None:
            raise ValueError("回测尚未运行")
        
        # 基本统计
        stats = self.vbt_portfolio.stats()
        
        start_date = self.vbt_data.index[0]
        end_date = self.vbt_data.index[-1]
        initial_capital = self.initial_capital
        final_capital = float(stats['End Value'])
        
        # 计算收益率
        total_return = float(stats['Total Return [%]']) / 100.0
        
        # 获取详细指标
        try:
            max_drawdown = abs(float(stats['Max Drawdown [%]']) / 100.0)
        except:
            max_drawdown = 0.0
        
        try:
            sharpe_ratio = float(stats['Sharpe Ratio'])
        except:
            sharpe_ratio = 0.0
        
        try:
            sortino_ratio = float(stats['Sortino Ratio'])
        except:
            sortino_ratio = 0.0
        
        # 交易统计
        trades_df = self.vbt_portfolio.trades.records_readable
        
        if not trades_df.empty:
            winning_trades = len(trades_df[trades_df['PnL'] > 0])
            losing_trades = len(trades_df[trades_df['PnL'] < 0])
            total_trades = len(trades_df)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_trade_return = trades_df['PnL'].mean() if not trades_df.empty else 0
            avg_winning_trade = trades_df[trades_df['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
            avg_losing_trade = trades_df[trades_df['PnL'] < 0]['PnL'].mean() if losing_trades > 0 else 0
            
            # 盈利因子
            total_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
            total_loss = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
            
            # 连续盈亏统计
            consecutive_wins, consecutive_losses = self._calculate_consecutive_stats_vbt(trades_df)
        else:
            winning_trades = losing_trades = total_trades = 0
            win_rate = avg_trade_return = avg_winning_trade = avg_losing_trade = 0
            profit_factor = 0
            consecutive_wins = consecutive_losses = 0
        
        # 权益曲线
        equity_curve = self.vbt_portfolio.value()
        
        # 日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 回撤序列
        rolling_max = equity_curve.expanding().max()
        drawdown_series = (equity_curve - rolling_max) / rolling_max
        
        # 年化收益率
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (final_capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # 转换交易记录
        trades_list = self._convert_vbt_trades(trades_df)
        
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
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_trade_return=avg_trade_return,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            max_consecutive_wins=consecutive_wins,
            max_consecutive_losses=consecutive_losses,
            equity_curve=equity_curve,
            trades=trades_list,
            positions=[],  # VectorBT不直接提供持仓历史
            orders=[],     # VectorBT不直接提供订单历史
            daily_returns=daily_returns,
            drawdown_series=drawdown_series,
            metadata={
                'vectorbt_stats': stats,
                'config': self.config,
                'engine_type': 'vectorbt'
            }
        )
    
    def _convert_vbt_trades(self, trades_df: pd.DataFrame) -> List[Trade]:
        """转换VectorBT交易记录"""
        trades_list = []
        
        for idx, trade_row in trades_df.iterrows():
            # 入场交易
            entry_trade = Trade(
                trade_id=f"entry_{idx}",
                order_id=f"order_entry_{idx}",
                symbol=str(trade_row.get('Column', 'default')),
                side=OrderSide.BUY,
                quantity=float(trade_row['Size']),
                price=float(trade_row['Entry Price']),
                timestamp=pd.to_datetime(trade_row['Entry Timestamp']),
                commission=float(trade_row.get('Entry Fees', 0)),
                metadata={'trade_type': 'entry', 'vbt_trade_id': idx}
            )
            trades_list.append(entry_trade)
            
            # 出场交易
            exit_trade = Trade(
                trade_id=f"exit_{idx}",
                order_id=f"order_exit_{idx}",
                symbol=str(trade_row.get('Column', 'default')),
                side=OrderSide.SELL,
                quantity=float(trade_row['Size']),
                price=float(trade_row['Exit Price']),
                timestamp=pd.to_datetime(trade_row['Exit Timestamp']),
                commission=float(trade_row.get('Exit Fees', 0)),
                metadata={'trade_type': 'exit', 'vbt_trade_id': idx, 'pnl': float(trade_row['PnL'])}
            )
            trades_list.append(exit_trade)
        
        return trades_list
    
    def _calculate_consecutive_stats_vbt(self, trades_df: pd.DataFrame) -> Tuple[int, int]:
        """计算连续盈亏统计"""
        if trades_df.empty:
            return 0, 0
        
        consecutive_wins = 0
        consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for _, trade in trades_df.iterrows():
            pnl = trade['PnL']
            
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                consecutive_wins = max(consecutive_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                consecutive_losses = max(consecutive_losses, current_losses)
        
        return consecutive_wins, consecutive_losses
    
    # VectorBT特有的方法
    def get_vectorbt_portfolio(self):
        """获取VectorBT组合对象"""
        return self.vbt_portfolio
    
    def plot_portfolio(self, **kwargs):
        """绘制组合图表"""
        if self.vbt_portfolio is None:
            raise ValueError("回测尚未运行")
        
        return self.vbt_portfolio.plot(**kwargs)
    
    def get_detailed_stats(self) -> pd.Series:
        """获取详细统计信息"""
        if self.vbt_portfolio is None:
            raise ValueError("回测尚未运行")
        
        return self.vbt_portfolio.stats()
    
    def get_trades_analysis(self) -> pd.DataFrame:
        """获取交易分析"""
        if self.vbt_portfolio is None:
            raise ValueError("回测尚未运行")
        
        return self.vbt_portfolio.trades.records_readable
    
    def get_positions_analysis(self) -> pd.DataFrame:
        """获取持仓分析"""
        if self.vbt_portfolio is None:
            raise ValueError("回测尚未运行")
        
        try:
            return self.vbt_portfolio.positions.records_readable
        except:
            return pd.DataFrame()
    
    def optimize_parameters(self, param_grid: Dict[str, List], 
                          metric: str = 'total_return') -> Dict:
        """
        参数优化
        
        Args:
            param_grid: 参数网格
            metric: 优化目标指标
            
        Returns:
            最优参数和结果
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("需要安装vectorbt进行参数优化")
        
        best_params = None
        best_score = float('-inf')
        results = []
        
        # 生成参数组合
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            try:
                # 使用当前参数运行回测
                temp_engine = VectorBTBacktestEngine(self.config)
                
                # 复制数据
                for symbol, data in self.data.items():
                    temp_engine.add_data(data, symbol)
                
                # 复制策略并设置参数
                for strategy in self.strategies:
                    temp_strategy = strategy.__class__(**params)
                    temp_engine.add_strategy(temp_strategy)
                
                # 运行回测
                result = temp_engine.run()
                
                # 评估指标
                if metric == 'total_return':
                    score = result.total_return
                elif metric == 'sharpe_ratio':
                    score = result.sharpe_ratio
                elif metric == 'sortino_ratio':
                    score = result.sortino_ratio
                elif metric == 'profit_factor':
                    score = result.profit_factor
                else:
                    score = result.total_return
                
                results.append({
                    'params': params,
                    'score': score,
                    'result': result
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                print(f"参数组合 {params} 优化失败: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }


class VectorBTStrategy:
    """VectorBT策略基类"""
    
    def __init__(self, **kwargs):
        """初始化策略参数"""
        self.params = kwargs
        self.engine = None
    
    def set_engine(self, engine):
        """设置引擎引用"""
        self.engine = engine
    
    def generate_signals(self, data: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        生成交易信号
        
        Args:
            data: 价格数据
            symbols: 标的列表
            
        Returns:
            包含buy、sell、size信号的字典
        """
        raise NotImplementedError("子类必须实现generate_signals方法")


class MovingAverageCrossStrategy(VectorBTStrategy):
    """移动平均线交叉策略"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20, **kwargs):
        """
        初始化移动平均线策略
        
        Args:
            fast_period: 快速移动平均线周期
            slow_period: 慢速移动平均线周期
        """
        super().__init__(fast_period=fast_period, slow_period=slow_period, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """生成移动平均线交叉信号"""
        buy_signals = pd.DataFrame(False, index=data.index, columns=symbols)
        sell_signals = pd.DataFrame(False, index=data.index, columns=symbols)
        size_signals = pd.DataFrame(1.0, index=data.index, columns=symbols)
        
        for symbol in symbols:
            close_col = f"{symbol}_close"
            if close_col not in data.columns:
                continue
            
            close_prices = data[close_col]
            
            # 计算移动平均线
            fast_ma = close_prices.rolling(window=self.fast_period).mean()
            slow_ma = close_prices.rolling(window=self.slow_period).mean()
            
            # 生成信号
            buy_signals[symbol] = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            sell_signals[symbol] = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        return {
            'buy': buy_signals,
            'sell': sell_signals,
            'size': size_signals
        }


class RSIStrategy(VectorBTStrategy):
    """RSI策略"""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70, **kwargs):
        """
        初始化RSI策略
        
        Args:
            rsi_period: RSI计算周期
            oversold: 超卖阈值
            overbought: 超买阈值
        """
        super().__init__(rsi_period=rsi_period, oversold=oversold, overbought=overbought, **kwargs)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """生成RSI信号"""
        buy_signals = pd.DataFrame(False, index=data.index, columns=symbols)
        sell_signals = pd.DataFrame(False, index=data.index, columns=symbols)
        size_signals = pd.DataFrame(1.0, index=data.index, columns=symbols)
        
        for symbol in symbols:
            close_col = f"{symbol}_close"
            if close_col not in data.columns:
                continue
            
            close_prices = data[close_col]
            
            # 计算RSI
            rsi = self._calculate_rsi(close_prices, self.rsi_period)
            
            # 生成信号
            buy_signals[symbol] = (rsi < self.oversold) & (rsi.shift(1) >= self.oversold)
            sell_signals[symbol] = (rsi > self.overbought) & (rsi.shift(1) <= self.overbought)
        
        return {
            'buy': buy_signals,
            'sell': sell_signals,
            'size': size_signals
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


# 便利函数
def create_vectorbt_engine(config: BacktestConfig = None) -> VectorBTBacktestEngine:
    """创建VectorBT回测引擎"""
    return VectorBTBacktestEngine(config)


def run_vectorbt_backtest(data: Dict[str, pd.DataFrame], 
                         strategies: List[VectorBTStrategy],
                         config: BacktestConfig = None) -> BacktestResult:
    """
    运行VectorBT回测的便利函数
    
    Args:
        data: 数据字典 {symbol: DataFrame}
        strategies: 策略列表
        config: 回测配置
        
    Returns:
        回测结果
    """
    engine = create_vectorbt_engine(config)
    
    # 添加数据
    for symbol, symbol_data in data.items():
        engine.add_data(symbol_data, symbol)
    
    # 添加策略
    for strategy in strategies:
        engine.add_strategy(strategy)
    
    # 运行回测
    return engine.run()