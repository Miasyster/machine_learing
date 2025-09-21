"""
高级策略回测示例

演示如何实现和回测复杂的量化策略，包括：
- 多因子策略
- 配对交易策略
- 动态对冲策略
- 机器学习策略
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from backtest import (
    BarBacktestEngine, BacktestConfig, Order, OrderType, OrderSide,
    PercentageCommissionModel, FixedSlippageModel, analyze_backtest_result
)


def generate_correlated_data(symbols: List[str], days: int = 252, correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, pd.DataFrame]:
    """
    生成具有相关性的多资产数据
    
    Args:
        symbols: 资产代码列表
        days: 天数
        correlation_matrix: 相关性矩阵
        
    Returns:
        多资产数据字典
    """
    np.random.seed(42)
    n_assets = len(symbols)
    
    # 默认相关性矩阵
    if correlation_matrix is None:
        correlation_matrix = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    correlation_matrix[i, j] = 0.3  # 默认相关性0.3
    
    # 生成相关的随机收益率
    mean_returns = np.array([0.0005] * n_assets)  # 日均收益率
    volatilities = np.array([0.02] * n_assets)    # 日波动率
    
    # 使用Cholesky分解生成相关的随机数
    L = np.linalg.cholesky(correlation_matrix)
    random_normals = np.random.normal(0, 1, (days, n_assets))
    correlated_returns = random_normals @ L.T
    
    # 调整为所需的均值和方差
    for i in range(n_assets):
        correlated_returns[:, i] = mean_returns[i] + volatilities[i] * correlated_returns[:, i]
    
    # 生成价格数据
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=days, freq='D')
    
    data = {}
    for i, symbol in enumerate(symbols):
        initial_price = 100.0 + i * 10
        
        # 计算累积价格
        prices = [initial_price]
        for j in range(1, days):
            prices.append(prices[-1] * (1 + correlated_returns[j, i]))
        
        prices = np.array(prices)
        
        # 生成OHLC数据
        open_prices = np.roll(prices, 1)
        open_prices[0] = initial_price
        
        high_prices = np.maximum(open_prices, prices) * (1 + np.abs(np.random.normal(0, 0.003, days)))
        low_prices = np.minimum(open_prices, prices) * (1 - np.abs(np.random.normal(0, 0.003, days)))
        close_prices = prices
        
        volume = np.random.lognormal(np.log(1000000), 0.3, days).astype(int)
        
        data[symbol] = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
    
    return data


class MultiFactorStrategy:
    """多因子策略"""
    
    def __init__(self, symbols: List[str], rebalance_freq: int = 20):
        """
        初始化多因子策略
        
        Args:
            symbols: 股票池
            rebalance_freq: 调仓频率（天）
        """
        self.symbols = symbols
        self.rebalance_freq = rebalance_freq
        self.engine = None
        self.current_positions = {symbol: 0 for symbol in symbols}
        self.last_rebalance = 0
        self.price_history = {symbol: [] for symbol in symbols}
        self.volume_history = {symbol: [] for symbol in symbols}
        
    def set_engine(self, engine):
        """设置引擎引用"""
        self.engine = engine
        
    def calculate_factors(self, symbol: str, current_data: dict) -> dict:
        """
        计算因子值
        
        Args:
            symbol: 股票代码
            current_data: 当前数据
            
        Returns:
            因子字典
        """
        if symbol not in current_data:
            return {}
        
        bar = current_data[symbol]
        price_hist = self.price_history[symbol]
        volume_hist = self.volume_history[symbol]
        
        factors = {}
        
        # 动量因子
        if len(price_hist) >= 20:
            factors['momentum_20'] = (price_hist[-1] / price_hist[-20] - 1) if price_hist[-20] != 0 else 0
        
        if len(price_hist) >= 60:
            factors['momentum_60'] = (price_hist[-1] / price_hist[-60] - 1) if price_hist[-60] != 0 else 0
        
        # 反转因子
        if len(price_hist) >= 5:
            factors['reversal_5'] = -(price_hist[-1] / price_hist[-5] - 1) if price_hist[-5] != 0 else 0
        
        # 波动率因子
        if len(price_hist) >= 20:
            returns = np.diff(price_hist[-20:]) / price_hist[-20:-1]
            factors['volatility'] = -np.std(returns)  # 负号表示低波动率更好
        
        # 成交量因子
        if len(volume_hist) >= 20:
            avg_volume = np.mean(volume_hist[-20:])
            current_volume = bar['volume']
            factors['volume_ratio'] = current_volume / avg_volume if avg_volume != 0 else 1
        
        # 技术指标因子
        if len(price_hist) >= 14:
            # RSI
            gains = []
            losses = []
            for i in range(1, min(15, len(price_hist))):
                change = price_hist[-i] - price_hist[-i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                factors['rsi'] = -(abs(rsi - 50) / 50)  # 偏好RSI接近50的股票
            
        return factors
    
    def calculate_composite_score(self, factors: dict) -> float:
        """
        计算综合评分
        
        Args:
            factors: 因子字典
            
        Returns:
            综合评分
        """
        # 因子权重
        weights = {
            'momentum_20': 0.2,
            'momentum_60': 0.15,
            'reversal_5': 0.1,
            'volatility': 0.2,
            'volume_ratio': 0.1,
            'rsi': 0.25
        }
        
        score = 0
        total_weight = 0
        
        for factor, value in factors.items():
            if factor in weights and not np.isnan(value):
                score += weights[factor] * value
                total_weight += weights[factor]
        
        return score / total_weight if total_weight > 0 else 0
    
    def on_bar(self, timestamp, bar_data):
        """处理每个bar的数据"""
        # 更新价格和成交量历史
        for symbol in self.symbols:
            if symbol in bar_data:
                self.price_history[symbol].append(bar_data[symbol]['close'])
                self.volume_history[symbol].append(bar_data[symbol]['volume'])
                
                # 限制历史长度
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol] = self.price_history[symbol][-100:]
                    self.volume_history[symbol] = self.volume_history[symbol][-100:]
        
        # 检查是否需要调仓
        if self.last_rebalance >= self.rebalance_freq:
            self.rebalance(timestamp, bar_data)
            self.last_rebalance = 0
        else:
            self.last_rebalance += 1
    
    def rebalance(self, timestamp, bar_data):
        """调仓"""
        # 计算所有股票的评分
        scores = {}
        for symbol in self.symbols:
            if symbol in bar_data:
                factors = self.calculate_factors(symbol, bar_data)
                scores[symbol] = self.calculate_composite_score(factors)
        
        if not scores:
            return
        
        # 选择前N只股票
        n_stocks = min(3, len(scores))  # 最多持有3只股票
        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_stocks = [stock[0] for stock in sorted_stocks[:n_stocks]]
        
        # 计算目标权重
        target_weights = {symbol: 1.0 / n_stocks if symbol in selected_stocks else 0 
                         for symbol in self.symbols}
        
        # 执行调仓
        available_capital = self.engine.current_capital
        
        for symbol in self.symbols:
            if symbol not in bar_data:
                continue
                
            current_price = bar_data[symbol]['close']
            target_value = available_capital * target_weights[symbol]
            target_quantity = int(target_value / current_price) if current_price > 0 else 0
            
            current_quantity = self.current_positions[symbol]
            quantity_diff = target_quantity - current_quantity
            
            if abs(quantity_diff) > 0:
                side = OrderSide.BUY if quantity_diff > 0 else OrderSide.SELL
                order = Order(
                    order_id=f"rebalance_{symbol}_{timestamp.strftime('%Y%m%d')}",
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=abs(quantity_diff),
                    timestamp=timestamp
                )
                self.engine.place_order(order)
                self.current_positions[symbol] = target_quantity
        
        print(f"{timestamp.date()}: 调仓完成，选择股票: {selected_stocks}")


class PairsTradeStrategy:
    """配对交易策略"""
    
    def __init__(self, symbol1: str, symbol2: str, lookback_period: int = 60, 
                 entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        """
        初始化配对交易策略
        
        Args:
            symbol1: 股票1
            symbol2: 股票2
            lookback_period: 回看周期
            entry_threshold: 开仓阈值
            exit_threshold: 平仓阈值
        """
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        self.engine = None
        self.price_history1 = []
        self.price_history2 = []
        self.spread_history = []
        self.position_symbol1 = 0
        self.position_symbol2 = 0
        self.in_trade = False
        
    def set_engine(self, engine):
        """设置引擎引用"""
        self.engine = engine
        
    def calculate_spread(self) -> Optional[float]:
        """计算价差"""
        if len(self.price_history1) < 2 or len(self.price_history2) < 2:
            return None
        
        # 计算对数价格比率
        log_ratio = np.log(self.price_history1[-1]) - np.log(self.price_history2[-1])
        return log_ratio
    
    def calculate_zscore(self) -> Optional[float]:
        """计算Z分数"""
        if len(self.spread_history) < self.lookback_period:
            return None
        
        recent_spreads = self.spread_history[-self.lookback_period:]
        mean_spread = np.mean(recent_spreads)
        std_spread = np.std(recent_spreads)
        
        if std_spread == 0:
            return 0
        
        current_spread = self.spread_history[-1]
        z_score = (current_spread - mean_spread) / std_spread
        return z_score
    
    def on_bar(self, timestamp, bar_data):
        """处理每个bar的数据"""
        # 检查数据可用性
        if self.symbol1 not in bar_data or self.symbol2 not in bar_data:
            return
        
        price1 = bar_data[self.symbol1]['close']
        price2 = bar_data[self.symbol2]['close']
        
        # 更新价格历史
        self.price_history1.append(price1)
        self.price_history2.append(price2)
        
        # 计算价差
        spread = self.calculate_spread()
        if spread is not None:
            self.spread_history.append(spread)
        
        # 限制历史长度
        max_history = self.lookback_period * 2
        if len(self.price_history1) > max_history:
            self.price_history1 = self.price_history1[-max_history:]
            self.price_history2 = self.price_history2[-max_history:]
            self.spread_history = self.spread_history[-max_history:]
        
        # 计算Z分数
        z_score = self.calculate_zscore()
        if z_score is None:
            return
        
        # 交易逻辑
        if not self.in_trade:
            # 开仓条件
            if z_score > self.entry_threshold:
                # 价差过高，做空价差：卖出symbol1，买入symbol2
                self.open_position(timestamp, "short_spread", price1, price2)
            elif z_score < -self.entry_threshold:
                # 价差过低，做多价差：买入symbol1，卖出symbol2
                self.open_position(timestamp, "long_spread", price1, price2)
        else:
            # 平仓条件
            if abs(z_score) < self.exit_threshold:
                self.close_position(timestamp, price1, price2)
    
    def open_position(self, timestamp, direction: str, price1: float, price2: float):
        """开仓"""
        # 计算仓位大小
        available_capital = self.engine.current_capital * 0.4  # 使用40%资金
        
        if direction == "long_spread":
            # 买入symbol1，卖出symbol2
            quantity1 = int(available_capital * 0.5 / price1)
            quantity2 = int(available_capital * 0.5 / price2)
            
            if quantity1 > 0:
                order1 = Order(
                    order_id=f"pairs_buy_{self.symbol1}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    symbol=self.symbol1,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity1,
                    timestamp=timestamp
                )
                self.engine.place_order(order1)
                self.position_symbol1 = quantity1
            
            if quantity2 > 0:
                order2 = Order(
                    order_id=f"pairs_sell_{self.symbol2}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    symbol=self.symbol2,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity2,
                    timestamp=timestamp
                )
                self.engine.place_order(order2)
                self.position_symbol2 = -quantity2
                
        else:  # short_spread
            # 卖出symbol1，买入symbol2
            quantity1 = int(available_capital * 0.5 / price1)
            quantity2 = int(available_capital * 0.5 / price2)
            
            if quantity1 > 0:
                order1 = Order(
                    order_id=f"pairs_sell_{self.symbol1}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    symbol=self.symbol1,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity1,
                    timestamp=timestamp
                )
                self.engine.place_order(order1)
                self.position_symbol1 = -quantity1
            
            if quantity2 > 0:
                order2 = Order(
                    order_id=f"pairs_buy_{self.symbol2}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    symbol=self.symbol2,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity2,
                    timestamp=timestamp
                )
                self.engine.place_order(order2)
                self.position_symbol2 = quantity2
        
        self.in_trade = True
        print(f"{timestamp.date()}: 开仓配对交易 {direction}")
    
    def close_position(self, timestamp, price1: float, price2: float):
        """平仓"""
        # 平仓symbol1
        if self.position_symbol1 != 0:
            side1 = OrderSide.SELL if self.position_symbol1 > 0 else OrderSide.BUY
            order1 = Order(
                order_id=f"pairs_close_{self.symbol1}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                symbol=self.symbol1,
                side=side1,
                order_type=OrderType.MARKET,
                quantity=abs(self.position_symbol1),
                timestamp=timestamp
            )
            self.engine.place_order(order1)
        
        # 平仓symbol2
        if self.position_symbol2 != 0:
            side2 = OrderSide.SELL if self.position_symbol2 > 0 else OrderSide.BUY
            order2 = Order(
                order_id=f"pairs_close_{self.symbol2}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                symbol=self.symbol2,
                side=side2,
                order_type=OrderType.MARKET,
                quantity=abs(self.position_symbol2),
                timestamp=timestamp
            )
            self.engine.place_order(order2)
        
        print(f"{timestamp.date()}: 平仓配对交易")
        
        # 重置状态
        self.position_symbol1 = 0
        self.position_symbol2 = 0
        self.in_trade = False


class MLPredictionStrategy:
    """机器学习预测策略"""
    
    def __init__(self, symbol: str, feature_window: int = 20, prediction_horizon: int = 5):
        """
        初始化机器学习策略
        
        Args:
            symbol: 交易标的
            feature_window: 特征窗口
            prediction_horizon: 预测周期
        """
        self.symbol = symbol
        self.feature_window = feature_window
        self.prediction_horizon = prediction_horizon
        
        self.engine = None
        self.price_history = []
        self.volume_history = []
        self.feature_history = []
        self.model = None
        self.model_trained = False
        self.position = 0
        
        # 尝试导入机器学习库
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            self.RandomForestRegressor = RandomForestRegressor
            self.StandardScaler = StandardScaler
            self.scaler = StandardScaler()
            self.ml_available = True
        except ImportError:
            print("scikit-learn不可用，ML策略将使用简化版本")
            self.ml_available = False
    
    def set_engine(self, engine):
        """设置引擎引用"""
        self.engine = engine
    
    def extract_features(self, prices: List[float], volumes: List[float]) -> List[float]:
        """提取特征"""
        if len(prices) < self.feature_window:
            return []
        
        features = []
        recent_prices = prices[-self.feature_window:]
        recent_volumes = volumes[-self.feature_window:]
        
        # 价格特征
        features.append(recent_prices[-1] / recent_prices[0] - 1)  # 总收益率
        features.append(np.mean(recent_prices))  # 平均价格
        features.append(np.std(recent_prices))   # 价格标准差
        
        # 技术指标特征
        if len(recent_prices) >= 5:
            ma5 = np.mean(recent_prices[-5:])
            features.append(recent_prices[-1] / ma5 - 1)  # 相对于5日均线
        else:
            features.append(0)
        
        if len(recent_prices) >= 10:
            ma10 = np.mean(recent_prices[-10:])
            features.append(recent_prices[-1] / ma10 - 1)  # 相对于10日均线
        else:
            features.append(0)
        
        # 动量特征
        if len(recent_prices) >= 3:
            momentum = (recent_prices[-1] - recent_prices[-3]) / recent_prices[-3]
            features.append(momentum)
        else:
            features.append(0)
        
        # 成交量特征
        features.append(np.mean(recent_volumes))  # 平均成交量
        features.append(recent_volumes[-1] / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 1)  # 成交量比率
        
        # RSI特征
        if len(recent_prices) >= 14:
            gains = []
            losses = []
            for i in range(1, 14):
                change = recent_prices[-i] - recent_prices[-i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi / 100)  # 归一化RSI
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        return features
    
    def train_model(self):
        """训练模型"""
        if not self.ml_available or len(self.feature_history) < 50:
            return False
        
        # 准备训练数据
        X = []
        y = []
        
        for i in range(len(self.feature_history) - self.prediction_horizon):
            features = self.feature_history[i]
            if len(features) > 0:
                # 目标：未来N天的收益率
                future_price = self.price_history[i + self.prediction_horizon]
                current_price = self.price_history[i]
                future_return = (future_price - current_price) / current_price
                
                X.append(features)
                y.append(future_return)
        
        if len(X) < 20:  # 需要足够的训练样本
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.model = self.RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        self.model_trained = True
        
        return True
    
    def predict(self, features: List[float]) -> float:
        """预测"""
        if not self.model_trained or not features:
            return 0
        
        try:
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            return prediction
        except:
            return 0
    
    def on_bar(self, timestamp, bar_data):
        """处理每个bar的数据"""
        if self.symbol not in bar_data:
            return
        
        current_price = bar_data[self.symbol]['close']
        current_volume = bar_data[self.symbol]['volume']
        
        # 更新历史数据
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        # 提取特征
        features = self.extract_features(self.price_history, self.volume_history)
        if features:
            self.feature_history.append(features)
        
        # 限制历史长度
        max_history = 200
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            self.feature_history = self.feature_history[-max_history:]
        
        # 训练模型（每50个bar重新训练一次）
        if len(self.feature_history) % 50 == 0 and len(self.feature_history) >= 50:
            if self.train_model():
                print(f"{timestamp.date()}: 模型重新训练完成")
        
        # 预测和交易
        if self.model_trained and features:
            prediction = self.predict(features)
            
            # 交易逻辑
            threshold = 0.02  # 2%的预测阈值
            
            if prediction > threshold and self.position <= 0:
                # 预测上涨，买入
                available_capital = self.engine.current_capital * 0.8
                quantity = int(available_capital / current_price)
                
                if quantity > 0:
                    # 先平空仓
                    if self.position < 0:
                        order = Order(
                            order_id=f"ml_cover_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                            symbol=self.symbol,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=abs(self.position),
                            timestamp=timestamp
                        )
                        self.engine.place_order(order)
                    
                    # 买入
                    order = Order(
                        order_id=f"ml_buy_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                        symbol=self.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        timestamp=timestamp
                    )
                    self.engine.place_order(order)
                    self.position = quantity
                    print(f"{timestamp.date()}: ML预测买入，预测收益率: {prediction:.3f}")
            
            elif prediction < -threshold and self.position >= 0:
                # 预测下跌，卖出
                if self.position > 0:
                    # 平多仓
                    order = Order(
                        order_id=f"ml_sell_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                        symbol=self.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=self.position,
                        timestamp=timestamp
                    )
                    self.engine.place_order(order)
                    self.position = 0
                    print(f"{timestamp.date()}: ML预测卖出，预测收益率: {prediction:.3f}")


def run_advanced_strategies_example():
    """运行高级策略示例"""
    print("=" * 60)
    print("高级策略回测示例")
    print("=" * 60)
    
    # 1. 生成相关性数据
    print("\n1. 生成多资产相关性数据...")
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    # 设置相关性矩阵
    correlation_matrix = np.array([
        [1.0, 0.6, 0.7, 0.3, 0.5],  # AAPL
        [0.6, 1.0, 0.8, 0.4, 0.7],  # GOOGL
        [0.7, 0.8, 1.0, 0.3, 0.6],  # MSFT
        [0.3, 0.4, 0.3, 1.0, 0.4],  # TSLA
        [0.5, 0.7, 0.6, 0.4, 1.0]   # AMZN
    ])
    
    data_dict = generate_correlated_data(symbols, days=252, correlation_matrix=correlation_matrix)
    
    print(f"生成了{len(symbols)}只股票的数据")
    print(f"数据范围: {data_dict[symbols[0]].index[0].date()} 到 {data_dict[symbols[0]].index[-1].date()}")
    
    # 2. 配置回测参数
    config = BacktestConfig(
        initial_capital=200000.0,  # 20万初始资金
        commission_model=PercentageCommissionModel(commission_rate=0.001),
        slippage_model=FixedSlippageModel(slippage_bps=5.0),
        max_position_size=0.95
    )
    
    # 3. 运行多因子策略
    print("\n2. 运行多因子策略...")
    mf_engine = BarBacktestEngine(config)
    
    # 添加数据
    for symbol, data in data_dict.items():
        mf_engine.add_data(data, symbol)
    
    mf_strategy = MultiFactorStrategy(symbols, rebalance_freq=20)
    mf_engine.add_strategy(mf_strategy)
    
    mf_result = mf_engine.run()
    
    # 4. 运行配对交易策略
    print("\n3. 运行配对交易策略...")
    pairs_engine = BarBacktestEngine(config)
    
    # 选择相关性较高的两只股票进行配对交易
    symbol1, symbol2 = "GOOGL", "MSFT"  # 相关性0.8
    pairs_engine.add_data(data_dict[symbol1], symbol1)
    pairs_engine.add_data(data_dict[symbol2], symbol2)
    
    pairs_strategy = PairsTradeStrategy(
        symbol1=symbol1, 
        symbol2=symbol2,
        lookback_period=60,
        entry_threshold=2.0,
        exit_threshold=0.5
    )
    pairs_engine.add_strategy(pairs_strategy)
    
    pairs_result = pairs_engine.run()
    
    # 5. 运行机器学习策略
    print("\n4. 运行机器学习策略...")
    ml_engine = BarBacktestEngine(config)
    
    # 使用AAPL进行ML策略测试
    ml_symbol = "AAPL"
    ml_engine.add_data(data_dict[ml_symbol], ml_symbol)
    
    ml_strategy = MLPredictionStrategy(
        symbol=ml_symbol,
        feature_window=20,
        prediction_horizon=5
    )
    ml_engine.add_strategy(ml_strategy)
    
    ml_result = ml_engine.run()
    
    # 6. 基准策略：买入持有
    print("\n5. 运行基准策略（等权重买入持有）...")
    benchmark_engine = BarBacktestEngine(config)
    
    for symbol, data in data_dict.items():
        benchmark_engine.add_data(data, symbol)
    
    class BenchmarkStrategy:
        def __init__(self, symbols):
            self.symbols = symbols
            self.engine = None
            self.bought = False
            
        def set_engine(self, engine):
            self.engine = engine
            
        def on_bar(self, timestamp, bar_data):
            if not self.bought:
                # 等权重买入所有股票
                available_capital = self.engine.current_capital
                capital_per_stock = available_capital / len(self.symbols)
                
                for symbol in self.symbols:
                    if symbol in bar_data:
                        price = bar_data[symbol]['close']
                        quantity = int(capital_per_stock * 0.95 / price)
                        
                        if quantity > 0:
                            order = Order(
                                order_id=f"benchmark_{symbol}_{timestamp.strftime('%Y%m%d')}",
                                symbol=symbol,
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                quantity=quantity,
                                timestamp=timestamp
                            )
                            self.engine.place_order(order)
                
                self.bought = True
    
    benchmark_strategy = BenchmarkStrategy(symbols)
    benchmark_engine.add_strategy(benchmark_strategy)
    
    benchmark_result = benchmark_engine.run()
    
    # 7. 结果对比分析
    print("\n6. 策略性能对比")
    print("=" * 80)
    
    strategies = {
        "多因子策略": mf_result,
        "配对交易策略": pairs_result,
        "机器学习策略": ml_result,
        "基准策略(等权重)": benchmark_result
    }
    
    print(f"{'策略名称':<15} {'总收益率':<10} {'年化收益率':<12} {'最大回撤':<10} {'夏普比率':<10} {'交易次数':<8}")
    print("-" * 80)
    
    for name, result in strategies.items():
        print(f"{name:<15} {result.total_return:<10.2%} {result.annualized_return:<12.2%} "
              f"{result.max_drawdown:<10.2%} {result.sharpe_ratio:<10.3f} {result.total_trades:<8}")
    
    # 8. 详细分析
    print("\n7. 详细策略分析")
    print("=" * 60)
    
    for name, result in strategies.items():
        print(f"\n{name}:")
        analyzer = analyze_backtest_result(result)
        advanced_metrics = analyzer.calculate_advanced_metrics()
        
        print(f"  最终资金: ${result.final_capital:,.2f}")
        print(f"  年化波动率: {advanced_metrics.get('volatility', 0):.2%}")
        print(f"  Sortino比率: {advanced_metrics.get('sortino_ratio', 0):.3f}")
        print(f"  Calmar比率: {advanced_metrics.get('calmar_ratio', 0):.3f}")
        print(f"  胜率: {result.win_rate:.2%}")
        print(f"  盈利因子: {result.profit_factor:.3f}")
        print(f"  最大连续盈利: {result.max_consecutive_wins}")
        print(f"  最大连续亏损: {result.max_consecutive_losses}")
    
    # 9. 生成可视化
    print("\n8. 生成策略对比图表...")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(16, 12))
        
        # 权益曲线对比
        plt.subplot(2, 3, 1)
        for name, result in strategies.items():
            if result.equity_curve is not None:
                plt.plot(result.equity_curve.index, result.equity_curve.values, 
                        label=name, linewidth=2)
        
        plt.axhline(y=config.initial_capital, color='gray', linestyle='--', alpha=0.7)
        plt.title('权益曲线对比')
        plt.xlabel('时间')
        plt.ylabel('组合价值 ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 回撤对比
        plt.subplot(2, 3, 2)
        for name, result in strategies.items():
            if result.drawdown_series is not None:
                plt.fill_between(result.drawdown_series.index, 
                               result.drawdown_series.values * 100, 0,
                               alpha=0.3, label=name)
        
        plt.title('回撤对比')
        plt.xlabel('时间')
        plt.ylabel('回撤 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 收益率分布
        plt.subplot(2, 3, 3)
        for name, result in strategies.items():
            if result.daily_returns is not None and not result.daily_returns.empty:
                plt.hist(result.daily_returns * 100, bins=20, alpha=0.5, 
                        label=name, density=True)
        
        plt.title('日收益率分布')
        plt.xlabel('收益率 (%)')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 风险收益散点图
        plt.subplot(2, 3, 4)
        returns = [result.annualized_return for result in strategies.values()]
        volatilities = [analyze_backtest_result(result).calculate_advanced_metrics().get('volatility', 0) 
                       for result in strategies.values()]
        
        plt.scatter(volatilities, returns, s=100)
        for i, name in enumerate(strategies.keys()):
            plt.annotate(name, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('风险收益散点图')
        plt.xlabel('年化波动率')
        plt.ylabel('年化收益率')
        plt.grid(True, alpha=0.3)
        
        # 月度收益热力图（以多因子策略为例）
        plt.subplot(2, 3, 5)
        if mf_result.daily_returns is not None and not mf_result.daily_returns.empty:
            monthly_returns = mf_result.daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns_matrix = monthly_returns.values.reshape(-1, 1)
            
            im = plt.imshow(monthly_returns_matrix.T, cmap='RdYlGn', aspect='auto')
            plt.colorbar(im, label='月度收益率')
            plt.title('多因子策略月度收益率')
            plt.xlabel('月份')
            plt.ylabel('收益率')
        
        # 滚动夏普比率
        plt.subplot(2, 3, 6)
        for name, result in strategies.items():
            if result.daily_returns is not None and not result.daily_returns.empty:
                rolling_sharpe = (result.daily_returns.rolling(60).mean() / 
                                result.daily_returns.rolling(60).std() * np.sqrt(252))
                plt.plot(rolling_sharpe.index, rolling_sharpe.values, 
                        label=name, linewidth=2)
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.title('滚动夏普比率 (60天)')
        plt.xlabel('时间')
        plt.ylabel('夏普比率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("图表已生成并显示")
        
    except ImportError:
        print("matplotlib未安装，跳过图表生成")
    except Exception as e:
        print(f"图表生成失败: {e}")
    
    print("\n" + "=" * 60)
    print("高级策略回测完成！")
    print("=" * 60)
    
    return {
        "multifactor_result": mf_result,
        "pairs_result": pairs_result,
        "ml_result": ml_result,
        "benchmark_result": benchmark_result
    }


if __name__ == "__main__":
    # 运行高级策略示例
    results = run_advanced_strategies_example()
    
    print("\n策略总结:")
    print("1. 多因子策略：基于动量、反转、波动率等多个因子进行选股和权重分配")
    print("2. 配对交易策略：利用股票间的协整关系进行统计套利")
    print("3. 机器学习策略：使用随机森林模型预测未来收益率")
    print("4. 基准策略：等权重买入持有所有股票")
    
    print("\n可以进一步分析:")
    print("- 策略的风险暴露分析")
    print("- 不同市场环境下的表现")
    print("- 参数敏感性分析")
    print("- 组合优化和资产配置")