"""
基础回测引擎使用示例

演示如何使用回测引擎进行简单的策略回测，包括：
- 数据准备
- 策略实现
- 回测执行
- 结果分析
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from backtest import (
    BarBacktestEngine, BacktestConfig, Order, OrderType, OrderSide,
    PercentageCommissionModel, FixedSlippageModel, analyze_backtest_result
)


def generate_sample_data(symbol: str = "AAPL", days: int = 252) -> pd.DataFrame:
    """
    生成示例股票数据
    
    Args:
        symbol: 股票代码
        days: 天数
        
    Returns:
        OHLCV数据
    """
    # 设置随机种子以获得可重复的结果
    np.random.seed(42)
    
    # 生成日期范围
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=days, freq='D')
    
    # 生成价格数据（几何布朗运动）
    initial_price = 100.0
    drift = 0.0005  # 日漂移率
    volatility = 0.02  # 日波动率
    
    # 生成随机收益率
    returns = np.random.normal(drift, volatility, days)
    
    # 计算价格序列
    prices = [initial_price]
    for i in range(1, days):
        prices.append(prices[-1] * (1 + returns[i]))
    
    prices = np.array(prices)
    
    # 生成OHLC数据
    # Open: 前一日收盘价 + 小幅随机变动
    open_prices = np.roll(prices, 1)
    open_prices[0] = initial_price
    open_prices = open_prices * (1 + np.random.normal(0, 0.001, days))
    
    # High: 当日最高价
    high_prices = np.maximum(open_prices, prices) * (1 + np.abs(np.random.normal(0, 0.005, days)))
    
    # Low: 当日最低价
    low_prices = np.minimum(open_prices, prices) * (1 - np.abs(np.random.normal(0, 0.005, days)))
    
    # Close: 收盘价
    close_prices = prices
    
    # Volume: 交易量
    base_volume = 1000000
    volume = np.random.lognormal(np.log(base_volume), 0.5, days).astype(int)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return data


class SimpleMovingAverageStrategy:
    """简单移动平均策略"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        初始化策略
        
        Args:
            short_window: 短期移动平均窗口
            long_window: 长期移动平均窗口
        """
        self.short_window = short_window
        self.long_window = long_window
        self.engine = None
        self.position = 0  # 当前持仓
        self.ma_short_history = []
        self.ma_long_history = []
        self.price_history = []
        
    def set_engine(self, engine):
        """设置引擎引用"""
        self.engine = engine
        
    def on_bar(self, timestamp, bar_data):
        """
        处理每个bar的数据
        
        Args:
            timestamp: 时间戳
            bar_data: bar数据字典
        """
        # 获取当前价格
        if "AAPL" not in bar_data:
            return
            
        current_bar = bar_data["AAPL"]
        current_price = current_bar['close']
        
        # 更新价格历史
        self.price_history.append(current_price)
        
        # 计算移动平均线
        if len(self.price_history) >= self.short_window:
            ma_short = np.mean(self.price_history[-self.short_window:])
            self.ma_short_history.append(ma_short)
        else:
            return
            
        if len(self.price_history) >= self.long_window:
            ma_long = np.mean(self.price_history[-self.long_window:])
            self.ma_long_history.append(ma_long)
        else:
            return
        
        # 生成交易信号
        if len(self.ma_short_history) >= 2 and len(self.ma_long_history) >= 2:
            # 当前和前一个移动平均值
            ma_short_current = self.ma_short_history[-1]
            ma_short_previous = self.ma_short_history[-2]
            ma_long_current = self.ma_long_history[-1]
            ma_long_previous = self.ma_long_history[-2]
            
            # 金叉：短期均线上穿长期均线，买入信号
            if (ma_short_previous <= ma_long_previous and 
                ma_short_current > ma_long_current and 
                self.position == 0):
                
                # 计算买入数量（使用可用资金的90%）
                available_cash = self.engine.current_capital
                quantity = int((available_cash * 0.9) / current_price)
                
                if quantity > 0:
                    order = Order(
                        order_id=f"buy_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                        symbol="AAPL",
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        timestamp=timestamp
                    )
                    self.engine.place_order(order)
                    self.position = quantity
                    print(f"{timestamp.date()}: 买入信号 - 数量: {quantity}, 价格: {current_price:.2f}")
            
            # 死叉：短期均线下穿长期均线，卖出信号
            elif (ma_short_previous >= ma_long_previous and 
                  ma_short_current < ma_long_current and 
                  self.position > 0):
                
                order = Order(
                    order_id=f"sell_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    symbol="AAPL",
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=self.position,
                    timestamp=timestamp
                )
                self.engine.place_order(order)
                print(f"{timestamp.date()}: 卖出信号 - 数量: {self.position}, 价格: {current_price:.2f}")
                self.position = 0


class BuyAndHoldStrategy:
    """买入并持有策略"""
    
    def __init__(self):
        self.engine = None
        self.bought = False
        
    def set_engine(self, engine):
        """设置引擎引用"""
        self.engine = engine
        
    def on_bar(self, timestamp, bar_data):
        """处理每个bar的数据"""
        if not self.bought and "AAPL" in bar_data:
            # 第一天买入
            current_price = bar_data["AAPL"]['close']
            available_cash = self.engine.current_capital
            quantity = int((available_cash * 0.95) / current_price)  # 使用95%的资金
            
            if quantity > 0:
                order = Order(
                    order_id=f"buy_and_hold_{timestamp.strftime('%Y%m%d')}",
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    timestamp=timestamp
                )
                self.engine.place_order(order)
                self.bought = True
                print(f"{timestamp.date()}: 买入并持有 - 数量: {quantity}, 价格: {current_price:.2f}")


def run_basic_backtest():
    """运行基础回测示例"""
    print("=" * 60)
    print("基础回测引擎示例")
    print("=" * 60)
    
    # 1. 生成示例数据
    print("\n1. 生成示例数据...")
    data = generate_sample_data("AAPL", days=252)  # 一年的数据
    print(f"数据范围: {data.index[0].date()} 到 {data.index[-1].date()}")
    print(f"数据点数: {len(data)}")
    print(f"价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 2. 配置回测参数
    print("\n2. 配置回测参数...")
    config = BacktestConfig(
        initial_capital=100000.0,  # 初始资金10万
        commission_model=PercentageCommissionModel(commission_rate=0.001),  # 0.1%手续费
        slippage_model=FixedSlippageModel(slippage_bps=5.0),  # 5个基点滑点
        max_position_size=0.95  # 最大仓位95%
    )
    print(f"初始资金: ${config.initial_capital:,.2f}")
    print(f"手续费率: {config.commission_model.commission_rate:.3%}")
    print(f"滑点: {config.slippage_model.slippage_bps} bps")
    
    # 3. 运行移动平均策略回测
    print("\n3. 运行移动平均策略回测...")
    ma_engine = BarBacktestEngine(config)
    ma_engine.add_data(data, "AAPL")
    
    ma_strategy = SimpleMovingAverageStrategy(short_window=20, long_window=50)
    ma_engine.add_strategy(ma_strategy)
    
    ma_result = ma_engine.run()
    
    # 4. 运行买入持有策略回测（作为基准）
    print("\n4. 运行买入持有策略回测...")
    bh_engine = BarBacktestEngine(config)
    bh_engine.add_data(data, "AAPL")
    
    bh_strategy = BuyAndHoldStrategy()
    bh_engine.add_strategy(bh_strategy)
    
    bh_result = bh_engine.run()
    
    # 5. 比较结果
    print("\n5. 回测结果比较")
    print("=" * 60)
    
    strategies = {
        "移动平均策略": ma_result,
        "买入持有策略": bh_result
    }
    
    for name, result in strategies.items():
        print(f"\n{name}:")
        print(f"  最终资金: ${result.final_capital:,.2f}")
        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  年化收益率: {result.annualized_return:.2%}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.3f}")
        print(f"  总交易次数: {result.total_trades}")
        print(f"  胜率: {result.win_rate:.2%}")
        print(f"  盈利因子: {result.profit_factor:.3f}")
    
    # 6. 详细分析
    print("\n6. 详细分析")
    print("=" * 60)
    
    # 分析移动平均策略
    print("\n移动平均策略详细分析:")
    ma_analyzer = analyze_backtest_result(ma_result)
    advanced_metrics = ma_analyzer.calculate_advanced_metrics()
    
    print(f"  年化波动率: {advanced_metrics.get('volatility', 0):.2%}")
    print(f"  Sortino比率: {advanced_metrics.get('sortino_ratio', 0):.3f}")
    print(f"  Calmar比率: {advanced_metrics.get('calmar_ratio', 0):.3f}")
    print(f"  VaR (95%): {advanced_metrics.get('var_95', 0):.2%}")
    print(f"  最大连续盈利: {result.max_consecutive_wins}")
    print(f"  最大连续亏损: {result.max_consecutive_losses}")
    
    # 7. 生成图表（如果可能）
    print("\n7. 生成分析图表...")
    try:
        # 绘制权益曲线对比
        import matplotlib.pyplot as plt
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
        
        plt.figure(figsize=(15, 10))
        
        # 权益曲线
        plt.subplot(2, 2, 1)
        plt.plot(ma_result.equity_curve.index, ma_result.equity_curve.values, 
                label='移动平均策略', linewidth=2)
        plt.plot(bh_result.equity_curve.index, bh_result.equity_curve.values, 
                label='买入持有策略', linewidth=2)
        plt.axhline(y=config.initial_capital, color='gray', linestyle='--', alpha=0.7, label='初始资金')
        plt.title('权益曲线对比')
        plt.xlabel('时间')
        plt.ylabel('组合价值 ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 回撤对比
        plt.subplot(2, 2, 2)
        if ma_result.drawdown_series is not None:
            plt.fill_between(ma_result.drawdown_series.index, 
                           ma_result.drawdown_series.values * 100, 0,
                           alpha=0.3, color='red', label='移动平均策略')
        if bh_result.drawdown_series is not None:
            plt.fill_between(bh_result.drawdown_series.index, 
                           bh_result.drawdown_series.values * 100, 0,
                           alpha=0.3, color='blue', label='买入持有策略')
        plt.title('回撤对比')
        plt.xlabel('时间')
        plt.ylabel('回撤 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 收益率分布
        plt.subplot(2, 2, 3)
        if ma_result.daily_returns is not None and not ma_result.daily_returns.empty:
            plt.hist(ma_result.daily_returns * 100, bins=30, alpha=0.7, 
                    label='移动平均策略', density=True)
        if bh_result.daily_returns is not None and not bh_result.daily_returns.empty:
            plt.hist(bh_result.daily_returns * 100, bins=30, alpha=0.7, 
                    label='买入持有策略', density=True)
        plt.title('日收益率分布')
        plt.xlabel('收益率 (%)')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 滚动夏普比率
        plt.subplot(2, 2, 4)
        if ma_result.daily_returns is not None and not ma_result.daily_returns.empty:
            rolling_sharpe_ma = ma_result.daily_returns.rolling(60).mean() / ma_result.daily_returns.rolling(60).std() * np.sqrt(252)
            plt.plot(rolling_sharpe_ma.index, rolling_sharpe_ma.values, 
                    label='移动平均策略', linewidth=2)
        if bh_result.daily_returns is not None and not bh_result.daily_returns.empty:
            rolling_sharpe_bh = bh_result.daily_returns.rolling(60).mean() / bh_result.daily_returns.rolling(60).std() * np.sqrt(252)
            plt.plot(rolling_sharpe_bh.index, rolling_sharpe_bh.values, 
                    label='买入持有策略', linewidth=2)
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
    
    # 8. 生成报告
    print("\n8. 生成详细报告...")
    report = ma_analyzer.generate_report()
    
    # 保存报告到文件
    report_path = "backtest_report.md"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"详细报告已保存到: {report_path}")
    except Exception as e:
        print(f"报告保存失败: {e}")
    
    print("\n" + "=" * 60)
    print("回测完成！")
    print("=" * 60)
    
    return {
        "ma_result": ma_result,
        "bh_result": bh_result,
        "ma_analyzer": ma_analyzer
    }


if __name__ == "__main__":
    # 运行示例
    results = run_basic_backtest()
    
    # 可以进一步分析结果
    print("\n如需进一步分析，可以使用返回的结果对象:")
    print("- results['ma_result']: 移动平均策略回测结果")
    print("- results['bh_result']: 买入持有策略回测结果") 
    print("- results['ma_analyzer']: 移动平均策略分析器")
    
    # 示例：获取更多指标
    ma_analyzer = results['ma_analyzer']
    advanced_metrics = ma_analyzer.calculate_advanced_metrics()
    
    print(f"\n额外指标示例:")
    print(f"信息比率: {advanced_metrics.get('information_ratio', 0):.3f}")
    print(f"下行偏差: {advanced_metrics.get('downside_deviation', 0):.2%}")
    print(f"偏度: {advanced_metrics.get('skewness', 0):.3f}")
    print(f"峰度: {advanced_metrics.get('kurtosis', 0):.3f}")