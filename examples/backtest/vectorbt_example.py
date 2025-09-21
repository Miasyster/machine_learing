"""
VectorBT集成回测示例

演示如何使用VectorBT集成模块进行高性能向量化回测，包括：
- 多策略并行回测
- 参数优化
- 性能对比分析
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from backtest.vectorbt_integration import (
        VectorBTBacktestEngine, VectorBTStrategy, MovingAverageCrossStrategy,
        RSIStrategy, create_vectorbt_engine, run_vectorbt_backtest
    )
    VECTORBT_AVAILABLE = True
except ImportError as e:
    print(f"VectorBT不可用: {e}")
    print("请安装vectorbt: pip install vectorbt")
    VECTORBT_AVAILABLE = False


def generate_multi_asset_data(symbols: list, days: int = 252) -> dict:
    """
    生成多资产示例数据
    
    Args:
        symbols: 资产代码列表
        days: 天数
        
    Returns:
        多资产OHLCV数据字典
    """
    np.random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=days, freq='D')
    
    data = {}
    
    for i, symbol in enumerate(symbols):
        # 为不同资产设置不同的参数
        initial_price = 100.0 + i * 20  # 不同的初始价格
        drift = 0.0005 + i * 0.0002  # 不同的漂移率
        volatility = 0.015 + i * 0.005  # 不同的波动率
        
        # 生成相关性（简单的相关性模拟）
        if i > 0:
            correlation = 0.3  # 与第一个资产的相关性
            base_returns = np.random.normal(drift, volatility, days)
            if symbol in data:
                prev_returns = data[symbols[0]]['returns']
                correlated_returns = (correlation * prev_returns + 
                                    np.sqrt(1 - correlation**2) * base_returns)
            else:
                correlated_returns = base_returns
        else:
            correlated_returns = np.random.normal(drift, volatility, days)
        
        # 计算价格序列
        prices = [initial_price]
        for j in range(1, days):
            prices.append(prices[-1] * (1 + correlated_returns[j]))
        
        prices = np.array(prices)
        
        # 生成OHLC数据
        open_prices = np.roll(prices, 1)
        open_prices[0] = initial_price
        open_prices = open_prices * (1 + np.random.normal(0, 0.001, days))
        
        high_prices = np.maximum(open_prices, prices) * (1 + np.abs(np.random.normal(0, 0.005, days)))
        low_prices = np.minimum(open_prices, prices) * (1 - np.abs(np.random.normal(0, 0.005, days)))
        close_prices = prices
        
        base_volume = 1000000 * (1 + i * 0.5)
        volume = np.random.lognormal(np.log(base_volume), 0.5, days).astype(int)
        
        data[symbol] = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
            'returns': correlated_returns
        }, index=dates)
    
    return data


class MeanReversionStrategy(VectorBTStrategy):
    """均值回归策略"""
    
    def __init__(self, lookback_period: int = 20, threshold: float = 2.0):
        """
        初始化均值回归策略
        
        Args:
            lookback_period: 回看周期
            threshold: 标准差阈值
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.threshold = threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 价格数据
            
        Returns:
            交易信号DataFrame
        """
        close_prices = data['close']
        
        # 计算移动平均和标准差
        rolling_mean = close_prices.rolling(window=self.lookback_period).mean()
        rolling_std = close_prices.rolling(window=self.lookback_period).std()
        
        # 计算Z分数
        z_score = (close_prices - rolling_mean) / rolling_std
        
        # 生成信号
        buy_signals = z_score < -self.threshold  # 价格过低，买入
        sell_signals = z_score > self.threshold   # 价格过高，卖出
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy'] = buy_signals
        signals['sell'] = sell_signals
        signals['position'] = 0
        
        # 计算持仓
        for i in range(1, len(signals)):
            if signals['buy'].iloc[i]:
                signals['position'].iloc[i] = 1
            elif signals['sell'].iloc[i]:
                signals['position'].iloc[i] = -1
            else:
                signals['position'].iloc[i] = signals['position'].iloc[i-1]
        
        return signals


class MomentumStrategy(VectorBTStrategy):
    """动量策略"""
    
    def __init__(self, lookback_period: int = 12, holding_period: int = 3):
        """
        初始化动量策略
        
        Args:
            lookback_period: 动量计算周期
            holding_period: 持有周期
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        close_prices = data['close']
        
        # 计算动量（过去N期收益率）
        momentum = close_prices.pct_change(self.lookback_period)
        
        # 生成信号：动量为正时买入，为负时卖出
        signals = pd.DataFrame(index=data.index)
        signals['momentum'] = momentum
        signals['buy'] = momentum > 0
        signals['sell'] = momentum < 0
        signals['position'] = 0
        
        # 计算持仓（考虑持有周期）
        position = 0
        hold_counter = 0
        
        for i in range(len(signals)):
            if hold_counter > 0:
                hold_counter -= 1
                signals['position'].iloc[i] = position
            elif signals['buy'].iloc[i]:
                position = 1
                hold_counter = self.holding_period
                signals['position'].iloc[i] = position
            elif signals['sell'].iloc[i]:
                position = -1
                hold_counter = self.holding_period
                signals['position'].iloc[i] = position
            else:
                position = 0
                signals['position'].iloc[i] = position
        
        return signals


def run_vectorbt_example():
    """运行VectorBT回测示例"""
    if not VECTORBT_AVAILABLE:
        print("VectorBT不可用，无法运行示例")
        return None
    
    print("=" * 60)
    print("VectorBT集成回测示例")
    print("=" * 60)
    
    # 1. 生成多资产数据
    print("\n1. 生成多资产数据...")
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    data_dict = generate_multi_asset_data(symbols, days=252)
    
    print(f"资产数量: {len(symbols)}")
    print(f"数据范围: {data_dict[symbols[0]].index[0].date()} 到 {data_dict[symbols[0]].index[-1].date()}")
    
    # 2. 创建VectorBT引擎
    print("\n2. 创建VectorBT引擎...")
    engine = create_vectorbt_engine(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005
    )
    
    # 3. 添加数据
    print("\n3. 添加数据到引擎...")
    for symbol, data in data_dict.items():
        engine.add_data(data, symbol)
    
    # 4. 创建多个策略
    print("\n4. 创建多个策略...")
    strategies = {
        "MA_Cross_10_30": MovingAverageCrossStrategy(short_window=10, long_window=30),
        "MA_Cross_20_50": MovingAverageCrossStrategy(short_window=20, long_window=50),
        "RSI_30_70": RSIStrategy(rsi_period=14, oversold_threshold=30, overbought_threshold=70),
        "RSI_20_80": RSIStrategy(rsi_period=14, oversold_threshold=20, overbought_threshold=80),
        "MeanReversion_20_2": MeanReversionStrategy(lookback_period=20, threshold=2.0),
        "MeanReversion_30_1.5": MeanReversionStrategy(lookback_period=30, threshold=1.5),
        "Momentum_12_3": MomentumStrategy(lookback_period=12, holding_period=3),
        "Momentum_20_5": MomentumStrategy(lookback_period=20, holding_period=5),
    }
    
    # 5. 运行多策略回测
    print("\n5. 运行多策略回测...")
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"  运行策略: {strategy_name}")
        engine.add_strategy(strategy, strategy_name)
        
        try:
            result = engine.run_backtest()
            results[strategy_name] = result
            print(f"    ✓ 完成")
        except Exception as e:
            print(f"    ✗ 失败: {e}")
            continue
    
    # 6. 结果分析
    print("\n6. 策略性能对比")
    print("=" * 80)
    print(f"{'策略名称':<20} {'总收益率':<10} {'年化收益率':<12} {'最大回撤':<10} {'夏普比率':<10} {'交易次数':<8}")
    print("-" * 80)
    
    performance_summary = []
    
    for strategy_name, result in results.items():
        if result is not None:
            total_return = result.get('total_return', 0)
            annual_return = result.get('annual_return', 0)
            max_drawdown = result.get('max_drawdown', 0)
            sharpe_ratio = result.get('sharpe_ratio', 0)
            total_trades = result.get('total_trades', 0)
            
            print(f"{strategy_name:<20} {total_return:<10.2%} {annual_return:<12.2%} "
                  f"{max_drawdown:<10.2%} {sharpe_ratio:<10.3f} {total_trades:<8}")
            
            performance_summary.append({
                'strategy': strategy_name,
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': total_trades
            })
    
    # 7. 找出最佳策略
    if performance_summary:
        print("\n7. 最佳策略分析")
        print("=" * 60)
        
        # 按不同指标排序
        best_return = max(performance_summary, key=lambda x: x['total_return'])
        best_sharpe = max(performance_summary, key=lambda x: x['sharpe_ratio'])
        best_drawdown = min(performance_summary, key=lambda x: abs(x['max_drawdown']))
        
        print(f"最高收益率策略: {best_return['strategy']} ({best_return['total_return']:.2%})")
        print(f"最高夏普比率策略: {best_sharpe['strategy']} ({best_sharpe['sharpe_ratio']:.3f})")
        print(f"最小回撤策略: {best_drawdown['strategy']} ({best_drawdown['max_drawdown']:.2%})")
    
    # 8. 参数优化示例
    print("\n8. 参数优化示例")
    print("=" * 60)
    
    print("对移动平均策略进行参数优化...")
    
    # 定义参数范围
    short_windows = [5, 10, 15, 20]
    long_windows = [30, 40, 50, 60]
    
    optimization_results = []
    
    for short_win in short_windows:
        for long_win in long_windows:
            if short_win >= long_win:
                continue
                
            strategy_name = f"MA_Opt_{short_win}_{long_win}"
            strategy = MovingAverageCrossStrategy(short_window=short_win, long_window=long_win)
            
            # 创建新引擎实例进行测试
            test_engine = create_vectorbt_engine(
                initial_capital=100000.0,
                commission=0.001,
                slippage=0.0005
            )
            
            # 只使用第一个资产进行优化
            test_engine.add_data(data_dict[symbols[0]], symbols[0])
            test_engine.add_strategy(strategy, strategy_name)
            
            try:
                result = test_engine.run_backtest()
                if result:
                    optimization_results.append({
                        'short_window': short_win,
                        'long_window': long_win,
                        'total_return': result.get('total_return', 0),
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'max_drawdown': result.get('max_drawdown', 0)
                    })
            except Exception as e:
                print(f"优化失败 {strategy_name}: {e}")
                continue
    
    # 显示优化结果
    if optimization_results:
        print(f"\n参数优化结果 (基于{symbols[0]}):")
        print(f"{'短窗口':<8} {'长窗口':<8} {'总收益率':<10} {'夏普比率':<10} {'最大回撤':<10}")
        print("-" * 50)
        
        # 按夏普比率排序
        optimization_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        for result in optimization_results[:10]:  # 显示前10个结果
            print(f"{result['short_window']:<8} {result['long_window']:<8} "
                  f"{result['total_return']:<10.2%} {result['sharpe_ratio']:<10.3f} "
                  f"{result['max_drawdown']:<10.2%}")
        
        best_params = optimization_results[0]
        print(f"\n最优参数: 短窗口={best_params['short_window']}, "
              f"长窗口={best_params['long_window']}")
    
    # 9. 生成可视化（如果可能）
    print("\n9. 生成可视化...")
    try:
        import matplotlib.pyplot as plt
        
        # 策略收益率对比
        plt.figure(figsize=(15, 10))
        
        # 收益率对比
        plt.subplot(2, 2, 1)
        strategy_names = list(results.keys())[:6]  # 只显示前6个策略
        returns = [results[name].get('total_return', 0) for name in strategy_names]
        
        plt.bar(range(len(strategy_names)), [r * 100 for r in returns])
        plt.title('策略总收益率对比 (%)')
        plt.xticks(range(len(strategy_names)), strategy_names, rotation=45)
        plt.ylabel('收益率 (%)')
        plt.grid(True, alpha=0.3)
        
        # 夏普比率对比
        plt.subplot(2, 2, 2)
        sharpe_ratios = [results[name].get('sharpe_ratio', 0) for name in strategy_names]
        
        plt.bar(range(len(strategy_names)), sharpe_ratios)
        plt.title('策略夏普比率对比')
        plt.xticks(range(len(strategy_names)), strategy_names, rotation=45)
        plt.ylabel('夏普比率')
        plt.grid(True, alpha=0.3)
        
        # 最大回撤对比
        plt.subplot(2, 2, 3)
        max_drawdowns = [abs(results[name].get('max_drawdown', 0)) * 100 for name in strategy_names]
        
        plt.bar(range(len(strategy_names)), max_drawdowns, color='red', alpha=0.7)
        plt.title('策略最大回撤对比 (%)')
        plt.xticks(range(len(strategy_names)), strategy_names, rotation=45)
        plt.ylabel('最大回撤 (%)')
        plt.grid(True, alpha=0.3)
        
        # 参数优化热力图
        if optimization_results:
            plt.subplot(2, 2, 4)
            
            # 创建热力图数据
            short_wins = sorted(list(set([r['short_window'] for r in optimization_results])))
            long_wins = sorted(list(set([r['long_window'] for r in optimization_results])))
            
            heatmap_data = np.zeros((len(short_wins), len(long_wins)))
            
            for result in optimization_results:
                i = short_wins.index(result['short_window'])
                j = long_wins.index(result['long_window'])
                heatmap_data[i, j] = result['sharpe_ratio']
            
            im = plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
            plt.colorbar(im, label='夏普比率')
            plt.title('移动平均参数优化热力图')
            plt.xlabel('长窗口')
            plt.ylabel('短窗口')
            plt.xticks(range(len(long_wins)), long_wins)
            plt.yticks(range(len(short_wins)), short_wins)
        
        plt.tight_layout()
        plt.show()
        
        print("可视化图表已生成")
        
    except ImportError:
        print("matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"可视化生成失败: {e}")
    
    print("\n" + "=" * 60)
    print("VectorBT回测示例完成！")
    print("=" * 60)
    
    return {
        "results": results,
        "optimization_results": optimization_results,
        "performance_summary": performance_summary
    }


def run_simple_vectorbt_example():
    """运行简单的VectorBT示例"""
    if not VECTORBT_AVAILABLE:
        print("VectorBT不可用，无法运行示例")
        return None
    
    print("运行简单VectorBT示例...")
    
    # 生成简单数据
    data = generate_multi_asset_data(["AAPL"], days=100)["AAPL"]
    
    # 使用便利函数运行回测
    result = run_vectorbt_backtest(
        data=data,
        strategy=MovingAverageCrossStrategy(short_window=10, long_window=20),
        initial_capital=10000.0,
        commission=0.001
    )
    
    if result:
        print(f"简单回测结果:")
        print(f"  总收益率: {result.get('total_return', 0):.2%}")
        print(f"  夏普比率: {result.get('sharpe_ratio', 0):.3f}")
        print(f"  最大回撤: {result.get('max_drawdown', 0):.2%}")
    
    return result


if __name__ == "__main__":
    # 运行完整示例
    print("选择运行模式:")
    print("1. 完整VectorBT示例 (包含多策略对比和参数优化)")
    print("2. 简单VectorBT示例")
    
    try:
        choice = input("请选择 (1 或 2): ").strip()
        
        if choice == "2":
            result = run_simple_vectorbt_example()
        else:
            result = run_vectorbt_example()
            
        if result:
            print("\n示例运行完成！")
        else:
            print("\n示例运行失败或被跳过")
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n运行出错: {e}")
        # 默认运行简单示例
        print("尝试运行简单示例...")
        run_simple_vectorbt_example()