"""
执行一致性模块使用示例

本文件展示如何使用执行一致性模块进行交易管理，包括：
1. 基础持仓期管理
2. 止损平仓规则
3. 完整交易执行流程
4. 多资产组合管理
5. 绩效分析和监控
6. 真实场景应用

主要功能演示：
- 3天持仓期限制
- 1%止损平仓规则
- 自动平仓触发
- 交易绩效分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from execution.execution_consistency import (
    ExecutionConsistencyManager, create_execution_manager, 
    simulate_trading_execution, Position, PositionStatus
)


def generate_sample_data(start_date='2024-01-01', periods=30, symbols=['AAPL', 'GOOGL', 'MSFT']):
    """
    生成示例价格数据
    
    Parameters:
    -----------
    start_date : str
        开始日期
    periods : int
        数据期数
    symbols : list
        股票代码列表
        
    Returns:
    --------
    pd.DataFrame
        多资产价格数据
    """
    dates = pd.date_range(start_date, periods=periods, freq='D')
    
    data = {}
    for symbol in symbols:
        # 生成随机价格走势
        np.random.seed(hash(symbol) % 1000)  # 确保可重复性
        returns = np.random.normal(0.001, 0.02, periods)  # 日收益率
        prices = [100.0]  # 初始价格
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
            
        data[f'{symbol}_close'] = prices
        
    return pd.DataFrame(data, index=dates)


def example_1_basic_holding_period():
    """示例1: 基础持仓期管理"""
    print("=" * 60)
    print("示例1: 基础持仓期管理")
    print("=" * 60)
    
    # 创建执行管理器（3天持仓期）
    manager = create_execution_manager(holding_days=3, stop_loss_threshold=0.01)
    
    # 模拟开仓
    entry_time = datetime(2024, 1, 1, 10, 0, 0)
    position_id = manager.open_position(
        symbol="AAPL",
        entry_price=150.0,
        quantity=100,
        direction="long",
        entry_time=entry_time
    )
    
    print(f"开仓成功: {position_id}")
    print(f"开仓时间: {entry_time}")
    print(f"开仓价格: $150.00")
    
    # 模拟价格变化和时间推移
    for day in range(1, 5):
        current_time = entry_time + timedelta(days=day)
        current_price = 150.0 + np.random.normal(0, 2)  # 价格随机波动
        current_prices = {"AAPL": current_price}
        
        # 更新持仓状态
        results = manager.update_positions(current_prices, current_time)
        
        print(f"\n第{day}天 ({current_time.date()}):")
        print(f"当前价格: ${current_price:.2f}")
        
        if results['expired']:
            print("触发到期平仓!")
            for pos in results['expired']:
                print(f"平仓原因: {pos.exit_reason}")
                print(f"平仓价格: ${pos.exit_price:.2f}")
                break
        elif results['stop_loss']:
            print("触发止损平仓!")
            for pos in results['stop_loss']:
                print(f"平仓原因: {pos.exit_reason}")
                print(f"平仓价格: ${pos.exit_price:.2f}")
                break
        else:
            print("持仓继续...")
    
    # 显示最终结果
    summary = manager.get_position_summary()
    print(f"\n持仓汇总:")
    print(summary[['symbol', 'entry_price', 'exit_price', 'status', 'exit_reason', 'pnl']])


def example_2_stop_loss_management():
    """示例2: 止损平仓规则"""
    print("\n" + "=" * 60)
    print("示例2: 止损平仓规则")
    print("=" * 60)
    
    # 创建执行管理器（1%止损）
    manager = create_execution_manager(holding_days=5, stop_loss_threshold=0.01)
    
    # 开仓多头和空头
    entry_time = datetime(2024, 1, 1, 10, 0, 0)
    
    long_id = manager.open_position(
        symbol="AAPL",
        entry_price=100.0,
        quantity=100,
        direction="long",
        entry_time=entry_time
    )
    
    short_id = manager.open_position(
        symbol="GOOGL",
        entry_price=2000.0,
        quantity=50,
        direction="short",
        entry_time=entry_time
    )
    
    print("开仓完成:")
    print(f"多头持仓: AAPL @ $100.00")
    print(f"空头持仓: GOOGL @ $2000.00")
    
    # 模拟不同的价格场景
    scenarios = [
        {"AAPL": 101.0, "GOOGL": 1990.0, "desc": "小幅波动"},
        {"AAPL": 98.5, "GOOGL": 2025.0, "desc": "触发止损"},
        {"AAPL": 95.0, "GOOGL": 2050.0, "desc": "大幅亏损"}
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        current_time = entry_time + timedelta(hours=i)
        current_prices = {k: v for k, v in scenario.items() if k != "desc"}
        
        results = manager.update_positions(current_prices, current_time)
        
        print(f"\n场景{i}: {scenario['desc']}")
        print(f"AAPL价格: ${scenario['AAPL']:.2f}")
        print(f"GOOGL价格: ${scenario['GOOGL']:.2f}")
        
        if results['stop_loss']:
            print("触发止损平仓:")
            for pos in results['stop_loss']:
                pnl_pct = ((pos.exit_price - pos.entry_price) / pos.entry_price) * (1 if pos.direction == 'long' else -1)
                print(f"  {pos.symbol} {pos.direction}: 盈亏 {pnl_pct:.2%}")
        else:
            print("未触发止损")
    
    # 显示绩效指标
    metrics = manager.get_performance_metrics()
    print(f"\n绩效指标:")
    print(f"总持仓数: {metrics.get('total_positions', 0)}")
    print(f"已平仓数: {metrics.get('closed_positions', 0)}")
    print(f"止损率: {metrics.get('stop_loss_rate', 0):.2%}")


def example_3_complete_trading_workflow():
    """示例3: 完整交易执行流程"""
    print("\n" + "=" * 60)
    print("示例3: 完整交易执行流程")
    print("=" * 60)
    
    # 生成价格数据
    data = generate_sample_data(periods=15, symbols=['AAPL'])
    
    # 生成简单的交易信号（移动平均策略）
    data['ma_5'] = data['AAPL_close'].rolling(5).mean()
    data['ma_10'] = data['AAPL_close'].rolling(10).mean()
    
    # 信号：5日均线上穿10日均线买入，下穿卖出
    signals = pd.Series(0, index=data.index)
    signals[data['ma_5'] > data['ma_10']] = 1
    signals[data['ma_5'] < data['ma_10']] = -1
    
    # 使用便捷函数进行交易模拟
    price_data = data[['AAPL_close']].rename(columns={'AAPL_close': 'close'})
    result = simulate_trading_execution(
        data=price_data,
        signals=signals,
        holding_days=3,
        stop_loss_threshold=0.01,
        initial_capital=100000
    )
    
    print("交易模拟完成!")
    print(f"\n持仓汇总:")
    summary = result['summary']
    if not summary.empty:
        print(summary[['symbol', 'direction', 'entry_price', 'exit_price', 'status', 'pnl']].head())
    else:
        print("无持仓记录")
    
    print(f"\n绩效指标:")
    metrics = result['metrics']
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


def example_4_multi_asset_portfolio():
    """示例4: 多资产组合管理"""
    print("\n" + "=" * 60)
    print("示例4: 多资产组合管理")
    print("=" * 60)
    
    # 创建执行管理器
    manager = create_execution_manager(holding_days=3, stop_loss_threshold=0.015)
    
    # 多资产开仓
    entry_time = datetime(2024, 1, 1, 10, 0, 0)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    entry_prices = [150.0, 2500.0, 300.0, 200.0]
    
    position_ids = []
    for symbol, price in zip(symbols, entry_prices):
        position_id = manager.open_position(
            symbol=symbol,
            entry_price=price,
            quantity=100,
            direction="long",
            entry_time=entry_time
        )
        position_ids.append(position_id)
    
    print(f"开仓{len(symbols)}个资产:")
    for symbol, price in zip(symbols, entry_prices):
        print(f"  {symbol}: ${price:.2f}")
    
    # 模拟市场波动
    for day in range(1, 5):
        current_time = entry_time + timedelta(days=day)
        
        # 生成当前价格（不同资产不同波动）
        current_prices = {}
        for i, (symbol, entry_price) in enumerate(zip(symbols, entry_prices)):
            # 不同资产不同的波动模式
            if symbol == 'AAPL':
                change = np.random.normal(-0.005, 0.02)  # 轻微下跌趋势
            elif symbol == 'GOOGL':
                change = np.random.normal(0.002, 0.015)  # 轻微上涨趋势
            elif symbol == 'MSFT':
                change = np.random.normal(0, 0.01)       # 横盘
            else:  # TSLA
                change = np.random.normal(0, 0.03)       # 高波动
                
            current_prices[symbol] = entry_price * (1 + change * day)
        
        # 更新持仓
        results = manager.update_positions(current_prices, current_time)
        
        print(f"\n第{day}天 ({current_time.date()}):")
        for symbol, price in current_prices.items():
            entry_price = entry_prices[symbols.index(symbol)]
            change_pct = (price - entry_price) / entry_price
            print(f"  {symbol}: ${price:.2f} ({change_pct:+.2%})")
        
        if results['expired']:
            print(f"  到期平仓: {len(results['expired'])}个")
        if results['stop_loss']:
            print(f"  止损平仓: {len(results['stop_loss'])}个")
            for pos in results['stop_loss']:
                print(f"    {pos.symbol}: {pos.exit_reason}")
    
    # 组合绩效分析
    summary = manager.get_position_summary()
    metrics = manager.get_performance_metrics()
    
    print(f"\n组合绩效分析:")
    print(f"总持仓数: {metrics.get('total_positions', 0)}")
    print(f"已平仓数: {metrics.get('closed_positions', 0)}")
    print(f"总盈亏: ${metrics.get('total_pnl', 0):.2f}")
    print(f"胜率: {metrics.get('win_rate', 0):.2%}")
    print(f"止损率: {metrics.get('stop_loss_rate', 0):.2%}")
    print(f"到期率: {metrics.get('expiry_rate', 0):.2%}")


def example_5_performance_monitoring():
    """示例5: 绩效分析和监控"""
    print("\n" + "=" * 60)
    print("示例5: 绩效分析和监控")
    print("=" * 60)
    
    # 创建执行管理器
    manager = create_execution_manager(holding_days=3, stop_loss_threshold=0.01)
    
    # 模拟一系列交易
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    # 交易记录
    trades = [
        {"symbol": "AAPL", "entry": 100.0, "exit": 105.0, "direction": "long", "reason": "手动"},
        {"symbol": "GOOGL", "entry": 2000.0, "exit": 1980.0, "direction": "long", "reason": "止损"},
        {"symbol": "MSFT", "entry": 300.0, "exit": 310.0, "direction": "long", "reason": "到期"},
        {"symbol": "TSLA", "entry": 200.0, "exit": 195.0, "direction": "short", "reason": "止损"},
        {"symbol": "NVDA", "entry": 500.0, "exit": 520.0, "direction": "long", "reason": "到期"},
    ]
    
    # 执行交易
    for i, trade in enumerate(trades):
        entry_time = base_time + timedelta(hours=i)
        exit_time = entry_time + timedelta(hours=1)
        
        # 开仓
        position_id = manager.open_position(
            symbol=trade["symbol"],
            entry_price=trade["entry"],
            quantity=100,
            direction=trade["direction"],
            entry_time=entry_time
        )
        
        # 平仓
        manager.holding_manager.close_position(
            position_id, trade["exit"], exit_time, trade["reason"]
        )
    
    # 获取详细分析
    summary = manager.get_position_summary()
    execution_log = manager.get_execution_log()
    metrics = manager.get_performance_metrics()
    
    print("交易汇总:")
    print(summary[['symbol', 'direction', 'entry_price', 'exit_price', 'exit_reason', 'pnl']])
    
    print(f"\n详细绩效指标:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key:
                print(f"  {key}: {value:.2%}")
            elif 'pnl' in key or 'avg' in key:
                print(f"  {key}: ${value:.2f}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 分析不同平仓原因的表现
    print(f"\n按平仓原因分析:")
    for reason in summary['exit_reason'].unique():
        if pd.notna(reason):
            subset = summary[summary['exit_reason'] == reason]
            avg_pnl = subset['pnl'].mean()
            count = len(subset)
            print(f"  {reason}: {count}笔, 平均盈亏 ${avg_pnl:.2f}")


def example_6_real_world_scenario():
    """示例6: 真实场景应用"""
    print("\n" + "=" * 60)
    print("示例6: 真实场景应用 - 日内交易策略")
    print("=" * 60)
    
    # 生成更真实的价格数据（包含日内波动）
    dates = pd.date_range('2024-01-01 09:30:00', periods=100, freq='30min')
    
    # 模拟日内价格走势
    np.random.seed(42)
    base_price = 150.0
    prices = [base_price]
    
    for i in range(1, len(dates)):
        # 模拟日内波动模式
        hour = dates[i].hour
        if 9 <= hour <= 10:  # 开盘波动大
            volatility = 0.015
        elif 11 <= hour <= 14:  # 中午波动小
            volatility = 0.005
        else:  # 收盘波动大
            volatility = 0.012
            
        change = np.random.normal(0, volatility)
        prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({'close': prices}, index=dates)
    
    # 生成交易信号（基于RSI策略）
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    data['rsi'] = calculate_rsi(data['close'])
    
    # RSI策略信号
    signals = pd.Series(0, index=data.index)
    signals[data['rsi'] < 30] = 1   # 超卖买入
    signals[data['rsi'] > 70] = -1  # 超买卖出
    
    # 执行交易模拟
    result = simulate_trading_execution(
        data=data,
        signals=signals,
        holding_days=1,  # 日内交易，1天持仓期
        stop_loss_threshold=0.005,  # 0.5%止损
        initial_capital=100000
    )
    
    print("日内交易策略回测结果:")
    
    summary = result['summary']
    if not summary.empty:
        print(f"\n交易统计:")
        print(f"总交易次数: {len(summary)}")
        print(f"盈利交易: {len(summary[summary['pnl'] > 0])}")
        print(f"亏损交易: {len(summary[summary['pnl'] <= 0])}")
        
        # 按小时分析交易表现
        summary['entry_hour'] = pd.to_datetime(summary['entry_time']).dt.hour
        hourly_performance = summary.groupby('entry_hour')['pnl'].agg(['count', 'mean', 'sum'])
        print(f"\n按小时交易表现:")
        print(hourly_performance)
    
    metrics = result['metrics']
    print(f"\n策略绩效:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key:
                print(f"  {key}: {value:.2%}")
            elif 'pnl' in key or 'avg' in key:
                print(f"  {key}: ${value:.2f}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def main():
    """运行所有示例"""
    print("执行一致性模块使用示例")
    print("=" * 80)
    
    try:
        example_1_basic_holding_period()
        example_2_stop_loss_management()
        example_3_complete_trading_workflow()
        example_4_multi_asset_portfolio()
        example_5_performance_monitoring()
        example_6_real_world_scenario()
        
        print("\n" + "=" * 80)
        print("所有示例运行完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()