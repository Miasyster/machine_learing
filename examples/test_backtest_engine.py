"""
测试回测引擎的交易执行
"""
import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import (
    BacktestConfig, Order, OrderType, OrderSide,
    PercentageCommissionModel, FixedSlippageModel
)
from src.backtest.engine import BarBacktestEngine

class SimpleTestStrategy:
    """简单测试策略：前5天买入，第6天卖出"""
    
    def __init__(self):
        self.engine = None
        self.day_count = 0
        self.position = 0
        
    def set_engine(self, engine):
        self.engine = engine
        
    def on_bar(self, timestamp, bar_data):
        """处理每个bar"""
        self.day_count += 1
        
        # 获取当前价格
        if "STOCK" not in bar_data:
            return
        current_price = bar_data["STOCK"]['close']
        
        print(f"第{self.day_count}天, 价格: {current_price:.2f}, 持仓: {self.position}")
        
        if self.day_count <= 5 and self.position == 0:
            # 前5天买入
            order = Order(
                symbol="STOCK",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100,
                timestamp=timestamp
            )
            self.engine.place_order(order)
            self.position = 100
            print(f"  -> 买入100股")
            
        elif self.day_count == 6 and self.position > 0:
            # 第6天卖出
            order = Order(
                symbol="STOCK",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=self.position,
                timestamp=timestamp
            )
            self.engine.place_order(order)
            print(f"  -> 卖出{self.position}股")
            self.position = 0

def generate_test_data(n_days=10):
    """生成测试数据"""
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    prices = [100 + i for i in range(n_days)]  # 简单递增价格
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000] * n_days
    })
    
    return data

def test_backtest_engine():
    """测试回测引擎"""
    print("=== 测试回测引擎 ===")
    
    # 生成测试数据
    data = generate_test_data(10)
    print(f"生成测试数据: {len(data)} 天")
    print(data[['date', 'close']].head())
    
    # 配置回测
    config = BacktestConfig(
        initial_capital=100000,
        commission_model=PercentageCommissionModel(0.001),
        slippage_model=FixedSlippageModel(0.01),
        start_date=data['date'].iloc[0],
        end_date=data['date'].iloc[-1]
    )
    
    # 创建回测引擎
    engine = BarBacktestEngine(config)
    
    # 添加数据 - 使用正确的API
    df_data = data.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
    engine.add_data(df_data, "STOCK")
    
    # 创建并添加策略
    strategy = SimpleTestStrategy()
    engine.add_strategy(strategy)
    
    # 运行回测
    print("\n开始回测...")
    results = engine.run()
    
    # 显示结果
    print(f"\n=== 回测结果 ===")
    print(f"初始资金: {config.initial_capital:,.2f}")
    print(f"最终资金: {results.final_capital:,.2f}")
    print(f"总收益率: {results.total_return:.2%}")
    print(f"交易次数: {results.total_trades}")
    
    # 显示交易详情
    if results.trades:
        print(f"\n=== 交易详情 ===")
        for i, trade in enumerate(results.trades):
            print(f"交易 {i+1}: {trade.side.name} {trade.quantity} @ {trade.price:.2f} on {trade.timestamp}")
    
    return results

if __name__ == "__main__":
    test_backtest_engine()