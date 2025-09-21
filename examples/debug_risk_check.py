#!/usr/bin/env python3
"""调试风险检查问题"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtest.engine import BarBacktestEngine
from src.backtest.base import BacktestConfig, Order, OrderType, OrderSide

# 创建测试数据
dates = pd.date_range('2023-01-01', periods=10, freq='D')
data = pd.DataFrame({
    'open': [100 + i for i in range(10)],
    'high': [105 + i for i in range(10)],
    'low': [95 + i for i in range(10)],
    'close': [102 + i for i in range(10)],
    'volume': [1000] * 10
}, index=dates)

print("测试数据:")
print(data.head())

# 创建回测引擎
config = BacktestConfig(initial_capital=100000.0, max_position_size=0.5)
engine = BarBacktestEngine(config)

# 添加数据
engine.add_data(data, "STOCK")

print(f"\n引擎数据符号: {list(engine.data.keys())}")

# 开始回测（只处理第一个bar）
engine.current_time = dates[0]
engine.current_bar = {"STOCK": 0}

print(f"\n当前时间: {engine.current_time}")
print(f"当前bar索引: {engine.current_bar}")

# 获取当前数据
current_data = engine._get_current_data()
print(f"\n当前数据: {current_data}")

# 创建订单
order = Order(
    symbol="STOCK",
    side=OrderSide.BUY,
    quantity=100,
    order_type=OrderType.MARKET
)

print(f"\n测试订单: {order.symbol}, 数量={order.quantity}, 类型={order.order_type}")

# 测试风险检查
result = engine._risk_check(order)
print(f"\n风险检查结果: {result}")