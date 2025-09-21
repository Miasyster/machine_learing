#!/usr/bin/env python3
"""
调试订单执行问题的简化测试脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest.engine import BarBacktestEngine
from backtest.base import Order, OrderType, OrderSide, BacktestConfig

def create_simple_data():
    """创建简单的测试数据"""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=dates)
    return data

def test_order_execution():
    """测试订单执行"""
    print("=== 开始调试订单执行 ===")
    
    # 创建回测引擎
    config = BacktestConfig(
        initial_capital=100000
    )
    engine = BarBacktestEngine(config)
    
    # 添加数据
    data = create_simple_data()
    engine.add_data(data, "TEST")
    
    print(f"初始资金: {engine.current_capital}")
    print(f"最大持仓限制: {engine.max_position_size}")
    
    # 开始回测
    engine.current_time = data.index[0]
    engine.current_bar = {"TEST": 0}
    
    # 更新市场价格
    engine._update_market_prices(engine.current_time)
    
    # 创建一个简单的买入订单
    order = Order(
        symbol="TEST",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100
    )
    
    print(f"\n=== 下单前状态 ===")
    print(f"当前资金: {engine.current_capital}")
    print(f"当前持仓: {engine.positions}")
    print(f"当前价格: {engine.get_current_price('TEST')}")
    
    # 下单
    order_id = engine.place_order(order)
    print(f"\n订单ID: {order_id}")
    print(f"待成交订单数量: {len(engine.pending_orders)}")
    
    # 处理订单
    engine._process_pending_orders(engine.current_time)
    
    print(f"\n=== 下单后状态 ===")
    print(f"当前资金: {engine.current_capital}")
    print(f"当前持仓: {engine.positions}")
    print(f"已成交订单数量: {len([o for o in engine.all_orders if o.status.name == 'FILLED'])}")
    print(f"待成交订单数量: {len(engine.pending_orders)}")

if __name__ == "__main__":
    test_order_execution()