"""
回测引擎单元测试

测试回测引擎的各个组件和功能，包括：
- 基础数据结构
- 回测引擎核心功能
- 事务管理系统
- VectorBT集成
- 结果分析
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest.base import (
    Order, Trade, Position, BacktestResult, BacktestConfig,
    OrderType, OrderSide, OrderStatus, PositionSide,
    FixedSlippageModel, PercentageSlippageModel,
    FixedCommissionModel, PercentageCommissionModel
)
from backtest.engine import BarBacktestEngine
from backtest.transaction import TransactionManager, RiskRule, StopLossType
from backtest.analysis import BacktestAnalyzer


class TestBaseClasses(unittest.TestCase):
    """测试基础数据结构"""
    
    def test_order_creation(self):
        """测试订单创建"""
        order = Order(
            order_id="test_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            price=150.0,
            timestamp=datetime.now()
        )
        
        self.assertEqual(order.order_id, "test_001")
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, 100)
        self.assertEqual(order.status, OrderStatus.PENDING)
    
    def test_trade_creation(self):
        """测试交易创建"""
        trade = Trade(
            trade_id="trade_001",
            order_id="order_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            commission=1.5
        )
        
        self.assertEqual(trade.trade_id, "trade_001")
        self.assertEqual(trade.symbol, "AAPL")
        self.assertEqual(trade.quantity, 100)
        self.assertEqual(trade.commission, 1.5)
    
    def test_position_creation(self):
        """测试持仓创建"""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=100,
            avg_price=150.0,
            market_value=15000.0,
            unrealized_pnl=500.0,
            timestamp=datetime.now()
        )
        
        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.side, PositionSide.LONG)
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.avg_price, 150.0)
    
    def test_slippage_models(self):
        """测试滑点模型"""
        # 固定滑点
        fixed_slippage = FixedSlippageModel(slippage_bps=10)
        slippage = fixed_slippage.calculate_slippage(100.0, 1000)
        self.assertEqual(slippage, 0.1)  # 10 bps = 0.1
        
        # 百分比滑点
        pct_slippage = PercentageSlippageModel(slippage_rate=0.001)
        slippage = pct_slippage.calculate_slippage(100.0, 1000)
        self.assertEqual(slippage, 0.1)  # 0.1% of 100
    
    def test_commission_models(self):
        """测试手续费模型"""
        # 固定手续费
        fixed_commission = FixedCommissionModel(commission_per_trade=5.0)
        commission = fixed_commission.calculate_commission(100.0, 1000)
        self.assertEqual(commission, 5.0)
        
        # 百分比手续费
        pct_commission = PercentageCommissionModel(commission_rate=0.001)
        commission = pct_commission.calculate_commission(100.0, 1000)
        self.assertEqual(commission, 0.1)  # 0.1% of 100


class TestBarBacktestEngine(unittest.TestCase):
    """测试Bar回测引擎"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = BacktestConfig(
            initial_capital=100000.0,
            commission_model=PercentageCommissionModel(0.001),
            slippage_model=FixedSlippageModel(5.0)
        )
        self.engine = BarBacktestEngine(self.config)
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        
        self.test_data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(100) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(100)) * 0.005),
            'low': prices * (1 - np.abs(np.random.randn(100)) * 0.005),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_add_data(self):
        """测试添加数据"""
        self.engine.add_data(self.test_data, "TEST")
        self.assertIn("TEST", self.engine.data)
        self.assertEqual(len(self.engine.data["TEST"]), 100)
    
    def test_place_order(self):
        """测试下单"""
        self.engine.add_data(self.test_data, "TEST")
        
        order = Order(
            order_id="test_order",
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            timestamp=self.test_data.index[0]
        )
        
        self.engine.place_order(order)
        self.assertEqual(len(self.engine.pending_orders), 1)
        self.assertEqual(self.engine.pending_orders[0].order_id, "test_order")
    
    def test_process_bar(self):
        """测试处理单个bar"""
        self.engine.add_data(self.test_data, "TEST")
        
        # 下一个市价买单
        order = Order(
            order_id="test_order",
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            timestamp=self.test_data.index[0]
        )
        self.engine.place_order(order)
        
        # 处理第一个bar
        bar_data = {"TEST": self.test_data.iloc[0]}
        self.engine._process_bar(self.test_data.index[0], bar_data)
        
        # 检查订单是否被执行
        self.assertEqual(len(self.engine.pending_orders), 0)
        self.assertEqual(len(self.engine.trades), 1)
        self.assertIn("TEST", self.engine.positions)
    
    def test_simple_strategy(self):
        """测试简单策略"""
        class SimpleStrategy:
            def __init__(self):
                self.engine = None
                self.position = False
            
            def set_engine(self, engine):
                self.engine = engine
            
            def on_bar(self, timestamp, bar_data):
                if not self.position and len(self.engine.trades) == 0:
                    # 第一次买入
                    order = Order(
                        order_id=f"buy_{timestamp}",
                        symbol="TEST",
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=100,
                        timestamp=timestamp
                    )
                    self.engine.place_order(order)
                    self.position = True
                elif self.position and len(self.engine.trades) == 1:
                    # 卖出
                    order = Order(
                        order_id=f"sell_{timestamp}",
                        symbol="TEST",
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=100,
                        timestamp=timestamp
                    )
                    self.engine.place_order(order)
                    self.position = False
        
        self.engine.add_data(self.test_data, "TEST")
        strategy = SimpleStrategy()
        self.engine.add_strategy(strategy)
        
        result = self.engine.run()
        
        # 检查回测结果
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.initial_capital, 100000.0)
        self.assertGreater(len(self.engine.trades), 0)


class TestTransactionManager(unittest.TestCase):
    """测试事务管理系统"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = BacktestConfig(initial_capital=100000.0)
        self.manager = TransactionManager(self.config)
    
    def test_position_size_limit(self):
        """测试仓位限制"""
        # 设置最大仓位为50%
        self.config.max_position_size = 0.5
        
        order = Order(
            order_id="test",
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,  # 假设价格100，总价值100000，超过50%限制
            price=100.0,
            timestamp=datetime.now()
        )
        
        # 检查仓位限制
        is_valid, reason = self.manager.check_position_limits(order, 100000.0)
        # 这里需要根据具体实现调整断言
    
    def test_stop_loss_rule(self):
        """测试止损规则"""
        rule = RiskRule(
            rule_id="stop_loss_test",
            symbol="TEST",
            stop_loss_type=StopLossType.PERCENTAGE,
            stop_loss_value=0.05,  # 5%止损
            is_active=True
        )
        
        self.manager.add_risk_rule(rule)
        
        # 模拟价格下跌触发止损
        current_price = 95.0  # 从100下跌到95，跌幅5%
        entry_price = 100.0
        
        should_trigger = self.manager._should_trigger_stop_loss(rule, current_price, entry_price)
        self.assertTrue(should_trigger)


class TestBacktestAnalyzer(unittest.TestCase):
    """测试回测分析器"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟回测结果
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 模拟权益曲线
        returns = np.random.randn(100) * 0.01
        equity_values = 100000 * (1 + returns).cumprod()
        
        self.result = BacktestResult(
            start_date=dates[0],
            end_date=dates[-1],
            initial_capital=100000.0,
            final_capital=equity_values[-1],
            total_return=(equity_values[-1] / 100000.0) - 1,
            annualized_return=0.1,
            max_drawdown=0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            win_rate=0.6,
            profit_factor=1.8,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            avg_trade_return=100.0,
            avg_winning_trade=200.0,
            avg_losing_trade=-50.0,
            max_consecutive_wins=5,
            max_consecutive_losses=3,
            equity_curve=pd.Series(equity_values, index=dates),
            trades=[],
            positions=[],
            orders=[],
            daily_returns=pd.Series(returns, index=dates),
            drawdown_series=pd.Series(np.random.randn(100) * 0.01, index=dates),
            metadata={}
        )
        
        self.analyzer = BacktestAnalyzer(self.result)
    
    def test_calculate_advanced_metrics(self):
        """测试高级指标计算"""
        metrics = self.analyzer.calculate_advanced_metrics()
        
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('var_95', metrics)
        self.assertIn('volatility', metrics)
        
        # 检查指标合理性
        self.assertIsInstance(metrics['total_return'], (int, float))
        self.assertIsInstance(metrics['sharpe_ratio'], (int, float))
    
    def test_generate_report(self):
        """测试报告生成"""
        report = self.analyzer.generate_report()
        
        self.assertIsInstance(report, str)
        self.assertIn('回测分析报告', report)
        self.assertIn('收益指标', report)
        self.assertIn('风险指标', report)
        self.assertIn('交易统计', report)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_equity_curve(self, mock_show):
        """测试权益曲线绘制"""
        fig = self.analyzer.plot_equity_curve(use_plotly=False)
        self.assertIsNotNone(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_drawdown(self, mock_show):
        """测试回撤图绘制"""
        fig = self.analyzer.plot_drawdown(use_plotly=False)
        self.assertIsNotNone(fig)


class TestVectorBTIntegration(unittest.TestCase):
    """测试VectorBT集成"""
    
    def setUp(self):
        """设置测试环境"""
        try:
            from backtest.vectorbt_integration import VectorBTBacktestEngine, MovingAverageCrossStrategy
            self.vectorbt_available = True
            self.VectorBTBacktestEngine = VectorBTBacktestEngine
            self.MovingAverageCrossStrategy = MovingAverageCrossStrategy
        except ImportError:
            self.vectorbt_available = False
            self.skipTest("VectorBT not available")
    
    def test_vectorbt_engine_creation(self):
        """测试VectorBT引擎创建"""
        if not self.vectorbt_available:
            self.skipTest("VectorBT not available")
        
        config = BacktestConfig(initial_capital=100000.0)
        engine = self.VectorBTBacktestEngine(config)
        
        self.assertEqual(engine.initial_capital, 100000.0)
        self.assertEqual(engine.config, config)
    
    def test_moving_average_strategy(self):
        """测试移动平均策略"""
        if not self.vectorbt_available:
            self.skipTest("VectorBT not available")
        
        strategy = self.MovingAverageCrossStrategy(fast_period=5, slow_period=20)
        
        self.assertEqual(strategy.fast_period, 5)
        self.assertEqual(strategy.slow_period, 20)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_backtest_workflow(self):
        """测试完整回测流程"""
        # 创建配置
        config = BacktestConfig(
            initial_capital=100000.0,
            commission_model=PercentageCommissionModel(0.001),
            slippage_model=FixedSlippageModel(5.0)
        )
        
        # 创建引擎
        engine = BarBacktestEngine(config)
        
        # 添加数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 0.02)
        
        test_data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(50) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(50)) * 0.005),
            'low': prices * (1 - np.abs(np.random.randn(50)) * 0.005),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        engine.add_data(test_data, "TEST")
        
        # 添加简单策略
        class BuyAndHoldStrategy:
            def __init__(self):
                self.engine = None
                self.bought = False
            
            def set_engine(self, engine):
                self.engine = engine
            
            def on_bar(self, timestamp, bar_data):
                if not self.bought:
                    order = Order(
                        order_id=f"buy_{timestamp}",
                        symbol="TEST",
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=100,
                        timestamp=timestamp
                    )
                    self.engine.place_order(order)
                    self.bought = True
        
        strategy = BuyAndHoldStrategy()
        engine.add_strategy(strategy)
        
        # 运行回测
        result = engine.run()
        
        # 验证结果
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.initial_capital, 100000.0)
        self.assertNotEqual(result.final_capital, result.initial_capital)
        
        # 分析结果
        analyzer = BacktestAnalyzer(result)
        metrics = analyzer.calculate_advanced_metrics()
        
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        
        # 生成报告
        report = analyzer.generate_report()
        self.assertIsInstance(report, str)
        self.assertIn('回测分析报告', report)


class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""
    
    def test_invalid_data(self):
        """测试无效数据处理"""
        engine = BarBacktestEngine()
        
        # 测试空数据
        with self.assertRaises(ValueError):
            engine.add_data(pd.DataFrame(), "TEST")
        
        # 测试缺少必要列的数据
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103]
            # 缺少high, low, volume
        }, index=pd.date_range('2023-01-01', periods=3))
        
        with self.assertRaises(ValueError):
            engine.add_data(invalid_data, "TEST")
    
    def test_invalid_order(self):
        """测试无效订单处理"""
        engine = BarBacktestEngine()
        
        # 测试无效数量的订单
        invalid_order = Order(
            order_id="invalid",
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0,  # 无效数量
            timestamp=datetime.now()
        )
        
        with self.assertRaises(ValueError):
            engine.place_order(invalid_order)
    
    def test_insufficient_capital(self):
        """测试资金不足处理"""
        config = BacktestConfig(initial_capital=1000.0)  # 较小的初始资金
        engine = BarBacktestEngine(config)
        
        # 添加测试数据
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        test_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        }, index=dates)
        
        engine.add_data(test_data, "TEST")
        
        # 尝试下一个超出资金能力的大单
        large_order = Order(
            order_id="large_order",
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,  # 需要100,000，但只有1,000资金
            timestamp=dates[0]
        )
        
        engine.place_order(large_order)
        
        # 处理bar，订单应该被拒绝
        bar_data = {"TEST": test_data.iloc[0]}
        engine._process_bar(dates[0], bar_data)
        
        # 检查订单是否被拒绝
        self.assertEqual(len(engine.trades), 0)


if __name__ == '__main__':
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestBaseClasses,
        TestBarBacktestEngine,
        TestTransactionManager,
        TestBacktestAnalyzer,
        TestVectorBTIntegration,
        TestIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print(f"\n{'='*50}")
    print(f"测试摘要:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
    
    print(f"{'='*50}")