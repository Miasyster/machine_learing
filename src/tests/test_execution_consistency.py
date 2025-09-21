"""
执行一致性模块单元测试

测试内容：
1. 持仓期管理器测试
2. 止损管理器测试
3. 执行一致性管理器测试
4. 便捷函数测试
5. 边界情况测试
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from execution.execution_consistency import (
    HoldingPeriodManager, StopLossManager, ExecutionConsistencyManager,
    Position, PositionStatus, create_execution_manager, simulate_trading_execution
)


class TestPosition(unittest.TestCase):
    """测试持仓数据类"""
    
    def test_position_creation(self):
        """测试持仓创建"""
        entry_time = datetime.now()
        position = Position(
            symbol="AAPL",
            entry_time=entry_time,
            entry_price=150.0,
            quantity=100,
            direction="long",
            position_id="test_001"
        )
        
        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.entry_price, 150.0)
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.direction, "long")
        self.assertEqual(position.status, PositionStatus.ACTIVE)
        self.assertIsNone(position.exit_time)
        self.assertIsNone(position.exit_price)


class TestHoldingPeriodManager(unittest.TestCase):
    """测试持仓期管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = HoldingPeriodManager(holding_days=3)
        self.entry_time = datetime(2024, 1, 1, 10, 0, 0)
        
    def test_add_position(self):
        """测试添加持仓"""
        position = Position(
            symbol="AAPL",
            entry_time=self.entry_time,
            entry_price=150.0,
            quantity=100,
            direction="long",
            position_id="test_001"
        )
        
        self.manager.add_position(position)
        self.assertIn("test_001", self.manager.positions)
        self.assertEqual(len(self.manager.positions), 1)
        
    def test_check_expired_positions(self):
        """测试检查到期持仓"""
        # 添加持仓
        position = Position(
            symbol="AAPL",
            entry_time=self.entry_time,
            entry_price=150.0,
            quantity=100,
            direction="long",
            position_id="test_001"
        )
        self.manager.add_position(position)
        
        # 测试未到期
        current_time = self.entry_time + timedelta(days=2)
        current_prices = {"AAPL": 155.0}
        expired = self.manager.check_expired_positions(current_time, current_prices)
        self.assertEqual(len(expired), 0)
        
        # 测试到期
        current_time = self.entry_time + timedelta(days=3)
        expired = self.manager.check_expired_positions(current_time, current_prices)
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].status, PositionStatus.EXPIRED)
        self.assertEqual(expired[0].exit_price, 155.0)
        
    def test_get_active_positions(self):
        """测试获取活跃持仓"""
        # 添加活跃持仓
        position1 = Position(
            symbol="AAPL",
            entry_time=self.entry_time,
            entry_price=150.0,
            quantity=100,
            direction="long",
            position_id="test_001"
        )
        
        # 添加已平仓持仓
        position2 = Position(
            symbol="GOOGL",
            entry_time=self.entry_time,
            entry_price=2500.0,
            quantity=50,
            direction="long",
            position_id="test_002"
        )
        position2.status = PositionStatus.CLOSED
        
        self.manager.add_position(position1)
        self.manager.add_position(position2)
        
        active_positions = self.manager.get_active_positions()
        self.assertEqual(len(active_positions), 1)
        self.assertEqual(active_positions[0].position_id, "test_001")
        
    def test_close_position(self):
        """测试平仓操作"""
        position = Position(
            symbol="AAPL",
            entry_time=self.entry_time,
            entry_price=150.0,
            quantity=100,
            direction="long",
            position_id="test_001"
        )
        self.manager.add_position(position)
        
        exit_time = self.entry_time + timedelta(hours=1)
        success = self.manager.close_position(
            "test_001", 155.0, exit_time, "手动平仓"
        )
        
        self.assertTrue(success)
        self.assertEqual(position.exit_price, 155.0)
        self.assertEqual(position.exit_time, exit_time)
        self.assertEqual(position.status, PositionStatus.CLOSED)


class TestStopLossManager(unittest.TestCase):
    """测试止损管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = StopLossManager(stop_loss_threshold=0.01)
        self.entry_time = datetime(2024, 1, 1, 10, 0, 0)
        
    def test_calculate_pnl_long(self):
        """测试多头盈亏计算"""
        position = Position(
            symbol="AAPL",
            entry_time=self.entry_time,
            entry_price=100.0,
            quantity=100,
            direction="long",
            position_id="test_001"
        )
        
        # 盈利情况
        pnl = self.manager.calculate_pnl(position, 105.0)
        self.assertAlmostEqual(pnl, 0.05, places=4)
        
        # 亏损情况
        pnl = self.manager.calculate_pnl(position, 95.0)
        self.assertAlmostEqual(pnl, -0.05, places=4)
        
    def test_calculate_pnl_short(self):
        """测试空头盈亏计算"""
        position = Position(
            symbol="AAPL",
            entry_time=self.entry_time,
            entry_price=100.0,
            quantity=100,
            direction="short",
            position_id="test_001"
        )
        
        # 盈利情况（价格下跌）
        pnl = self.manager.calculate_pnl(position, 95.0)
        self.assertAlmostEqual(pnl, 0.05, places=4)
        
        # 亏损情况（价格上涨）
        pnl = self.manager.calculate_pnl(position, 105.0)
        self.assertAlmostEqual(pnl, -0.05, places=4)
        
    def test_check_stop_loss(self):
        """测试止损检查"""
        # 创建持仓
        position1 = Position(
            symbol="AAPL",
            entry_time=self.entry_time,
            entry_price=100.0,
            quantity=100,
            direction="long",
            position_id="test_001"
        )
        
        position2 = Position(
            symbol="GOOGL",
            entry_time=self.entry_time,
            entry_price=2000.0,
            quantity=50,
            direction="short",
            position_id="test_002"
        )
        
        positions = [position1, position2]
        current_prices = {"AAPL": 98.0, "GOOGL": 2025.0}  # 都触发止损
        current_time = datetime.now()
        
        stop_loss_positions = self.manager.check_stop_loss(
            positions, current_prices, current_time
        )
        
        self.assertEqual(len(stop_loss_positions), 2)
        for pos in stop_loss_positions:
            self.assertEqual(pos.status, PositionStatus.STOP_LOSS)
            self.assertIsNotNone(pos.exit_time)
            self.assertIsNotNone(pos.exit_price)


class TestExecutionConsistencyManager(unittest.TestCase):
    """测试执行一致性管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = ExecutionConsistencyManager(
            holding_days=3, stop_loss_threshold=0.01
        )
        self.entry_time = datetime(2024, 1, 1, 10, 0, 0)
        
    def test_open_position(self):
        """测试开仓操作"""
        position_id = self.manager.open_position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            direction="long",
            entry_time=self.entry_time
        )
        
        self.assertIsNotNone(position_id)
        self.assertIn(position_id, self.manager.holding_manager.positions)
        self.assertEqual(len(self.manager.execution_log), 1)
        
    def test_update_positions_expiry(self):
        """测试持仓到期更新"""
        # 开仓
        position_id = self.manager.open_position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            direction="long",
            entry_time=self.entry_time
        )
        
        # 3天后检查
        current_time = self.entry_time + timedelta(days=3)
        current_prices = {"AAPL": 155.0}
        
        results = self.manager.update_positions(current_prices, current_time)
        
        self.assertEqual(len(results['expired']), 1)
        self.assertEqual(len(results['stop_loss']), 0)
        
    def test_update_positions_stop_loss(self):
        """测试止损更新"""
        # 开仓
        position_id = self.manager.open_position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=100,
            direction="long",
            entry_time=self.entry_time
        )
        
        # 价格下跌触发止损
        current_time = self.entry_time + timedelta(hours=1)
        current_prices = {"AAPL": 98.0}  # 下跌2%，触发1%止损
        
        results = self.manager.update_positions(current_prices, current_time)
        
        self.assertEqual(len(results['expired']), 0)
        self.assertEqual(len(results['stop_loss']), 1)
        
    def test_get_position_summary(self):
        """测试获取持仓汇总"""
        # 开仓
        self.manager.open_position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            direction="long",
            entry_time=self.entry_time
        )
        
        summary = self.manager.get_position_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 1)
        self.assertIn('position_id', summary.columns)
        self.assertIn('symbol', summary.columns)
        self.assertIn('pnl', summary.columns)
        
    def test_get_execution_log(self):
        """测试获取执行日志"""
        # 开仓
        self.manager.open_position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            direction="long",
            entry_time=self.entry_time
        )
        
        log = self.manager.get_execution_log()
        
        self.assertIsInstance(log, pd.DataFrame)
        self.assertEqual(len(log), 1)
        self.assertIn('action', log.columns)
        self.assertIn('symbol', log.columns)
        
    def test_get_performance_metrics(self):
        """测试获取绩效指标"""
        # 开仓并平仓
        position_id = self.manager.open_position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=100,
            direction="long",
            entry_time=self.entry_time
        )
        
        # 触发止损
        current_time = self.entry_time + timedelta(hours=1)
        current_prices = {"AAPL": 98.0}
        self.manager.update_positions(current_prices, current_time)
        
        metrics = self.manager.get_performance_metrics()
        
        self.assertIn('total_positions', metrics)
        self.assertIn('closed_positions', metrics)
        self.assertIn('total_pnl', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('stop_loss_rate', metrics)


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""
    
    def test_create_execution_manager(self):
        """测试创建执行管理器"""
        manager = create_execution_manager(holding_days=5, stop_loss_threshold=0.02)
        
        self.assertIsInstance(manager, ExecutionConsistencyManager)
        self.assertEqual(manager.holding_manager.holding_days, 5)
        self.assertEqual(manager.stop_loss_manager.stop_loss_threshold, 0.02)
        
    def test_simulate_trading_execution(self):
        """测试交易执行模拟"""
        # 创建测试数据
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        prices = np.random.uniform(95, 105, 10)
        data = pd.DataFrame({'close': prices}, index=dates)
        
        # 创建交易信号
        signals = pd.Series([0, 1, 0, 0, -1, 0, 1, 0, 0, 0], index=dates)
        
        result = simulate_trading_execution(
            data=data,
            signals=signals,
            holding_days=3,
            stop_loss_threshold=0.01
        )
        
        self.assertIn('manager', result)
        self.assertIn('summary', result)
        self.assertIn('execution_log', result)
        self.assertIn('metrics', result)
        
        self.assertIsInstance(result['manager'], ExecutionConsistencyManager)
        self.assertIsInstance(result['summary'], pd.DataFrame)
        self.assertIsInstance(result['execution_log'], pd.DataFrame)
        self.assertIsInstance(result['metrics'], dict)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_positions(self):
        """测试空持仓情况"""
        manager = ExecutionConsistencyManager()
        
        # 空持仓检查
        current_time = datetime.now()
        current_prices = {"AAPL": 150.0}
        results = manager.update_positions(current_prices, current_time)
        
        self.assertEqual(len(results['expired']), 0)
        self.assertEqual(len(results['stop_loss']), 0)
        
        # 空持仓汇总
        summary = manager.get_position_summary()
        self.assertTrue(summary.empty)
        
        # 空绩效指标
        metrics = manager.get_performance_metrics()
        self.assertEqual(metrics, {})
        
    def test_missing_price_data(self):
        """测试缺失价格数据"""
        manager = ExecutionConsistencyManager()
        
        # 开仓
        manager.open_position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            direction="long",
            entry_time=datetime.now()
        )
        
        # 缺失价格数据
        current_time = datetime.now()
        current_prices = {"GOOGL": 2500.0}  # 没有AAPL价格
        results = manager.update_positions(current_prices, current_time)
        
        # 应该没有止损触发
        self.assertEqual(len(results['stop_loss']), 0)
        
    def test_zero_threshold(self):
        """测试零阈值"""
        manager = StopLossManager(stop_loss_threshold=0.0)
        
        position = Position(
            symbol="AAPL",
            entry_time=datetime.now(),
            entry_price=100.0,
            quantity=100,
            direction="long",
            position_id="test_001"
        )
        
        # 任何亏损都应该触发止损
        current_prices = {"AAPL": 99.99}
        current_time = datetime.now()
        
        stop_loss_positions = manager.check_stop_loss(
            [position], current_prices, current_time
        )
        
        self.assertEqual(len(stop_loss_positions), 1)
        
    def test_large_threshold(self):
        """测试大阈值"""
        manager = StopLossManager(stop_loss_threshold=0.5)  # 50%止损
        
        position = Position(
            symbol="AAPL",
            entry_time=datetime.now(),
            entry_price=100.0,
            quantity=100,
            direction="long",
            position_id="test_001"
        )
        
        # 小幅亏损不应该触发止损
        current_prices = {"AAPL": 90.0}  # 10%亏损
        current_time = datetime.now()
        
        stop_loss_positions = manager.check_stop_loss(
            [position], current_prices, current_time
        )
        
        self.assertEqual(len(stop_loss_positions), 0)


class TestPerformanceMetrics(unittest.TestCase):
    """测试绩效指标计算"""
    
    def test_win_rate_calculation(self):
        """测试胜率计算"""
        manager = ExecutionConsistencyManager()
        
        # 创建盈利持仓
        manager.open_position("AAPL", 100.0, 100, "long", datetime.now(), "win_1")
        manager.holding_manager.close_position("win_1", 105.0, datetime.now(), "手动")
        
        # 创建亏损持仓
        manager.open_position("GOOGL", 2000.0, 50, "long", datetime.now(), "loss_1")
        manager.holding_manager.close_position("loss_1", 1950.0, datetime.now(), "止损")
        
        metrics = manager.get_performance_metrics()
        
        self.assertEqual(metrics['win_rate'], 0.5)
        self.assertEqual(metrics['closed_positions'], 2)
        self.assertGreater(metrics['avg_win'], 0)
        self.assertLess(metrics['avg_loss'], 0)
        
    def test_stop_loss_rate_calculation(self):
        """测试止损率计算"""
        manager = ExecutionConsistencyManager()
        
        # 创建止损持仓
        manager.open_position("AAPL", 100.0, 100, "long", datetime.now(), "stop_1")
        manager.holding_manager.close_position("stop_1", 98.0, datetime.now(), "止损平仓")
        
        # 创建到期持仓
        manager.open_position("GOOGL", 2000.0, 50, "long", datetime.now(), "exp_1")
        manager.holding_manager.close_position("exp_1", 2010.0, datetime.now(), "持仓期到期")
        
        metrics = manager.get_performance_metrics()
        
        self.assertEqual(metrics['stop_loss_rate'], 0.5)
        self.assertEqual(metrics['expiry_rate'], 0.5)


if __name__ == '__main__':
    unittest.main()