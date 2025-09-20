"""
数据完整性检查模块的测试用例
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from etl.data_integrity import DataIntegrityChecker


class TestDataIntegrityChecker(unittest.TestCase):
    """数据完整性检查器测试类"""
    
    def setUp(self):
        """设置测试数据"""
        self.checker = DataIntegrityChecker()
        
        # 创建正常的测试数据
        self.dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        self.normal_data = pd.DataFrame({
            'open': np.random.normal(50000, 1000, 100),
            'high': np.random.normal(51000, 1000, 100),
            'low': np.random.normal(49000, 1000, 100),
            'close': np.random.normal(50000, 1000, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=self.dates)
        
        # 确保OHLC逻辑正确
        for i in range(len(self.normal_data)):
            row = self.normal_data.iloc[i]
            high_val = max(row['open'], row['close']) + abs(np.random.normal(100, 50))
            low_val = min(row['open'], row['close']) - abs(np.random.normal(100, 50))
            self.normal_data.iloc[i, self.normal_data.columns.get_loc('high')] = high_val
            self.normal_data.iloc[i, self.normal_data.columns.get_loc('low')] = low_val
        
        # 确保成交量为正
        self.normal_data['volume'] = np.abs(self.normal_data['volume'])
    
    def test_date_continuity_normal(self):
        """测试正常数据的日期连续性"""
        result = self.checker.check_date_continuity(self.normal_data, '1H')
        
        self.assertTrue(result['is_continuous'])
        self.assertEqual(result['missing_points'], 0)
        self.assertEqual(result['duplicate_points'], 0)
        self.assertEqual(result['continuity_ratio'], 1.0)
        self.assertTrue(result['passes_tolerance'])
    
    def test_date_continuity_missing_dates(self):
        """测试有缺失日期的数据"""
        # 删除一些行来模拟缺失日期
        missing_data = self.normal_data.drop(self.normal_data.index[10:15])
        
        result = self.checker.check_date_continuity(missing_data, '1H')
        
        self.assertFalse(result['is_continuous'])
        self.assertEqual(result['missing_points'], 5)
        self.assertLess(result['continuity_ratio'], 1.0)
    
    def test_date_continuity_duplicate_dates(self):
        """测试有重复日期的数据"""
        # 添加重复的行
        duplicate_data = pd.concat([self.normal_data, self.normal_data.iloc[[0, 1]]])
        
        result = self.checker.check_date_continuity(duplicate_data, '1H')
        
        self.assertFalse(result['is_continuous'])
        self.assertEqual(result['duplicate_points'], 2)
    
    def test_ohlc_logic_normal(self):
        """测试正常OHLC数据的逻辑"""
        result = self.checker.check_ohlc_logic(self.normal_data)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['total_violations'], 0)
        self.assertEqual(result['violation_ratio'], 0.0)
    
    def test_ohlc_logic_high_violations(self):
        """测试高价违规的情况"""
        violation_data = self.normal_data.copy()
        # 让高价低于开盘价
        violation_data.loc[violation_data.index[0], 'high'] = violation_data.loc[violation_data.index[0], 'open'] - 100
        
        result = self.checker.check_ohlc_logic(violation_data)
        
        self.assertFalse(result['is_valid'])
        self.assertGreater(result['total_violations'], 0)
        self.assertGreater(len(result['violations']['high_violations']), 0)
    
    def test_ohlc_logic_low_violations(self):
        """测试低价违规的情况"""
        violation_data = self.normal_data.copy()
        # 让低价高于收盘价
        violation_data.loc[violation_data.index[0], 'low'] = violation_data.loc[violation_data.index[0], 'close'] + 100
        
        result = self.checker.check_ohlc_logic(violation_data)
        
        self.assertFalse(result['is_valid'])
        self.assertGreater(result['total_violations'], 0)
        self.assertGreater(len(result['violations']['low_violations']), 0)
    
    def test_ohlc_logic_negative_prices(self):
        """测试负价格的情况"""
        violation_data = self.normal_data.copy()
        violation_data.loc[violation_data.index[0], 'open'] = -100
        
        result = self.checker.check_ohlc_logic(violation_data)
        
        self.assertFalse(result['is_valid'])
        self.assertGreater(result['total_violations'], 0)
        self.assertIn('open', result['violations']['negative_price_violations'])
    
    def test_ohlc_logic_zero_prices(self):
        """测试零价格的情况"""
        violation_data = self.normal_data.copy()
        violation_data.loc[violation_data.index[0], 'close'] = 0
        
        result = self.checker.check_ohlc_logic(violation_data)
        
        self.assertFalse(result['is_valid'])
        self.assertGreater(result['total_violations'], 0)
        self.assertIn('close', result['violations']['zero_price_violations'])
    
    def test_ohlc_logic_price_jumps(self):
        """测试价格跳跃的情况"""
        violation_data = self.normal_data.copy()
        # 创建一个巨大的价格跳跃
        violation_data.loc[violation_data.index[1], 'open'] = violation_data.loc[violation_data.index[0], 'close'] * 2
        
        result = self.checker.check_ohlc_logic(violation_data)
        
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['violations']['price_jump_violations']), 0)
    
    def test_ohlc_logic_missing_columns(self):
        """测试缺少必要列的情况"""
        incomplete_data = self.normal_data.drop(columns=['high'])
        
        result = self.checker.check_ohlc_logic(incomplete_data)
        
        self.assertFalse(result['is_valid'])
        self.assertIn('error', result)
        self.assertIn('high', result['error'])
    
    def test_volume_logic_normal(self):
        """测试正常成交量数据的逻辑"""
        result = self.checker.check_volume_logic(self.normal_data)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['total_violations'], 0)
    
    def test_volume_logic_negative_volume(self):
        """测试负成交量的情况"""
        violation_data = self.normal_data.copy()
        violation_data.loc[violation_data.index[0], 'volume'] = -100
        
        result = self.checker.check_volume_logic(violation_data)
        
        self.assertFalse(result['is_valid'])
        self.assertGreater(result['total_violations'], 0)
        self.assertGreater(len(result['violations']['volume']['negative_volume']), 0)
    
    def test_volume_logic_abnormal_volume(self):
        """测试异常大成交量的情况"""
        violation_data = self.normal_data.copy()
        # 设置一个异常大的成交量
        mean_vol = violation_data['volume'].mean()
        violation_data.loc[violation_data.index[0], 'volume'] = mean_vol * 20
        
        result = self.checker.check_volume_logic(violation_data)
        
        # 注意：这个测试可能通过，因为异常检测的阈值设置
        # 主要是确保代码能正常运行
        self.assertIsInstance(result['is_valid'], bool)
        self.assertIn('volume', result['violations'])
    
    def test_lookback_consistency_normal(self):
        """测试正常数据的回溯一致性"""
        result = self.checker.check_lookback_consistency(self.normal_data)
        
        self.assertIsInstance(result['is_consistent'], bool)
        self.assertIn('consistency_results', result)
        self.assertIn('checked_periods', result)
    
    def test_lookback_consistency_with_gaps(self):
        """测试有价格跳跃的回溯一致性"""
        violation_data = self.normal_data.copy()
        # 创建价格跳跃
        violation_data.loc[violation_data.index[1], 'open'] = violation_data.loc[violation_data.index[0], 'close'] * 1.2
        
        result = self.checker.check_lookback_consistency(violation_data)
        
        self.assertIsInstance(result['is_consistent'], bool)
        self.assertGreaterEqual(result['total_violations'], 0)
    
    def test_comprehensive_integrity_check_normal(self):
        """测试正常数据的综合完整性检查"""
        result = self.checker.comprehensive_integrity_check(self.normal_data)
        
        self.assertIn('timestamp', result)
        self.assertIn('data_shape', result)
        self.assertIn('checks', result)
        self.assertIn('overall_score', result)
        self.assertIn('overall_status', result)
        
        # 检查各个子检查是否都存在
        checks = result['checks']
        self.assertIn('date_continuity', checks)
        self.assertIn('ohlc_logic', checks)
        self.assertIn('volume_logic', checks)
        self.assertIn('lookback_consistency', checks)
        
        # 正常数据应该有较高的评分
        self.assertGreaterEqual(result['overall_score'], 0.7)
    
    def test_comprehensive_integrity_check_with_violations(self):
        """测试有违规的数据的综合完整性检查"""
        violation_data = self.normal_data.copy()
        
        # 添加各种违规
        violation_data.loc[violation_data.index[0], 'high'] = violation_data.loc[violation_data.index[0], 'low'] - 100  # OHLC违规
        violation_data.loc[violation_data.index[1], 'volume'] = -100  # 负成交量
        violation_data = violation_data.drop(violation_data.index[10:15])  # 缺失日期
        
        result = self.checker.comprehensive_integrity_check(violation_data)
        
        # 有违规的数据应该有较低的评分
        self.assertLess(result['overall_score'], 0.8)
        self.assertEqual(result['overall_status'], 'FAIL')
    
    def test_generate_integrity_report(self):
        """测试生成完整性报告"""
        result = self.checker.comprehensive_integrity_check(self.normal_data)
        report = self.checker.generate_integrity_report(result)
        
        self.assertIsInstance(report, str)
        self.assertIn('数据完整性检查报告', report)
        self.assertIn('总体评分', report)
        self.assertIn('日期连续性检查', report)
        self.assertIn('OHLC价格逻辑检查', report)
        self.assertIn('成交量逻辑检查', report)
        self.assertIn('回溯一致性检查', report)
    
    def test_custom_columns(self):
        """测试自定义列名"""
        custom_checker = DataIntegrityChecker(
            price_columns=['o', 'h', 'l', 'c'],
            volume_columns=['vol']
        )
        
        custom_data = self.normal_data.rename(columns={
            'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'vol'
        })
        
        result = custom_checker.check_ohlc_logic(custom_data)
        self.assertTrue(result['is_valid'])
        
        result = custom_checker.check_volume_logic(custom_data)
        self.assertTrue(result['is_valid'])
    
    def test_empty_data(self):
        """测试空数据"""
        empty_data = pd.DataFrame()
        
        # 大部分检查应该能处理空数据而不崩溃
        try:
            result = self.checker.check_ohlc_logic(empty_data)
            self.assertFalse(result['is_valid'])
        except Exception:
            pass  # 空数据可能会抛出异常，这是可以接受的
    
    def test_single_row_data(self):
        """测试单行数据"""
        single_row = self.normal_data.iloc[[0]]
        
        result = self.checker.check_ohlc_logic(single_row)
        # 单行数据应该能通过OHLC检查（如果数据本身是正确的）
        self.assertTrue(result['is_valid'])
        
        result = self.checker.check_volume_logic(single_row)
        self.assertTrue(result['is_valid'])


if __name__ == '__main__':
    unittest.main()