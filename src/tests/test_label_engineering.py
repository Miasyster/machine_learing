"""
标签构建模块单元测试

测试标签构建功能的正确性、边界情况和性能
"""

import unittest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch
import sys
import os

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.label_engineering import (
    ReturnLabelBuilder,
    DirectionLabelBuilder,
    ExcessReturnLabelBuilder,
    LabelSmoother,
    NoiseReducer,
    LabelEngineeringManager,
    create_return_labels,
    create_direction_labels
)


class TestReturnLabelBuilder(unittest.TestCase):
    """测试收益率标签构建器"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'close': [100, 102, 101, 105, 103, 108, 106, 110, 109, 112],
            'date': pd.date_range('2023-01-01', periods=10, freq='D')
        })
    
    def test_simple_return_calculation(self):
        """测试简单收益率计算"""
        builder = ReturnLabelBuilder(periods=1, method='simple')
        builder.fit(self.data)
        returns = builder.transform(self.data)
        
        # 验证第一个收益率: (102-100)/100 = 0.02
        self.assertAlmostEqual(returns.iloc[0], 0.02, places=4)
        
        # 验证最后几个值为NaN
        self.assertTrue(pd.isna(returns.iloc[-1]))
    
    def test_log_return_calculation(self):
        """测试对数收益率计算"""
        builder = ReturnLabelBuilder(periods=1, method='log')
        builder.fit(self.data)
        returns = builder.transform(self.data)
        
        # 验证第一个对数收益率
        expected = np.log(102/100)
        self.assertAlmostEqual(returns.iloc[0], expected, places=4)
    
    def test_multi_period_returns(self):
        """测试多期收益率"""
        builder = ReturnLabelBuilder(periods=3, method='simple')
        builder.fit(self.data)
        returns = builder.transform(self.data)
        
        # 验证第一个3期收益率: (105-100)/100 = 0.05
        self.assertAlmostEqual(returns.iloc[0], 0.05, places=4)
        
        # 验证最后3个值为NaN
        for i in range(-3, 0):
            self.assertTrue(pd.isna(returns.iloc[i]))
    
    def test_invalid_price_column(self):
        """测试无效价格列"""
        builder = ReturnLabelBuilder(price_column='invalid')
        
        with self.assertRaises(ValueError):
            builder.fit(self.data)
    
    def test_unfitted_transform(self):
        """测试未拟合就转换"""
        builder = ReturnLabelBuilder()
        
        with self.assertRaises(ValueError):
            builder.transform(self.data)
    
    def test_fit_transform(self):
        """测试fit_transform方法"""
        builder = ReturnLabelBuilder(periods=1)
        returns = builder.fit_transform(self.data)
        
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), len(self.data))


class TestDirectionLabelBuilder(unittest.TestCase):
    """测试方向标签构建器"""
    
    def setUp(self):
        """设置测试数据"""
        self.data = pd.DataFrame({
            'close': [100, 105, 95, 110, 90, 115, 85, 120, 80, 125],  # 明显的涨跌模式
            'date': pd.date_range('2023-01-01', periods=10, freq='D')
        })
    
    def test_direction_labels_without_neutral_zone(self):
        """测试无中性区间的方向标签"""
        builder = DirectionLabelBuilder(periods=1, threshold=0.0)
        builder.fit(self.data)
        directions = builder.transform(self.data)
        
        # 第一个应该是+1 (105 > 100)
        self.assertEqual(directions.iloc[0], 1)
        
        # 第二个应该是-1 (95 < 105)
        self.assertEqual(directions.iloc[1], -1)
    
    def test_direction_labels_with_neutral_zone(self):
        """测试有中性区间的方向标签"""
        builder = DirectionLabelBuilder(periods=1, neutral_zone=0.02)  # 2%中性区间
        builder.fit(self.data)
        directions = builder.transform(self.data)
        
        # 验证标签只包含-1, 0, 1
        unique_values = set(directions.dropna().unique())
        self.assertTrue(unique_values.issubset({-1, 0, 1}))
    
    def test_direction_statistics(self):
        """测试方向标签统计"""
        builder = DirectionLabelBuilder(periods=1, neutral_zone=0.01)
        builder.fit(self.data)
        
        # 验证统计信息存在
        self.assertIn('positive_ratio', builder.label_stats_)
        self.assertIn('negative_ratio', builder.label_stats_)
        self.assertIn('neutral_ratio', builder.label_stats_)


class TestExcessReturnLabelBuilder(unittest.TestCase):
    """测试超额收益标签构建器"""
    
    def setUp(self):
        """设置测试数据"""
        self.data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108],
            'benchmark': [100, 101, 102, 103, 104],
            'date': pd.date_range('2023-01-01', periods=5, freq='D')
        })
        
        self.benchmark_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'date': pd.date_range('2023-01-01', periods=5, freq='D')
        })
    
    def test_excess_return_same_dataframe(self):
        """测试同一DataFrame中的超额收益"""
        builder = ExcessReturnLabelBuilder(
            periods=1,
            benchmark_column='benchmark'
        )
        builder.fit(self.data)
        excess_returns = builder.transform(self.data)
        
        # 第一个超额收益: (102-100)/100 - (101-100)/100 = 0.02 - 0.01 = 0.01
        self.assertAlmostEqual(excess_returns.iloc[0], 0.01, places=4)
    
    def test_excess_return_separate_dataframe(self):
        """测试单独DataFrame中的超额收益"""
        builder = ExcessReturnLabelBuilder(
            periods=1,
            benchmark_data=self.benchmark_data
        )
        builder.fit(self.data)
        excess_returns = builder.transform(self.data)
        
        # 验证结果不全为NaN
        self.assertFalse(excess_returns.isna().all())
    
    def test_missing_benchmark_error(self):
        """测试缺少基准数据的错误"""
        builder = ExcessReturnLabelBuilder(periods=1)
        
        with self.assertRaises(ValueError):
            builder.fit(self.data)
    
    def test_invalid_benchmark_column(self):
        """测试无效基准列"""
        builder = ExcessReturnLabelBuilder(
            periods=1,
            benchmark_column='invalid'
        )
        
        with self.assertRaises(ValueError):
            builder.fit(self.data)


class TestLabelSmoother(unittest.TestCase):
    """测试标签平滑器"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.labels = pd.Series(np.random.randn(100))
    
    def test_rolling_mean_smooth(self):
        """测试滚动均值平滑"""
        smoothed = LabelSmoother.rolling_mean_smooth(self.labels, window=5)
        
        # 平滑后的标准差应该更小
        self.assertLess(smoothed.std(), self.labels.std())
    
    def test_exponential_smooth(self):
        """测试指数平滑"""
        smoothed = LabelSmoother.exponential_smooth(self.labels, alpha=0.3)
        
        # 验证结果长度相同
        self.assertEqual(len(smoothed), len(self.labels))
        
        # 平滑后的标准差应该更小
        self.assertLess(smoothed.std(), self.labels.std())
    
    @patch('scipy.ndimage.gaussian_filter1d')
    def test_gaussian_smooth(self, mock_filter):
        """测试高斯平滑（模拟scipy）"""
        mock_filter.return_value = np.ones(len(self.labels))
        
        smoothed = LabelSmoother.gaussian_smooth(self.labels, window=5, std=1.0)
        
        # 验证scipy函数被调用
        mock_filter.assert_called_once()
        
        # 验证结果长度
        self.assertEqual(len(smoothed), len(self.labels))


class TestNoiseReducer(unittest.TestCase):
    """测试噪声处理器"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建包含异常值的数据
        normal_data = np.random.normal(0, 1, 95)
        outliers = np.array([10, -10, 15, -15, 20])  # 明显的异常值
        self.labels = pd.Series(np.concatenate([normal_data, outliers]))
    
    def test_outlier_clip(self):
        """测试异常值裁剪"""
        clipped = NoiseReducer.outlier_clip(
            self.labels, 
            lower_quantile=0.05, 
            upper_quantile=0.95
        )
        
        # 裁剪后的极值应该被限制
        self.assertLess(clipped.max(), self.labels.max())
        self.assertGreater(clipped.min(), self.labels.min())
    
    @patch('scipy.stats.mstats.winsorize')
    def test_winsorize(self, mock_winsorize):
        """测试Winsorize处理（模拟scipy）"""
        mock_winsorize.return_value = np.ones(len(self.labels.dropna()))
        
        winsorized = NoiseReducer.winsorize(self.labels, limits=(0.05, 0.05))
        
        # 验证scipy函数被调用
        mock_winsorize.assert_called_once()
        
        # 验证结果长度
        self.assertEqual(len(winsorized), len(self.labels))
    
    def test_z_score_filter(self):
        """测试Z-score过滤"""
        filtered = NoiseReducer.z_score_filter(self.labels, threshold=2.0)
        
        # 过滤后应该有一些NaN值（异常值被过滤）
        self.assertGreater(filtered.isna().sum(), 0)
        
        # 验证过滤前后的数据量变化
        original_count = self.labels.count()
        filtered_count = filtered.count()
        self.assertLess(filtered_count, original_count)
        
        # 验证极端异常值被过滤掉
        self.assertNotIn(20, filtered.values)
        self.assertNotIn(-20, filtered.values)


class TestLabelEngineeringManager(unittest.TestCase):
    """测试标签构建管理器"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100,
            'benchmark': np.random.randn(50).cumsum() + 100,
            'date': pd.date_range('2023-01-01', periods=50, freq='D')
        })
        
        self.manager = LabelEngineeringManager()
    
    def test_add_return_label(self):
        """测试添加收益率标签"""
        self.manager.add_return_label('test_return', periods=1)
        
        self.assertIn('test_return', self.manager.builders)
        self.assertIsInstance(
            self.manager.builders['test_return'], 
            ReturnLabelBuilder
        )
    
    def test_add_direction_label(self):
        """测试添加方向标签"""
        self.manager.add_direction_label('test_direction', periods=1)
        
        self.assertIn('test_direction', self.manager.builders)
        self.assertIsInstance(
            self.manager.builders['test_direction'], 
            DirectionLabelBuilder
        )
    
    def test_add_excess_return_label(self):
        """测试添加超额收益标签"""
        self.manager.add_excess_return_label(
            'test_excess', 
            periods=1, 
            benchmark_column='benchmark'
        )
        
        self.assertIn('test_excess', self.manager.builders)
        self.assertIsInstance(
            self.manager.builders['test_excess'], 
            ExcessReturnLabelBuilder
        )
    
    def test_fit_transform_workflow(self):
        """测试完整的拟合转换流程"""
        # 添加多个标签
        self.manager.add_return_label('return_1d', periods=1)
        self.manager.add_direction_label('direction_1d', periods=1)
        self.manager.add_excess_return_label(
            'excess_1d', 
            periods=1, 
            benchmark_column='benchmark'
        )
        
        # 拟合并转换
        labels = self.manager.fit_transform(self.data)
        
        # 验证结果
        self.assertIsInstance(labels, pd.DataFrame)
        self.assertEqual(len(labels), len(self.data))
        self.assertEqual(len(labels.columns), 3)
        
        # 验证列名
        expected_columns = ['return_1d', 'direction_1d', 'excess_1d']
        self.assertEqual(list(labels.columns), expected_columns)
    
    def test_get_label_summary(self):
        """测试获取标签统计摘要"""
        self.manager.add_return_label('return_1d', periods=1)
        self.manager.fit_transform(self.data)
        
        summary = self.manager.get_label_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('return_1d', summary.index)
        
        # 验证统计列
        expected_stats = ['count', 'mean', 'std', 'min', 'max', 'null_ratio']
        for stat in expected_stats:
            self.assertIn(stat, summary.columns)
    
    def test_apply_smoothing(self):
        """测试应用标签平滑"""
        self.manager.add_return_label('return_1d', periods=1)
        self.manager.fit_transform(self.data)
        
        # 测试滚动均值平滑
        smoothed = self.manager.apply_smoothing(
            'return_1d', 
            method='rolling_mean', 
            window=5
        )
        
        self.assertIsInstance(smoothed, pd.Series)
        self.assertEqual(len(smoothed), len(self.data))
    
    def test_apply_noise_reduction(self):
        """测试应用噪声处理"""
        self.manager.add_return_label('return_1d', periods=1)
        self.manager.fit_transform(self.data)
        
        # 测试异常值裁剪
        cleaned = self.manager.apply_noise_reduction(
            'return_1d', 
            method='outlier_clip'
        )
        
        self.assertIsInstance(cleaned, pd.Series)
        self.assertEqual(len(cleaned), len(self.data))
    
    def test_invalid_label_operations(self):
        """测试无效标签操作"""
        self.manager.add_return_label('return_1d', periods=1)
        self.manager.fit_transform(self.data)
        
        # 测试不存在的标签
        with self.assertRaises(ValueError):
            self.manager.apply_smoothing('invalid_label')
        
        with self.assertRaises(ValueError):
            self.manager.apply_noise_reduction('invalid_label')


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'close': np.random.randn(30).cumsum() + 100,
            'date': pd.date_range('2023-01-01', periods=30, freq='D')
        })
    
    def test_create_return_labels(self):
        """测试批量创建收益率标签"""
        labels = create_return_labels(
            self.data, 
            periods=[1, 3, 5], 
            method='simple'
        )
        
        self.assertIsInstance(labels, pd.DataFrame)
        self.assertEqual(len(labels.columns), 3)
        
        expected_columns = ['return_1d', 'return_3d', 'return_5d']
        self.assertEqual(list(labels.columns), expected_columns)
    
    def test_create_direction_labels(self):
        """测试批量创建方向标签"""
        labels = create_direction_labels(
            self.data, 
            periods=[1, 3, 5], 
            neutral_zone=0.01
        )
        
        self.assertIsInstance(labels, pd.DataFrame)
        self.assertEqual(len(labels.columns), 3)
        
        expected_columns = ['direction_1d', 'direction_3d', 'direction_5d']
        self.assertEqual(list(labels.columns), expected_columns)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_data(self):
        """测试空数据"""
        empty_data = pd.DataFrame({'close': []})
        builder = ReturnLabelBuilder()
        
        # 空数据应该能拟合但产生空结果
        builder.fit(empty_data)
        result = builder.transform(empty_data)
        
        self.assertEqual(len(result), 0)
    
    def test_single_row_data(self):
        """测试单行数据"""
        single_data = pd.DataFrame({'close': [100]})
        builder = ReturnLabelBuilder(periods=1)
        
        builder.fit(single_data)
        result = builder.transform(single_data)
        
        # 单行数据无法计算收益率
        self.assertTrue(pd.isna(result.iloc[0]))
    
    def test_missing_values_in_prices(self):
        """测试价格中的缺失值"""
        data_with_nan = pd.DataFrame({
            'close': [100, np.nan, 102, 103, np.nan, 105]
        })
        
        builder = ReturnLabelBuilder(periods=1)
        builder.fit(data_with_nan)
        result = builder.transform(data_with_nan)
        
        # 包含NaN的收益率计算应该产生NaN
        self.assertTrue(pd.isna(result.iloc[0]))  # 100 -> NaN
        self.assertTrue(pd.isna(result.iloc[1]))  # NaN -> 102
    
    def test_large_periods(self):
        """测试大期数"""
        data = pd.DataFrame({'close': range(10)})
        builder = ReturnLabelBuilder(periods=20)  # 期数大于数据长度
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.fit(data)
            result = builder.transform(data)
            
            # 应该产生警告
            self.assertTrue(len(w) > 0)
            
            # 所有值都应该是NaN
            self.assertTrue(result.isna().all())


class TestPerformance(unittest.TestCase):
    """测试性能"""
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 创建大数据集
        np.random.seed(42)
        large_data = pd.DataFrame({
            'close': np.random.randn(10000).cumsum() + 100,
            'benchmark': np.random.randn(10000).cumsum() + 100
        })
        
        manager = LabelEngineeringManager()
        manager.add_return_label('return_1d', periods=1)
        manager.add_return_label('return_5d', periods=5)
        manager.add_direction_label('direction_1d', periods=1)
        manager.add_excess_return_label('excess_1d', periods=1, benchmark_column='benchmark')
        
        # 测试执行时间（应该在合理时间内完成）
        import time
        start_time = time.time()
        
        labels = manager.fit_transform(large_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证结果
        self.assertEqual(len(labels), len(large_data))
        self.assertEqual(len(labels.columns), 4)
        
        # 性能要求：10000行数据应该在10秒内完成
        self.assertLess(execution_time, 10.0, 
                       f"Performance test failed: {execution_time:.2f}s > 10.0s")
        
        print(f"Large dataset performance: {execution_time:.2f}s for {len(large_data)} rows")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)