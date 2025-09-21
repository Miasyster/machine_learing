"""
多目标标签构建模块单元测试

测试多目标标签构建功能的正确性、边界情况和性能
"""

import unittest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, MagicMock

# 添加src路径以便导入模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from features.multi_objective_labels import (
    SharpeRatioLabelBuilder,
    VolatilityLabelBuilder,
    RiskAdjustedReturnLabelBuilder,
    MultiObjectiveOptimizationLabelBuilder,
    MultiObjectiveLabelManager,
    create_sharpe_maximization_labels,
    create_multi_objective_labels
)


class TestSharpeRatioLabelBuilder(unittest.TestCase):
    """测试夏普比率标签构建器"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.data = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        self.data.set_index('date', inplace=True)
    
    def test_sharpe_ratio_labels_basic(self):
        """测试基础夏普比率标签生成"""
        builder = SharpeRatioLabelBuilder(periods=1, risk_free_rate=0.02)
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        # 检查标签基本属性
        self.assertIsInstance(labels, pd.Series)
        self.assertEqual(len(labels), len(self.data))
        
        # 检查统计信息
        self.assertIn('mean_sharpe', builder.label_stats_)
        self.assertIn('positive_sharpe_ratio', builder.label_stats_)
    
    def test_sharpe_ratio_labels_multi_period(self):
        """测试多期夏普比率标签"""
        builder = SharpeRatioLabelBuilder(periods=5, risk_free_rate=0.02)
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        # 检查标签数量（应该少于原数据，因为需要未来数据）
        valid_labels = labels.dropna()
        self.assertLess(len(valid_labels), len(self.data))
        self.assertGreater(len(valid_labels), 0)
    
    def test_sharpe_ratio_invalid_column(self):
        """测试无效价格列"""
        builder = SharpeRatioLabelBuilder(price_column='invalid')
        
        with self.assertRaises(ValueError):
            builder.fit(self.data)
    
    def test_sharpe_ratio_not_fitted(self):
        """测试未拟合时的错误"""
        builder = SharpeRatioLabelBuilder()
        
        with self.assertRaises(ValueError):
            builder.transform(self.data)
    
    def test_sharpe_ratio_fit_transform(self):
        """测试拟合和转换一体化"""
        builder = SharpeRatioLabelBuilder(periods=1)
        labels = builder.fit_transform(self.data)
        
        self.assertIsInstance(labels, pd.Series)
        self.assertTrue(builder.is_fitted_)


class TestVolatilityLabelBuilder(unittest.TestCase):
    """测试波动率标签构建器"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        highs = prices * 1.01
        lows = prices * 0.99
        
        self.data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': highs,
            'low': lows
        })
        self.data.set_index('date', inplace=True)
    
    def test_realized_volatility_labels(self):
        """测试已实现波动率标签"""
        builder = VolatilityLabelBuilder(periods=1, volatility_type='realized')
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        self.assertIsInstance(labels, pd.Series)
        self.assertIn('mean_volatility', builder.label_stats_)
    
    def test_parkinson_volatility_labels(self):
        """测试Parkinson波动率标签"""
        builder = VolatilityLabelBuilder(periods=1, volatility_type='parkinson')
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        self.assertIsInstance(labels, pd.Series)
        # Parkinson估计器可能产生较少的有效结果，检查是否为Series即可
        self.assertEqual(len(labels), len(self.data))
    
    def test_volatility_without_high_low(self):
        """测试没有高低价时的波动率计算"""
        data_no_hl = self.data[['close']].copy()
        builder = VolatilityLabelBuilder(periods=1, volatility_type='parkinson')
        builder.fit(data_no_hl)
        labels = builder.transform(data_no_hl)
        
        # 应该回退到已实现波动率
        self.assertIsInstance(labels, pd.Series)


class TestRiskAdjustedReturnLabelBuilder(unittest.TestCase):
    """测试风险调整收益标签构建器"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        
        benchmark_returns = np.random.normal(0.0005, 0.015, 100)
        benchmark_prices = 100 * np.exp(np.cumsum(benchmark_returns))
        
        self.data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'benchmark': benchmark_prices
        })
        self.data.set_index('date', inplace=True)
    
    def test_information_ratio_labels(self):
        """测试信息比率标签"""
        builder = RiskAdjustedReturnLabelBuilder(
            periods=1, 
            metric='information_ratio',
            benchmark_column='benchmark'
        )
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        self.assertIsInstance(labels, pd.Series)
        self.assertIn('mean_information_ratio', builder.label_stats_)
    
    def test_sortino_ratio_labels(self):
        """测试索提诺比率标签"""
        builder = RiskAdjustedReturnLabelBuilder(
            periods=1,
            metric='sortino_ratio',
            benchmark_column='benchmark'
        )
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        self.assertIsInstance(labels, pd.Series)
        self.assertIn('mean_sortino_ratio', builder.label_stats_)
    
    def test_calmar_ratio_labels(self):
        """测试卡尔马比率标签"""
        builder = RiskAdjustedReturnLabelBuilder(
            periods=1,
            metric='calmar_ratio'
        )
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        self.assertIsInstance(labels, pd.Series)
        self.assertIn('mean_calmar_ratio', builder.label_stats_)
    
    def test_fixed_benchmark_return(self):
        """测试固定基准收益率"""
        builder = RiskAdjustedReturnLabelBuilder(
            periods=1,
            metric='information_ratio',
            benchmark_return=0.03
        )
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        self.assertIsInstance(labels, pd.Series)
    
    def test_unsupported_metric(self):
        """测试不支持的指标"""
        builder = RiskAdjustedReturnLabelBuilder(metric='invalid_metric')
        
        with self.assertRaises(ValueError):
            builder.fit(self.data)


class TestMultiObjectiveOptimizationLabelBuilder(unittest.TestCase):
    """测试多目标优化标签构建器"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.data = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        self.data.set_index('date', inplace=True)
    
    def test_weighted_sum_optimization(self):
        """测试加权求和优化"""
        builder = MultiObjectiveOptimizationLabelBuilder(
            periods=1,
            objectives=['return', 'sharpe'],
            weights=[0.6, 0.4],
            optimization_method='weighted_sum'
        )
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        self.assertIsInstance(labels, pd.Series)
        self.assertTrue(builder.is_fitted_)
    
    def test_pareto_ranking_optimization(self):
        """测试帕累托排序优化"""
        builder = MultiObjectiveOptimizationLabelBuilder(
            periods=1,
            objectives=['return', 'sharpe'],
            optimization_method='pareto_ranking'
        )
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        self.assertIsInstance(labels, pd.Series)
    
    def test_unsupported_optimization_method(self):
        """测试不支持的优化方法"""
        builder = MultiObjectiveOptimizationLabelBuilder(
            optimization_method='invalid_method'
        )
        builder.fit(self.data)
        
        with self.assertRaises(ValueError):
            builder.transform(self.data)
    
    def test_multi_objective_with_volatility(self):
        """测试包含波动率的多目标优化"""
        builder = MultiObjectiveOptimizationLabelBuilder(
            periods=1,
            objectives=['return', 'sharpe', 'volatility']
        )
        builder.fit(self.data)
        labels = builder.transform(self.data)
        
        self.assertIsInstance(labels, pd.Series)


class TestMultiObjectiveLabelManager(unittest.TestCase):
    """测试多目标标签管理器"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        
        benchmark_returns = np.random.normal(0.0005, 0.015, 100)
        benchmark_prices = 100 * np.exp(np.cumsum(benchmark_returns))
        
        self.data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'benchmark': benchmark_prices
        })
        self.data.set_index('date', inplace=True)
        
        self.manager = MultiObjectiveLabelManager()
    
    def test_add_sharpe_label(self):
        """测试添加夏普比率标签"""
        self.manager.add_sharpe_label('test_sharpe', periods=1)
        
        self.assertIn('test_sharpe', self.manager.builders)
        self.assertIsInstance(
            self.manager.builders['test_sharpe'], 
            SharpeRatioLabelBuilder
        )
    
    def test_add_volatility_label(self):
        """测试添加波动率标签"""
        self.manager.add_volatility_label('test_vol', periods=1)
        
        self.assertIn('test_vol', self.manager.builders)
        self.assertIsInstance(
            self.manager.builders['test_vol'],
            VolatilityLabelBuilder
        )
    
    def test_add_risk_adjusted_label(self):
        """测试添加风险调整标签"""
        self.manager.add_risk_adjusted_label(
            'test_risk_adj',
            periods=1,
            metric='information_ratio'
        )
        
        self.assertIn('test_risk_adj', self.manager.builders)
        self.assertIsInstance(
            self.manager.builders['test_risk_adj'],
            RiskAdjustedReturnLabelBuilder
        )
    
    def test_add_multi_objective_label(self):
        """测试添加多目标标签"""
        self.manager.add_multi_objective_label(
            'test_multi_obj',
            periods=1,
            objectives=['return', 'sharpe']
        )
        
        self.assertIn('test_multi_obj', self.manager.builders)
        self.assertIsInstance(
            self.manager.builders['test_multi_obj'],
            MultiObjectiveOptimizationLabelBuilder
        )
    
    def test_comprehensive_fit_transform(self):
        """测试综合拟合和转换"""
        self.manager.add_sharpe_label('sharpe_1d', periods=1)
        self.manager.add_volatility_label('vol_1d', periods=1)
        self.manager.add_risk_adjusted_label(
            'info_ratio',
            periods=1,
            metric='information_ratio',
            benchmark_column='benchmark'
        )
        
        labels = self.manager.fit_transform(self.data)
        
        self.assertIsInstance(labels, pd.DataFrame)
        self.assertEqual(len(labels), len(self.data))
        expected_columns = ['sharpe_1d', 'vol_1d', 'info_ratio']
        
        # 检查至少有一些列被成功生成
        generated_columns = [col for col in expected_columns if col in labels.columns]
        self.assertGreater(len(generated_columns), 0)
    
    def test_get_optimization_summary(self):
        """测试获取优化摘要"""
        self.manager.add_sharpe_label('sharpe_1d', periods=1)
        labels = self.manager.fit_transform(self.data)
        
        summary = self.manager.get_optimization_summary()
        
        if not labels.empty:
            self.assertIsInstance(summary, pd.DataFrame)
            if len(summary) > 0:
                self.assertIn('count', summary.columns)
                self.assertIn('mean', summary.columns)


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        returns = np.random.normal(0.001, 0.02, 50)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.data = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        self.data.set_index('date', inplace=True)
    
    def test_create_sharpe_maximization_labels(self):
        """测试创建夏普比率最大化标签"""
        labels = create_sharpe_maximization_labels(
            self.data,
            periods=[1, 3],
            risk_free_rate=0.02
        )
        
        self.assertIsInstance(labels, pd.DataFrame)
        expected_columns = ['sharpe_1d', 'sharpe_3d']
        
        # 检查至少有一些列被生成
        generated_columns = [col for col in expected_columns if col in labels.columns]
        self.assertGreater(len(generated_columns), 0)
    
    def test_create_multi_objective_labels(self):
        """测试创建多目标标签"""
        labels = create_multi_objective_labels(
            self.data,
            periods=[1],
            objectives=['return', 'sharpe']
        )
        
        self.assertIsInstance(labels, pd.DataFrame)
        expected_columns = ['multi_obj_1d']
        
        # 检查至少有一些列被生成
        generated_columns = [col for col in expected_columns if col in labels.columns]
        self.assertGreater(len(generated_columns), 0)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_data(self):
        """测试空数据"""
        empty_data = pd.DataFrame({'close': []})
        builder = SharpeRatioLabelBuilder()
        
        # 空数据应该能够拟合但产生空的标签
        builder.fit(empty_data)
        labels = builder.transform(empty_data)
        self.assertIsInstance(labels, pd.Series)
        self.assertEqual(len(labels), 0)
    
    def test_insufficient_data(self):
        """测试数据不足"""
        small_data = pd.DataFrame({'close': [100, 101, 102]})
        builder = SharpeRatioLabelBuilder(rolling_window=20)
        
        # 应该能够拟合，但可能产生很少的有效标签
        builder.fit(small_data)
        labels = builder.transform(small_data)
        self.assertIsInstance(labels, pd.Series)
    
    def test_constant_prices(self):
        """测试恒定价格"""
        constant_data = pd.DataFrame({'close': [100] * 50})
        builder = SharpeRatioLabelBuilder()
        
        builder.fit(constant_data)
        labels = builder.transform(constant_data)
        
        # 恒定价格应该产生零或NaN的夏普比率
        self.assertIsInstance(labels, pd.Series)
    
    def test_extreme_volatility(self):
        """测试极端波动率"""
        np.random.seed(42)
        extreme_returns = np.random.normal(0, 0.5, 50)  # 极高波动率
        extreme_prices = 100 * np.exp(np.cumsum(extreme_returns))
        extreme_data = pd.DataFrame({'close': extreme_prices})
        
        builder = SharpeRatioLabelBuilder()
        builder.fit(extreme_data)
        labels = builder.transform(extreme_data)
        
        self.assertIsInstance(labels, pd.Series)


class TestPerformance(unittest.TestCase):
    """测试性能"""
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 创建较大的数据集
        np.random.seed(42)
        large_size = 5000
        dates = pd.date_range('2010-01-01', periods=large_size, freq='D')
        returns = np.random.normal(0.001, 0.02, large_size)
        prices = 100 * np.exp(np.cumsum(returns))
        
        large_data = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        large_data.set_index('date', inplace=True)
        
        # 测试多目标标签管理器性能
        manager = MultiObjectiveLabelManager()
        manager.add_sharpe_label('sharpe_1d', periods=1)
        manager.add_volatility_label('vol_1d', periods=1)
        
        import time
        start_time = time.time()
        labels = manager.fit_transform(large_data)
        end_time = time.time()
        
        # 检查结果
        self.assertIsInstance(labels, pd.DataFrame)
        self.assertEqual(len(labels), len(large_data))
        
        # 性能检查（应该在合理时间内完成）
        execution_time = end_time - start_time
        self.assertLess(execution_time, 30)  # 应该在30秒内完成
        
        print(f"大数据集处理时间: {execution_time:.2f}秒")


if __name__ == '__main__':
    # 忽略警告以保持测试输出清洁
    warnings.filterwarnings('ignore')
    
    # 运行测试
    unittest.main(verbosity=2)