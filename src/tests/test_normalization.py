"""
特征标准化模块的单元测试
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from features.normalization import (
    RollingZScoreNormalizer,
    RollingQuantileNormalizer,
    RobustZScoreNormalizer,
    FeatureNormalizer,
    rolling_zscore_normalize,
    rolling_quantile_normalize,
    robust_zscore_normalize
)


class TestNormalization(unittest.TestCase):
    """标准化功能测试类"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        
        # 创建测试数据
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        symbols = ['AAPL', 'GOOGL']
        
        data_list = []
        for symbol in symbols:
            base_price = 100 if symbol == 'AAPL' else 1000
            for i, date in enumerate(dates):
                # 创建有趋势的价格数据
                trend = i * 0.1
                noise = np.random.randn() * 5
                price = base_price + trend + noise
                
                data_list.append({
                    'date': date,
                    'symbol': symbol,
                    'close': price,
                    'volume': 1000000 + np.random.randn() * 100000,
                    'feature1': np.random.randn() * 2,
                    'feature2': np.random.randn() * 10 + 5
                })
        
        self.test_data = pd.DataFrame(data_list)
        self.feature_columns = ['close', 'volume', 'feature1', 'feature2']
    
    def test_rolling_zscore_normalizer(self):
        """测试滚动Z-Score标准化器"""
        normalizer = RollingZScoreNormalizer(window=60, min_periods=30)
        
        # 测试fit_transform
        result = normalizer.fit_transform(
            self.test_data, self.feature_columns, 'symbol'
        )
        
        # 检查结果形状
        self.assertEqual(result.shape, self.test_data.shape)
        
        # 检查标准化后的数据特性（滚动标准化的特性）
        for symbol in ['AAPL', 'GOOGL']:
            symbol_data = result[result['symbol'] == symbol]
            for col in self.feature_columns:
                col_data = symbol_data[col].dropna()
                if len(col_data) > 100:  # 有足够数据时检查
                    # 滚动标准化后的均值可能不完全为0，但应该相对较小
                    self.assertLess(abs(col_data.mean()), 1.0)
                    # 标准差应该相对合理
                    self.assertGreater(col_data.std(), 0.5)
                    self.assertLess(col_data.std(), 2.0)
    
    def test_rolling_quantile_normalizer(self):
        """测试滚动分位数标准化器"""
        normalizer = RollingQuantileNormalizer(window=60, min_periods=30)
        
        # 测试fit_transform
        result = normalizer.fit_transform(
            self.test_data, self.feature_columns, 'symbol'
        )
        
        # 检查结果形状
        self.assertEqual(result.shape, self.test_data.shape)
        
        # 检查标准化后的数据范围（应该在0-1之间）
        for col in self.feature_columns:
            col_data = result[col].dropna()
            if len(col_data) > 0:
                self.assertGreaterEqual(col_data.min(), 0)
                self.assertLessEqual(col_data.max(), 1)
    
    def test_robust_zscore_normalizer(self):
        """测试鲁棒Z-Score标准化器"""
        normalizer = RobustZScoreNormalizer(window=60, min_periods=30)
        
        # 测试fit_transform
        result = normalizer.fit_transform(
            self.test_data, self.feature_columns, 'symbol'
        )
        
        # 检查结果形状
        self.assertEqual(result.shape, self.test_data.shape)
        
        # 检查标准化后的数据（应该对异常值更鲁棒）
        for symbol in ['AAPL', 'GOOGL']:
            symbol_data = result[result['symbol'] == symbol]
            for col in self.feature_columns:
                col_data = symbol_data[col].dropna()
                if len(col_data) > 100:
                    # 中位数应该接近0
                    self.assertAlmostEqual(col_data.median(), 0, delta=0.5)
    
    def test_feature_normalizer(self):
        """测试特征标准化管理器"""
        normalizer = FeatureNormalizer()
        
        # 测试不同的标准化方法
        methods = ['rolling_zscore', 'rolling_quantile', 'robust_zscore']
        
        for method in methods:
            with self.subTest(method=method):
                result = normalizer.normalize_features(
                    self.test_data, self.feature_columns, method, 'symbol',
                    window=60, min_periods=30
                )
                
                # 检查结果形状
                self.assertEqual(result.shape, self.test_data.shape)
                
                # 检查标准化器是否被保存
                self.assertIn(method, normalizer.fitted_normalizers)
    
    def test_batch_normalize(self):
        """测试批量标准化"""
        normalizer = FeatureNormalizer()
        
        # 配置不同特征组使用不同的标准化方法
        feature_configs = {
            'price_features': {
                'columns': ['close'],
                'method': 'rolling_zscore',
                'params': {'window': 60, 'min_periods': 30}
            },
            'volume_features': {
                'columns': ['volume'],
                'method': 'rolling_quantile',
                'params': {'window': 60, 'min_periods': 30}
            },
            'other_features': {
                'columns': ['feature1', 'feature2'],
                'method': 'robust_zscore',
                'params': {'window': 60, 'min_periods': 30}
            }
        }
        
        result = normalizer.batch_normalize(
            self.test_data, feature_configs, 'symbol'
        )
        
        # 检查结果形状
        self.assertEqual(result.shape, self.test_data.shape)
        
        # 检查不同特征是否使用了不同的标准化方法
        # close列应该是z-score标准化（均值相对较小）
        close_data = result[result['symbol'] == 'AAPL']['close'].dropna()
        if len(close_data) > 100:
            self.assertLess(abs(close_data.mean()), 1.0)
        
        # volume列应该是分位数标准化（范围0-1）
        volume_data = result['volume'].dropna()
        if len(volume_data) > 0:
            self.assertGreaterEqual(volume_data.min(), 0)
            self.assertLessEqual(volume_data.max(), 1)
    
    def test_normalization_stats(self):
        """测试标准化统计信息"""
        normalizer = FeatureNormalizer()
        
        stats = normalizer.get_normalization_stats(
            self.test_data, self.feature_columns, 'symbol'
        )
        
        # 检查统计信息结构
        self.assertIn('AAPL', stats)
        self.assertIn('GOOGL', stats)
        
        for symbol in ['AAPL', 'GOOGL']:
            symbol_stats = stats[symbol]
            for col in self.feature_columns:
                if col in symbol_stats:
                    col_stats = symbol_stats[col]
                    # 检查必要的统计量
                    required_stats = ['mean', 'std', 'median', 'mad', 'min', 'max', 'count']
                    for stat in required_stats:
                        self.assertIn(stat, col_stats)
    
    def test_convenience_functions(self):
        """测试便捷函数"""
        # 测试滚动Z-Score便捷函数
        result1 = rolling_zscore_normalize(
            self.test_data, self.feature_columns, 'symbol', window=60
        )
        self.assertEqual(result1.shape, self.test_data.shape)
        
        # 测试滚动分位数便捷函数
        result2 = rolling_quantile_normalize(
            self.test_data, self.feature_columns, 'symbol', window=60
        )
        self.assertEqual(result2.shape, self.test_data.shape)
        
        # 测试鲁棒Z-Score便捷函数
        result3 = robust_zscore_normalize(
            self.test_data, self.feature_columns, 'symbol', window=60
        )
        self.assertEqual(result3.shape, self.test_data.shape)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空数据
        empty_data = pd.DataFrame()
        normalizer = RollingZScoreNormalizer()
        
        with self.assertRaises(ValueError):
            normalizer.fit_transform(empty_data, self.feature_columns, 'symbol')
        
        # 测试缺少列
        incomplete_data = self.test_data[['date', 'symbol']].copy()
        
        with self.assertRaises(ValueError):
            normalizer.fit_transform(incomplete_data, self.feature_columns, 'symbol')
        
        # 测试数据量不足的情况
        small_data = self.test_data.head(10).copy()
        normalizer_large_window = RollingZScoreNormalizer(window=100, min_periods=50)
        
        # 应该产生警告但不报错
        with self.assertWarns(UserWarning):
            result = normalizer_large_window.fit_transform(
                small_data, self.feature_columns, 'symbol'
            )
            self.assertEqual(result.shape, small_data.shape)
    
    def test_data_consistency(self):
        """测试数据一致性"""
        normalizer = RollingZScoreNormalizer(window=60, min_periods=30)
        
        # 多次运行应该得到相同结果
        result1 = normalizer.fit_transform(
            self.test_data, self.feature_columns, 'symbol'
        )
        result2 = normalizer.fit_transform(
            self.test_data, self.feature_columns, 'symbol'
        )
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_cross_symbol_independence(self):
        """测试跨品种独立性"""
        normalizer = RollingZScoreNormalizer(window=60, min_periods=30)
        
        # 分别对每个品种标准化
        aapl_data = self.test_data[self.test_data['symbol'] == 'AAPL'].copy()
        googl_data = self.test_data[self.test_data['symbol'] == 'GOOGL'].copy()
        
        aapl_result = normalizer.fit_transform(aapl_data, self.feature_columns, 'symbol')
        googl_result = normalizer.fit_transform(googl_data, self.feature_columns, 'symbol')
        
        # 合并结果
        separate_result = pd.concat([aapl_result, googl_result]).sort_index()
        
        # 一起标准化
        combined_result = normalizer.fit_transform(
            self.test_data, self.feature_columns, 'symbol'
        )
        
        # 结果应该相同（证明品种间独立）
        for col in self.feature_columns:
            pd.testing.assert_series_equal(
                separate_result[col].sort_index(),
                combined_result[col].sort_index(),
                check_names=False
            )


class TestNormalizationPerformance(unittest.TestCase):
    """标准化性能测试"""
    
    def setUp(self):
        """设置大规模测试数据"""
        np.random.seed(42)
        
        # 创建较大的测试数据集
        dates = pd.date_range('2015-01-01', periods=2000, freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        data_list = []
        for symbol in symbols:
            base_price = np.random.uniform(50, 1000)
            for i, date in enumerate(dates):
                data_list.append({
                    'date': date,
                    'symbol': symbol,
                    'close': base_price + np.random.randn() * 10,
                    'volume': 1000000 + np.random.randn() * 100000,
                    'feature1': np.random.randn(),
                    'feature2': np.random.randn() * 5,
                    'feature3': np.random.randn() * 2,
                    'feature4': np.random.randn() * 8
                })
        
        self.large_data = pd.DataFrame(data_list)
        self.feature_columns = ['close', 'volume', 'feature1', 'feature2', 'feature3', 'feature4']
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        import time
        
        normalizer = FeatureNormalizer()
        
        start_time = time.time()
        result = normalizer.normalize_features(
            self.large_data, self.feature_columns, 'rolling_zscore', 'symbol',
            window=252, min_periods=60
        )
        end_time = time.time()
        
        # 检查结果正确性
        self.assertEqual(result.shape, self.large_data.shape)
        
        # 性能检查（应该在合理时间内完成）
        processing_time = end_time - start_time
        print(f"大数据集标准化耗时: {processing_time:.2f}秒")
        
        # 对于10000行数据，应该在10秒内完成
        self.assertLess(processing_time, 10.0)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)