"""
数据清洗模块测试用例
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加src路径到系统路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from etl.data_cleaner import DataCleaner


class TestDataCleaner:
    """数据清洗器测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': np.random.normal(50000, 1000, 100),
            'high': np.random.normal(51000, 1000, 100),
            'low': np.random.normal(49000, 1000, 100),
            'close': np.random.normal(50000, 1000, 100),
            'volume': np.random.normal(1000, 200, 100),
            'quote_asset_volume': np.random.normal(50000000, 10000000, 100)
        }, index=dates)
        
        # 确保OHLC关系正确
        for i in range(len(data)):
            prices = [data.iloc[i]['open'], data.iloc[i]['close']]
            data.iloc[i, data.columns.get_loc('high')] = max(prices) + np.random.uniform(0, 100)
            data.iloc[i, data.columns.get_loc('low')] = min(prices) - np.random.uniform(0, 100)
        
        return data
    
    @pytest.fixture
    def data_with_missing(self, sample_data):
        """创建有缺失值的测试数据"""
        data = sample_data.copy()
        # 添加缺失值
        data.loc[data.index[10:15], 'close'] = np.nan
        data.loc[data.index[20:22], 'volume'] = np.nan
        data.loc[data.index[0], 'open'] = np.nan
        return data
    
    @pytest.fixture
    def data_with_outliers(self, sample_data):
        """创建有异常值的测试数据"""
        data = sample_data.copy()
        # 添加异常值
        data.loc[data.index[20], 'high'] = 100000  # 价格异常高
        data.loc[data.index[30], 'volume'] = -100  # 负成交量
        data.loc[data.index[40], 'low'] = data.loc[data.index[40], 'high'] + 100  # low > high
        return data
    
    @pytest.fixture
    def cleaner(self):
        """创建数据清洗器实例"""
        return DataCleaner()
    
    def test_init(self, cleaner):
        """测试初始化"""
        assert cleaner.price_columns == ['open', 'high', 'low', 'close']
        assert cleaner.volume_columns == ['volume', 'quote_asset_volume']
    
    def test_detect_missing_values(self, cleaner, data_with_missing):
        """测试缺失值检测"""
        missing_info = cleaner.detect_missing_values(data_with_missing)
        
        assert missing_info['total_rows'] == 100
        assert missing_info['missing_by_column']['close'] == 5
        assert missing_info['missing_by_column']['volume'] == 2
        assert missing_info['missing_by_column']['open'] == 1
        assert missing_info['rows_with_missing'] > 0
        assert missing_info['complete_rows'] < 100
    
    def test_fill_missing_values_forward(self, cleaner, data_with_missing):
        """测试前向填充"""
        filled_data = cleaner.fill_missing_values(data_with_missing, method='forward')
        
        # 检查是否减少了缺失值
        original_missing = data_with_missing.isnull().sum().sum()
        filled_missing = filled_data.isnull().sum().sum()
        assert filled_missing <= original_missing
    
    def test_fill_missing_values_interpolate(self, cleaner, data_with_missing):
        """测试插值填充"""
        filled_data = cleaner.fill_missing_values(data_with_missing, method='interpolate')
        
        # 价格列应该被插值填充
        for col in cleaner.price_columns:
            if col in data_with_missing.columns:
                original_missing = data_with_missing[col].isnull().sum()
                filled_missing = filled_data[col].isnull().sum()
                assert filled_missing <= original_missing
    
    def test_fill_missing_values_smart(self, cleaner, data_with_missing):
        """测试智能填充"""
        filled_data = cleaner.fill_missing_values(data_with_missing, method='smart')
        
        # 检查成交量列是否被正确处理（可能填充为0）
        volume_missing = filled_data['volume'].isnull().sum()
        assert volume_missing == 0  # 智能填充应该处理所有成交量缺失值
    
    def test_detect_outliers_zscore(self, cleaner, data_with_outliers):
        """测试Z分数异常值检测"""
        outlier_info = cleaner.detect_outliers(data_with_outliers, method='z_score')
        
        assert outlier_info['method'] == 'z_score'
        assert outlier_info['total_outliers'] > 0
        assert 'outliers_by_column' in outlier_info
    
    def test_detect_outliers_iqr(self, cleaner, data_with_outliers):
        """测试IQR异常值检测"""
        outlier_info = cleaner.detect_outliers(data_with_outliers, method='iqr')
        
        assert outlier_info['method'] == 'iqr'
        assert outlier_info['total_outliers'] > 0
    
    def test_detect_outliers_business_rules(self, cleaner, data_with_outliers):
        """测试业务规则异常值检测"""
        outlier_info = cleaner.detect_outliers(data_with_outliers, method='business_rules')
        
        assert outlier_info['method'] == 'business_rules'
        # 应该检测到负成交量和价格关系异常
        assert outlier_info['total_outliers'] > 0
    
    def test_handle_outliers_clip(self, cleaner, data_with_outliers):
        """测试异常值截断处理"""
        outlier_info = cleaner.detect_outliers(data_with_outliers, method='iqr')
        cleaned_data = cleaner.handle_outliers(data_with_outliers, outlier_info, method='clip')
        
        # 检查数据形状没有改变
        assert cleaned_data.shape == data_with_outliers.shape
        
        # 检查极端值是否被截断
        for col in cleaner.price_columns:
            if col in cleaned_data.columns:
                assert cleaned_data[col].max() <= data_with_outliers[col].quantile(0.95) * 1.1
    
    def test_handle_outliers_remove(self, cleaner, data_with_outliers):
        """测试异常值删除处理"""
        outlier_info = cleaner.detect_outliers(data_with_outliers, method='business_rules')
        cleaned_data = cleaner.handle_outliers(data_with_outliers, outlier_info, method='remove')
        
        # 检查行数是否减少
        assert len(cleaned_data) <= len(data_with_outliers)
    
    def test_handle_outliers_interpolate(self, cleaner, data_with_outliers):
        """测试异常值插值处理"""
        outlier_info = cleaner.detect_outliers(data_with_outliers, method='iqr')
        cleaned_data = cleaner.handle_outliers(data_with_outliers, outlier_info, method='interpolate')
        
        # 检查数据形状没有改变
        assert cleaned_data.shape == data_with_outliers.shape
    
    def test_align_time_series_single(self, cleaner, sample_data):
        """测试单个DataFrame时序对齐"""
        # 创建有缺失时间点的数据
        incomplete_data = sample_data.iloc[::2].copy()  # 每隔一个取一个
        
        aligned_data = cleaner.align_time_series(incomplete_data, freq='1H')
        
        # 检查时间序列是否完整
        expected_length = (sample_data.index.max() - sample_data.index.min()).total_seconds() / 3600 + 1
        assert len(aligned_data) == expected_length
    
    def test_align_time_series_multiple(self, cleaner, sample_data):
        """测试多个DataFrame时序对齐"""
        # 创建两个不同时间范围的数据集
        data1 = sample_data.iloc[:50].copy()
        data2 = sample_data.iloc[25:75].copy()
        
        data_dict = {'symbol1': data1, 'symbol2': data2}
        aligned_dict = cleaner.align_time_series(data_dict, freq='1H')
        
        # 检查所有数据集的长度是否一致
        lengths = [len(df) for df in aligned_dict.values()]
        assert len(set(lengths)) == 1  # 所有长度应该相同
        
        # 检查时间索引是否一致
        indices = [df.index for df in aligned_dict.values()]
        for i in range(1, len(indices)):
            assert indices[0].equals(indices[i])
    
    def test_validate_data_quality(self, cleaner, sample_data):
        """测试数据质量验证"""
        quality_report = cleaner.validate_data_quality(sample_data)
        
        assert 'basic_info' in quality_report
        assert 'missing_values' in quality_report
        assert 'price_validation' in quality_report
        assert 'volume_validation' in quality_report
        assert 'ohlc_validation' in quality_report
        
        # 检查基本信息
        assert quality_report['basic_info']['total_rows'] == len(sample_data)
        assert quality_report['basic_info']['total_columns'] == len(sample_data.columns)
    
    def test_clean_data_pipeline(self, cleaner, data_with_missing, data_with_outliers):
        """测试完整数据清洗流水线"""
        # 创建同时有缺失值和异常值的数据
        problematic_data = data_with_missing.copy()
        problematic_data.loc[problematic_data.index[20], 'high'] = 100000
        problematic_data.loc[problematic_data.index[30], 'volume'] = -100
        
        cleaned_data, report = cleaner.clean_data_pipeline(
            problematic_data,
            missing_method='smart',
            outlier_detection='business_rules',
            outlier_handling='clip'
        )
        
        # 检查报告结构
        assert 'original_shape' in report
        assert 'final_shape' in report
        assert 'steps_performed' in report
        assert 'quality_before' in report
        assert 'quality_after' in report
        
        # 检查数据质量是否改善
        before_missing = report['quality_before']['missing_values']['rows_with_missing']
        after_missing = report['quality_after']['missing_values']['rows_with_missing']
        assert after_missing <= before_missing
    
    def test_generate_cleaning_summary(self, cleaner, data_with_missing):
        """测试清洗摘要生成"""
        cleaned_data, report = cleaner.clean_data_pipeline(data_with_missing)
        summary = cleaner.generate_cleaning_summary(report)
        
        assert isinstance(summary, str)
        assert "数据清洗摘要报告" in summary
        assert "原始数据形状" in summary
        assert "清洗后形状" in summary
    
    def test_edge_cases(self, cleaner):
        """测试边界情况"""
        # 空数据
        empty_data = pd.DataFrame()
        with pytest.raises((ValueError, IndexError)):
            cleaner.detect_missing_values(empty_data)
        
        # 单行数据
        single_row = pd.DataFrame({
            'open': [50000], 'high': [51000], 'low': [49000], 'close': [50500]
        }, index=pd.date_range('2024-01-01', periods=1))
        
        missing_info = cleaner.detect_missing_values(single_row)
        assert missing_info['total_rows'] == 1
    
    def test_invalid_methods(self, cleaner, sample_data):
        """测试无效方法参数"""
        with pytest.raises(ValueError):
            cleaner.fill_missing_values(sample_data, method='invalid_method')
        
        with pytest.raises(ValueError):
            cleaner.detect_outliers(sample_data, method='invalid_method')
        
        with pytest.raises(ValueError):
            outlier_info = {'outliers_by_column': {}, 'outlier_indices': set()}
            cleaner.handle_outliers(sample_data, outlier_info, method='invalid_method')


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])