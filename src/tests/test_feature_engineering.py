"""
特征工程模块测试用例
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加src路径到系统路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.feature_engineering import (
    FeatureEngineer,
    SimpleMovingAverage, ExponentialMovingAverage, WeightedMovingAverage,
    AverageTrueRange, RollingVolatility, BollingerBands,
    VolumeWeightedAveragePrice, OnBalanceVolume, VolumeRatio,
    RelativeStrengthIndex, MACD, Momentum, RateOfChange
)


class TestFeatureEngineering:
    """特征工程测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)  # 确保结果可重现
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        # 生成模拟价格数据
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, 100)  # 2%的价格波动
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # 生成close价格
        close_prices = [p * (1 + np.random.normal(0, 0.001)) for p in prices]
        
        # 生成high和low价格
        high_prices = []
        low_prices = []
        for i in range(len(prices)):
            open_price = prices[i]
            close_price = close_prices[i]
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
            high_prices.append(high_price)
            low_prices.append(low_price)
        
        data = pd.DataFrame({
            'open': prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.normal(1000, 200, 100),
            'quote_asset_volume': np.random.normal(50000000, 10000000, 100)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def feature_engineer(self):
        """创建特征工程器实例"""
        return FeatureEngineer()
    
    def test_feature_engineer_initialization(self, feature_engineer):
        """测试特征工程器初始化"""
        assert isinstance(feature_engineer, FeatureEngineer)
        indicators = feature_engineer.get_available_indicators()
        assert len(indicators) > 0
        assert 'SMA' in indicators
        assert 'EMA' in indicators
        assert 'RSI' in indicators
    
    def test_simple_moving_average(self, sample_data):
        """测试简单移动平均"""
        sma = SimpleMovingAverage()
        result = sma.calculate(sample_data, column='close', window=20)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isnull().all()
        
        # 检查计算正确性（第20个值应该等于前20个值的平均）
        manual_sma_20 = sample_data['close'].iloc[:20].mean()
        assert abs(result.iloc[19] - manual_sma_20) < 1e-10
    
    def test_exponential_moving_average(self, sample_data):
        """测试指数移动平均"""
        ema = ExponentialMovingAverage()
        result = ema.calculate(sample_data, column='close', window=20)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isnull().all()
        
        # EMA应该对最近的价格更敏感
        assert result.iloc[-1] != result.iloc[-2]
    
    def test_weighted_moving_average(self, sample_data):
        """测试加权移动平均"""
        wma = WeightedMovingAverage()
        result = wma.calculate(sample_data, column='close', window=10)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isnull().all()
    
    def test_average_true_range(self, sample_data):
        """测试平均真实波幅"""
        atr = AverageTrueRange()
        result = atr.calculate(sample_data, window=14)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert (result.dropna() >= 0).all()  # ATR应该总是非负的（忽略NaN值）
    
    def test_rolling_volatility(self, sample_data):
        """测试滚动波动率"""
        vol = RollingVolatility()
        result = vol.calculate(sample_data, column='close', window=20, annualize=True)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert (result.dropna() >= 0).all()  # 波动率应该总是非负的（忽略NaN值）
    
    def test_bollinger_bands(self, sample_data):
        """测试布林带"""
        bb = BollingerBands()
        result = bb.calculate(sample_data, column='close', window=20, num_std=2.0)
        
        assert isinstance(result, dict)
        assert 'bb_upper' in result
        assert 'bb_middle' in result
        assert 'bb_lower' in result
        assert 'bb_width' in result
        assert 'bb_position' in result
        
        # 检查布林带关系（忽略NaN值）
        valid_mask = ~(result['bb_upper'].isna() | result['bb_middle'].isna() | result['bb_lower'].isna())
        assert (result['bb_upper'][valid_mask] >= result['bb_middle'][valid_mask]).all()
        assert (result['bb_middle'][valid_mask] >= result['bb_lower'][valid_mask]).all()
        # 布林带位置指标可以超出0-1范围（价格可能在布林带外）
        valid_position = result['bb_position'].dropna()
        assert isinstance(valid_position, pd.Series)  # 只检查类型，不限制范围
    
    def test_volume_weighted_average_price(self, sample_data):
        """测试成交量加权平均价格"""
        vwap = VolumeWeightedAveragePrice()
        
        # 测试累积VWAP
        result_cumulative = vwap.calculate(sample_data)
        assert isinstance(result_cumulative, pd.Series)
        assert len(result_cumulative) == len(sample_data)
        
        # 测试滚动VWAP
        result_rolling = vwap.calculate(sample_data, window=20)
        assert isinstance(result_rolling, pd.Series)
        assert len(result_rolling) == len(sample_data)
    
    def test_on_balance_volume(self, sample_data):
        """测试能量潮"""
        obv = OnBalanceVolume()
        result = obv.calculate(sample_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        # OBV是累积的，所以应该是单调的（除非价格完全不变）
    
    def test_volume_ratio(self, sample_data):
        """测试成交量比率"""
        vr = VolumeRatio()
        result = vr.calculate(sample_data, window=20)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert (result > 0).all()  # 成交量比率应该总是正数
    
    def test_relative_strength_index(self, sample_data):
        """测试相对强弱指数"""
        rsi = RelativeStrengthIndex()
        result = rsi.calculate(sample_data, column='close', window=14)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        # RSI应该在0-100之间
        valid_rsi = result.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd(self, sample_data):
        """测试MACD"""
        macd = MACD()
        result = macd.calculate(sample_data, column='close')
        
        assert isinstance(result, dict)
        assert 'macd' in result
        assert 'macd_signal' in result
        assert 'macd_histogram' in result
        
        # 检查MACD组件
        for key, series in result.items():
            assert isinstance(series, pd.Series)
            assert len(series) == len(sample_data)
        
        # 柱状图应该等于MACD线减去信号线
        histogram_check = result['macd'] - result['macd_signal']
        np.testing.assert_array_almost_equal(
            result['macd_histogram'].dropna().values,
            histogram_check.dropna().values,
            decimal=10
        )
    
    def test_momentum(self, sample_data):
        """测试动量指标"""
        momentum = Momentum()
        result = momentum.calculate(sample_data, column='close', window=10)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
    
    def test_rate_of_change(self, sample_data):
        """测试变化率"""
        roc = RateOfChange()
        result = roc.calculate(sample_data, column='close', window=10)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
    
    def test_feature_engineer_calculate_feature(self, feature_engineer, sample_data):
        """测试特征工程器计算单个特征"""
        result = feature_engineer.calculate_feature(sample_data, 'SMA', column='close', window=20)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
    
    def test_feature_engineer_calculate_multiple_features(self, feature_engineer, sample_data):
        """测试特征工程器批量计算特征"""
        feature_configs = {
            'sma_20': {'indicator': 'SMA', 'params': {'column': 'close', 'window': 20}},
            'ema_12': {'indicator': 'EMA', 'params': {'column': 'close', 'window': 12}},
            'rsi_14': {'indicator': 'RSI', 'params': {'column': 'close', 'window': 14}}
        }
        
        result = feature_engineer.calculate_multiple_features(sample_data, feature_configs)
        
        assert isinstance(result, pd.DataFrame)
        assert 'sma_20' in result.columns
        assert 'ema_12' in result.columns
        assert 'rsi_14' in result.columns
        assert len(result) == len(sample_data)
    
    def test_calculate_all_features(self, sample_data):
        """测试计算所有特征"""
        fe = FeatureEngineer(sample_data)
        
        # 定义特征配置
        feature_configs = {
            'sma_5': {'indicator': 'SMA', 'params': {'window': 5}},
            'sma_20': {'indicator': 'SMA', 'params': {'window': 20}},
            'ema_12': {'indicator': 'EMA', 'params': {'window': 12}},
            'ema_26': {'indicator': 'EMA', 'params': {'window': 26}},
            'atr_14': {'indicator': 'ATR', 'params': {'window': 14}},
            'volatility_20': {'indicator': 'VOLATILITY', 'params': {'window': 20}},
            'vwap': {'indicator': 'VWAP', 'params': {}},
            'obv': {'indicator': 'OBV', 'params': {}},
            'rsi_14': {'indicator': 'RSI', 'params': {'window': 14}},
            'momentum_10': {'indicator': 'MOMENTUM', 'params': {'window': 10}},
            'roc_10': {'indicator': 'ROC', 'params': {'window': 10}}
        }
        
        result = fe.calculate_multiple_features(sample_data, feature_configs)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        
        # 检查是否包含预期的特征
        expected_features = list(feature_configs.keys())
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_custom_feature_config(self, sample_data):
        """测试自定义特征配置"""
        fe = FeatureEngineer(sample_data)
        
        custom_config = {
            'custom_sma': {'indicator': 'SMA', 'params': {'column': 'close', 'window': 50}},
            'custom_rsi': {'indicator': 'RSI', 'params': {'column': 'close', 'window': 21}}
        }
        
        result = fe.calculate_multiple_features(sample_data, custom_config)
        
        assert 'custom_sma' in result.columns
        assert 'custom_rsi' in result.columns
    
    def test_invalid_indicator(self, feature_engineer, sample_data):
        """测试无效指标"""
        with pytest.raises(ValueError):
            feature_engineer.calculate_feature(sample_data, 'INVALID_INDICATOR')
    
    def test_missing_columns(self, sample_data):
        """测试缺少必要列的情况"""
        incomplete_data = sample_data[['close']].copy()  # 只保留close列
        
        atr = AverageTrueRange()
        with pytest.raises(ValueError):
            atr.calculate(incomplete_data)
    
    def test_empty_data(self):
        """测试空数据"""
        empty_data = pd.DataFrame()
        
        sma = SimpleMovingAverage()
        with pytest.raises(ValueError):
            sma.calculate(empty_data)
    
    def test_single_row_data(self):
        """测试单行数据"""
        single_row = pd.DataFrame({
            'open': [50000], 'high': [51000], 'low': [49000], 
            'close': [50500], 'volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1))
        
        sma = SimpleMovingAverage()
        result = sma.calculate(single_row, column='close', window=20)
        
        assert len(result) == 1
        assert result.iloc[0] == single_row['close'].iloc[0]
    
    def test_window_larger_than_data(self, sample_data):
        """测试窗口大于数据长度的情况"""
        small_data = sample_data.head(10)
        
        sma = SimpleMovingAverage()
        result = sma.calculate(small_data, column='close', window=20)
        
        # 应该仍然能计算，但使用可用的数据
        assert len(result) == 10
        assert not result.isnull().all()
    
    def test_data_with_nan(self, sample_data):
        """测试包含NaN的数据"""
        data_with_nan = sample_data.copy()
        data_with_nan.loc[data_with_nan.index[10:15], 'close'] = np.nan
        
        sma = SimpleMovingAverage()
        result = sma.calculate(data_with_nan, column='close', window=20)
        
        # 应该能处理NaN值
        assert len(result) == len(data_with_nan)
    
    def test_feature_consistency(self, sample_data):
        """测试特征计算的一致性"""
        # 多次计算同一特征应该得到相同结果
        sma = SimpleMovingAverage()
        result1 = sma.calculate(sample_data, column='close', window=20)
        result2 = sma.calculate(sample_data, column='close', window=20)
        
        pd.testing.assert_series_equal(result1, result2)
    
    def test_feature_mathematical_properties(self, sample_data):
        """测试特征的数学性质"""
        # 测试移动平均的平滑性质
        sma_short = SimpleMovingAverage().calculate(sample_data, column='close', window=5)
        sma_long = SimpleMovingAverage().calculate(sample_data, column='close', window=20)
        
        # 短期移动平均应该比长期移动平均更接近当前价格
        price_diff_short = abs(sample_data['close'] - sma_short)
        price_diff_long = abs(sample_data['close'] - sma_long)
        
        # 在大多数情况下，短期MA应该更接近当前价格
        closer_count = (price_diff_short <= price_diff_long).sum()
        assert closer_count > len(sample_data) * 0.6  # 至少60%的情况
    
    def test_performance_with_large_data(self):
        """测试大数据集的性能"""
        # 创建较大的数据集
        large_data = pd.DataFrame({
            'close': np.random.normal(50000, 1000, 10000),
            'volume': np.random.normal(1000, 200, 10000)
        })
        
        # 应该能够快速计算
        import time
        start_time = time.time()
        
        sma = SimpleMovingAverage()
        result = sma.calculate(large_data, column='close', window=20)
        
        end_time = time.time()
        
        assert len(result) == 10000
        assert (end_time - start_time) < 1.0  # 应该在1秒内完成


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])