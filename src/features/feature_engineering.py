"""
特征工程模块 - 技术指标计算
实现各种技术分析指标，包括移动平均、波动率、成交量等衍生特征
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional, Tuple
import warnings
from abc import ABC, abstractmethod


class TechnicalIndicator(ABC):
    """技术指标基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算技术指标"""
        pass
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """验证数据格式"""
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        if data.empty:
            raise ValueError("数据不能为空")


class FeatureEngineer:
    """特征工程主类"""
    
    def __init__(self, data: pd.DataFrame = None):
        self.indicators = {}
        self.feature_cache = {}
        self.data = data
        
        # 自动注册所有指标
        self._register_default_indicators()
        
    def register_indicator(self, indicator: TechnicalIndicator) -> None:
        """注册技术指标"""
        self.indicators[indicator.name] = indicator
    
    def calculate_feature(self, data: pd.DataFrame, indicator_name: str, **kwargs) -> pd.Series:
        """计算单个特征"""
        if indicator_name not in self.indicators:
            raise ValueError(f"未知的指标: {indicator_name}")
        
        indicator = self.indicators[indicator_name]
        return indicator.calculate(data, **kwargs)
    
    def calculate_multiple_features(self, data: pd.DataFrame, 
                                  feature_configs: Dict[str, Dict]) -> pd.DataFrame:
        """批量计算多个特征"""
        result_data = data.copy()
        
        for feature_name, config in feature_configs.items():
            indicator_name = config.get('indicator')
            params = config.get('params', {})
            
            if indicator_name in self.indicators:
                try:
                    feature_values = self.calculate_feature(data, indicator_name, **params)
                    result_data[feature_name] = feature_values
                except Exception as e:
                    warnings.warn(f"计算特征 {feature_name} 失败: {str(e)}")
                    result_data[feature_name] = np.nan
        
        return result_data
    
    def get_available_indicators(self) -> List[str]:
        """获取可用的指标列表"""
        return list(self.indicators.keys())
    
    def _register_default_indicators(self):
        """注册默认的技术指标"""
        # 移动平均指标
        self.register_indicator(SimpleMovingAverage())
        self.register_indicator(ExponentialMovingAverage())
        self.register_indicator(WeightedMovingAverage())
        self.register_indicator(TripleExponentialMovingAverage())
        
        # 波动率指标
        self.register_indicator(AverageTrueRange())
        self.register_indicator(BollingerBands())
        self.register_indicator(RollingVolatility())
        
        # 成交量指标
        self.register_indicator(VolumeWeightedAveragePrice())
        self.register_indicator(OnBalanceVolume())
        self.register_indicator(VolumeRatio())
        self.register_indicator(AccumulationDistributionLine())
        
        # 价格相关指标
        self.register_indicator(RelativeStrengthIndex())
        self.register_indicator(MACD())
        self.register_indicator(Momentum())
        self.register_indicator(RateOfChange())
    
    def calculate_sma(self, window: int = 20, column: str = 'close') -> pd.DataFrame:
        """计算简单移动平均"""
        if self.data is None:
            raise ValueError("请先设置数据")
        result = self.data.copy()
        sma_values = self.calculate_feature(self.data, "SMA", column=column, window=window)
        result[f'SMA_{window}'] = sma_values
        return result
    
    def calculate_ema(self, window: int = 20, column: str = 'close') -> pd.DataFrame:
        """计算指数移动平均"""
        if self.data is None:
            raise ValueError("请先设置数据")
        result = self.data.copy()
        ema_values = self.calculate_feature(self.data, "EMA", column=column, window=window)
        result[f'EMA_{window}'] = ema_values
        return result
    
    def calculate_rsi(self, window: int = 14, column: str = 'close') -> pd.DataFrame:
        """计算相对强弱指数"""
        if self.data is None:
            raise ValueError("请先设置数据")
        result = self.data.copy()
        rsi_values = self.calculate_feature(self.data, "RSI", column=column, window=window)
        result[f'RSI_{window}'] = rsi_values
        return result


# ==================== 移动平均指标 ====================

class SimpleMovingAverage(TechnicalIndicator):
    """简单移动平均 (SMA)"""
    
    def __init__(self):
        super().__init__("SMA")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
        """计算简单移动平均"""
        self.validate_data(data, [column])
        return data[column].rolling(window=window, min_periods=1).mean()


class ExponentialMovingAverage(TechnicalIndicator):
    """指数移动平均 (EMA)"""
    
    def __init__(self):
        super().__init__("EMA")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
        """计算指数移动平均"""
        self.validate_data(data, [column])
        return data[column].ewm(span=window, adjust=False).mean()


class WeightedMovingAverage(TechnicalIndicator):
    """加权移动平均 (WMA)"""
    
    def __init__(self):
        super().__init__("WMA")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
        """计算加权移动平均"""
        self.validate_data(data, [column])
        
        def wma(series):
            weights = np.arange(1, len(series) + 1)
            return np.average(series, weights=weights)
        
        return data[column].rolling(window=window, min_periods=1).apply(wma, raw=True)


class TripleExponentialMovingAverage(TechnicalIndicator):
    """三重指数移动平均 (TEMA)"""
    
    def __init__(self):
        super().__init__("TEMA")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
        """计算三重指数移动平均"""
        self.validate_data(data, [column])
        
        ema1 = data[column].ewm(span=window).mean()
        ema2 = ema1.ewm(span=window).mean()
        ema3 = ema2.ewm(span=window).mean()
        
        tema = 3 * ema1 - 3 * ema2 + ema3
        return tema


# ==================== 波动率指标 ====================

class AverageTrueRange(TechnicalIndicator):
    """平均真实波幅 (ATR)"""
    
    def __init__(self):
        super().__init__("ATR")
    
    def calculate(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算平均真实波幅"""
        self.validate_data(data, ['high', 'low', 'close'])
        
        # 计算真实波幅
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # 计算ATR
        atr = true_range.rolling(window=window, min_periods=1).mean()
        return atr


class RollingVolatility(TechnicalIndicator):
    """滚动波动率"""
    
    def __init__(self):
        super().__init__("VOLATILITY")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close', 
                 window: int = 20, annualize: bool = True) -> pd.Series:
        """计算滚动波动率"""
        self.validate_data(data, [column])
        
        # 计算收益率
        returns = data[column].pct_change()
        
        # 计算滚动标准差
        volatility = returns.rolling(window=window, min_periods=1).std()
        
        # 年化波动率（假设一年252个交易日）
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        return volatility


class BollingerBands(TechnicalIndicator):
    """布林带"""
    
    def __init__(self):
        super().__init__("BOLLINGER")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close', 
                 window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        """计算布林带"""
        self.validate_data(data, [column])
        
        # 中轨（移动平均）
        middle = data[column].rolling(window=window, min_periods=1).mean()
        
        # 标准差
        std = data[column].rolling(window=window, min_periods=1).std()
        
        # 上轨和下轨
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return {
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower,
            'bb_width': (upper - lower) / middle,  # 布林带宽度
            'bb_position': (data[column] - lower) / (upper - lower)  # 价格在布林带中的位置
        }


# ==================== 成交量指标 ====================

class VolumeWeightedAveragePrice(TechnicalIndicator):
    """成交量加权平均价格 (VWAP)"""
    
    def __init__(self):
        super().__init__("VWAP")
    
    def calculate(self, data: pd.DataFrame, window: Optional[int] = None) -> pd.Series:
        """计算VWAP"""
        self.validate_data(data, ['high', 'low', 'close', 'volume'])
        
        # 典型价格
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # 价格*成交量
        pv = typical_price * data['volume']
        
        if window is None:
            # 累积VWAP
            vwap = pv.cumsum() / data['volume'].cumsum()
        else:
            # 滚动VWAP
            pv_sum = pv.rolling(window=window, min_periods=1).sum()
            volume_sum = data['volume'].rolling(window=window, min_periods=1).sum()
            vwap = pv_sum / volume_sum
        
        return vwap


class OnBalanceVolume(TechnicalIndicator):
    """能量潮 (OBV)"""
    
    def __init__(self):
        super().__init__("OBV")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算OBV"""
        self.validate_data(data, ['close', 'volume'])
        
        # 价格变化方向
        price_change = data['close'].diff()
        
        # OBV变化
        obv_change = np.where(price_change > 0, data['volume'],
                             np.where(price_change < 0, -data['volume'], 0))
        
        # 累积OBV
        obv = pd.Series(obv_change.cumsum(), index=data.index)
        return obv


class VolumeRatio(TechnicalIndicator):
    """成交量比率"""
    
    def __init__(self):
        super().__init__("VOLUME_RATIO")
    
    def calculate(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算成交量比率"""
        self.validate_data(data, ['volume'])
        
        # 当前成交量 / 平均成交量
        avg_volume = data['volume'].rolling(window=window, min_periods=1).mean()
        volume_ratio = data['volume'] / avg_volume
        
        return volume_ratio


class AccumulationDistributionLine(TechnicalIndicator):
    """累积/派发线 (A/D Line)"""
    
    def __init__(self):
        super().__init__("ADL")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算A/D线"""
        self.validate_data(data, ['high', 'low', 'close', 'volume'])
        
        # 资金流量乘数
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        
        # 处理分母为0的情况
        clv = clv.fillna(0)
        
        # 资金流量
        money_flow = clv * data['volume']
        
        # 累积A/D线
        adl = money_flow.cumsum()
        return adl


# ==================== 价格相关指标 ====================

class RelativeStrengthIndex(TechnicalIndicator):
    """相对强弱指数 (RSI)"""
    
    def __init__(self):
        super().__init__("RSI")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close', window: int = 14) -> pd.Series:
        """计算RSI"""
        self.validate_data(data, [column])
        
        # 价格变化
        delta = data[column].diff()
        
        # 上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 平均收益和损失
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        # RSI计算
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class MACD(TechnicalIndicator):
    """移动平均收敛散度 (MACD)"""
    
    def __init__(self):
        super().__init__("MACD")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close',
                 fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9) -> Dict[str, pd.Series]:
        """计算MACD"""
        self.validate_data(data, [column])
        
        # 快速和慢速EMA
        ema_fast = data[column].ewm(span=fast_period).mean()
        ema_slow = data[column].ewm(span=slow_period).mean()
        
        # MACD线
        macd_line = ema_fast - ema_slow
        
        # 信号线
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # 柱状图
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }


class Momentum(TechnicalIndicator):
    """动量指标"""
    
    def __init__(self):
        super().__init__("MOMENTUM")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close', window: int = 10) -> pd.Series:
        """计算动量"""
        self.validate_data(data, [column])
        
        # 当前价格 - N期前价格
        momentum = data[column] - data[column].shift(window)
        return momentum


class RateOfChange(TechnicalIndicator):
    """变化率 (ROC)"""
    
    def __init__(self):
        super().__init__("ROC")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close', window: int = 10) -> pd.Series:
        """计算变化率"""
        self.validate_data(data, [column])
        
        # (当前价格 - N期前价格) / N期前价格 * 100
        roc = ((data[column] - data[column].shift(window)) / data[column].shift(window)) * 100
        return roc


# ==================== 工厂函数 ====================

def create_feature_engineer() -> FeatureEngineer:
    """创建配置好的特征工程器"""
    engineer = FeatureEngineer()
    
    # 注册所有指标
    indicators = [
        SimpleMovingAverage(),
        ExponentialMovingAverage(),
        WeightedMovingAverage(),
        TripleExponentialMovingAverage(),
        AverageTrueRange(),
        RollingVolatility(),
        BollingerBands(),
        VolumeWeightedAveragePrice(),
        OnBalanceVolume(),
        VolumeRatio(),
        AccumulationDistributionLine(),
        RelativeStrengthIndex(),
        MACD(),
        Momentum(),
        RateOfChange()
    ]
    
    for indicator in indicators:
        engineer.register_indicator(indicator)
    
    return engineer


# ==================== 便捷函数 ====================

def calculate_all_features(data: pd.DataFrame, 
                          feature_config: Optional[Dict] = None) -> pd.DataFrame:
    """计算所有常用技术指标"""
    
    if feature_config is None:
        # 默认特征配置
        feature_config = {
            # 移动平均
            'sma_5': {'indicator': 'SMA', 'params': {'window': 5}},
            'sma_20': {'indicator': 'SMA', 'params': {'window': 20}},
            'sma_50': {'indicator': 'SMA', 'params': {'window': 50}},
            'ema_12': {'indicator': 'EMA', 'params': {'window': 12}},
            'ema_26': {'indicator': 'EMA', 'params': {'window': 26}},
            
            # 波动率
            'atr_14': {'indicator': 'ATR', 'params': {'window': 14}},
            'volatility_20': {'indicator': 'VOLATILITY', 'params': {'window': 20}},
            
            # 成交量
            'vwap': {'indicator': 'VWAP', 'params': {}},
            'obv': {'indicator': 'OBV', 'params': {}},
            'volume_ratio_20': {'indicator': 'VOLUME_RATIO', 'params': {'window': 20}},
            
            # 价格指标
            'rsi_14': {'indicator': 'RSI', 'params': {'window': 14}},
            'momentum_10': {'indicator': 'MOMENTUM', 'params': {'window': 10}},
            'roc_10': {'indicator': 'ROC', 'params': {'window': 10}}
        }
    
    engineer = create_feature_engineer()
    result = engineer.calculate_multiple_features(data, feature_config)
    
    # 处理布林带和MACD等返回字典的指标
    bollinger = engineer.indicators['BOLLINGER'].calculate(data)
    for key, values in bollinger.items():
        result[key] = values
    
    macd = engineer.indicators['MACD'].calculate(data)
    for key, values in macd.items():
        result[key] = values
    
    return result


if __name__ == "__main__":
    # 示例用法
    print("特征工程模块已加载")
    print("可用指标:", create_feature_engineer().get_available_indicators())