"""
特征标准化模块
实现按品种分组的滚动z-score和分位数化标准化，避免跨品种尺度问题
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional, Tuple, Callable
import warnings
from abc import ABC, abstractmethod


class Normalizer(ABC):
    """标准化器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.fitted_params = {}
    
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame, 
                     feature_columns: List[str],
                     group_column: str = 'symbol',
                     **kwargs) -> pd.DataFrame:
        """拟合并转换数据"""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame,
                 feature_columns: List[str],
                 group_column: str = 'symbol',
                 **kwargs) -> pd.DataFrame:
        """使用已拟合的参数转换数据"""
        pass
    
    def validate_data(self, data: pd.DataFrame, 
                     feature_columns: List[str],
                     group_column: str) -> None:
        """验证数据格式"""
        if data.empty:
            raise ValueError("数据不能为空")
        
        missing_columns = [col for col in feature_columns + [group_column] 
                          if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")


class RollingZScoreNormalizer(Normalizer):
    """滚动Z-Score标准化器
    
    按品种分组计算滚动均值和标准差，进行z-score标准化
    公式: (x - rolling_mean) / rolling_std
    """
    
    def __init__(self, window: int = 252, min_periods: Optional[int] = None):
        super().__init__("rolling_zscore")
        self.window = window
        self.min_periods = min_periods or max(1, window // 4)
    
    def fit_transform(self, data: pd.DataFrame,
                     feature_columns: List[str],
                     group_column: str = 'symbol',
                     **kwargs) -> pd.DataFrame:
        """拟合并转换数据"""
        self.validate_data(data, feature_columns, group_column)
        
        result_data = data.copy()
        
        # 按品种分组处理
        for symbol in data[group_column].unique():
            symbol_mask = data[group_column] == symbol
            symbol_data = data[symbol_mask].copy().sort_index()
            
            if len(symbol_data) < self.min_periods:
                warnings.warn(f"品种 {symbol} 的数据量({len(symbol_data)})小于最小周期({self.min_periods})")
                continue
            
            # 对每个特征列进行滚动z-score标准化
            for col in feature_columns:
                if col in symbol_data.columns:
                    # 计算滚动均值和标准差
                    rolling_mean = symbol_data[col].rolling(
                        window=self.window, 
                        min_periods=self.min_periods
                    ).mean()
                    
                    rolling_std = symbol_data[col].rolling(
                        window=self.window, 
                        min_periods=self.min_periods
                    ).std()
                    
                    # Z-score标准化
                    normalized_values = (symbol_data[col] - rolling_mean) / rolling_std
                    
                    # 处理除零和无穷大情况
                    normalized_values = normalized_values.fillna(0)
                    normalized_values = normalized_values.replace([np.inf, -np.inf], 0)
                    
                    # 对于标准差为0的情况，设置为0
                    zero_std_mask = rolling_std == 0
                    normalized_values[zero_std_mask] = 0
                    
                    # 更新结果
                    result_data.loc[symbol_mask, col] = normalized_values
        
        return result_data
    
    def transform(self, data: pd.DataFrame,
                 feature_columns: List[str],
                 group_column: str = 'symbol',
                 **kwargs) -> pd.DataFrame:
        """使用已拟合的参数转换数据（对于滚动标准化，等同于fit_transform）"""
        return self.fit_transform(data, feature_columns, group_column, **kwargs)


class RollingQuantileNormalizer(Normalizer):
    """滚动分位数标准化器
    
    按品种分组计算滚动分位数，将数据转换为0-1之间的分位数值
    """
    
    def __init__(self, window: int = 252, min_periods: Optional[int] = None,
                 quantile_range: Tuple[float, float] = (0.01, 0.99)):
        super().__init__("rolling_quantile")
        self.window = window
        self.min_periods = min_periods or max(1, window // 4)
        self.quantile_range = quantile_range
    
    def fit_transform(self, data: pd.DataFrame,
                     feature_columns: List[str],
                     group_column: str = 'symbol',
                     **kwargs) -> pd.DataFrame:
        """拟合并转换数据"""
        self.validate_data(data, feature_columns, group_column)
        
        result_data = data.copy()
        
        # 按品种分组处理
        for symbol in data[group_column].unique():
            symbol_mask = data[group_column] == symbol
            symbol_data = data[symbol_mask].copy()
            
            if len(symbol_data) < self.min_periods:
                warnings.warn(f"品种 {symbol} 的数据量({len(symbol_data)})小于最小周期({self.min_periods})")
                continue
            
            # 对每个特征列进行滚动分位数标准化
            for col in feature_columns:
                if col in symbol_data.columns:
                    normalized_values = self._rolling_quantile_transform(
                        symbol_data[col], self.window, self.min_periods
                    )
                    
                    # 更新结果
                    result_data.loc[symbol_mask, col] = normalized_values
        
        return result_data
    
    def _rolling_quantile_transform(self, series: pd.Series, 
                                   window: int, min_periods: int) -> pd.Series:
        """计算滚动分位数标准化"""
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            # 获取滚动窗口数据
            start_idx = max(0, i - window + 1)
            window_data = series.iloc[start_idx:i+1]
            
            if len(window_data) < min_periods:
                result.iloc[i] = np.nan
                continue
            
            # 计算当前值在窗口中的分位数
            current_value = series.iloc[i]
            if pd.isna(current_value):
                result.iloc[i] = np.nan
                continue
            
            # 计算分位数
            rank = (window_data < current_value).sum()
            quantile = rank / len(window_data)
            
            # 应用分位数范围限制
            min_q, max_q = self.quantile_range
            quantile = np.clip(quantile, min_q, max_q)
            
            # 重新缩放到0-1范围
            quantile = (quantile - min_q) / (max_q - min_q)
            
            result.iloc[i] = quantile
        
        return result.fillna(0.5)  # 缺失值用中位数填充
    
    def transform(self, data: pd.DataFrame,
                 feature_columns: List[str],
                 group_column: str = 'symbol',
                 **kwargs) -> pd.DataFrame:
        """使用已拟合的参数转换数据（对于滚动标准化，等同于fit_transform）"""
        return self.fit_transform(data, feature_columns, group_column, **kwargs)


class RobustZScoreNormalizer(Normalizer):
    """鲁棒Z-Score标准化器
    
    使用中位数和MAD（中位数绝对偏差）进行标准化，对异常值更鲁棒
    公式: (x - rolling_median) / (1.4826 * rolling_mad)
    """
    
    def __init__(self, window: int = 252, min_periods: Optional[int] = None):
        super().__init__("robust_zscore")
        self.window = window
        self.min_periods = min_periods or max(1, window // 4)
        self.mad_constant = 1.4826  # 使MAD等价于正态分布的标准差
    
    def fit_transform(self, data: pd.DataFrame,
                     feature_columns: List[str],
                     group_column: str = 'symbol',
                     **kwargs) -> pd.DataFrame:
        """拟合并转换数据"""
        self.validate_data(data, feature_columns, group_column)
        
        result_data = data.copy()
        
        # 按品种分组处理
        for symbol in data[group_column].unique():
            symbol_mask = data[group_column] == symbol
            symbol_data = data[symbol_mask].copy()
            
            if len(symbol_data) < self.min_periods:
                warnings.warn(f"品种 {symbol} 的数据量({len(symbol_data)})小于最小周期({self.min_periods})")
                continue
            
            # 对每个特征列进行鲁棒z-score标准化
            for col in feature_columns:
                if col in symbol_data.columns:
                    # 计算滚动中位数
                    rolling_median = symbol_data[col].rolling(
                        window=self.window, 
                        min_periods=self.min_periods
                    ).median()
                    
                    # 计算滚动MAD
                    rolling_mad = symbol_data[col].rolling(
                        window=self.window, 
                        min_periods=self.min_periods
                    ).apply(lambda x: np.median(np.abs(x - np.median(x))))
                    
                    # 鲁棒Z-score标准化
                    normalized_values = (symbol_data[col] - rolling_median) / (
                        self.mad_constant * rolling_mad
                    )
                    
                    # 处理除零情况
                    normalized_values = normalized_values.fillna(0)
                    normalized_values = normalized_values.replace([np.inf, -np.inf], 0)
                    
                    # 更新结果
                    result_data.loc[symbol_mask, col] = normalized_values
        
        return result_data
    
    def transform(self, data: pd.DataFrame,
                 feature_columns: List[str],
                 group_column: str = 'symbol',
                 **kwargs) -> pd.DataFrame:
        """使用已拟合的参数转换数据（对于滚动标准化，等同于fit_transform）"""
        return self.fit_transform(data, feature_columns, group_column, **kwargs)


class FeatureNormalizer:
    """特征标准化管理器"""
    
    def __init__(self):
        self.normalizers = {
            'rolling_zscore': RollingZScoreNormalizer,
            'rolling_quantile': RollingQuantileNormalizer,
            'robust_zscore': RobustZScoreNormalizer
        }
        self.fitted_normalizers = {}
    
    def normalize_features(self, data: pd.DataFrame,
                          feature_columns: List[str],
                          method: str = 'rolling_zscore',
                          group_column: str = 'symbol',
                          **kwargs) -> pd.DataFrame:
        """标准化特征
        
        Args:
            data: 输入数据
            feature_columns: 需要标准化的特征列
            method: 标准化方法 ('rolling_zscore', 'rolling_quantile', 'robust_zscore')
            group_column: 分组列（品种列）
            **kwargs: 标准化器的额外参数
        
        Returns:
            标准化后的数据
        """
        if method not in self.normalizers:
            raise ValueError(f"不支持的标准化方法: {method}. 可用方法: {list(self.normalizers.keys())}")
        
        # 创建标准化器
        normalizer_class = self.normalizers[method]
        normalizer = normalizer_class(**kwargs)
        
        # 执行标准化
        normalized_data = normalizer.fit_transform(
            data, feature_columns, group_column
        )
        
        # 保存拟合的标准化器
        self.fitted_normalizers[method] = normalizer
        
        return normalized_data
    
    def get_normalization_stats(self, data: pd.DataFrame,
                               feature_columns: List[str],
                               group_column: str = 'symbol') -> Dict:
        """获取标准化统计信息"""
        stats = {}
        
        for symbol in data[group_column].unique():
            symbol_data = data[data[group_column] == symbol]
            symbol_stats = {}
            
            for col in feature_columns:
                if col in symbol_data.columns:
                    col_data = symbol_data[col].dropna()
                    if len(col_data) > 0:
                        symbol_stats[col] = {
                            'mean': col_data.mean(),
                            'std': col_data.std(),
                            'median': col_data.median(),
                            'mad': np.median(np.abs(col_data - col_data.median())),
                            'min': col_data.min(),
                            'max': col_data.max(),
                            'count': len(col_data)
                        }
            
            stats[symbol] = symbol_stats
        
        return stats
    
    def batch_normalize(self, data: pd.DataFrame,
                       feature_configs: Dict[str, Dict],
                       group_column: str = 'symbol') -> pd.DataFrame:
        """批量标准化多个特征
        
        Args:
            data: 输入数据
            feature_configs: 特征配置字典
                格式: {
                    'feature_group_1': {
                        'columns': ['col1', 'col2'],
                        'method': 'rolling_zscore',
                        'params': {'window': 252}
                    }
                }
            group_column: 分组列
        
        Returns:
            标准化后的数据
        """
        result_data = data.copy()
        
        for group_name, config in feature_configs.items():
            columns = config.get('columns', [])
            method = config.get('method', 'rolling_zscore')
            params = config.get('params', {})
            
            if not columns:
                continue
            
            try:
                # 标准化当前特征组
                normalized_data = self.normalize_features(
                    result_data, columns, method, group_column, **params
                )
                
                # 更新结果数据
                for col in columns:
                    if col in normalized_data.columns:
                        result_data[col] = normalized_data[col]
                        
            except Exception as e:
                warnings.warn(f"标准化特征组 {group_name} 失败: {str(e)}")
        
        return result_data


def create_feature_normalizer() -> FeatureNormalizer:
    """创建特征标准化器实例"""
    return FeatureNormalizer()


# 便捷函数
def rolling_zscore_normalize(data: pd.DataFrame,
                           feature_columns: List[str],
                           group_column: str = 'symbol',
                           window: int = 252,
                           min_periods: Optional[int] = None) -> pd.DataFrame:
    """滚动Z-Score标准化便捷函数"""
    normalizer = FeatureNormalizer()
    return normalizer.normalize_features(
        data, feature_columns, 'rolling_zscore', group_column,
        window=window, min_periods=min_periods
    )


def rolling_quantile_normalize(data: pd.DataFrame,
                             feature_columns: List[str],
                             group_column: str = 'symbol',
                             window: int = 252,
                             min_periods: Optional[int] = None,
                             quantile_range: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """滚动分位数标准化便捷函数"""
    normalizer = FeatureNormalizer()
    return normalizer.normalize_features(
        data, feature_columns, 'rolling_quantile', group_column,
        window=window, min_periods=min_periods, quantile_range=quantile_range
    )


def robust_zscore_normalize(data: pd.DataFrame,
                          feature_columns: List[str],
                          group_column: str = 'symbol',
                          window: int = 252,
                          min_periods: Optional[int] = None) -> pd.DataFrame:
    """鲁棒Z-Score标准化便捷函数"""
    normalizer = FeatureNormalizer()
    return normalizer.normalize_features(
        data, feature_columns, 'robust_zscore', group_column,
        window=window, min_periods=min_periods
    )


if __name__ == "__main__":
    # 示例用法
    print("特征标准化模块已加载")
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    data_list = []
    for symbol in symbols:
        for date in dates:
            data_list.append({
                'date': date,
                'symbol': symbol,
                'close': 100 + np.random.randn() * 10,
                'volume': 1000000 + np.random.randn() * 100000,
                'feature1': np.random.randn(),
                'feature2': np.random.randn() * 5
            })
    
    sample_data = pd.DataFrame(data_list)
    
    # 测试标准化
    normalizer = create_feature_normalizer()
    feature_cols = ['close', 'volume', 'feature1', 'feature2']
    
    print(f"\n原始数据统计:")
    print(sample_data[feature_cols].describe())
    
    # 滚动Z-Score标准化
    normalized_data = normalizer.normalize_features(
        sample_data, feature_cols, 'rolling_zscore', window=60
    )
    
    print(f"\n滚动Z-Score标准化后统计:")
    print(normalized_data[feature_cols].describe())
    
    print("\n标准化模块测试完成")