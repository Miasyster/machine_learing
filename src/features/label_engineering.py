"""
标签构建模块

实现量化交易中常用的标签构建功能：
- 未来收益率标签
- 方向标签（+1/0/-1）
- 超额收益标签（相对基准）
- 标签平滑和噪声处理

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
import warnings
from abc import ABC, abstractmethod


class LabelBuilder(ABC):
    """标签构建器基类"""
    
    def __init__(self):
        self.is_fitted_ = False
        self.label_stats_ = {}
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'LabelBuilder':
        """拟合标签构建器"""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """构建标签"""
        pass
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """拟合并构建标签"""
        return self.fit(data, **kwargs).transform(data, **kwargs)


class ReturnLabelBuilder(LabelBuilder):
    """收益率标签构建器"""
    
    def __init__(self, 
                 periods: int = 1,
                 price_column: str = 'close',
                 method: str = 'simple',
                 min_periods: Optional[int] = None):
        """
        初始化收益率标签构建器
        
        Args:
            periods: 未来N期的收益率
            price_column: 价格列名
            method: 计算方法 ('simple', 'log')
            min_periods: 最小期数要求
        """
        super().__init__()
        self.periods = periods
        self.price_column = price_column
        self.method = method
        self.min_periods = min_periods or periods
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'ReturnLabelBuilder':
        """
        拟合收益率标签构建器
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            self
        """
        if self.price_column not in data.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        # 计算统计信息
        prices = data[self.price_column]
        self.label_stats_ = {
            'price_mean': prices.mean(),
            'price_std': prices.std(),
            'price_min': prices.min(),
            'price_max': prices.max(),
            'data_length': len(data)
        }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        构建收益率标签
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            收益率标签Series
        """
        if not self.is_fitted_:
            raise ValueError("LabelBuilder must be fitted first")
        
        if self.price_column not in data.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        prices = data[self.price_column]
        
        # 计算未来价格
        future_prices = prices.shift(-self.periods)
        
        # 计算收益率
        if self.method == 'simple':
            returns = (future_prices - prices) / prices
        elif self.method == 'log':
            returns = np.log(future_prices / prices)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # 处理边界情况
        if len(returns) < self.min_periods:
            warnings.warn(f"Data length ({len(returns)}) is less than min_periods ({self.min_periods})")
        
        return returns


class DirectionLabelBuilder(LabelBuilder):
    """方向标签构建器（+1/0/-1）"""
    
    def __init__(self,
                 periods: int = 1,
                 price_column: str = 'close',
                 threshold: float = 0.0,
                 method: str = 'simple',
                 neutral_zone: Optional[float] = None):
        """
        初始化方向标签构建器
        
        Args:
            periods: 未来N期
            price_column: 价格列名
            threshold: 方向判断阈值
            method: 计算方法 ('simple', 'log')
            neutral_zone: 中性区间半径，如果设置则在[-neutral_zone, +neutral_zone]内为0
        """
        super().__init__()
        self.periods = periods
        self.price_column = price_column
        self.threshold = threshold
        self.method = method
        self.neutral_zone = neutral_zone
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'DirectionLabelBuilder':
        """拟合方向标签构建器"""
        if self.price_column not in data.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        # 计算收益率分布统计
        prices = data[self.price_column]
        future_prices = prices.shift(-self.periods)
        
        if self.method == 'simple':
            returns = (future_prices - prices) / prices
        else:  # log
            returns = np.log(future_prices / prices)
        
        returns_clean = returns.dropna()
        
        self.label_stats_ = {
            'return_mean': returns_clean.mean(),
            'return_std': returns_clean.std(),
            'return_skew': returns_clean.skew(),
            'return_kurt': returns_clean.kurtosis(),
            'positive_ratio': (returns_clean > self.threshold).mean(),
            'negative_ratio': (returns_clean < -self.threshold).mean(),
            'neutral_ratio': None
        }
        
        if self.neutral_zone is not None:
            neutral_mask = (returns_clean.abs() <= self.neutral_zone)
            self.label_stats_['neutral_ratio'] = neutral_mask.mean()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """构建方向标签"""
        if not self.is_fitted_:
            raise ValueError("LabelBuilder must be fitted first")
        
        # 先计算收益率
        return_builder = ReturnLabelBuilder(
            periods=self.periods,
            price_column=self.price_column,
            method=self.method
        )
        returns = return_builder.fit_transform(data)
        
        # 构建方向标签
        if self.neutral_zone is not None:
            # 三分类：+1, 0, -1
            labels = pd.Series(0, index=returns.index)
            labels[returns > self.neutral_zone] = 1
            labels[returns < -self.neutral_zone] = -1
        else:
            # 二分类转三分类
            labels = pd.Series(0, index=returns.index)
            labels[returns > self.threshold] = 1
            labels[returns < self.threshold] = -1
        
        return labels


class ExcessReturnLabelBuilder(LabelBuilder):
    """超额收益标签构建器"""
    
    def __init__(self,
                 periods: int = 1,
                 price_column: str = 'close',
                 benchmark_column: Optional[str] = None,
                 benchmark_data: Optional[pd.DataFrame] = None,
                 method: str = 'simple'):
        """
        初始化超额收益标签构建器
        
        Args:
            periods: 未来N期
            price_column: 价格列名
            benchmark_column: 基准列名（如果基准在同一DataFrame中）
            benchmark_data: 基准数据DataFrame（如果基准在单独的DataFrame中）
            method: 计算方法 ('simple', 'log')
        """
        super().__init__()
        self.periods = periods
        self.price_column = price_column
        self.benchmark_column = benchmark_column
        self.benchmark_data = benchmark_data
        self.method = method
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'ExcessReturnLabelBuilder':
        """拟合超额收益标签构建器"""
        if self.price_column not in data.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        # 验证基准数据
        if self.benchmark_column is not None:
            if self.benchmark_column not in data.columns:
                raise ValueError(f"Benchmark column '{self.benchmark_column}' not found in data")
        elif self.benchmark_data is not None:
            if self.price_column not in self.benchmark_data.columns:
                raise ValueError(f"Price column '{self.price_column}' not found in benchmark_data")
        else:
            raise ValueError("Either benchmark_column or benchmark_data must be provided")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """构建超额收益标签"""
        if not self.is_fitted_:
            raise ValueError("LabelBuilder must be fitted first")
        
        # 计算资产收益率
        asset_return_builder = ReturnLabelBuilder(
            periods=self.periods,
            price_column=self.price_column,
            method=self.method
        )
        asset_returns = asset_return_builder.fit_transform(data)
        
        # 计算基准收益率
        if self.benchmark_column is not None:
            # 基准在同一DataFrame中
            benchmark_return_builder = ReturnLabelBuilder(
                periods=self.periods,
                price_column=self.benchmark_column,
                method=self.method
            )
            benchmark_returns = benchmark_return_builder.fit_transform(data)
        else:
            # 基准在单独的DataFrame中
            benchmark_return_builder = ReturnLabelBuilder(
                periods=self.periods,
                price_column=self.price_column,
                method=self.method
            )
            benchmark_returns = benchmark_return_builder.fit_transform(self.benchmark_data)
            
            # 对齐时间索引
            benchmark_returns = benchmark_returns.reindex(data.index)
        
        # 计算超额收益
        excess_returns = asset_returns - benchmark_returns
        
        return excess_returns


class LabelSmoother:
    """标签平滑器"""
    
    @staticmethod
    def gaussian_smooth(labels: pd.Series, 
                       window: int = 5, 
                       std: float = 1.0) -> pd.Series:
        """
        高斯平滑
        
        Args:
            labels: 原始标签
            window: 窗口大小
            std: 高斯标准差
            
        Returns:
            平滑后的标签
        """
        from scipy import ndimage
        
        # 处理缺失值
        labels_filled = labels.ffill().bfill()
        
        # 高斯平滑
        smoothed = ndimage.gaussian_filter1d(labels_filled.values, sigma=std)
        
        return pd.Series(smoothed, index=labels.index)
    
    @staticmethod
    def rolling_mean_smooth(labels: pd.Series, 
                           window: int = 5,
                           center: bool = True) -> pd.Series:
        """
        滚动均值平滑
        
        Args:
            labels: 原始标签
            window: 窗口大小
            center: 是否居中
            
        Returns:
            平滑后的标签
        """
        return labels.rolling(window=window, center=center).mean()
    
    @staticmethod
    def exponential_smooth(labels: pd.Series, 
                          alpha: float = 0.3) -> pd.Series:
        """
        指数平滑
        
        Args:
            labels: 原始标签
            alpha: 平滑参数
            
        Returns:
            平滑后的标签
        """
        return labels.ewm(alpha=alpha).mean()


class NoiseReducer:
    """噪声处理器"""
    
    @staticmethod
    def outlier_clip(labels: pd.Series, 
                    lower_quantile: float = 0.01,
                    upper_quantile: float = 0.99) -> pd.Series:
        """
        异常值裁剪
        
        Args:
            labels: 原始标签
            lower_quantile: 下分位数
            upper_quantile: 上分位数
            
        Returns:
            处理后的标签
        """
        lower_bound = labels.quantile(lower_quantile)
        upper_bound = labels.quantile(upper_quantile)
        
        return labels.clip(lower=lower_bound, upper=upper_bound)
    
    @staticmethod
    def winsorize(labels: pd.Series,
                 limits: Tuple[float, float] = (0.01, 0.01)) -> pd.Series:
        """
        Winsorize处理
        
        Args:
            labels: 原始标签
            limits: (下限比例, 上限比例)
            
        Returns:
            处理后的标签
        """
        from scipy.stats import mstats
        
        winsorized = mstats.winsorize(labels.dropna().values, limits=limits)
        result = labels.copy()
        result.loc[labels.notna()] = winsorized
        
        return result
    
    @staticmethod
    def z_score_filter(labels: pd.Series, 
                      threshold: float = 3.0) -> pd.Series:
        """
        Z-score异常值过滤
        
        Args:
            labels: 原始标签
            threshold: Z-score阈值
            
        Returns:
            处理后的标签
        """
        z_scores = np.abs((labels - labels.mean()) / labels.std())
        return labels.where(z_scores <= threshold)


class LabelEngineeringManager:
    """标签构建管理器"""
    
    def __init__(self):
        self.builders = {}
        self.labels = {}
        self.label_stats = {}
    
    def add_return_label(self, 
                        name: str,
                        periods: int = 1,
                        price_column: str = 'close',
                        method: str = 'simple',
                        **kwargs) -> 'LabelEngineeringManager':
        """添加收益率标签"""
        self.builders[name] = ReturnLabelBuilder(
            periods=periods,
            price_column=price_column,
            method=method,
            **kwargs
        )
        return self
    
    def add_direction_label(self,
                           name: str,
                           periods: int = 1,
                           price_column: str = 'close',
                           threshold: float = 0.0,
                           neutral_zone: Optional[float] = None,
                           **kwargs) -> 'LabelEngineeringManager':
        """添加方向标签"""
        self.builders[name] = DirectionLabelBuilder(
            periods=periods,
            price_column=price_column,
            threshold=threshold,
            neutral_zone=neutral_zone,
            **kwargs
        )
        return self
    
    def add_excess_return_label(self,
                               name: str,
                               periods: int = 1,
                               price_column: str = 'close',
                               benchmark_column: Optional[str] = None,
                               benchmark_data: Optional[pd.DataFrame] = None,
                               **kwargs) -> 'LabelEngineeringManager':
        """添加超额收益标签"""
        self.builders[name] = ExcessReturnLabelBuilder(
            periods=periods,
            price_column=price_column,
            benchmark_column=benchmark_column,
            benchmark_data=benchmark_data,
            **kwargs
        )
        return self
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'LabelEngineeringManager':
        """拟合所有标签构建器"""
        for name, builder in self.builders.items():
            try:
                builder.fit(data, **kwargs)
                print(f"✓ Fitted label builder: {name}")
            except Exception as e:
                print(f"✗ Failed to fit label builder {name}: {e}")
        
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """构建所有标签"""
        results = {}
        
        for name, builder in self.builders.items():
            try:
                labels = builder.transform(data, **kwargs)
                results[name] = labels
                
                # 保存统计信息
                self.label_stats[name] = {
                    'count': labels.count(),
                    'mean': labels.mean(),
                    'std': labels.std(),
                    'min': labels.min(),
                    'max': labels.max(),
                    'null_ratio': labels.isnull().mean()
                }
                
                print(f"✓ Generated label: {name} ({labels.count()} valid values)")
            except Exception as e:
                print(f"✗ Failed to generate label {name}: {e}")
                results[name] = pd.Series(np.nan, index=data.index)
        
        self.labels = pd.DataFrame(results)
        return self.labels
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """拟合并构建所有标签"""
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    def get_label_summary(self) -> pd.DataFrame:
        """获取标签统计摘要"""
        if not self.label_stats:
            return pd.DataFrame()
        
        return pd.DataFrame(self.label_stats).T
    
    def apply_smoothing(self, 
                       label_name: str,
                       method: str = 'gaussian',
                       **kwargs) -> pd.Series:
        """应用标签平滑"""
        if label_name not in self.labels.columns:
            raise ValueError(f"Label '{label_name}' not found")
        
        labels = self.labels[label_name]
        
        if method == 'gaussian':
            return LabelSmoother.gaussian_smooth(labels, **kwargs)
        elif method == 'rolling_mean':
            return LabelSmoother.rolling_mean_smooth(labels, **kwargs)
        elif method == 'exponential':
            return LabelSmoother.exponential_smooth(labels, **kwargs)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    def apply_noise_reduction(self,
                             label_name: str,
                             method: str = 'outlier_clip',
                             **kwargs) -> pd.Series:
        """应用噪声处理"""
        if label_name not in self.labels.columns:
            raise ValueError(f"Label '{label_name}' not found")
        
        labels = self.labels[label_name]
        
        if method == 'outlier_clip':
            return NoiseReducer.outlier_clip(labels, **kwargs)
        elif method == 'winsorize':
            return NoiseReducer.winsorize(labels, **kwargs)
        elif method == 'z_score_filter':
            return NoiseReducer.z_score_filter(labels, **kwargs)
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")


# 便捷函数
def create_return_labels(data: pd.DataFrame,
                        periods: List[int] = [1, 5, 10],
                        price_column: str = 'close',
                        method: str = 'simple') -> pd.DataFrame:
    """
    批量创建收益率标签
    
    Args:
        data: 价格数据
        periods: 期数列表
        price_column: 价格列名
        method: 计算方法
        
    Returns:
        包含所有收益率标签的DataFrame
    """
    manager = LabelEngineeringManager()
    
    for period in periods:
        manager.add_return_label(
            name=f'return_{period}d',
            periods=period,
            price_column=price_column,
            method=method
        )
    
    return manager.fit_transform(data)


def create_direction_labels(data: pd.DataFrame,
                           periods: List[int] = [1, 5, 10],
                           price_column: str = 'close',
                           neutral_zone: Optional[float] = None) -> pd.DataFrame:
    """
    批量创建方向标签
    
    Args:
        data: 价格数据
        periods: 期数列表
        price_column: 价格列名
        neutral_zone: 中性区间
        
    Returns:
        包含所有方向标签的DataFrame
    """
    manager = LabelEngineeringManager()
    
    for period in periods:
        manager.add_direction_label(
            name=f'direction_{period}d',
            periods=period,
            price_column=price_column,
            neutral_zone=neutral_zone
        )
    
    return manager.fit_transform(data)


if __name__ == "__main__":
    print("标签构建模块已加载")
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    
    # 模拟价格数据（带趋势和噪声）
    trend = np.linspace(100, 150, 1000)
    noise = np.random.normal(0, 2, 1000)
    prices = trend + noise + np.random.normal(0, 0.5, 1000)
    
    # 模拟基准数据
    benchmark_trend = np.linspace(100, 140, 1000)
    benchmark_noise = np.random.normal(0, 1.5, 1000)
    benchmark_prices = benchmark_trend + benchmark_noise
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'benchmark': benchmark_prices
    })
    
    print(f"\\n示例数据: {len(data)} 行")
    print(f"价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 创建标签构建管理器
    manager = LabelEngineeringManager()
    
    # 添加各种标签
    manager.add_return_label('return_1d', periods=1, method='simple')
    manager.add_return_label('return_5d', periods=5, method='simple')
    manager.add_return_label('return_log_1d', periods=1, method='log')
    
    manager.add_direction_label('direction_1d', periods=1, neutral_zone=0.01)
    manager.add_direction_label('direction_5d', periods=5, neutral_zone=0.02)
    
    manager.add_excess_return_label('excess_1d', periods=1, benchmark_column='benchmark')
    manager.add_excess_return_label('excess_5d', periods=5, benchmark_column='benchmark')
    
    # 构建标签
    print("\\n构建标签...")
    labels = manager.fit_transform(data)
    
    print(f"\\n生成的标签: {list(labels.columns)}")
    print(f"标签数据形状: {labels.shape}")
    
    # 显示标签统计
    print("\\n============================================================")
    print("标签统计摘要")
    print("============================================================")
    summary = manager.get_label_summary()
    print(summary.round(4))
    
    # 显示标签分布
    print("\\n============================================================")
    print("方向标签分布")
    print("============================================================")
    for col in ['direction_1d', 'direction_5d']:
        if col in labels.columns:
            dist = labels[col].value_counts().sort_index()
            print(f"\\n{col}:")
            for value, count in dist.items():
                print(f"  {value:2.0f}: {count:4d} ({count/len(labels)*100:.1f}%)")
    
    # 测试标签平滑
    print("\\n============================================================")
    print("标签平滑测试")
    print("============================================================")
    
    if 'return_1d' in labels.columns:
        original = labels['return_1d']
        smoothed = manager.apply_smoothing('return_1d', method='gaussian', window=5, std=1.0)
        
        print(f"原始标签标准差: {original.std():.6f}")
        print(f"平滑后标准差: {smoothed.std():.6f}")
        print(f"平滑效果: {(1 - smoothed.std()/original.std())*100:.1f}% 降噪")
    
    print("\\n标签构建模块测试完成")