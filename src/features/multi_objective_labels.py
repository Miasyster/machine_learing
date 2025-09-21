"""
多目标标签构建模块

实现量化交易中的多目标标签构建功能：
- 夏普比率最大化标签
- 波动率预测标签
- 风险调整收益标签
- 多目标联合优化标签

作者: AI Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from scipy import optimize
from sklearn.preprocessing import StandardScaler
import warnings


class MultiObjectiveLabelBuilder(ABC):
    """多目标标签构建器基类"""
    
    def __init__(self):
        self.is_fitted_ = False
        self.label_stats_ = {}
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'MultiObjectiveLabelBuilder':
        """拟合标签构建器"""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """转换数据为标签"""
        pass
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """拟合并转换"""
        return self.fit(data, **kwargs).transform(data, **kwargs)


class SharpeRatioLabelBuilder(MultiObjectiveLabelBuilder):
    """夏普比率标签构建器"""
    
    def __init__(self, 
                 periods: int = 1,
                 risk_free_rate: float = 0.0,
                 rolling_window: int = 20,
                 min_periods: Optional[int] = None,
                 price_column: str = 'close'):
        """
        初始化夏普比率标签构建器
        
        参数:
            periods: 预测期数
            risk_free_rate: 无风险利率（年化）
            rolling_window: 滚动窗口大小
            min_periods: 最小期数要求
            price_column: 价格列名
        """
        super().__init__()
        self.periods = periods
        self.risk_free_rate = risk_free_rate
        self.rolling_window = rolling_window
        self.min_periods = min_periods or max(10, rolling_window // 2)
        self.price_column = price_column
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'SharpeRatioLabelBuilder':
        """拟合夏普比率标签构建器"""
        if self.price_column not in data.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        # 计算收益率
        prices = data[self.price_column]
        returns = prices.pct_change()
        
        # 计算滚动夏普比率统计
        rolling_mean = returns.rolling(window=self.rolling_window, min_periods=self.min_periods).mean()
        rolling_std = returns.rolling(window=self.rolling_window, min_periods=self.min_periods).std()
        
        # 年化调整
        trading_days = 252
        daily_rf_rate = self.risk_free_rate / trading_days
        
        # 计算夏普比率
        excess_returns = rolling_mean - daily_rf_rate
        sharpe_ratios = excess_returns / rolling_std
        
        self.label_stats_ = {
            'mean_sharpe': sharpe_ratios.mean(),
            'std_sharpe': sharpe_ratios.std(),
            'min_sharpe': sharpe_ratios.min(),
            'max_sharpe': sharpe_ratios.max(),
            'positive_sharpe_ratio': (sharpe_ratios > 0).mean(),
            'high_sharpe_ratio': (sharpe_ratios > 1.0).mean()
        }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """转换为夏普比率标签"""
        if not self.is_fitted_:
            raise ValueError("SharpeRatioLabelBuilder must be fitted first")
        
        prices = data[self.price_column]
        
        # 计算未来收益率
        future_returns = prices.pct_change(periods=self.periods).shift(-self.periods)
        
        # 计算历史波动率
        returns = prices.pct_change()
        rolling_volatility = returns.rolling(
            window=self.rolling_window, 
            min_periods=self.min_periods
        ).std()
        
        # 计算未来夏普比率标签
        daily_rf_rate = self.risk_free_rate / 252
        excess_future_returns = future_returns - daily_rf_rate * self.periods
        
        # 使用历史波动率估计未来夏普比率
        sharpe_labels = excess_future_returns / (rolling_volatility * np.sqrt(self.periods))
        
        return sharpe_labels


class VolatilityLabelBuilder(MultiObjectiveLabelBuilder):
    """波动率预测标签构建器"""
    
    def __init__(self,
                 periods: int = 1,
                 volatility_type: str = 'realized',
                 rolling_window: int = 20,
                 price_column: str = 'close'):
        """
        初始化波动率标签构建器
        
        参数:
            periods: 预测期数
            volatility_type: 波动率类型 ('realized', 'garch', 'parkinson')
            rolling_window: 滚动窗口大小
            price_column: 价格列名
        """
        super().__init__()
        self.periods = periods
        self.volatility_type = volatility_type
        self.rolling_window = rolling_window
        self.price_column = price_column
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'VolatilityLabelBuilder':
        """拟合波动率标签构建器"""
        if self.price_column not in data.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        prices = data[self.price_column]
        volatilities = self._calculate_volatility(data)
        
        self.label_stats_ = {
            'mean_volatility': volatilities.mean(),
            'std_volatility': volatilities.std(),
            'min_volatility': volatilities.min(),
            'max_volatility': volatilities.max(),
            'volatility_persistence': volatilities.autocorr(lag=1)
        }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """转换为波动率标签"""
        if not self.is_fitted_:
            raise ValueError("VolatilityLabelBuilder must be fitted first")
        
        # 计算未来波动率
        future_volatility = self._calculate_future_volatility(data)
        
        return future_volatility
    
    def _calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        """计算波动率"""
        prices = data[self.price_column]
        
        if self.volatility_type == 'realized':
            returns = prices.pct_change()
            volatility = returns.rolling(window=self.rolling_window).std() * np.sqrt(252)
        
        elif self.volatility_type == 'parkinson':
            # Parkinson估计器（需要高低价）
            if 'high' in data.columns and 'low' in data.columns:
                high_low_ratio = np.log(data['high'] / data['low'])
                volatility = high_low_ratio.rolling(window=self.rolling_window).apply(
                    lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))) * np.sqrt(252)
                )
            else:
                # 回退到已实现波动率
                returns = prices.pct_change()
                volatility = returns.rolling(window=self.rolling_window).std() * np.sqrt(252)
        
        else:  # 默认使用已实现波动率
            returns = prices.pct_change()
            volatility = returns.rolling(window=self.rolling_window).std() * np.sqrt(252)
        
        return volatility
    
    def _calculate_future_volatility(self, data: pd.DataFrame) -> pd.Series:
        """计算未来波动率"""
        prices = data[self.price_column]
        
        # 计算未来periods天的已实现波动率
        future_returns = []
        for i in range(self.periods):
            future_return = prices.pct_change().shift(-(i+1))
            future_returns.append(future_return)
        
        future_returns_df = pd.concat(future_returns, axis=1)
        future_volatility = future_returns_df.std(axis=1) * np.sqrt(252)
        
        return future_volatility


class RiskAdjustedReturnLabelBuilder(MultiObjectiveLabelBuilder):
    """风险调整收益标签构建器"""
    
    def __init__(self,
                 periods: int = 1,
                 metric: str = 'information_ratio',
                 benchmark_column: Optional[str] = None,
                 benchmark_return: float = 0.0,
                 rolling_window: int = 20,
                 price_column: str = 'close'):
        """
        初始化风险调整收益标签构建器
        
        参数:
            periods: 预测期数
            metric: 风险调整指标 ('information_ratio', 'calmar_ratio', 'sortino_ratio')
            benchmark_column: 基准列名
            benchmark_return: 固定基准收益率
            rolling_window: 滚动窗口大小
            price_column: 价格列名
        """
        super().__init__()
        self.periods = periods
        self.metric = metric
        self.benchmark_column = benchmark_column
        self.benchmark_return = benchmark_return
        self.rolling_window = rolling_window
        self.price_column = price_column
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'RiskAdjustedReturnLabelBuilder':
        """拟合风险调整收益标签构建器"""
        if self.price_column not in data.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        prices = data[self.price_column]
        risk_adjusted_returns = self._calculate_risk_adjusted_returns(data)
        
        self.label_stats_ = {
            f'mean_{self.metric}': risk_adjusted_returns.mean(),
            f'std_{self.metric}': risk_adjusted_returns.std(),
            f'min_{self.metric}': risk_adjusted_returns.min(),
            f'max_{self.metric}': risk_adjusted_returns.max(),
            f'positive_{self.metric}_ratio': (risk_adjusted_returns > 0).mean()
        }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """转换为风险调整收益标签"""
        if not self.is_fitted_:
            raise ValueError("RiskAdjustedReturnLabelBuilder must be fitted first")
        
        # 计算未来风险调整收益
        future_risk_adjusted_returns = self._calculate_future_risk_adjusted_returns(data)
        
        return future_risk_adjusted_returns
    
    def _calculate_risk_adjusted_returns(self, data: pd.DataFrame) -> pd.Series:
        """计算风险调整收益"""
        prices = data[self.price_column]
        returns = prices.pct_change()
        
        # 获取基准收益
        if self.benchmark_column and self.benchmark_column in data.columns:
            benchmark_returns = data[self.benchmark_column].pct_change()
        else:
            benchmark_returns = pd.Series(self.benchmark_return / 252, index=returns.index)
        
        excess_returns = returns - benchmark_returns
        
        if self.metric == 'information_ratio':
            # 信息比率 = 超额收益 / 跟踪误差
            tracking_error = excess_returns.rolling(window=self.rolling_window).std()
            rolling_excess_return = excess_returns.rolling(window=self.rolling_window).mean()
            risk_adjusted_returns = rolling_excess_return / tracking_error
        
        elif self.metric == 'calmar_ratio':
            # 卡尔马比率 = 年化收益 / 最大回撤
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.rolling(window=self.rolling_window, min_periods=1).max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.rolling(window=self.rolling_window).min().abs()
            
            annual_return = returns.rolling(window=self.rolling_window).mean() * 252
            risk_adjusted_returns = annual_return / max_drawdown
        
        elif self.metric == 'sortino_ratio':
            # 索提诺比率 = 超额收益 / 下行标准差
            downside_returns = excess_returns.where(excess_returns < 0, 0)
            downside_std = downside_returns.rolling(window=self.rolling_window).std()
            rolling_excess_return = excess_returns.rolling(window=self.rolling_window).mean()
            risk_adjusted_returns = rolling_excess_return / downside_std
        
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return risk_adjusted_returns
    
    def _calculate_future_risk_adjusted_returns(self, data: pd.DataFrame) -> pd.Series:
        """计算未来风险调整收益"""
        prices = data[self.price_column]
        
        # 计算未来收益率
        future_returns = prices.pct_change(periods=self.periods).shift(-self.periods)
        
        # 计算历史风险度量
        returns = prices.pct_change()
        
        if self.metric == 'information_ratio':
            # 使用历史跟踪误差
            if self.benchmark_column and self.benchmark_column in data.columns:
                benchmark_returns = data[self.benchmark_column].pct_change()
            else:
                benchmark_returns = pd.Series(self.benchmark_return / 252, index=returns.index)
            
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.rolling(window=self.rolling_window).std()
            
            future_excess_returns = future_returns - benchmark_returns.shift(-self.periods)
            future_risk_adjusted = future_excess_returns / tracking_error
        
        elif self.metric == 'sortino_ratio':
            # 使用历史下行标准差
            if self.benchmark_column and self.benchmark_column in data.columns:
                benchmark_returns = data[self.benchmark_column].pct_change()
            else:
                benchmark_returns = pd.Series(self.benchmark_return / 252, index=returns.index)
            
            excess_returns = returns - benchmark_returns
            downside_returns = excess_returns.where(excess_returns < 0, 0)
            downside_std = downside_returns.rolling(window=self.rolling_window).std()
            
            future_excess_returns = future_returns - benchmark_returns.shift(-self.periods)
            future_risk_adjusted = future_excess_returns / downside_std
        
        else:
            # 对于其他指标，使用简化计算
            volatility = returns.rolling(window=self.rolling_window).std()
            future_risk_adjusted = future_returns / volatility
        
        return future_risk_adjusted


class MultiObjectiveOptimizationLabelBuilder(MultiObjectiveLabelBuilder):
    """多目标优化标签构建器"""
    
    def __init__(self,
                 periods: int = 1,
                 objectives: List[str] = ['return', 'sharpe', 'volatility'],
                 weights: Optional[List[float]] = None,
                 optimization_method: str = 'weighted_sum',
                 price_column: str = 'close'):
        """
        初始化多目标优化标签构建器
        
        参数:
            periods: 预测期数
            objectives: 目标列表
            weights: 目标权重
            optimization_method: 优化方法 ('weighted_sum', 'pareto_ranking')
            price_column: 价格列名
        """
        super().__init__()
        self.periods = periods
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
        self.optimization_method = optimization_method
        self.price_column = price_column
        
        # 初始化子构建器
        self.sub_builders = {}
        self._initialize_sub_builders()
    
    def _initialize_sub_builders(self):
        """初始化子构建器"""
        if 'return' in self.objectives:
            try:
                from .label_engineering import ReturnLabelBuilder
            except ImportError:
                from label_engineering import ReturnLabelBuilder
            self.sub_builders['return'] = ReturnLabelBuilder(periods=self.periods)
        
        if 'sharpe' in self.objectives:
            self.sub_builders['sharpe'] = SharpeRatioLabelBuilder(periods=self.periods)
        
        if 'volatility' in self.objectives:
            self.sub_builders['volatility'] = VolatilityLabelBuilder(periods=self.periods)
        
        if 'information_ratio' in self.objectives:
            self.sub_builders['information_ratio'] = RiskAdjustedReturnLabelBuilder(
                periods=self.periods, metric='information_ratio'
            )
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'MultiObjectiveOptimizationLabelBuilder':
        """拟合多目标优化标签构建器"""
        # 拟合所有子构建器
        for name, builder in self.sub_builders.items():
            try:
                builder.fit(data, **kwargs)
                print(f"✓ Fitted {name} builder")
            except Exception as e:
                print(f"✗ Failed to fit {name} builder: {e}")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """转换为多目标优化标签"""
        if not self.is_fitted_:
            raise ValueError("MultiObjectiveOptimizationLabelBuilder must be fitted first")
        
        # 获取所有目标的标签
        objective_labels = {}
        for name, builder in self.sub_builders.items():
            try:
                labels = builder.transform(data, **kwargs)
                objective_labels[name] = labels
            except Exception as e:
                print(f"Warning: Failed to generate {name} labels: {e}")
                continue
        
        if not objective_labels:
            raise ValueError("No objective labels could be generated")
        
        # 标准化标签
        normalized_labels = self._normalize_labels(objective_labels)
        
        # 多目标优化
        if self.optimization_method == 'weighted_sum':
            optimized_labels = self._weighted_sum_optimization(normalized_labels)
        elif self.optimization_method == 'pareto_ranking':
            optimized_labels = self._pareto_ranking_optimization(normalized_labels)
        else:
            raise ValueError(f"Unsupported optimization method: {self.optimization_method}")
        
        return optimized_labels
    
    def _normalize_labels(self, objective_labels: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """标准化标签"""
        normalized = {}
        
        for name, labels in objective_labels.items():
            # 移除无穷值和NaN
            clean_labels = labels.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean_labels) == 0:
                continue
            
            # Z-score标准化
            mean_val = clean_labels.mean()
            std_val = clean_labels.std()
            
            if std_val > 0:
                normalized_labels = (labels - mean_val) / std_val
            else:
                normalized_labels = labels - mean_val
            
            # 对于波动率，取负值（因为我们希望低波动率）
            if name == 'volatility':
                normalized_labels = -normalized_labels
            
            normalized[name] = normalized_labels
        
        return normalized
    
    def _weighted_sum_optimization(self, normalized_labels: Dict[str, pd.Series]) -> pd.Series:
        """加权求和优化"""
        result = None
        
        for i, (name, labels) in enumerate(normalized_labels.items()):
            weight = self.weights[i] if i < len(self.weights) else 1.0
            
            if result is None:
                result = weight * labels
            else:
                result = result + weight * labels
        
        return result
    
    def _pareto_ranking_optimization(self, normalized_labels: Dict[str, pd.Series]) -> pd.Series:
        """帕累托排序优化"""
        # 将所有标签合并为DataFrame
        labels_df = pd.DataFrame(normalized_labels)
        labels_df = labels_df.dropna()
        
        if len(labels_df) == 0:
            return pd.Series(dtype=float)
        
        # 计算帕累托前沿
        pareto_ranks = self._calculate_pareto_ranks(labels_df.values)
        
        # 将排序结果映射回原始索引
        result = pd.Series(index=labels_df.index, data=pareto_ranks)
        
        # 重新索引到原始数据
        full_result = pd.Series(index=list(normalized_labels.values())[0].index, dtype=float)
        full_result.loc[result.index] = result
        
        return full_result
    
    def _calculate_pareto_ranks(self, objectives: np.ndarray) -> np.ndarray:
        """计算帕累托排序"""
        n = len(objectives)
        ranks = np.zeros(n)
        
        for i in range(n):
            rank = 1
            for j in range(n):
                if i != j:
                    # 检查j是否支配i
                    if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                        rank += 1
            ranks[i] = rank
        
        # 转换为分数（排序越高分数越高）
        max_rank = ranks.max()
        scores = (max_rank - ranks + 1) / max_rank
        
        return scores


class MultiObjectiveLabelManager:
    """多目标标签管理器"""
    
    def __init__(self):
        self.builders = {}
        self.labels = {}
        self.optimization_results = {}
    
    def add_sharpe_label(self, name: str, **kwargs) -> 'MultiObjectiveLabelManager':
        """添加夏普比率标签"""
        self.builders[name] = SharpeRatioLabelBuilder(**kwargs)
        return self
    
    def add_volatility_label(self, name: str, **kwargs) -> 'MultiObjectiveLabelManager':
        """添加波动率标签"""
        self.builders[name] = VolatilityLabelBuilder(**kwargs)
        return self
    
    def add_risk_adjusted_label(self, name: str, **kwargs) -> 'MultiObjectiveLabelManager':
        """添加风险调整收益标签"""
        self.builders[name] = RiskAdjustedReturnLabelBuilder(**kwargs)
        return self
    
    def add_multi_objective_label(self, name: str, **kwargs) -> 'MultiObjectiveLabelManager':
        """添加多目标优化标签"""
        self.builders[name] = MultiObjectiveOptimizationLabelBuilder(**kwargs)
        return self
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'MultiObjectiveLabelManager':
        """拟合所有构建器"""
        for name, builder in self.builders.items():
            try:
                builder.fit(data, **kwargs)
                print(f"✓ Fitted {name} builder")
            except Exception as e:
                print(f"✗ Failed to fit {name} builder: {e}")
        return self
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """拟合并转换所有标签"""
        self.fit(data, **kwargs)
        
        results = {}
        for name, builder in self.builders.items():
            try:
                labels = builder.transform(data, **kwargs)
                results[name] = labels
                print(f"✓ Generated {name} labels ({labels.count()} valid values)")
            except Exception as e:
                print(f"✗ Failed to generate {name} labels: {e}")
        
        self.labels = pd.DataFrame(results)
        return self.labels
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """获取优化摘要"""
        if self.labels.empty:
            return pd.DataFrame()
        
        summary_stats = {}
        for col in self.labels.columns:
            labels = self.labels[col].dropna()
            if len(labels) > 0:
                summary_stats[col] = {
                    'count': len(labels),
                    'mean': labels.mean(),
                    'std': labels.std(),
                    'min': labels.min(),
                    'max': labels.max(),
                    'skewness': labels.skew(),
                    'kurtosis': labels.kurtosis()
                }
        
        return pd.DataFrame(summary_stats).T


# 便捷函数
def create_sharpe_maximization_labels(data: pd.DataFrame,
                                     periods: List[int] = [1, 5, 20],
                                     risk_free_rate: float = 0.02,
                                     **kwargs) -> pd.DataFrame:
    """创建夏普比率最大化标签"""
    manager = MultiObjectiveLabelManager()
    
    for period in periods:
        manager.add_sharpe_label(
            f'sharpe_{period}d',
            periods=period,
            risk_free_rate=risk_free_rate,
            **kwargs
        )
    
    return manager.fit_transform(data)


def create_multi_objective_labels(data: pd.DataFrame,
                                 periods: List[int] = [1, 5],
                                 objectives: List[str] = ['return', 'sharpe', 'volatility'],
                                 **kwargs) -> pd.DataFrame:
    """创建多目标优化标签"""
    manager = MultiObjectiveLabelManager()
    
    for period in periods:
        manager.add_multi_objective_label(
            f'multi_obj_{period}d',
            periods=period,
            objectives=objectives,
            **kwargs
        )
    
    return manager.fit_transform(data)


if __name__ == "__main__":
    print("多目标标签构建模块已加载")
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # 模拟价格数据（带趋势和波动）
    returns = np.random.normal(0.0005, 0.02, 1000)
    returns[100:200] += 0.001  # 添加一个上涨趋势
    returns[500:600] -= 0.001  # 添加一个下跌趋势
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # 添加高低价
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, 1000)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, 1000)))
    
    # 创建基准数据
    benchmark_returns = np.random.normal(0.0003, 0.015, 1000)
    benchmark_prices = 100 * np.exp(np.cumsum(benchmark_returns))
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': highs,
        'low': lows,
        'benchmark': benchmark_prices
    })
    data.set_index('date', inplace=True)
    
    print(f"创建示例数据: {data.shape}")
    print(f"数据范围: {data.index[0]} 到 {data.index[-1]}")
    
    # 创建多目标标签管理器
    manager = MultiObjectiveLabelManager()
    
    # 添加各种标签
    manager.add_sharpe_label('sharpe_1d', periods=1, risk_free_rate=0.02)
    manager.add_sharpe_label('sharpe_5d', periods=5, risk_free_rate=0.02)
    
    manager.add_volatility_label('volatility_1d', periods=1, volatility_type='realized')
    manager.add_volatility_label('volatility_parkinson', periods=1, volatility_type='parkinson')
    
    manager.add_risk_adjusted_label('info_ratio', periods=1, metric='information_ratio', benchmark_column='benchmark')
    manager.add_risk_adjusted_label('sortino_ratio', periods=1, metric='sortino_ratio', benchmark_column='benchmark')
    
    manager.add_multi_objective_label('multi_obj_1d', periods=1, objectives=['return', 'sharpe', 'volatility'])
    
    # 拟合和转换
    labels = manager.fit_transform(data)
    
    print(f"\n生成的多目标标签: {list(labels.columns)}")
    print(f"标签数据形状: {labels.shape}")
    
    # 显示标签统计
    print("\n=== 多目标标签统计摘要 ===")
    summary = manager.get_optimization_summary()
    print(summary.round(4))
    
    # 分析夏普比率标签
    print("\n=== 夏普比率标签分析 ===")
    sharpe_cols = [col for col in labels.columns if 'sharpe' in col]
    for col in sharpe_cols:
        if col in labels.columns:
            sharpe_data = labels[col].dropna()
            if len(sharpe_data) > 0:
                print(f"{col}:")
                print(f"  均值: {sharpe_data.mean():.4f}")
                print(f"  标准差: {sharpe_data.std():.4f}")
                print(f"  正值比例: {(sharpe_data > 0).mean():.2%}")
                print(f"  高夏普比例 (>1): {(sharpe_data > 1).mean():.2%}")
    
    # 分析多目标优化结果
    print("\n=== 多目标优化分析 ===")
    if 'multi_obj_1d' in labels.columns:
        multi_obj = labels['multi_obj_1d'].dropna()
        if len(multi_obj) > 0:
            print(f"多目标分数分布:")
            print(f"  均值: {multi_obj.mean():.4f}")
            print(f"  标准差: {multi_obj.std():.4f}")
            print(f"  最小值: {multi_obj.min():.4f}")
            print(f"  最大值: {multi_obj.max():.4f}")
            
            # 分析分数分布
            percentiles = [10, 25, 50, 75, 90]
            print(f"  分位数分布:")
            for p in percentiles:
                val = np.percentile(multi_obj, p)
                print(f"    {p}%: {val:.4f}")
    
    print("\n多目标标签构建模块测试完成")