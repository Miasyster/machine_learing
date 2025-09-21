"""
时间序列验证模块

实现时间序列数据的特殊划分和验证策略，包括：
1. 时间序列数据划分（保持时间顺序）
2. 时间序列交叉验证（TimeSeriesSplit）
3. Walk-forward验证
4. 滑动窗口验证
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator, Generator
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator
import warnings

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesConfig:
    """时间序列验证配置"""
    
    # 基础配置
    time_column: Optional[str] = None  # 时间列名
    sort_by_time: bool = True  # 是否按时间排序
    
    # 数据划分比例
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # 交叉验证配置
    n_splits: int = 5
    max_train_size: Optional[int] = None
    test_size: Optional[int] = None
    gap: int = 0  # 训练集和测试集之间的间隔
    
    # Walk-forward配置
    initial_window: Optional[int] = None  # 初始训练窗口大小
    step_size: int = 1  # 每次前进的步数
    expanding_window: bool = True  # 是否使用扩展窗口（False为滑动窗口）
    
    # 验证策略
    validation_strategy: str = 'time_series_split'  # 'time_series_split', 'walk_forward', 'sliding_window'
    
    def __post_init__(self):
        """验证配置参数"""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("训练、验证、测试比例之和必须等于1.0")
        
        if self.train_ratio <= 0 or self.val_ratio < 0 or self.test_ratio <= 0:
            raise ValueError("所有比例必须为正数，验证集比例可以为0")


class TimeSeriesDataSplitter:
    """时间序列数据划分器"""
    
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        
    def split_data(self, 
                   X: Union[np.ndarray, pd.DataFrame], 
                   y: Union[np.ndarray, pd.Series],
                   time_index: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]] = None
                   ) -> Tuple[np.ndarray, ...]:
        """
        按时间顺序划分数据
        
        Args:
            X: 特征数据
            y: 目标变量
            time_index: 时间索引（可选）
            
        Returns:
            划分后的数据集
        """
        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
            
        n_samples = len(X_array)
        
        # 如果提供了时间索引，按时间排序
        if time_index is not None:
            if self.config.sort_by_time:
                sort_indices = np.argsort(time_index)
                X_array = X_array[sort_indices]
                y_array = y_array[sort_indices]
                if time_index is not None:
                    time_index = time_index[sort_indices]
        
        # 计算划分点
        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))
        
        # 划分数据
        X_train = X_array[:train_end]
        y_train = y_array[:train_end]
        
        if self.config.val_ratio > 0:
            X_val = X_array[train_end:val_end]
            y_val = y_array[train_end:val_end]
            X_test = X_array[val_end:]
            y_test = y_array[val_end:]
            
            logger.info(f"数据划分完成: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_test = X_array[train_end:]
            y_test = y_array[train_end:]
            
            logger.info(f"数据划分完成: 训练集 {len(X_train)}, 测试集 {len(X_test)}")
            return X_train, X_test, y_train, y_test
    
    def get_split_info(self, n_samples: int) -> Dict[str, Any]:
        """获取划分信息"""
        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))
        
        info = {
            'total_samples': n_samples,
            'train_samples': train_end,
            'train_ratio_actual': train_end / n_samples,
            'test_samples': n_samples - val_end if self.config.val_ratio > 0 else n_samples - train_end,
        }
        
        if self.config.val_ratio > 0:
            info.update({
                'val_samples': val_end - train_end,
                'val_ratio_actual': (val_end - train_end) / n_samples,
                'test_ratio_actual': (n_samples - val_end) / n_samples,
            })
        else:
            info['test_ratio_actual'] = (n_samples - train_end) / n_samples
            
        return info


class TimeSeriesCrossValidator:
    """时间序列交叉验证器"""
    
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        
    def split(self, 
              X: Union[np.ndarray, pd.DataFrame], 
              y: Optional[Union[np.ndarray, pd.Series]] = None
              ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成时间序列交叉验证的训练/验证索引
        
        Args:
            X: 特征数据
            y: 目标变量（可选）
            
        Yields:
            (train_indices, val_indices): 训练和验证索引
        """
        n_samples = len(X)
        
        if self.config.validation_strategy == 'time_series_split':
            # 使用sklearn的TimeSeriesSplit
            tscv = TimeSeriesSplit(
                n_splits=self.config.n_splits,
                max_train_size=self.config.max_train_size,
                test_size=self.config.test_size,
                gap=self.config.gap
            )
            
            for train_idx, val_idx in tscv.split(X):
                yield train_idx, val_idx
                
        elif self.config.validation_strategy == 'walk_forward':
            # Walk-forward验证
            yield from self._walk_forward_split(n_samples)
            
        elif self.config.validation_strategy == 'sliding_window':
            # 滑动窗口验证
            yield from self._sliding_window_split(n_samples)
            
        else:
            raise ValueError(f"不支持的验证策略: {self.config.validation_strategy}")
    
    def _walk_forward_split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Walk-forward验证划分"""
        if self.config.initial_window is None:
            initial_window = max(n_samples // (self.config.n_splits + 1), 50)
        else:
            initial_window = self.config.initial_window
            
        test_size = self.config.test_size or max(n_samples // (self.config.n_splits * 2), 10)
        
        for i in range(self.config.n_splits):
            if self.config.expanding_window:
                # 扩展窗口：训练集逐渐增大
                train_start = 0
                train_end = initial_window + i * self.config.step_size
            else:
                # 滑动窗口：训练集大小固定
                train_end = initial_window + i * self.config.step_size
                train_start = train_end - initial_window
            
            # 添加间隔
            val_start = train_end + self.config.gap
            val_end = val_start + test_size
            
            # 检查边界
            if val_end > n_samples:
                break
                
            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)
            
            yield train_idx, val_idx
    
    def _sliding_window_split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """滑动窗口验证划分"""
        if self.config.initial_window is None:
            window_size = n_samples // (self.config.n_splits + 1)
        else:
            window_size = self.config.initial_window
            
        test_size = self.config.test_size or max(window_size // 4, 10)
        
        for i in range(self.config.n_splits):
            train_start = i * self.config.step_size
            train_end = train_start + window_size
            
            val_start = train_end + self.config.gap
            val_end = val_start + test_size
            
            if val_end > n_samples:
                break
                
            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)
            
            yield train_idx, val_idx
    
    def get_n_splits(self, X: Union[np.ndarray, pd.DataFrame]) -> int:
        """获取实际的分割数量"""
        splits = list(self.split(X))
        return len(splits)


class WalkForwardValidator:
    """Walk-forward验证器"""
    
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.results = []
        
    def validate(self, 
                 model: BaseEstimator,
                 X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 scoring_func: Optional[callable] = None) -> Dict[str, Any]:
        """
        执行walk-forward验证
        
        Args:
            model: 要验证的模型
            X: 特征数据
            y: 目标变量
            scoring_func: 评分函数
            
        Returns:
            验证结果
        """
        cv = TimeSeriesCrossValidator(self.config)
        scores = []
        predictions = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # 获取训练和验证数据
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                
            if isinstance(y, pd.Series):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练模型
            model_copy = self._clone_model(model)
            model_copy.fit(X_train, y_train)
            
            # 预测
            y_pred = model_copy.predict(X_val)
            predictions.extend(y_pred)
            
            # 评分
            if scoring_func:
                score = scoring_func(y_val, y_pred)
            else:
                score = model_copy.score(X_val, y_val)
            
            scores.append(score)
            
            fold_results.append({
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score,
                'train_period': (train_idx[0], train_idx[-1]),
                'val_period': (val_idx[0], val_idx[-1])
            })
            
            logger.info(f"Fold {fold}: 训练集大小 {len(train_idx)}, 验证集大小 {len(val_idx)}, 得分 {score:.4f}")
        
        results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'fold_results': fold_results,
            'predictions': predictions,
            'n_splits': len(scores)
        }
        
        self.results.append(results)
        return results
    
    def _clone_model(self, model: BaseEstimator) -> BaseEstimator:
        """克隆模型"""
        from sklearn.base import clone
        return clone(model)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """获取验证摘要"""
        if not self.results:
            return {}
            
        all_scores = []
        for result in self.results:
            all_scores.extend(result['scores'])
        
        return {
            'total_validations': len(self.results),
            'total_folds': sum(r['n_splits'] for r in self.results),
            'overall_mean_score': np.mean(all_scores),
            'overall_std_score': np.std(all_scores),
            'best_score': max(all_scores),
            'worst_score': min(all_scores),
            'score_range': max(all_scores) - min(all_scores)
        }


def create_time_series_splits(X: Union[np.ndarray, pd.DataFrame],
                             y: Union[np.ndarray, pd.Series],
                             config: TimeSeriesConfig) -> Dict[str, Any]:
    """
    创建时间序列数据划分的便捷函数
    
    Args:
        X: 特征数据
        y: 目标变量
        config: 时间序列配置
        
    Returns:
        包含所有划分的字典
    """
    splitter = TimeSeriesDataSplitter(config)
    
    # 基础划分
    if config.val_ratio > 0:
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(X, y)
        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    else:
        X_train, X_test, y_train, y_test = splitter.split_data(X, y)
        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }
    
    # 交叉验证划分
    cv = TimeSeriesCrossValidator(config)
    cv_splits = list(cv.split(X_train))
    
    splits.update({
        'cv_splits': cv_splits,
        'n_cv_splits': len(cv_splits),
        'split_info': splitter.get_split_info(len(X))
    })
    
    return splits


# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = np.random.randn(n_samples)
    
    # 配置时间序列验证
    config = TimeSeriesConfig(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        n_splits=5,
        validation_strategy='walk_forward',
        expanding_window=True
    )
    
    # 创建数据划分
    splits = create_time_series_splits(X, y, config)
    
    print("数据划分信息:")
    print(f"训练集: {splits['X_train'].shape}")
    print(f"验证集: {splits['X_val'].shape}")
    print(f"测试集: {splits['X_test'].shape}")
    print(f"交叉验证折数: {splits['n_cv_splits']}")