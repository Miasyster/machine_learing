#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样本平衡模块

实现各种样本平衡技术，包括：
- SMOTE（合成少数类过采样技术）
- 随机过采样
- 随机欠采样
- 编辑最近邻欠采样
- Tomek链接清理
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any, List
from collections import Counter
import warnings

from .base import BaseAugmenter


class SampleBalancer(BaseAugmenter):
    """样本平衡器"""
    
    def __init__(self, 
                 strategy: str = 'auto',
                 sampling_strategy: Union[str, float, Dict] = 'auto',
                 k_neighbors: int = 5,
                 random_state: Optional[int] = None):
        """
        初始化样本平衡器
        
        Args:
            strategy: 平衡策略 ('smote', 'random_over', 'random_under', 'edited_nn', 'tomek', 'auto')
            sampling_strategy: 采样策略
            k_neighbors: SMOTE算法的邻居数量
            random_state: 随机种子
        """
        super().__init__(random_state)
        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.class_counts_ = None
        self.target_counts_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'SampleBalancer':
        """
        拟合平衡器
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            self
        """
        self._validate_input(X, y)
        
        if y is None:
            raise ValueError("y is required for sample balancing")
        
        y_array = self._to_numpy(y)
        self.class_counts_ = Counter(y_array)
        self.target_counts_ = self._calculate_target_counts(self.class_counts_)
        
        self.logger.info(f"Original class distribution: {dict(self.class_counts_)}")
        self.logger.info(f"Target class distribution: {dict(self.target_counts_)}")
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame], 
                  y: Union[np.ndarray, pd.Series]) -> Tuple[Union[np.ndarray, pd.DataFrame], 
                                                           Union[np.ndarray, pd.Series]]:
        """
        应用样本平衡
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            平衡后的(X, y)
        """
        self._validate_input(X, y)
        
        if self.class_counts_ is None:
            raise ValueError("Must fit before transform")
        
        X_array = self._to_numpy(X)
        y_array = self._to_numpy(y)
        
        # 根据策略选择方法
        if self.strategy == 'auto':
            strategy = self._auto_select_strategy()
        else:
            strategy = self.strategy
        
        if strategy == 'smote':
            X_balanced, y_balanced = self._smote(X_array, y_array)
        elif strategy == 'random_over':
            X_balanced, y_balanced = self._random_oversample(X_array, y_array)
        elif strategy == 'random_under':
            X_balanced, y_balanced = self._random_undersample(X_array, y_array)
        elif strategy == 'edited_nn':
            X_balanced, y_balanced = self._edited_nearest_neighbors(X_array, y_array)
        elif strategy == 'tomek':
            X_balanced, y_balanced = self._tomek_links(X_array, y_array)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 保持原始数据类型
        X_result = self._preserve_type(X, X_balanced)
        y_result = self._preserve_type(y, y_balanced)
        
        self.logger.info(f"Balanced data shape: {X_balanced.shape}")
        self.logger.info(f"Balanced class distribution: {dict(Counter(y_balanced))}")
        
        return X_result, y_result
    
    def _calculate_target_counts(self, class_counts: Counter) -> Counter:
        """计算目标类别数量"""
        if isinstance(self.sampling_strategy, str):
            if self.sampling_strategy == 'auto':
                # 自动平衡到多数类数量
                max_count = max(class_counts.values())
                return Counter({cls: max_count for cls in class_counts.keys()})
            elif self.sampling_strategy == 'minority':
                # 只增加少数类
                max_count = max(class_counts.values())
                target_counts = class_counts.copy()
                for cls, count in class_counts.items():
                    if count < max_count:
                        target_counts[cls] = max_count
                return target_counts
        elif isinstance(self.sampling_strategy, float):
            # 按比例调整
            max_count = max(class_counts.values())
            target_count = int(max_count * self.sampling_strategy)
            return Counter({cls: target_count for cls in class_counts.keys()})
        elif isinstance(self.sampling_strategy, dict):
            # 指定每个类的目标数量
            return Counter(self.sampling_strategy)
        
        return class_counts
    
    def _auto_select_strategy(self) -> str:
        """自动选择平衡策略"""
        total_samples = sum(self.class_counts_.values())
        num_classes = len(self.class_counts_)
        imbalance_ratio = max(self.class_counts_.values()) / min(self.class_counts_.values())
        
        if total_samples < 1000:
            return 'random_over'
        elif imbalance_ratio > 10:
            return 'smote'
        elif num_classes > 5:
            return 'random_under'
        else:
            return 'smote'
    
    def _smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE过采样"""
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)
            return smote.fit_resample(X, y)
        except ImportError:
            self.logger.warning("imbalanced-learn not available, using manual SMOTE implementation")
            return self._manual_smote(X, y)
    
    def _manual_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """手动实现SMOTE"""
        from sklearn.neighbors import NearestNeighbors
        
        X_new = []
        y_new = []
        
        for class_label, target_count in self.target_counts_.items():
            class_mask = y == class_label
            class_samples = X[class_mask]
            current_count = len(class_samples)
            
            if current_count >= target_count:
                # 不需要过采样
                indices = np.random.choice(current_count, target_count, replace=False)
                X_new.append(class_samples[indices])
                y_new.append(np.full(target_count, class_label))
            else:
                # 需要过采样
                X_new.append(class_samples)
                y_new.append(np.full(current_count, class_label))
                
                # 生成合成样本
                needed = target_count - current_count
                if current_count >= self.k_neighbors:
                    nn = NearestNeighbors(n_neighbors=self.k_neighbors)
                    nn.fit(class_samples)
                    
                    for _ in range(needed):
                        # 随机选择一个样本
                        idx = np.random.randint(current_count)
                        sample = class_samples[idx]
                        
                        # 找到最近邻
                        _, indices = nn.kneighbors([sample])
                        neighbor_idx = np.random.choice(indices[0][1:])  # 排除自己
                        neighbor = class_samples[neighbor_idx]
                        
                        # 生成合成样本
                        alpha = np.random.random()
                        synthetic = sample + alpha * (neighbor - sample)
                        
                        X_new.append([synthetic])
                        y_new.append([class_label])
                else:
                    # 样本太少，使用随机过采样
                    indices = np.random.choice(current_count, needed, replace=True)
                    X_new.append(class_samples[indices])
                    y_new.append(np.full(needed, class_label))
        
        X_balanced = np.vstack(X_new)
        y_balanced = np.hstack(y_new)
        
        # 随机打乱
        indices = np.random.permutation(len(X_balanced))
        return X_balanced[indices], y_balanced[indices]
    
    def _random_oversample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """随机过采样"""
        X_new = []
        y_new = []
        
        for class_label, target_count in self.target_counts_.items():
            class_mask = y == class_label
            class_samples = X[class_mask]
            current_count = len(class_samples)
            
            if current_count >= target_count:
                # 随机选择
                indices = np.random.choice(current_count, target_count, replace=False)
                X_new.append(class_samples[indices])
                y_new.append(np.full(target_count, class_label))
            else:
                # 随机重复采样
                indices = np.random.choice(current_count, target_count, replace=True)
                X_new.append(class_samples[indices])
                y_new.append(np.full(target_count, class_label))
        
        X_balanced = np.vstack(X_new)
        y_balanced = np.hstack(y_new)
        
        # 随机打乱
        indices = np.random.permutation(len(X_balanced))
        return X_balanced[indices], y_balanced[indices]
    
    def _random_undersample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """随机欠采样"""
        min_count = min(self.class_counts_.values())
        
        X_new = []
        y_new = []
        
        for class_label in self.class_counts_.keys():
            class_mask = y == class_label
            class_samples = X[class_mask]
            
            # 随机选择min_count个样本
            indices = np.random.choice(len(class_samples), min_count, replace=False)
            X_new.append(class_samples[indices])
            y_new.append(np.full(min_count, class_label))
        
        X_balanced = np.vstack(X_new)
        y_balanced = np.hstack(y_new)
        
        # 随机打乱
        indices = np.random.permutation(len(X_balanced))
        return X_balanced[indices], y_balanced[indices]
    
    def _edited_nearest_neighbors(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """编辑最近邻欠采样"""
        try:
            from imblearn.under_sampling import EditedNearestNeighbours
            enn = EditedNearestNeighbours(n_neighbors=self.k_neighbors)
            return enn.fit_resample(X, y)
        except ImportError:
            self.logger.warning("imbalanced-learn not available, using random undersampling")
            return self._random_undersample(X, y)
    
    def _tomek_links(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tomek链接清理"""
        try:
            from imblearn.under_sampling import TomekLinks
            tomek = TomekLinks()
            return tomek.fit_resample(X, y)
        except ImportError:
            self.logger.warning("imbalanced-learn not available, using random undersampling")
            return self._random_undersample(X, y)
    
    def get_class_distribution(self, y: Union[np.ndarray, pd.Series]) -> Dict[Any, int]:
        """获取类别分布"""
        y_array = self._to_numpy(y)
        return dict(Counter(y_array))
    
    def calculate_imbalance_ratio(self, y: Union[np.ndarray, pd.Series]) -> float:
        """计算不平衡比率"""
        distribution = self.get_class_distribution(y)
        return max(distribution.values()) / min(distribution.values())
    
    def get_params(self) -> Dict[str, Any]:
        """获取参数"""
        params = super().get_params()
        params.update({
            'strategy': self.strategy,
            'sampling_strategy': self.sampling_strategy,
            'k_neighbors': self.k_neighbors
        })
        return params