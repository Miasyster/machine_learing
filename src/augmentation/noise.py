#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
噪声注入模块

实现各种噪声注入技术，包括：
- 高斯噪声
- 均匀噪声
- 椒盐噪声
- 泊松噪声
- 拉普拉斯噪声
- 时间序列噪声
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any, List
import warnings

from .base import BaseAugmenter


class NoiseInjector(BaseAugmenter):
    """噪声注入器"""
    
    def __init__(self, 
                 noise_type: str = 'gaussian',
                 noise_level: float = 0.1,
                 noise_ratio: float = 1.0,
                 feature_wise: bool = False,
                 preserve_range: bool = True,
                 random_state: Optional[int] = None):
        """
        初始化噪声注入器
        
        Args:
            noise_type: 噪声类型 ('gaussian', 'uniform', 'salt_pepper', 'poisson', 'laplace')
            noise_level: 噪声强度
            noise_ratio: 添加噪声的样本比例
            feature_wise: 是否按特征独立添加噪声
            preserve_range: 是否保持原始数据范围
            random_state: 随机种子
        """
        super().__init__(random_state)
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.noise_ratio = noise_ratio
        self.feature_wise = feature_wise
        self.preserve_range = preserve_range
        
        self.data_min_ = None
        self.data_max_ = None
        self.data_std_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'NoiseInjector':
        """
        拟合噪声注入器
        
        Args:
            X: 特征数据
            y: 标签数据（可选）
            
        Returns:
            self
        """
        self._validate_input(X, y)
        
        X_array = self._to_numpy(X)
        
        if self.preserve_range:
            self.data_min_ = np.min(X_array, axis=0)
            self.data_max_ = np.max(X_array, axis=0)
        
        if self.feature_wise:
            self.data_std_ = np.std(X_array, axis=0)
        else:
            self.data_std_ = np.std(X_array)
        
        self.logger.info(f"Fitted noise injector with {self.noise_type} noise")
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame], 
                  y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[Union[np.ndarray, pd.DataFrame], 
                                                                            Optional[Union[np.ndarray, pd.Series]]]:
        """
        应用噪声注入
        
        Args:
            X: 特征数据
            y: 标签数据（可选）
            
        Returns:
            添加噪声后的(X, y)
        """
        self._validate_input(X, y)
        
        X_array = self._to_numpy(X)
        
        # 选择要添加噪声的样本
        n_samples = len(X_array)
        n_noisy_samples = int(n_samples * self.noise_ratio)
        noisy_indices = np.random.choice(n_samples, n_noisy_samples, replace=False)
        
        X_noisy = X_array.copy()
        
        # 根据噪声类型添加噪声
        if self.noise_type == 'gaussian':
            noise = self._generate_gaussian_noise(X_array[noisy_indices])
        elif self.noise_type == 'uniform':
            noise = self._generate_uniform_noise(X_array[noisy_indices])
        elif self.noise_type == 'salt_pepper':
            noise = self._generate_salt_pepper_noise(X_array[noisy_indices])
        elif self.noise_type == 'poisson':
            noise = self._generate_poisson_noise(X_array[noisy_indices])
        elif self.noise_type == 'laplace':
            noise = self._generate_laplace_noise(X_array[noisy_indices])
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        X_noisy[noisy_indices] = X_array[noisy_indices] + noise
        
        # 保持原始数据范围
        if self.preserve_range and self.data_min_ is not None and self.data_max_ is not None:
            X_noisy = np.clip(X_noisy, self.data_min_, self.data_max_)
        
        # 保持原始数据类型
        X_result = self._preserve_type(X, X_noisy)
        
        self.logger.info(f"Added {self.noise_type} noise to {n_noisy_samples}/{n_samples} samples")
        
        return X_result, y
    
    def _generate_gaussian_noise(self, X: np.ndarray) -> np.ndarray:
        """生成高斯噪声"""
        if self.feature_wise:
            noise = np.random.normal(0, self.noise_level * self.data_std_, X.shape)
        else:
            noise = np.random.normal(0, self.noise_level * self.data_std_, X.shape)
        return noise
    
    def _generate_uniform_noise(self, X: np.ndarray) -> np.ndarray:
        """生成均匀噪声"""
        if self.feature_wise:
            noise_scale = self.noise_level * self.data_std_
            noise = np.random.uniform(-noise_scale, noise_scale, X.shape)
        else:
            noise_scale = self.noise_level * self.data_std_
            noise = np.random.uniform(-noise_scale, noise_scale, X.shape)
        return noise
    
    def _generate_salt_pepper_noise(self, X: np.ndarray) -> np.ndarray:
        """生成椒盐噪声"""
        noise = np.zeros_like(X)
        
        # 盐噪声（设为最大值）
        salt_mask = np.random.random(X.shape) < self.noise_level / 2
        if self.data_max_ is not None:
            if len(self.data_max_.shape) == 0:  # 标量
                noise[salt_mask] = self.data_max_ - X[salt_mask]
            else:  # 向量
                for i in range(X.shape[1]):
                    mask_i = salt_mask[:, i] if len(X.shape) > 1 else salt_mask
                    noise[mask_i, i] = self.data_max_[i] - X[mask_i, i]
        else:
            noise[salt_mask] = np.max(X) - X[salt_mask]
        
        # 椒噪声（设为最小值）
        pepper_mask = np.random.random(X.shape) < self.noise_level / 2
        if self.data_min_ is not None:
            if len(self.data_min_.shape) == 0:  # 标量
                noise[pepper_mask] = self.data_min_ - X[pepper_mask]
            else:  # 向量
                for i in range(X.shape[1]):
                    mask_i = pepper_mask[:, i] if len(X.shape) > 1 else pepper_mask
                    noise[mask_i, i] = self.data_min_[i] - X[mask_i, i]
        else:
            noise[pepper_mask] = np.min(X) - X[pepper_mask]
        
        return noise
    
    def _generate_poisson_noise(self, X: np.ndarray) -> np.ndarray:
        """生成泊松噪声"""
        # 将数据缩放到正值范围
        X_scaled = X - np.min(X) + 1
        
        # 生成泊松噪声
        noise_samples = np.random.poisson(X_scaled * self.noise_level)
        noise = noise_samples - X_scaled * self.noise_level
        
        return noise
    
    def _generate_laplace_noise(self, X: np.ndarray) -> np.ndarray:
        """生成拉普拉斯噪声"""
        if self.feature_wise:
            scale = self.noise_level * self.data_std_
            noise = np.random.laplace(0, scale, X.shape)
        else:
            scale = self.noise_level * self.data_std_
            noise = np.random.laplace(0, scale, X.shape)
        return noise
    
    def add_temporal_noise(self, X: Union[np.ndarray, pd.DataFrame], 
                          correlation: float = 0.5) -> Union[np.ndarray, pd.DataFrame]:
        """
        添加时间相关噪声
        
        Args:
            X: 时间序列数据
            correlation: 时间相关性
            
        Returns:
            添加时间相关噪声的数据
        """
        X_array = self._to_numpy(X)
        n_samples, n_features = X_array.shape
        
        # 生成时间相关噪声
        noise = np.zeros_like(X_array)
        
        for i in range(n_features):
            # 第一个时间点的噪声
            noise[0, i] = np.random.normal(0, self.noise_level * self.data_std_)
            
            # 后续时间点的噪声（带相关性）
            for t in range(1, n_samples):
                prev_noise = noise[t-1, i]
                new_noise = np.random.normal(0, self.noise_level * self.data_std_)
                noise[t, i] = correlation * prev_noise + np.sqrt(1 - correlation**2) * new_noise
        
        X_noisy = X_array + noise
        
        # 保持原始数据范围
        if self.preserve_range and self.data_min_ is not None and self.data_max_ is not None:
            X_noisy = np.clip(X_noisy, self.data_min_, self.data_max_)
        
        return self._preserve_type(X, X_noisy)
    
    def add_multiplicative_noise(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        添加乘性噪声
        
        Args:
            X: 输入数据
            
        Returns:
            添加乘性噪声的数据
        """
        X_array = self._to_numpy(X)
        
        # 生成乘性噪声因子
        noise_factor = np.random.normal(1, self.noise_level, X_array.shape)
        X_noisy = X_array * noise_factor
        
        # 保持原始数据范围
        if self.preserve_range and self.data_min_ is not None and self.data_max_ is not None:
            X_noisy = np.clip(X_noisy, self.data_min_, self.data_max_)
        
        return self._preserve_type(X, X_noisy)
    
    def add_outlier_noise(self, X: Union[np.ndarray, pd.DataFrame], 
                         outlier_ratio: float = 0.05) -> Union[np.ndarray, pd.DataFrame]:
        """
        添加异常值噪声
        
        Args:
            X: 输入数据
            outlier_ratio: 异常值比例
            
        Returns:
            添加异常值噪声的数据
        """
        X_array = self._to_numpy(X)
        n_samples = len(X_array)
        n_outliers = int(n_samples * outlier_ratio)
        
        X_noisy = X_array.copy()
        
        # 随机选择异常值位置
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        
        for idx in outlier_indices:
            # 生成极端值
            if self.data_min_ is not None and self.data_max_ is not None:
                data_range = self.data_max_ - self.data_min_
                # 随机选择极端方向
                direction = np.random.choice([-1, 1])
                if direction == -1:
                    extreme_value = self.data_min_ - data_range * self.noise_level
                else:
                    extreme_value = self.data_max_ + data_range * self.noise_level
                X_noisy[idx] = extreme_value
            else:
                # 使用标准差的倍数
                extreme_multiplier = np.random.choice([-1, 1]) * (3 + self.noise_level * 5)
                if hasattr(self.data_std_, '__len__'):
                    X_noisy[idx] = X_array[idx] + extreme_multiplier * self.data_std_
                else:
                    X_noisy[idx] = X_array[idx] + extreme_multiplier * self.data_std_
        
        return self._preserve_type(X, X_noisy)
    
    def get_noise_statistics(self, X_original: Union[np.ndarray, pd.DataFrame], 
                           X_noisy: Union[np.ndarray, pd.DataFrame]) -> Dict[str, float]:
        """
        计算噪声统计信息
        
        Args:
            X_original: 原始数据
            X_noisy: 添加噪声后的数据
            
        Returns:
            噪声统计信息
        """
        X_orig = self._to_numpy(X_original)
        X_noise = self._to_numpy(X_noisy)
        
        noise = X_noise - X_orig
        
        stats = {
            'noise_mean': np.mean(noise),
            'noise_std': np.std(noise),
            'noise_max': np.max(np.abs(noise)),
            'snr': 20 * np.log10(np.std(X_orig) / np.std(noise)) if np.std(noise) > 0 else float('inf'),
            'mse': np.mean(noise ** 2),
            'mae': np.mean(np.abs(noise))
        }
        
        return stats
    
    def get_params(self) -> Dict[str, Any]:
        """获取参数"""
        params = super().get_params()
        params.update({
            'noise_type': self.noise_type,
            'noise_level': self.noise_level,
            'noise_ratio': self.noise_ratio,
            'feature_wise': self.feature_wise,
            'preserve_range': self.preserve_range
        })
        return params