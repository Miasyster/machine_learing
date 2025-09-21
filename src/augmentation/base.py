#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强基础类

提供数据增强的基础接口和通用功能
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseAugmenter(ABC):
    """数据增强基础类"""
    
    def __init__(self, random_state: Optional[int] = None):
        """
        初始化数据增强器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'BaseAugmenter':
        """
        拟合增强器
        
        Args:
            X: 特征数据
            y: 标签数据（可选）
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, X: Union[np.ndarray, pd.DataFrame], 
                  y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[Union[np.ndarray, pd.DataFrame], 
                                                                            Optional[Union[np.ndarray, pd.Series]]]:
        """
        应用数据增强
        
        Args:
            X: 特征数据
            y: 标签数据（可选）
            
        Returns:
            增强后的(X, y)
        """
        pass
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[Union[np.ndarray, pd.DataFrame], 
                                                                                Optional[Union[np.ndarray, pd.Series]]]:
        """
        拟合并应用数据增强
        
        Args:
            X: 特征数据
            y: 标签数据（可选）
            
        Returns:
            增强后的(X, y)
        """
        return self.fit(X, y).transform(X, y)
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame], 
                       y: Optional[Union[np.ndarray, pd.Series]] = None) -> None:
        """
        验证输入数据
        
        Args:
            X: 特征数据
            y: 标签数据（可选）
        """
        if X is None:
            raise ValueError("X cannot be None")
        
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            if len(X) == 0:
                raise ValueError("X cannot be empty")
        else:
            raise TypeError("X must be numpy array or pandas DataFrame")
        
        if y is not None:
            if isinstance(y, (np.ndarray, pd.Series)):
                if len(y) != len(X):
                    raise ValueError("X and y must have the same length")
            else:
                raise TypeError("y must be numpy array or pandas Series")
    
    def _to_numpy(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        转换为numpy数组
        
        Args:
            data: 输入数据
            
        Returns:
            numpy数组
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return data
    
    def _preserve_type(self, original: Union[np.ndarray, pd.DataFrame, pd.Series], 
                      transformed: np.ndarray) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        保持原始数据类型
        
        Args:
            original: 原始数据
            transformed: 转换后的数据
            
        Returns:
            保持原始类型的数据
        """
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(transformed, columns=original.columns)
        elif isinstance(original, pd.Series):
            return pd.Series(transformed, name=original.name)
        return transformed
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取参数
        
        Returns:
            参数字典
        """
        return {'random_state': self.random_state}
    
    def set_params(self, **params) -> 'BaseAugmenter':
        """
        设置参数
        
        Args:
            **params: 参数
            
        Returns:
            self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self