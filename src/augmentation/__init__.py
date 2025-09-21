#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强模块

提供各种数据增强技术，包括：
- 样本平衡（SMOTE、过采样、欠采样）
- 噪声注入（高斯、均匀、椒盐等）
- 数据变换和预处理
"""

from .base import BaseAugmenter
from .balancing import SampleBalancer
from .noise import NoiseInjector

__all__ = [
    'BaseAugmenter',
    'SampleBalancer', 
    'NoiseInjector'
]

__version__ = '1.0.0'