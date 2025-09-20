#!/usr/bin/env python3
"""
简单的特征工程测试脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer

def test_feature_engineering():
    """测试特征工程功能"""
    print("开始测试特征工程功能...")
    
    # 创建测试数据
    np.random.seed(42)  # 设置随机种子以获得可重复的结果
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='h'),
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(150, 250, 100),
        'low': np.random.uniform(50, 150, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }
    df = pd.DataFrame(data)
    print(f"✓ 创建测试数据成功，形状: {df.shape}")
    
    # 测试特征工程初始化
    fe = FeatureEngineer(df)
    print(f"✓ 特征工程初始化成功")
    print(f"✓ 可用指标数量: {len(fe.get_available_indicators())}")
    print(f"✓ 可用指标: {fe.get_available_indicators()}")
    
    # 测试SMA计算
    try:
        result_sma = fe.calculate_sma(window=5)
        print(f"✓ SMA计算成功，新增列: {[col for col in result_sma.columns if col not in df.columns]}")
    except Exception as e:
        print(f"✗ SMA计算失败: {e}")
    
    # 测试EMA计算
    try:
        result_ema = fe.calculate_ema(window=10)
        print(f"✓ EMA计算成功，新增列: {[col for col in result_ema.columns if col not in df.columns]}")
    except Exception as e:
        print(f"✗ EMA计算失败: {e}")
    
    # 测试RSI计算
    try:
        result_rsi = fe.calculate_rsi(window=14)
        print(f"✓ RSI计算成功，新增列: {[col for col in result_rsi.columns if col not in df.columns]}")
    except Exception as e:
        print(f"✗ RSI计算失败: {e}")
    
    # 测试批量计算
    try:
        feature_configs = {
            'sma_20': {'indicator': 'SMA', 'params': {'window': 20}},
            'ema_12': {'indicator': 'EMA', 'params': {'window': 12}},
            'rsi_14': {'indicator': 'RSI', 'params': {'window': 14}}
        }
        result_batch = fe.calculate_multiple_features(df, feature_configs)
        print(f"✓ 批量计算成功，新增列: {[col for col in result_batch.columns if col not in df.columns]}")
    except Exception as e:
        print(f"✗ 批量计算失败: {e}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_feature_engineering()