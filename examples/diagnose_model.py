#!/usr/bin/env python3
"""
模型诊断脚本 - 分析机器学习模型的预测问题
"""

import numpy as np
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.supervised import ClassificationTrainer
from src.features.feature_engineering import calculate_all_features

def generate_sample_data(n_days=1000):
    """生成模拟市场数据"""
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    np.random.seed(42)
    
    # 生成价格数据
    returns = np.random.normal(0.0005, 0.02, n_days)
    trend = np.linspace(0, 0.3, n_days)
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 252)
    
    log_returns = returns + trend/n_days + seasonal/n_days
    prices = 100 * np.exp(np.cumsum(log_returns))
    
    # 生成OHLCV数据
    data = []
    for i, price in enumerate(prices):
        daily_volatility = np.random.uniform(0.005, 0.03)
        high = price * (1 + daily_volatility)
        low = price * (1 - daily_volatility)
        open_price = price * np.random.uniform(0.99, 1.01)
        close_price = price
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)

def prepare_training_data(data):
    """准备训练数据"""
    # 计算技术指标
    features = calculate_all_features(data)
    
    # 添加额外特征
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    features['volatility'] = features['returns'].rolling(20).std()
    features['price_position'] = (data['close'] - data['close'].rolling(20).min()) / \
                                (data['close'].rolling(20).max() - data['close'].rolling(20).min())
    features['volume_sma'] = data['volume'].rolling(20).mean()
    features['volume_ratio'] = data['volume'] / features['volume_sma']
    
    # 创建目标变量（未来收益率分类）
    future_returns = data['close'].shift(-5) / data['close'] - 1
    target = pd.cut(future_returns, 
                   bins=[-np.inf, -0.01, 0.01, np.inf], 
                   labels=[0, 1, 2])
    
    # 删除包含NaN的行
    valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
    features_clean = features[valid_idx]
    target_clean = target[valid_idx]
    
    return features_clean, target_clean

def main():
    print("=== 模型诊断脚本 ===")
    
    # 1. 生成数据
    print("1. 生成模拟数据...")
    data = generate_sample_data()
    print(f"   数据长度: {len(data)}")
    print(f"   价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 2. 准备特征
    print("2. 准备特征...")
    features, target = prepare_training_data(data)
    print(f"   特征数量: {features.shape[1]}")
    print(f"   样本数量: {len(features)}")
    print(f"   目标分布: {dict(target.value_counts())}")
    
    # 3. 训练模型
    print("3. 训练模型...")
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
    
    trainer = ClassificationTrainer(
        models=['random_forest'],
        param_grids={
            'random_forest': {
                'n_estimators': [200],
                'max_depth': [15],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'random_state': [42]
            }
        }
    )
    
    trainer.fit(X_train.values, y_train.values)
    model = trainer.best_model_
    
    train_score = model.score(X_train.values, y_train.values)
    test_score = model.score(X_test.values, y_test.values)
    
    print(f"   训练准确率: {train_score:.4f}")
    print(f"   测试准确率: {test_score:.4f}")
    
    # 4. 分析预测
    print("4. 分析预测...")
    predictions = model.predict(X_test.values)
    probabilities = model.predict_proba(X_test.values)
    
    print(f"   预测分布: {dict(pd.Series(predictions).value_counts())}")
    print(f"   实际分布: {dict(y_test.value_counts())}")
    
    # 分析概率分布
    print("5. 分析概率分布...")
    max_probs = np.max(probabilities, axis=1)
    print(f"   最高概率统计:")
    print(f"     均值: {max_probs.mean():.4f}")
    print(f"     中位数: {np.median(max_probs):.4f}")
    print(f"     最小值: {max_probs.min():.4f}")
    print(f"     最大值: {max_probs.max():.4f}")
    
    # 分析不同阈值下的信号分布
    print("6. 分析不同阈值下的信号分布...")
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        buy_signals = np.sum(probabilities[:, 2] > threshold)  # 类别2为上涨
        sell_signals = np.sum(probabilities[:, 2] < (1 - threshold))
        hold_signals = len(probabilities) - buy_signals - sell_signals
        
        print(f"   阈值 {threshold}: 买入={buy_signals}, 卖出={sell_signals}, 持有={hold_signals}")
    
    print("\n=== 诊断完成 ===")

if __name__ == "__main__":
    main()