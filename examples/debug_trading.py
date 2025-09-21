"""
调试交易问题的简化脚本
"""
import numpy as np
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.supervised import ClassificationTrainer
from sklearn.ensemble import RandomForestClassifier

def generate_sample_data(n_days=200):
    """生成样本数据"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # 生成价格数据
    price = 100
    prices = []
    volumes = []
    
    for i in range(n_days):
        # 随机游走价格
        change = np.random.normal(0, 0.02)
        price *= (1 + change)
        prices.append(price)
        volumes.append(np.random.randint(1000, 10000))
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    return data

def prepare_features_and_target(data):
    """准备特征和目标变量"""
    # 简单的技术指标
    data['sma_5'] = data['close'].rolling(5).mean()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['rsi'] = 50 + np.random.normal(0, 15, len(data))  # 简化的RSI
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
    
    # 特征
    features = data[['sma_5', 'sma_20', 'rsi', 'volume_ratio']].copy()
    
    # 目标变量 - 使用分位数方法
    future_returns = data['close'].shift(-3) / data['close'] - 1
    q33 = future_returns.quantile(0.33)
    q67 = future_returns.quantile(0.67)
    
    print(f"目标分位数: 33%={q33:.4f}, 67%={q67:.4f}")
    
    target = pd.cut(future_returns, 
                   bins=[-np.inf, q33, q67, np.inf], 
                   labels=[0, 1, 2])
    
    # 删除NaN
    valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
    features = features[valid_idx]
    target = target[valid_idx]
    
    print(f"目标分布:")
    print(target.value_counts().sort_index())
    
    return features, target

def test_model_training(features, target):
    """测试模型训练"""
    print("\n=== 测试模型训练 ===")
    
    # 分割数据
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = target[:split_idx], target[split_idx:]
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"训练集目标分布:")
    print(y_train.value_counts().sort_index())
    
    # 训练模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 预测
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)
    
    # 评估
    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()
    
    print(f"训练准确率: {train_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    
    print(f"\n测试集预测分布:")
    unique, counts = np.unique(test_pred, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  类别 {u}: {c} 个")
    
    # 分析概率分布
    print(f"\n测试集概率分析:")
    for i in range(test_proba.shape[1]):
        probs = test_proba[:, i]
        print(f"  类别 {i} 概率: 均值={probs.mean():.4f}, 最大={probs.max():.4f}")
    
    return model, test_proba

def test_signal_generation(model, features, thresholds=[0.3, 0.4, 0.5, 0.6]):
    """测试信号生成"""
    print("\n=== 测试信号生成 ===")
    
    # 使用最后50个样本测试
    test_features = features[-50:]
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)
    
    for threshold in thresholds:
        signals = []
        for i in range(len(test_features)):
            pred = predictions[i]
            probs = probabilities[i]
            
            if len(probs) >= 3:
                buy_prob = probs[2]  # 上涨概率
                sell_prob = probs[0]  # 下跌概率
                
                if buy_prob > threshold:
                    signal = 1  # 买入
                elif sell_prob > threshold:
                    signal = -1  # 卖出
                else:
                    signal = 0  # 持有
            else:
                signal = 0
            
            signals.append(signal)
        
        signal_counts = np.unique(signals, return_counts=True)
        print(f"阈值 {threshold}: ", end="")
        for sig, count in zip(signal_counts[0], signal_counts[1]):
            if sig == 1:
                print(f"买入={count} ", end="")
            elif sig == -1:
                print(f"卖出={count} ", end="")
            else:
                print(f"持有={count} ", end="")
        print()

def main():
    print("=== 调试交易问题 ===")
    
    # 生成数据
    data = generate_sample_data(200)
    print(f"生成数据: {len(data)} 天")
    
    # 准备特征和目标
    features, target = prepare_features_and_target(data)
    print(f"有效样本: {len(features)}")
    
    # 测试模型训练
    model, test_proba = test_model_training(features, target)
    
    # 测试信号生成
    test_signal_generation(model, features)

if __name__ == "__main__":
    main()