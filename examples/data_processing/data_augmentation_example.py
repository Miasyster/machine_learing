#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强示例脚本

展示样本平衡和噪声注入技术的使用，包括：
1. 样本平衡（SMOTE、过采样、欠采样）
2. 噪声注入（高斯、均匀、椒盐等）
3. 时间序列噪声
4. 效果可视化和评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from augmentation.balancing import SampleBalancer
from augmentation.noise import NoiseInjector


def create_imbalanced_dataset():
    """创建不平衡数据集"""
    print("=== 创建不平衡数据集 ===")
    
    # 创建不平衡分类数据集
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],  # 不平衡比例
        random_state=42
    )
    
    print(f"原始数据集形状: {X.shape}")
    print(f"类别分布: {np.bincount(y)}")
    print(f"不平衡比例: {np.bincount(y)[0]/np.bincount(y)[1]:.2f}:1")
    
    return X, y


def demonstrate_sample_balancing():
    """演示样本平衡技术"""
    print("\n=== 样本平衡技术演示 ===")
    
    X, y = create_imbalanced_dataset()
    
    # 不同的平衡策略
    strategies = ['smote', 'random_over', 'random_under', 'edited_nn', 'tomek']
    
    results = {}
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} 策略 ---")
        
        try:
            # 创建样本平衡器
            balancer = SampleBalancer(strategy=strategy, random_state=42)
            
            # 拟合和转换
            balancer.fit(X, y)
            X_balanced, y_balanced = balancer.transform(X, y)
            
            print(f"平衡后数据形状: {X_balanced.shape}")
            print(f"平衡后类别分布: {np.bincount(y_balanced)}")
            
            # 训练模型评估效果
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42
            )
            
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)
            
            # 在原始测试集上评估
            X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            y_pred = clf.predict(X_orig_test)
            
            results[strategy] = {
                'balanced_shape': X_balanced.shape,
                'balanced_distribution': np.bincount(y_balanced),
                'test_accuracy': clf.score(X_orig_test, y_orig_test)
            }
            
            print(f"测试准确率: {results[strategy]['test_accuracy']:.4f}")
            
        except Exception as e:
            print(f"策略 {strategy} 执行失败: {e}")
            results[strategy] = {'error': str(e)}
    
    return results


def demonstrate_noise_injection():
    """演示噪声注入技术"""
    print("\n=== 噪声注入技术演示 ===")
    
    # 创建回归数据集
    X, y = make_regression(
        n_samples=500,
        n_features=5,
        noise=0.1,
        random_state=42
    )
    
    print(f"原始数据集形状: {X.shape}")
    
    # 不同的噪声类型
    noise_types = ['gaussian', 'uniform', 'salt_pepper', 'laplace']
    noise_levels = [0.05, 0.1, 0.2]
    
    results = {}
    
    # 原始模型性能
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    original_model = RandomForestRegressor(random_state=42)
    original_model.fit(X_train_scaled, y_train)
    original_pred = original_model.predict(X_test_scaled)
    original_r2 = r2_score(y_test, original_pred)
    
    print(f"原始模型 R2 分数: {original_r2:.4f}")
    
    for noise_type in noise_types:
        print(f"\n--- {noise_type.upper()} 噪声 ---")
        
        for noise_level in noise_levels:
            try:
                # 创建噪声注入器
                noise_injector = NoiseInjector(
                    noise_type=noise_type,
                    noise_level=noise_level,
                    noise_ratio=0.8,  # 80%的样本添加噪声
                    random_state=42
                )
                
                # 拟合和转换
                noise_injector.fit(X_train)
                X_train_noisy, _ = noise_injector.transform(X_train)
                
                # 训练带噪声的模型
                X_train_noisy_scaled = scaler.fit_transform(X_train_noisy)
                
                noisy_model = RandomForestRegressor(random_state=42)
                noisy_model.fit(X_train_noisy_scaled, y_train)
                noisy_pred = noisy_model.predict(X_test_scaled)
                noisy_r2 = r2_score(y_test, noisy_pred)
                
                # 计算噪声统计
                noise_stats = noise_injector.get_noise_statistics(X_train, X_train_noisy)
                
                key = f"{noise_type}_{noise_level}"
                results[key] = {
                    'noise_level': noise_level,
                    'r2_score': noisy_r2,
                    'r2_change': noisy_r2 - original_r2,
                    'noise_stats': noise_stats
                }
                
                print(f"  噪声强度 {noise_level}: R2={noisy_r2:.4f} (变化: {noisy_r2-original_r2:+.4f})")
                print(f"    SNR: {noise_stats['snr']:.2f} dB")
                
            except Exception as e:
                print(f"  噪声强度 {noise_level} 失败: {e}")
    
    return results, original_r2


def demonstrate_temporal_noise():
    """演示时间序列噪声"""
    print("\n=== 时间序列噪声演示 ===")
    
    # 创建时间序列数据
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    signal = np.sin(t) + 0.5 * np.sin(3*t) + 0.2 * np.sin(5*t)
    X_ts = signal.reshape(-1, 1)
    
    print(f"时间序列数据形状: {X_ts.shape}")
    
    # 创建噪声注入器
    noise_injector = NoiseInjector(
        noise_type='gaussian',
        noise_level=0.1,
        random_state=42
    )
    
    noise_injector.fit(X_ts)
    
    # 添加不同类型的噪声
    X_temporal = noise_injector.add_temporal_noise(X_ts, correlation=0.7)
    X_multiplicative = noise_injector.add_multiplicative_noise(X_ts)
    X_outlier = noise_injector.add_outlier_noise(X_ts, outlier_ratio=0.05)
    
    return t, signal, X_ts, X_temporal, X_multiplicative, X_outlier


def visualize_results():
    """可视化结果"""
    print("\n=== 结果可视化 ===")
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 样本平衡效果
    print("生成样本平衡效果图...")
    X, y = create_imbalanced_dataset()
    
    # 原始数据分布
    ax1 = plt.subplot(3, 4, 1)
    plt.scatter(X[y==0, 0], X[y==0, 1], alpha=0.6, label=f'类别 0 ({np.sum(y==0)})')
    plt.scatter(X[y==1, 0], X[y==1, 1], alpha=0.6, label=f'类别 1 ({np.sum(y==1)})')
    plt.title('原始不平衡数据')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SMOTE平衡后
    try:
        balancer = SampleBalancer(strategy='smote', random_state=42)
        balancer.fit(X, y)
        X_smote, y_smote = balancer.transform(X, y)
        
        ax2 = plt.subplot(3, 4, 2)
        plt.scatter(X_smote[y_smote==0, 0], X_smote[y_smote==0, 1], alpha=0.6, 
                   label=f'类别 0 ({np.sum(y_smote==0)})')
        plt.scatter(X_smote[y_smote==1, 0], X_smote[y_smote==1, 1], alpha=0.6, 
                   label=f'类别 1 ({np.sum(y_smote==1)})')
        plt.title('SMOTE平衡后')
        plt.legend()
        plt.grid(True, alpha=0.3)
    except Exception as e:
        print(f"SMOTE可视化失败: {e}")
    
    # 2. 噪声注入效果
    print("生成噪声注入效果图...")
    
    # 创建简单的2D数据用于可视化
    np.random.seed(42)
    X_simple = np.random.randn(200, 2)
    
    noise_types = ['gaussian', 'uniform', 'salt_pepper']
    
    for i, noise_type in enumerate(noise_types):
        try:
            noise_injector = NoiseInjector(
                noise_type=noise_type,
                noise_level=0.2,
                random_state=42
            )
            
            noise_injector.fit(X_simple)
            X_noisy, _ = noise_injector.transform(X_simple)
            
            ax = plt.subplot(3, 4, 3 + i)
            plt.scatter(X_simple[:, 0], X_simple[:, 1], alpha=0.5, label='原始')
            plt.scatter(X_noisy[:, 0], X_noisy[:, 1], alpha=0.5, label='加噪声')
            plt.title(f'{noise_type.title()} 噪声')
            plt.legend()
            plt.grid(True, alpha=0.3)
        except Exception as e:
            print(f"{noise_type}噪声可视化失败: {e}")
    
    # 3. 时间序列噪声
    print("生成时间序列噪声效果图...")
    try:
        t, signal, X_ts, X_temporal, X_multiplicative, X_outlier = demonstrate_temporal_noise()
        
        # 原始信号
        ax7 = plt.subplot(3, 4, 7)
        plt.plot(t, signal, 'b-', label='原始信号', linewidth=2)
        plt.title('原始时间序列')
        plt.xlabel('时间')
        plt.ylabel('幅值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 时间相关噪声
        ax8 = plt.subplot(3, 4, 8)
        plt.plot(t, signal, 'b-', label='原始信号', alpha=0.7)
        plt.plot(t, X_temporal.flatten(), 'r-', label='时间相关噪声', alpha=0.7)
        plt.title('时间相关噪声')
        plt.xlabel('时间')
        plt.ylabel('幅值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 乘性噪声
        ax9 = plt.subplot(3, 4, 9)
        plt.plot(t, signal, 'b-', label='原始信号', alpha=0.7)
        plt.plot(t, X_multiplicative.flatten(), 'g-', label='乘性噪声', alpha=0.7)
        plt.title('乘性噪声')
        plt.xlabel('时间')
        plt.ylabel('幅值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 异常值噪声
        ax10 = plt.subplot(3, 4, 10)
        plt.plot(t, signal, 'b-', label='原始信号', alpha=0.7)
        plt.plot(t, X_outlier.flatten(), 'm-', label='异常值噪声', alpha=0.7)
        plt.title('异常值噪声')
        plt.xlabel('时间')
        plt.ylabel('幅值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"时间序列可视化失败: {e}")
    
    # 4. 噪声强度对比
    print("生成噪声强度对比图...")
    try:
        ax11 = plt.subplot(3, 4, 11)
        
        # 创建不同强度的高斯噪声
        noise_levels = [0.05, 0.1, 0.2, 0.3]
        X_base = np.random.randn(100, 1)
        
        for i, level in enumerate(noise_levels):
            noise_injector = NoiseInjector(
                noise_type='gaussian',
                noise_level=level,
                random_state=42
            )
            noise_injector.fit(X_base)
            X_noisy, _ = noise_injector.transform(X_base)
            
            plt.plot(X_noisy[:50], alpha=0.7, label=f'噪声强度 {level}')
        
        plt.title('不同噪声强度对比')
        plt.xlabel('样本索引')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"噪声强度对比可视化失败: {e}")
    
    # 5. 性能对比
    print("生成性能对比图...")
    try:
        ax12 = plt.subplot(3, 4, 12)
        
        # 模拟不同增强技术的性能
        techniques = ['原始', 'SMOTE', '高斯噪声', '均匀噪声', '椒盐噪声']
        accuracies = [0.85, 0.88, 0.86, 0.87, 0.84]  # 模拟数据
        
        bars = plt.bar(techniques, accuracies, alpha=0.7, 
                      color=['blue', 'green', 'red', 'orange', 'purple'])
        plt.title('不同增强技术性能对比')
        plt.ylabel('准确率')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
    except Exception as e:
        print(f"性能对比可视化失败: {e}")
    
    plt.tight_layout()
    plt.savefig('data_augmentation_results.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存到 data_augmentation_results.png")
    
    return fig


def generate_comprehensive_report():
    """生成综合报告"""
    print("\n=== 生成综合报告 ===")
    
    report = []
    report.append("# 数据增强技术综合报告\n")
    
    # 1. 样本平衡结果
    report.append("## 1. 样本平衡技术结果\n")
    balance_results = demonstrate_sample_balancing()
    
    for strategy, result in balance_results.items():
        if 'error' not in result:
            report.append(f"### {strategy.upper()}\n")
            report.append(f"- 平衡后数据形状: {result['balanced_shape']}\n")
            report.append(f"- 类别分布: {result['balanced_distribution']}\n")
            report.append(f"- 测试准确率: {result['test_accuracy']:.4f}\n")
        else:
            report.append(f"### {strategy.upper()}\n")
            report.append(f"- 执行失败: {result['error']}\n")
    
    # 2. 噪声注入结果
    report.append("\n## 2. 噪声注入技术结果\n")
    noise_results, original_r2 = demonstrate_noise_injection()
    
    report.append(f"原始模型 R2 分数: {original_r2:.4f}\n\n")
    
    for key, result in noise_results.items():
        parts = key.split('_')
        noise_type = parts[0]
        noise_level = '_'.join(parts[1:])  # 处理可能包含多个下划线的情况
        report.append(f"### {noise_type.upper()} 噪声 (强度: {noise_level})\n")
        report.append(f"- R2 分数: {result['r2_score']:.4f}\n")
        report.append(f"- R2 变化: {result['r2_change']:+.4f}\n")
        report.append(f"- 信噪比: {result['noise_stats']['snr']:.2f} dB\n")
        report.append(f"- 均方误差: {result['noise_stats']['mse']:.6f}\n")
    
    # 3. 建议和总结
    report.append("\n## 3. 建议和总结\n")
    report.append("### 样本平衡建议\n")
    report.append("- SMOTE适用于大多数不平衡问题\n")
    report.append("- 随机过采样简单有效，但可能过拟合\n")
    report.append("- 欠采样适用于大数据集\n")
    
    report.append("\n### 噪声注入建议\n")
    report.append("- 高斯噪声最常用，适合大多数场景\n")
    report.append("- 噪声强度需要根据数据特性调整\n")
    report.append("- 适量噪声可以提高模型泛化能力\n")
    
    report.append("\n### 使用指南\n")
    report.append("1. 根据数据特性选择合适的增强技术\n")
    report.append("2. 通过交叉验证确定最佳参数\n")
    report.append("3. 监控增强后的模型性能变化\n")
    report.append("4. 结合多种技术可能获得更好效果\n")
    
    # 保存报告
    with open('data_augmentation_report.md', 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print("综合报告已保存到 data_augmentation_report.md")
    
    return ''.join(report)


def main():
    """主函数"""
    print("数据增强技术演示开始...")
    print("=" * 60)
    
    try:
        # 1. 演示样本平衡
        balance_results = demonstrate_sample_balancing()
        
        # 2. 演示噪声注入
        noise_results, original_r2 = demonstrate_noise_injection()
        
        # 3. 演示时间序列噪声
        temporal_results = demonstrate_temporal_noise()
        
        # 4. 可视化结果
        fig = visualize_results()
        
        # 5. 生成综合报告
        report = generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("数据增强技术演示完成！")
        print("\n生成的文件:")
        print("- data_augmentation_results.png (可视化结果)")
        print("- data_augmentation_report.md (综合报告)")
        
        print("\n主要发现:")
        print("1. 样本平衡技术可以有效处理不平衡数据")
        print("2. 噪声注入可以提高模型的鲁棒性")
        print("3. 不同技术适用于不同的数据场景")
        print("4. 参数调优对效果影响显著")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()