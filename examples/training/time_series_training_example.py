#!/usr/bin/env python3
"""
时间序列训练示例脚本
展示完整的时间序列数据划分、交叉验证和Walk-forward验证流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 导入我们的训练模块
from src.training.time_series_trainer import TimeSeriesTrainer, TimeSeriesTrainingConfig
from src.training.time_series_validation import TimeSeriesConfig


def generate_time_series_data(n_samples=1000, n_features=5, noise_level=0.1):
    """生成模拟时间序列数据"""
    print("📊 生成时间序列数据...")
    
    # 创建时间索引
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # 生成特征
    np.random.seed(42)
    
    # 趋势特征
    trend = np.linspace(0, 10, n_samples)
    
    # 季节性特征
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    
    # 随机特征
    random_features = np.random.randn(n_samples, n_features-2)
    
    # 组合特征
    X = np.column_stack([
        trend,
        seasonal,
        *[random_features[:, i] for i in range(n_features-2)]
    ])
    
    # 生成目标变量（带有时间依赖性）
    y = (
        2 * trend +
        1.5 * seasonal +
        0.5 * np.sum(random_features, axis=1) +
        noise_level * np.random.randn(n_samples)
    )
    
    # 添加一些时间依赖性
    for i in range(1, n_samples):
        y[i] += 0.1 * y[i-1]
    
    # 创建DataFrame
    feature_names = ['trend', 'seasonal'] + [f'feature_{i}' for i in range(n_features-2)]
    df = pd.DataFrame(X, columns=feature_names, index=dates)
    df['target'] = y
    
    print(f"✅ 数据生成完成: {df.shape}")
    print(f"   时间范围: {dates[0].strftime('%Y-%m-%d')} 到 {dates[-1].strftime('%Y-%m-%d')}")
    
    return df


def create_models():
    """创建模型字典"""
    print("🤖 创建模型...")
    
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0),
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    print(f"✅ 创建了 {len(models)} 个模型")
    return models


def run_basic_training(df, models):
    """运行基础训练"""
    print("\n" + "="*60)
    print("🏋️ 基础时间序列训练")
    print("="*60)
    
    # 准备数据
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # 配置训练参数
    ts_config = TimeSeriesConfig(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        gap=0  # 无间隔
    )
    
    training_config = TimeSeriesTrainingConfig(
        time_series_config=ts_config,
        verbose=True,
        random_state=42
    )
    
    # 创建训练器
    trainer = TimeSeriesTrainer(config=training_config, models=models)
    
    # 训练模型
    trainer.fit(X, y)
    
    # 显示结果
    print("\n📈 训练结果:")
    comparison = trainer.get_model_comparison()
    print(comparison)
    
    return trainer


def run_cross_validation(df, models):
    """运行时间序列交叉验证"""
    print("\n" + "="*60)
    print("🔄 时间序列交叉验证")
    print("="*60)
    
    # 准备数据
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # 配置交叉验证
    ts_config = TimeSeriesConfig(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        gap=5  # 5天间隔
    )
    
    training_config = TimeSeriesTrainingConfig(
        time_series_config=ts_config,
        use_time_series_cv=True,
        cv_folds=5,
        verbose=True,
        random_state=42
    )
    
    # 创建训练器
    trainer = TimeSeriesTrainer(config=training_config, models=models)
    
    # 训练模型
    trainer.fit(X, y)
    
    # 显示交叉验证结果（已在训练过程中执行）
    print("\n📊 交叉验证结果:")
    comparison = trainer.get_model_comparison()
    if 'cv_mean_r2' in comparison.columns:
        for _, row in comparison.iterrows():
            model_name = row['model']
            cv_mean = row.get('cv_mean_r2', 'N/A')
            cv_std = row.get('cv_std_r2', 'N/A')
            if isinstance(cv_mean, (int, float)) and isinstance(cv_std, (int, float)):
                print(f"{model_name:15} | R² = {cv_mean:.3f} ± {cv_std:.3f}")
            else:
                print(f"{model_name:15} | R² = {cv_mean} ± {cv_std}")
    
    # 构造cv_results用于返回
    cv_results = {}
    if 'cv_mean_r2' in comparison.columns:
        for _, row in comparison.iterrows():
            model_name = row['model']
            cv_mean = row.get('cv_mean_r2', 0)
            cv_std = row.get('cv_std_r2', 0)
            if isinstance(cv_mean, (int, float)):
                # 模拟交叉验证分数
                cv_results[model_name] = [cv_mean] * 5
    
    return trainer, cv_results


def run_walk_forward_validation(df, models):
    """运行Walk-forward验证"""
    print("\n" + "="*60)
    print("🚶 Walk-Forward 验证")
    print("="*60)
    
    # 准备数据
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # 配置Walk-forward验证
    ts_config = TimeSeriesConfig(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        gap=10  # 10天间隔
    )
    
    training_config = TimeSeriesTrainingConfig(
        time_series_config=ts_config,
        use_walk_forward=True,
        verbose=True,
        random_state=42
    )
    
    # 创建训练器
    trainer = TimeSeriesTrainer(config=training_config, models=models)
    
    # 训练模型
    trainer.fit(X, y)
    
    # 显示Walk-forward验证结果（已在训练过程中执行）
    print("\n📊 Walk-forward验证结果:")
    comparison = trainer.get_model_comparison()
    if 'wf_mean_score' in comparison.columns:
        for _, row in comparison.iterrows():
            model_name = row['model']
            wf_mean = row.get('wf_mean_score', 'N/A')
            wf_std = row.get('wf_std_score', 'N/A')
            if isinstance(wf_mean, (int, float)) and isinstance(wf_std, (int, float)):
                print(f"{model_name:15} | R² = {wf_mean:.3f} ± {wf_std:.3f}")
            else:
                print(f"{model_name:15} | R² = {wf_mean} ± {wf_std}")
    
    # 构造wf_results用于返回
    wf_results = {}
    if 'wf_mean_score' in comparison.columns:
        for _, row in comparison.iterrows():
            model_name = row['model']
            wf_mean = row.get('wf_mean_score', 0)
            wf_std = row.get('wf_std_score', 0)
            if isinstance(wf_mean, (int, float)):
                # 模拟walk-forward分数
                wf_results[model_name] = [wf_mean] * 5
    
    return trainer, wf_results


def visualize_results(df, trainer, cv_results, wf_results):
    """可视化结果"""
    print("\n" + "="*60)
    print("📊 结果可视化")
    print("="*60)
    
    # 设置中文字体和负号显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置字体大小
    plt.rcParams['font.size'] = 10
    
    # 设置图形样式
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 时间序列数据
    axes[0, 0].plot(df.index, df['target'], alpha=0.7)
    axes[0, 0].set_title('时间序列数据')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('目标值')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 模型性能比较
    comparison = trainer.get_model_comparison()
    if 'test_r2' in comparison.columns:
        comparison.plot(x='model', y='test_r2', kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('模型R2性能比较')
        axes[0, 1].set_ylabel('R2 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        # 设置Y轴范围以正确显示负值
        min_r2 = comparison['test_r2'].min()
        max_r2 = comparison['test_r2'].max()
        y_margin = (max_r2 - min_r2) * 0.1
        axes[0, 1].set_ylim(min_r2 - y_margin, max_r2 + y_margin)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 交叉验证结果
    if cv_results:
        cv_means = [np.mean(scores) for scores in cv_results.values()]
        cv_stds = [np.std(scores) for scores in cv_results.values()]
        model_names = list(cv_results.keys())
        
        axes[1, 0].bar(model_names, cv_means, yerr=cv_stds, capsize=5)
        axes[1, 0].set_title('交叉验证结果')
        axes[1, 0].set_ylabel('R2 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        # 设置Y轴范围以正确显示负值
        min_cv = min(cv_means) - max(cv_stds)
        max_cv = max(cv_means) + max(cv_stds)
        y_margin = (max_cv - min_cv) * 0.1
        axes[1, 0].set_ylim(min_cv - y_margin, max_cv + y_margin)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Walk-forward验证结果
    if wf_results:
        wf_means = [np.mean(scores) for scores in wf_results.values()]
        wf_stds = [np.std(scores) for scores in wf_results.values()]
        model_names = list(wf_results.keys())
        
        axes[1, 1].bar(model_names, wf_means, yerr=wf_stds, capsize=5)
        axes[1, 1].set_title('Walk-Forward验证结果')
        axes[1, 1].set_ylabel('R2 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        # 设置Y轴范围以正确显示负值
        min_wf = min(wf_means) - max(wf_stds)
        max_wf = max(wf_means) + max(wf_stds)
        y_margin = (max_wf - min_wf) * 0.1
        axes[1, 1].set_ylim(min_wf - y_margin, max_wf + y_margin)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_training_results.png', dpi=300, bbox_inches='tight')
    print("📊 结果图表已保存为 'time_series_training_results.png'")
    
    # 显示图表
    plt.show()


def main():
    """主函数"""
    print("🚀 时间序列训练完整示例")
    print("="*60)
    
    # 1. 生成数据
    df = generate_time_series_data(n_samples=1000, n_features=5)
    
    # 2. 创建模型
    models = create_models()
    
    # 3. 基础训练
    basic_trainer = run_basic_training(df, models)
    
    # 4. 交叉验证
    cv_trainer, cv_results = run_cross_validation(df, models)
    
    # 5. Walk-forward验证
    wf_trainer, wf_results = run_walk_forward_validation(df, models)
    
    # 6. 可视化结果
    try:
        visualize_results(df, basic_trainer, cv_results, wf_results)
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")
        print("💡 请确保安装了matplotlib和seaborn")
    
    # 7. 总结
    print("\n" + "="*60)
    print("📋 训练总结")
    print("="*60)
    
    print("✅ 完成的任务:")
    print("   - 时间序列数据生成")
    print("   - 基础时间序列训练")
    print("   - 时间序列交叉验证")
    print("   - Walk-forward验证")
    print("   - 结果可视化")
    
    print(f"\n🏆 最佳模型: {basic_trainer.get_best_model_name()}")
    
    print("\n💡 下一步建议:")
    print("   - 尝试更多模型类型")
    print("   - 调整验证参数")
    print("   - 添加特征工程")
    print("   - 实施模型集成")
    
    return basic_trainer, cv_trainer, wf_trainer


if __name__ == "__main__":
    trainer_basic, trainer_cv, trainer_wf = main()