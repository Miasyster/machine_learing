"""
时间序列训练示例

展示如何使用时间序列训练器进行完整的模型训练和验证，包括：
1. 数据准备和预处理
2. 时间序列数据划分
3. 多模型训练和比较
4. 时间序列交叉验证
5. Walk-forward验证
6. 结果分析和可视化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from sklearn.neural_network import MLPRegressor

# 导入我们的训练模块
from src.training.time_series_trainer import TimeSeriesTrainer, TimeSeriesTrainingConfig
from src.training.time_series_validation import TimeSeriesConfig


def generate_time_series_data(n_samples: int = 1000, 
                            n_features: int = 5,
                            trend: bool = True,
                            seasonality: bool = True,
                            noise_level: float = 0.1) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    生成模拟的时间序列数据
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        trend: 是否包含趋势
        seasonality: 是否包含季节性
        noise_level: 噪声水平
        
    Returns:
        特征数据, 目标变量, 时间索引
    """
    # 创建时间索引
    start_date = datetime(2020, 1, 1)
    time_index = pd.date_range(start=start_date, periods=n_samples, freq='D')
    
    # 生成基础特征
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # 添加时间相关特征
    X[:, 0] = np.arange(n_samples) / n_samples  # 趋势特征
    X[:, 1] = np.sin(2 * np.pi * np.arange(n_samples) / 365.25)  # 年度季节性
    if n_features > 2:
        X[:, 2] = np.cos(2 * np.pi * np.arange(n_samples) / 365.25)  # 年度季节性
    if n_features > 3:
        X[:, 3] = np.sin(2 * np.pi * np.arange(n_samples) / 7)  # 周季节性
    
    # 生成目标变量
    y = np.zeros(n_samples)
    
    # 添加趋势
    if trend:
        y += 2 * X[:, 0]
    
    # 添加季节性
    if seasonality:
        y += 0.5 * X[:, 1]
        if n_features > 2:
            y += 0.3 * X[:, 2]
        if n_features > 3:
            y += 0.2 * X[:, 3]
    
    # 添加特征的线性组合
    y += 0.5 * X[:, -1]  # 最后一个特征
    
    # 添加噪声
    y += noise_level * np.random.randn(n_samples)
    
    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names, index=time_index)
    y_series = pd.Series(y, index=time_index, name='target')
    
    return X_df, y_series, time_index


def create_models() -> Dict[str, BaseEstimator]:
    """创建模型字典"""
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0, random_state=42),
        'lasso': Lasso(alpha=0.1, random_state=42),
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
        'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'mlp': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    return models


def plot_data_splits(X: pd.DataFrame, y: pd.Series, splits_info: Dict[str, Any]):
    """可视化数据划分"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 绘制原始数据
    axes[0].plot(X.index, y, label='Target Variable', alpha=0.7)
    axes[0].set_title('Time Series Data')
    axes[0].set_ylabel('Target Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制数据划分
    n_samples = len(X)
    train_end = int(n_samples * 0.6)
    val_end = int(n_samples * 0.8)
    
    axes[1].plot(X.index[:train_end], y[:train_end], label='Training Set', color='blue')
    axes[1].plot(X.index[train_end:val_end], y[train_end:val_end], label='Validation Set', color='orange')
    axes[1].plot(X.index[val_end:], y[val_end:], label='Test Set', color='red')
    
    axes[1].axvline(x=X.index[train_end], color='blue', linestyle='--', alpha=0.7)
    axes[1].axvline(x=X.index[val_end], color='orange', linestyle='--', alpha=0.7)
    
    axes[1].set_title('Data Splits')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Target Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison(comparison_df: pd.DataFrame):
    """可视化模型比较结果"""
    # 选择主要指标进行比较
    metrics_to_plot = ['test_r2', 'test_mse', 'cv_mean_r2', 'wf_mean_score']
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
    
    if not available_metrics:
        print("没有可用的指标进行可视化")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics):
        if i >= 4:
            break
            
        ax = axes[i]
        data = comparison_df.set_index('model')[metric].sort_values(ascending=False)
        
        bars = ax.bar(range(len(data)), data.values)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha='right')
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    # 隐藏多余的子图
    for i in range(len(available_metrics), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_predictions(trainer: TimeSeriesTrainer, X_test: pd.DataFrame, y_test: pd.Series):
    """可视化预测结果"""
    best_model_name = trainer.get_best_model_name()
    predictions = trainer.predict(X_test.values, best_model_name)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 预测 vs 实际
    axes[0].plot(X_test.index, y_test, label='Actual', alpha=0.7)
    axes[0].plot(X_test.index, predictions, label=f'Predicted ({best_model_name})', alpha=0.7)
    axes[0].set_title('Predictions vs Actual (Test Set)')
    axes[0].set_ylabel('Target Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 残差图
    residuals = y_test - predictions
    axes[1].plot(X_test.index, residuals, alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_title('Residuals')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Residual')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("🚀 开始时间序列训练示例")
    print("=" * 50)
    
    # 1. 生成数据
    print("📊 生成时间序列数据...")
    X, y, time_index = generate_time_series_data(
        n_samples=1000,
        n_features=5,
        trend=True,
        seasonality=True,
        noise_level=0.1
    )
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"时间范围: {time_index[0]} 到 {time_index[-1]}")
    
    # 2. 配置训练参数
    print("\n⚙️ 配置训练参数...")
    
    # 时间序列配置
    ts_config = TimeSeriesConfig(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        n_splits=5,
        validation_strategy='walk_forward',
        expanding_window=True,
        gap=0
    )
    
    # 训练配置
    training_config = TimeSeriesTrainingConfig(
        time_series_config=ts_config,
        use_time_series_cv=True,
        use_walk_forward=True,
        regression_metrics=['mse', 'mae', 'r2', 'rmse'],
        save_predictions=True,
        verbose=True
    )
    
    # 3. 创建模型
    print("\n🤖 创建模型...")
    models = create_models()
    print(f"创建了 {len(models)} 个模型: {list(models.keys())}")
    
    # 4. 创建训练器并训练
    print("\n🏋️ 开始训练...")
    trainer = TimeSeriesTrainer(config=training_config)
    
    # 添加模型
    for name, model in models.items():
        trainer.add_model(name, model)
    
    # 训练
    trainer.fit(X, y, time_index)
    
    # 5. 获取结果
    print("\n📈 分析结果...")
    comparison_df = trainer.get_model_comparison()
    print("\n模型比较结果:")
    print(comparison_df.round(4))
    
    # 6. 获取最佳模型
    best_model = trainer.get_best_model_name()
    print(f"\n🏆 最佳模型: {best_model}")
    
    # 7. 可视化结果
    print("\n📊 生成可视化...")
    
    # 数据划分可视化
    plot_data_splits(X, y, trainer.data_splits['split_info'])
    
    # 模型比较可视化
    plot_model_comparison(comparison_df)
    
    # 预测结果可视化
    X_test = trainer.data_splits['X_test']
    y_test = trainer.data_splits['y_test']
    
    # 转换为DataFrame以便可视化
    test_index = time_index[int(len(time_index) * 0.8):]
    X_test_df = pd.DataFrame(X_test, columns=X.columns, index=test_index)
    y_test_series = pd.Series(y_test, index=test_index)
    
    plot_predictions(trainer, X_test_df, y_test_series)
    
    # 8. 保存结果
    print("\n💾 保存结果...")
    save_dir = "results/time_series_training"
    trainer.save_models(save_dir)
    
    # 保存比较结果
    os.makedirs(save_dir, exist_ok=True)
    comparison_df.to_csv(f"{save_dir}/detailed_comparison.csv", index=False)
    
    print(f"结果已保存到: {save_dir}")
    
    # 9. 详细分析
    print("\n🔍 详细分析:")
    print("-" * 30)
    
    for model_name, result in trainer.training_results.items():
        print(f"\n模型: {model_name}")
        print(f"  训练时间: {result.training_time:.2f}秒")
        print(f"  测试集R²: {result.test_scores.get('r2', 'N/A'):.4f}")
        print(f"  测试集RMSE: {result.test_scores.get('rmse', 'N/A'):.4f}")
        
        # 交叉验证结果
        if model_name in trainer.cv_results:
            cv_r2 = trainer.cv_results[model_name]['mean_scores'].get('r2', 'N/A')
            cv_r2_std = trainer.cv_results[model_name]['std_scores'].get('r2', 'N/A')
            print(f"  CV R²: {cv_r2:.4f} (±{cv_r2_std:.4f})")
        
        # Walk-forward结果
        if model_name in trainer.walk_forward_results:
            wf_score = trainer.walk_forward_results[model_name]['mean_score']
            wf_std = trainer.walk_forward_results[model_name]['std_score']
            print(f"  Walk-forward得分: {wf_score:.4f} (±{wf_std:.4f})")
    
    print("\n✅ 时间序列训练示例完成！")


if __name__ == "__main__":
    main()