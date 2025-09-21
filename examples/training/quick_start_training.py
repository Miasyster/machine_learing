"""
快速开始训练脚本

提供简单易用的时间序列模型训练入口
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

# 导入训练模块
from src.training.time_series_trainer import TimeSeriesTrainer, TimeSeriesTrainingConfig
from src.training.time_series_validation import TimeSeriesConfig


def create_sample_data(n_samples=500):
    """创建示例数据"""
    np.random.seed(42)
    
    # 创建时间索引
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # 创建特征
    X = pd.DataFrame({
        'trend': np.arange(n_samples) / n_samples,
        'seasonal': np.sin(2 * np.pi * np.arange(n_samples) / 365),
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples)
    }, index=dates)
    
    # 创建目标变量
    y = (2 * X['trend'] + 
         0.5 * X['seasonal'] + 
         0.3 * X['feature_1'] + 
         0.1 * np.random.randn(n_samples))
    
    return X, y


def main():
    """主函数"""
    print("🚀 快速开始时间序列训练")
    print("=" * 40)
    
    # 1. 创建数据
    print("📊 创建示例数据...")
    X, y = create_sample_data(500)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 2. 配置参数
    print("\n⚙️ 配置训练参数...")
    
    # 时间序列配置
    ts_config = TimeSeriesConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        n_splits=3,
        validation_strategy='walk_forward'
    )
    
    # 训练配置
    config = TimeSeriesTrainingConfig(
        time_series_config=ts_config,
        use_time_series_cv=True,
        use_walk_forward=True
    )
    
    # 3. 创建模型
    print("\n🤖 创建模型...")
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
    }
    
    # 4. 训练
    print("\n🏋️ 开始训练...")
    trainer = TimeSeriesTrainer(config=config)
    
    for name, model in models.items():
        trainer.add_model(name, model)
    
    trainer.fit(X, y)
    
    # 5. 查看结果
    print("\n📈 训练结果:")
    comparison = trainer.get_model_comparison()
    
    # 显示主要指标
    print("\n模型性能比较:")
    print("-" * 50)
    for _, row in comparison.iterrows():
        model_name = row['model']
        test_r2 = row.get('test_r2', 'N/A')
        test_rmse = row.get('test_rmse', 'N/A')
        training_time = row.get('training_time', 'N/A')
        
        # 格式化数值，处理N/A情况
        r2_str = f"{test_r2:6.3f}" if isinstance(test_r2, (int, float)) else f"{test_r2:>6}"
        rmse_str = f"{test_rmse:6.3f}" if isinstance(test_rmse, (int, float)) else f"{test_rmse:>6}"
        time_str = f"{training_time:5.1f}s" if isinstance(training_time, (int, float)) else f"{training_time:>6}"
        
        print(f"{model_name:15} | R²: {r2_str} | RMSE: {rmse_str} | 时间: {time_str}")
    
    # 6. 最佳模型
    best_model = trainer.get_best_model_name()
    print(f"\n🏆 最佳模型: {best_model}")
    
    # 7. 预测示例
    print("\n🔮 预测示例:")
    X_test = trainer.data_splits['X_test']
    predictions = trainer.predict(X_test, best_model)
    
    print(f"测试集前5个预测值: {predictions[:5]}")
    print(f"测试集前5个真实值: {trainer.data_splits['y_test'][:5]}")
    
    print("\n✅ 训练完成！")
    
    return trainer


if __name__ == "__main__":
    trainer = main()