"""
多目标标签构建模块使用示例

本文件展示了如何使用多目标标签构建功能进行夏普比率最大化、
风险调整收益优化和多目标联合优化等高级标签构建任务。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加src路径以便导入模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.multi_objective_labels import (
    SharpeRatioLabelBuilder,
    VolatilityLabelBuilder,
    RiskAdjustedReturnLabelBuilder,
    MultiObjectiveOptimizationLabelBuilder,
    MultiObjectiveLabelManager,
    create_sharpe_maximization_labels,
    create_multi_objective_labels
)


def generate_sample_data(n_days=500, n_assets=3):
    """生成示例金融数据"""
    np.random.seed(42)
    
    # 生成日期序列
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    # 生成多资产价格数据
    data = {}
    
    for i in range(n_assets):
        # 不同资产具有不同的收益率和波动率特征
        if i == 0:  # 高收益高风险资产
            mu, sigma = 0.0015, 0.025
        elif i == 1:  # 中等收益中等风险资产
            mu, sigma = 0.001, 0.018
        else:  # 低收益低风险资产
            mu, sigma = 0.0005, 0.012
        
        # 生成收益率序列（带有一些自相关性）
        returns = np.random.normal(mu, sigma, n_days)
        for j in range(1, len(returns)):
            returns[j] += 0.1 * returns[j-1]  # 添加轻微的自相关性
        
        # 计算价格序列
        prices = 100 * np.exp(np.cumsum(returns))
        
        # 生成高低价（基于收盘价的小幅波动）
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        
        # 生成成交量
        volumes = np.random.lognormal(10, 0.5, n_days)
        
        asset_name = f'asset_{i+1}'
        data[f'{asset_name}_close'] = prices
        data[f'{asset_name}_high'] = highs
        data[f'{asset_name}_low'] = lows
        data[f'{asset_name}_volume'] = volumes
    
    # 生成基准指数数据
    benchmark_returns = np.random.normal(0.0008, 0.015, n_days)
    benchmark_prices = 100 * np.exp(np.cumsum(benchmark_returns))
    data['benchmark_close'] = benchmark_prices
    
    # 创建DataFrame
    df = pd.DataFrame(data, index=dates)
    
    return df


def example_1_basic_sharpe_optimization():
    """示例1: 基础夏普比率优化标签"""
    print("=" * 60)
    print("示例1: 基础夏普比率优化标签")
    print("=" * 60)
    
    # 生成示例数据
    data = generate_sample_data(n_days=200, n_assets=1)
    asset_data = data[['asset_1_close']].rename(columns={'asset_1_close': 'close'})
    
    # 创建夏普比率标签构建器
    sharpe_builder = SharpeRatioLabelBuilder(
        periods=5,  # 5日前瞻期
        rolling_window=20,  # 20日滚动窗口
        risk_free_rate=0.03,  # 年化无风险利率3%
        price_column='close'
    )
    
    # 拟合和转换
    sharpe_labels = sharpe_builder.fit_transform(asset_data)
    
    # 输出统计信息
    print(f"夏普比率标签统计:")
    print(f"  有效标签数量: {sharpe_labels.count()}")
    print(f"  平均夏普比率: {sharpe_labels.mean():.4f}")
    print(f"  夏普比率标准差: {sharpe_labels.std():.4f}")
    print(f"  正夏普比率占比: {(sharpe_labels > 0).mean():.2%}")
    
    # 显示标签分布
    print(f"\n夏普比率分布:")
    print(sharpe_labels.describe())
    
    return sharpe_labels


def example_2_volatility_prediction():
    """示例2: 波动率预测标签"""
    print("\n" + "=" * 60)
    print("示例2: 波动率预测标签")
    print("=" * 60)
    
    # 生成示例数据
    data = generate_sample_data(n_days=200, n_assets=1)
    asset_data = data[['asset_1_close', 'asset_1_high', 'asset_1_low']].rename(columns={
        'asset_1_close': 'close',
        'asset_1_high': 'high',
        'asset_1_low': 'low'
    })
    
    # 创建不同类型的波动率标签
    vol_builders = {
        'realized': VolatilityLabelBuilder(
            periods=3,
            volatility_type='realized',
            rolling_window=15
        ),
        'parkinson': VolatilityLabelBuilder(
            periods=3,
            volatility_type='parkinson',
            rolling_window=15
        )
    }
    
    vol_labels = {}
    for vol_type, builder in vol_builders.items():
        labels = builder.fit_transform(asset_data)
        vol_labels[vol_type] = labels
        
        print(f"\n{vol_type.upper()}波动率标签:")
        print(f"  有效标签数量: {labels.count()}")
        print(f"  平均波动率: {labels.mean():.6f}")
        print(f"  波动率标准差: {labels.std():.6f}")
    
    # 比较不同波动率估计方法
    vol_df = pd.DataFrame(vol_labels)
    correlation = vol_df.corr()
    print(f"\n波动率估计方法相关性:")
    print(correlation)
    
    return vol_labels


def example_3_risk_adjusted_returns():
    """示例3: 风险调整收益标签"""
    print("\n" + "=" * 60)
    print("示例3: 风险调整收益标签")
    print("=" * 60)
    
    # 生成示例数据
    data = generate_sample_data(n_days=200, n_assets=1)
    asset_data = data[['asset_1_close', 'benchmark_close']].rename(columns={
        'asset_1_close': 'close',
        'benchmark_close': 'benchmark'
    })
    
    # 创建不同类型的风险调整收益标签
    risk_adj_builders = {
        'information_ratio': RiskAdjustedReturnLabelBuilder(
            periods=5,
            metric='information_ratio',
            benchmark_column='benchmark',
            rolling_window=20
        ),
        'sortino_ratio': RiskAdjustedReturnLabelBuilder(
            periods=5,
            metric='sortino_ratio',
            benchmark_return=0.02,  # 使用固定基准收益率
            rolling_window=20
        ),
        'calmar_ratio': RiskAdjustedReturnLabelBuilder(
            periods=5,
            metric='calmar_ratio',
            rolling_window=30  # 卡尔马比率需要更长的窗口
        )
    }
    
    risk_adj_labels = {}
    for metric_name, builder in risk_adj_builders.items():
        labels = builder.fit_transform(asset_data)
        risk_adj_labels[metric_name] = labels
        
        print(f"\n{metric_name.upper()}标签:")
        print(f"  有效标签数量: {labels.count()}")
        print(f"  平均值: {labels.mean():.4f}")
        print(f"  标准差: {labels.std():.4f}")
        print(f"  正值占比: {(labels > 0).mean():.2%}")
    
    return risk_adj_labels


def example_4_multi_objective_optimization():
    """示例4: 多目标优化标签"""
    print("\n" + "=" * 60)
    print("示例4: 多目标优化标签")
    print("=" * 60)
    
    # 生成示例数据
    data = generate_sample_data(n_days=200, n_assets=1)
    asset_data = data[['asset_1_close']].rename(columns={'asset_1_close': 'close'})
    
    # 创建多目标优化标签构建器
    multi_obj_builders = {
        'weighted_sum': MultiObjectiveOptimizationLabelBuilder(
            periods=5,
            objectives=['return', 'sharpe', 'volatility'],
            weights=[0.4, 0.4, 0.2],  # 收益40%，夏普40%，波动率20%
            optimization_method='weighted_sum'
        ),
        'pareto_ranking': MultiObjectiveOptimizationLabelBuilder(
            periods=5,
            objectives=['return', 'sharpe'],
            optimization_method='pareto_ranking'
        )
    }
    
    multi_obj_labels = {}
    for method_name, builder in multi_obj_builders.items():
        labels = builder.fit_transform(asset_data)
        multi_obj_labels[method_name] = labels
        
        print(f"\n{method_name.upper()}多目标优化标签:")
        print(f"  有效标签数量: {labels.count()}")
        print(f"  平均优化分数: {labels.mean():.4f}")
        print(f"  分数标准差: {labels.std():.4f}")
        
        # 显示目标函数统计
        if hasattr(builder, 'objective_stats_'):
            print(f"  目标函数统计:")
            for obj_name, stats in builder.objective_stats_.items():
                print(f"    {obj_name}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
    
    return multi_obj_labels


def example_5_comprehensive_label_manager():
    """示例5: 综合标签管理器"""
    print("\n" + "=" * 60)
    print("示例5: 综合标签管理器")
    print("=" * 60)
    
    # 生成示例数据
    data = generate_sample_data(n_days=200, n_assets=1)
    asset_data = data[['asset_1_close', 'benchmark_close']].rename(columns={
        'asset_1_close': 'close',
        'benchmark_close': 'benchmark'
    })
    
    # 创建多目标标签管理器
    manager = MultiObjectiveLabelManager()
    
    # 添加各种类型的标签
    manager.add_sharpe_label('sharpe_3d', periods=3, risk_free_rate=0.03)
    manager.add_sharpe_label('sharpe_5d', periods=5, risk_free_rate=0.03)
    
    manager.add_volatility_label('vol_3d', periods=3, volatility_type='realized')
    manager.add_volatility_label('vol_parkinson_5d', periods=5, volatility_type='parkinson')
    
    manager.add_risk_adjusted_label(
        'info_ratio_5d',
        periods=5,
        metric='information_ratio',
        benchmark_column='benchmark'
    )
    
    manager.add_multi_objective_label(
        'multi_obj_5d',
        periods=5,
        objectives=['return', 'sharpe', 'volatility'],
        weights=[0.3, 0.5, 0.2]
    )
    
    # 批量生成所有标签
    all_labels = manager.fit_transform(asset_data)
    
    print(f"生成的标签类型: {list(all_labels.columns)}")
    print(f"标签数据形状: {all_labels.shape}")
    
    # 显示标签相关性矩阵
    correlation_matrix = all_labels.corr()
    print(f"\n标签相关性矩阵:")
    print(correlation_matrix.round(3))
    
    # 获取优化摘要
    summary = manager.get_optimization_summary()
    print(f"\n标签统计摘要:")
    print(summary)
    
    return all_labels, manager


def example_6_convenience_functions():
    """示例6: 便捷函数使用"""
    print("\n" + "=" * 60)
    print("示例6: 便捷函数使用")
    print("=" * 60)
    
    # 生成示例数据
    data = generate_sample_data(n_days=150, n_assets=1)
    asset_data = data[['asset_1_close']].rename(columns={'asset_1_close': 'close'})
    
    # 使用便捷函数创建夏普比率最大化标签
    print("使用便捷函数创建夏普比率最大化标签...")
    sharpe_labels = create_sharpe_maximization_labels(
        asset_data,
        periods=[1, 3, 5],
        risk_free_rate=0.03,
        rolling_window=15
    )
    
    print(f"夏普比率标签: {list(sharpe_labels.columns)}")
    print(f"标签数据形状: {sharpe_labels.shape}")
    
    # 使用便捷函数创建多目标标签
    print("\n使用便捷函数创建多目标标签...")
    multi_obj_labels = create_multi_objective_labels(
        asset_data,
        periods=[3, 5],
        objectives=['return', 'sharpe'],
        optimization_method='weighted_sum',
        weights=[0.6, 0.4]
    )
    
    print(f"多目标标签: {list(multi_obj_labels.columns)}")
    print(f"标签数据形状: {multi_obj_labels.shape}")
    
    # 合并所有标签
    combined_labels = pd.concat([sharpe_labels, multi_obj_labels], axis=1)
    
    print(f"\n合并后的标签:")
    print(f"  总标签数: {combined_labels.shape[1]}")
    print(f"  有效样本数: {combined_labels.dropna().shape[0]}")
    
    return combined_labels


def example_7_multi_asset_analysis():
    """示例7: 多资产分析"""
    print("\n" + "=" * 60)
    print("示例7: 多资产分析")
    print("=" * 60)
    
    # 生成多资产数据
    data = generate_sample_data(n_days=200, n_assets=3)
    
    # 为每个资产创建标签
    asset_results = {}
    
    for i in range(3):
        asset_name = f'asset_{i+1}'
        asset_data = data[[f'{asset_name}_close']].rename(columns={
            f'{asset_name}_close': 'close'
        })
        
        print(f"\n分析 {asset_name.upper()}:")
        
        # 创建管理器
        manager = MultiObjectiveLabelManager()
        manager.add_sharpe_label('sharpe_5d', periods=5)
        manager.add_volatility_label('vol_5d', periods=5)
        manager.add_multi_objective_label(
            'multi_obj_5d',
            periods=5,
            objectives=['return', 'sharpe'],
            weights=[0.5, 0.5]
        )
        
        # 生成标签
        labels = manager.fit_transform(asset_data)
        asset_results[asset_name] = labels
        
        # 显示统计信息
        print(f"  夏普比率均值: {labels['sharpe_5d'].mean():.4f}")
        print(f"  波动率均值: {labels['vol_5d'].mean():.6f}")
        print(f"  多目标分数均值: {labels['multi_obj_5d'].mean():.4f}")
    
    # 比较不同资产的表现
    print(f"\n资产表现比较:")
    comparison_data = []
    for asset_name, labels in asset_results.items():
        comparison_data.append({
            'Asset': asset_name,
            'Avg_Sharpe': labels['sharpe_5d'].mean(),
            'Avg_Volatility': labels['vol_5d'].mean(),
            'Avg_MultiObj': labels['multi_obj_5d'].mean()
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4))
    
    return asset_results


def example_8_real_world_scenario():
    """示例8: 真实场景应用"""
    print("\n" + "=" * 60)
    print("示例8: 真实场景应用 - 量化策略标签构建")
    print("=" * 60)
    
    # 生成更真实的市场数据（包含趋势和周期性）
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # 模拟市场趋势和周期
    trend = np.linspace(0, 0.2, n_days)  # 长期上升趋势
    cycle = 0.05 * np.sin(2 * np.pi * np.arange(n_days) / 60)  # 60日周期
    noise = np.random.normal(0, 0.02, n_days)
    
    returns = trend / n_days + cycle / n_days + noise
    prices = 100 * np.exp(np.cumsum(returns))
    
    # 创建数据
    market_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.lognormal(12, 0.3, n_days)
    }, index=dates)
    
    print(f"市场数据概览:")
    print(f"  数据期间: {dates[0].date()} 至 {dates[-1].date()}")
    print(f"  总收益率: {(prices[-1] / prices[0] - 1):.2%}")
    print(f"  年化波动率: {np.std(returns) * np.sqrt(252):.2%}")
    
    # 构建量化策略标签
    print(f"\n构建量化策略标签...")
    
    # 创建专门的策略标签管理器
    strategy_manager = MultiObjectiveLabelManager()
    
    # 短期动量策略标签
    strategy_manager.add_sharpe_label('momentum_3d', periods=3, rolling_window=10)
    strategy_manager.add_sharpe_label('momentum_5d', periods=5, rolling_window=15)
    
    # 风险管理标签
    strategy_manager.add_volatility_label('risk_3d', periods=3, rolling_window=10)
    strategy_manager.add_volatility_label('risk_5d', periods=5, rolling_window=15)
    
    # 综合策略标签（平衡收益和风险）
    strategy_manager.add_multi_objective_label(
        'balanced_strategy',
        periods=5,
        objectives=['return', 'sharpe', 'volatility'],
        weights=[0.3, 0.5, 0.2]  # 重点关注夏普比率
    )
    
    # 激进策略标签（重点关注收益）
    strategy_manager.add_multi_objective_label(
        'aggressive_strategy',
        periods=3,
        objectives=['return', 'sharpe'],
        weights=[0.7, 0.3]  # 重点关注收益
    )
    
    # 保守策略标签（重点关注风险控制）
    strategy_manager.add_multi_objective_label(
        'conservative_strategy',
        periods=7,
        objectives=['sharpe', 'volatility'],
        weights=[0.6, 0.4]  # 重点关注风险调整收益
    )
    
    # 生成所有策略标签
    strategy_labels = strategy_manager.fit_transform(market_data)
    
    print(f"\n策略标签生成完成:")
    print(f"  生成标签数: {strategy_labels.shape[1]}")
    print(f"  有效样本数: {strategy_labels.dropna().shape[0]}")
    
    # 策略表现分析
    print(f"\n策略表现分析:")
    strategy_performance = {}
    
    for strategy in ['balanced_strategy', 'aggressive_strategy', 'conservative_strategy']:
        if strategy in strategy_labels.columns:
            labels = strategy_labels[strategy].dropna()
            if len(labels) > 0:
                strategy_performance[strategy] = {
                    'mean_score': labels.mean(),
                    'std_score': labels.std(),
                    'positive_ratio': (labels > 0).mean(),
                    'sharpe_of_scores': labels.mean() / labels.std() if labels.std() > 0 else 0
                }
    
    for strategy, metrics in strategy_performance.items():
        print(f"\n{strategy.upper()}:")
        print(f"  平均分数: {metrics['mean_score']:.4f}")
        print(f"  分数波动: {metrics['std_score']:.4f}")
        print(f"  正分数占比: {metrics['positive_ratio']:.2%}")
        print(f"  分数夏普比率: {metrics['sharpe_of_scores']:.4f}")
    
    # 标签相关性分析
    correlation_matrix = strategy_labels.corr()
    print(f"\n策略标签相关性分析:")
    print("主要相关性:")
    
    # 找出高相关性的标签对
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.5:  # 相关性阈值
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                print(f"  {col1} vs {col2}: {corr_value:.3f}")
    
    return strategy_labels, strategy_performance


def main():
    """主函数 - 运行所有示例"""
    print("多目标标签构建模块使用示例")
    print("=" * 80)
    print("本示例展示了如何使用多目标标签构建功能进行:")
    print("1. 夏普比率优化")
    print("2. 波动率预测")
    print("3. 风险调整收益计算")
    print("4. 多目标优化")
    print("5. 综合标签管理")
    print("6. 便捷函数使用")
    print("7. 多资产分析")
    print("8. 真实场景应用")
    print("=" * 80)
    
    try:
        # 运行所有示例
        example_1_basic_sharpe_optimization()
        example_2_volatility_prediction()
        example_3_risk_adjusted_returns()
        example_4_multi_objective_optimization()
        example_5_comprehensive_label_manager()
        example_6_convenience_functions()
        example_7_multi_asset_analysis()
        example_8_real_world_scenario()
        
        print("\n" + "=" * 80)
        print("所有示例运行完成！")
        print("=" * 80)
        print("\n多目标标签构建模块功能特点:")
        print("✓ 夏普比率最大化标签构建")
        print("✓ 多种波动率估计方法")
        print("✓ 丰富的风险调整收益指标")
        print("✓ 灵活的多目标优化框架")
        print("✓ 统一的标签管理接口")
        print("✓ 便捷的批量处理函数")
        print("✓ 支持多资产并行分析")
        print("✓ 适用于真实量化策略场景")
        
        print("\n技术亮点:")
        print("• 模块化设计，易于扩展")
        print("• 内存高效的滚动计算")
        print("• 完善的错误处理机制")
        print("• 丰富的统计信息输出")
        print("• 支持多种优化方法")
        print("• 灵活的参数配置")
        
    except Exception as e:
        print(f"\n运行示例时出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()