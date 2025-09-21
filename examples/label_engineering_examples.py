"""
标签构建模块使用示例

展示如何使用标签构建模块创建各种量化交易标签
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.label_engineering import (
    ReturnLabelBuilder,
    DirectionLabelBuilder,
    ExcessReturnLabelBuilder,
    LabelEngineeringManager,
    LabelSmoother,
    NoiseReducer,
    create_return_labels,
    create_direction_labels
)


def create_sample_data():
    """创建示例股票数据"""
    print("创建示例数据...")
    
    np.random.seed(42)
    n_days = 1000
    
    # 创建日期索引
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    # 模拟股票价格（几何布朗运动）
    dt = 1/252  # 日频率
    mu = 0.08   # 年化收益率
    sigma = 0.2 # 年化波动率
    
    # 生成价格序列
    returns = np.random.normal(mu*dt, sigma*np.sqrt(dt), n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # 模拟基准指数（市场指数）
    market_returns = np.random.normal(0.06*dt, 0.15*np.sqrt(dt), n_days)
    market_prices = 100 * np.exp(np.cumsum(market_returns))
    
    # 模拟成交量
    volume = np.random.lognormal(10, 0.5, n_days)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.01, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_days))),
        'close': prices,
        'volume': volume,
        'market_index': market_prices
    })
    
    # 确保high >= close >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    print(f"生成了 {len(data)} 天的股票数据")
    print(f"价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"基准范围: {data['market_index'].min():.2f} - {data['market_index'].max():.2f}")
    
    return data


def example_1_basic_return_labels():
    """示例1: 基础收益率标签"""
    print("\n" + "="*60)
    print("示例1: 基础收益率标签")
    print("="*60)
    
    data = create_sample_data()
    
    # 创建收益率标签构建器
    return_builder = ReturnLabelBuilder(periods=1, method='simple')
    
    # 拟合并生成标签
    returns_1d = return_builder.fit_transform(data)
    
    print(f"\n1日收益率统计:")
    print(f"均值: {returns_1d.mean():.6f}")
    print(f"标准差: {returns_1d.std():.6f}")
    print(f"最小值: {returns_1d.min():.6f}")
    print(f"最大值: {returns_1d.max():.6f}")
    print(f"偏度: {returns_1d.skew():.4f}")
    print(f"峰度: {returns_1d.kurtosis():.4f}")
    
    # 创建多期收益率
    periods = [1, 5, 10, 20]
    returns_df = create_return_labels(data, periods=periods)
    
    print(f"\n多期收益率相关性:")
    correlation_matrix = returns_df.corr()
    print(correlation_matrix.round(4))
    
    return returns_df


def example_2_direction_labels():
    """示例2: 方向标签"""
    print("\n" + "="*60)
    print("示例2: 方向标签")
    print("="*60)
    
    data = create_sample_data()
    
    # 创建方向标签（无中性区间）
    direction_builder = DirectionLabelBuilder(periods=1, threshold=0.0)
    directions_1d = direction_builder.fit_transform(data)
    
    print(f"\n1日方向标签分布（无中性区间）:")
    direction_counts = directions_1d.value_counts().sort_index()
    for direction, count in direction_counts.items():
        label = {-1: "下跌", 0: "平盘", 1: "上涨"}[direction]
        print(f"{label} ({direction:2d}): {count:4d} ({count/len(directions_1d)*100:.1f}%)")
    
    # 创建方向标签（有中性区间）
    direction_builder_neutral = DirectionLabelBuilder(
        periods=1, 
        neutral_zone=0.01  # 1%中性区间
    )
    directions_neutral = direction_builder_neutral.fit_transform(data)
    
    print(f"\n1日方向标签分布（1%中性区间）:")
    direction_counts_neutral = directions_neutral.value_counts().sort_index()
    for direction, count in direction_counts_neutral.items():
        label = {-1: "下跌", 0: "中性", 1: "上涨"}[direction]
        print(f"{label} ({direction:2d}): {count:4d} ({count/len(directions_neutral)*100:.1f}%)")
    
    # 创建多期方向标签
    periods = [1, 5, 10]
    directions_df = create_direction_labels(data, periods=periods, neutral_zone=0.01)
    
    print(f"\n多期方向标签统计:")
    for col in directions_df.columns:
        dist = directions_df[col].value_counts().sort_index()
        print(f"\n{col}:")
        for direction, count in dist.items():
            print(f"  {direction:2.0f}: {count:4d} ({count/len(directions_df)*100:.1f}%)")
    
    return directions_df


def example_3_excess_return_labels():
    """示例3: 超额收益标签"""
    print("\n" + "="*60)
    print("示例3: 超额收益标签")
    print("="*60)
    
    data = create_sample_data()
    
    # 创建超额收益标签
    excess_builder = ExcessReturnLabelBuilder(
        periods=1,
        benchmark_column='market_index'
    )
    excess_returns = excess_builder.fit_transform(data)
    
    print(f"\n1日超额收益统计:")
    print(f"均值: {excess_returns.mean():.6f}")
    print(f"标准差: {excess_returns.std():.6f}")
    print(f"信息比率: {excess_returns.mean()/excess_returns.std():.4f}")
    print(f"胜率: {(excess_returns > 0).mean():.4f}")
    
    # 计算累积超额收益
    cumulative_excess = excess_returns.cumsum()
    
    print(f"\n累积超额收益:")
    print(f"总超额收益: {cumulative_excess.iloc[-1]:.4f}")
    print(f"最大回撤: {(cumulative_excess - cumulative_excess.cummax()).min():.4f}")
    
    # 分析超额收益分布
    print(f"\n超额收益分位数:")
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    for q in quantiles:
        print(f"{q*100:4.0f}%: {excess_returns.quantile(q):.6f}")
    
    return excess_returns


def example_4_comprehensive_label_engineering():
    """示例4: 综合标签构建"""
    print("\n" + "="*60)
    print("示例4: 综合标签构建")
    print("="*60)
    
    data = create_sample_data()
    
    # 创建标签构建管理器
    manager = LabelEngineeringManager()
    
    # 添加各种标签
    periods = [1, 5, 10, 20]
    
    # 收益率标签
    for period in periods:
        manager.add_return_label(f'return_{period}d', periods=period)
        manager.add_return_label(f'log_return_{period}d', periods=period, method='log')
    
    # 方向标签
    for period in periods:
        manager.add_direction_label(f'direction_{period}d', periods=period, neutral_zone=0.01)
    
    # 超额收益标签
    for period in periods:
        manager.add_excess_return_label(
            f'excess_{period}d', 
            periods=period, 
            benchmark_column='market_index'
        )
    
    # 生成所有标签
    print("\n生成标签...")
    labels = manager.fit_transform(data)
    
    print(f"\n生成的标签: {list(labels.columns)}")
    print(f"标签数据形状: {labels.shape}")
    
    # 获取标签统计摘要
    print("\n标签统计摘要:")
    summary = manager.get_label_summary()
    print(summary.round(6))
    
    # 分析标签相关性
    print("\n收益率标签相关性:")
    return_cols = [col for col in labels.columns if col.startswith('return_')]
    if return_cols:
        return_corr = labels[return_cols].corr()
        print(return_corr.round(4))
    
    return labels, manager


def example_5_label_smoothing_and_noise_reduction():
    """示例5: 标签平滑和噪声处理"""
    print("\n" + "="*60)
    print("示例5: 标签平滑和噪声处理")
    print("="*60)
    
    data = create_sample_data()
    
    # 创建带噪声的标签
    manager = LabelEngineeringManager()
    manager.add_return_label('return_1d', periods=1)
    labels = manager.fit_transform(data)
    
    original_returns = labels['return_1d']
    
    print(f"原始收益率统计:")
    print(f"标准差: {original_returns.std():.6f}")
    print(f"偏度: {original_returns.skew():.4f}")
    print(f"峰度: {original_returns.kurtosis():.4f}")
    
    # 应用不同的平滑方法
    smoothing_methods = {
        'rolling_mean': {'method': 'rolling_mean', 'window': 5},
        'exponential': {'method': 'exponential', 'alpha': 0.3},
    }
    
    print(f"\n标签平滑效果:")
    for name, params in smoothing_methods.items():
        smoothed = manager.apply_smoothing('return_1d', **params)
        reduction = (1 - smoothed.std() / original_returns.std()) * 100
        print(f"{name:15s}: 标准差降低 {reduction:.1f}%")
    
    # 应用噪声处理
    print(f"\n噪声处理效果:")
    
    # 异常值裁剪
    clipped = manager.apply_noise_reduction('return_1d', method='outlier_clip')
    print(f"异常值裁剪: 极值范围 {clipped.min():.6f} ~ {clipped.max():.6f}")
    
    # Z-score过滤
    filtered = manager.apply_noise_reduction('return_1d', method='z_score_filter', threshold=2.5)
    filtered_ratio = filtered.count() / original_returns.count()
    print(f"Z-score过滤: 保留 {filtered_ratio:.1%} 的数据")
    
    return original_returns, smoothed, clipped, filtered


def example_6_advanced_applications():
    """示例6: 高级应用"""
    print("\n" + "="*60)
    print("示例6: 高级应用")
    print("="*60)
    
    data = create_sample_data()
    
    # 创建多时间框架标签
    print("\n多时间框架标签分析:")
    
    manager = LabelEngineeringManager()
    
    # 短期标签（1-5天）
    short_periods = [1, 2, 3, 5]
    for period in short_periods:
        manager.add_return_label(f'short_return_{period}d', periods=period)
        manager.add_direction_label(f'short_direction_{period}d', periods=period, neutral_zone=0.005)
    
    # 中期标签（1-4周）
    medium_periods = [5, 10, 15, 20]
    for period in medium_periods:
        manager.add_return_label(f'medium_return_{period}d', periods=period)
        manager.add_direction_label(f'medium_direction_{period}d', periods=period, neutral_zone=0.02)
    
    # 长期标签（1-3月）
    long_periods = [20, 40, 60]
    for period in long_periods:
        manager.add_return_label(f'long_return_{period}d', periods=period)
        manager.add_direction_label(f'long_direction_{period}d', periods=period, neutral_zone=0.05)
    
    labels = manager.fit_transform(data)
    
    # 分析不同时间框架的预测一致性
    print(f"\n时间框架方向一致性分析:")
    
    # 短期方向一致性
    short_direction_cols = [col for col in labels.columns if 'short_direction' in col]
    if len(short_direction_cols) > 1:
        short_consistency = (labels[short_direction_cols].std(axis=1) == 0).mean()
        print(f"短期方向一致性: {short_consistency:.3f}")
    
    # 中期方向一致性
    medium_direction_cols = [col for col in labels.columns if 'medium_direction' in col]
    if len(medium_direction_cols) > 1:
        medium_consistency = (labels[medium_direction_cols].std(axis=1) == 0).mean()
        print(f"中期方向一致性: {medium_consistency:.3f}")
    
    # 跨时间框架分析
    if short_direction_cols and medium_direction_cols:
        # 计算短期和中期方向的相关性
        short_avg = labels[short_direction_cols].mean(axis=1)
        medium_avg = labels[medium_direction_cols].mean(axis=1)
        cross_corr = short_avg.corr(medium_avg)
        print(f"短期-中期方向相关性: {cross_corr:.3f}")
    
    # 标签质量评估
    print(f"\n标签质量评估:")
    
    return_cols = [col for col in labels.columns if 'return' in col]
    for col in return_cols[:5]:  # 只显示前5个
        returns = labels[col].dropna()
        if len(returns) > 0:
            # 计算信噪比
            signal_to_noise = abs(returns.mean()) / returns.std()
            # 计算自相关性
            autocorr = returns.autocorr(lag=1)
            print(f"{col:20s}: 信噪比={signal_to_noise:.4f}, 自相关={autocorr:.4f}")
    
    return labels


def example_7_real_world_scenarios():
    """示例7: 真实场景应用"""
    print("\n" + "="*60)
    print("示例7: 真实场景应用")
    print("="*60)
    
    data = create_sample_data()
    
    # 场景1: 日内交易标签
    print("\n场景1: 日内交易标签构建")
    
    # 模拟分钟级数据
    minute_data = data.iloc[:100].copy()  # 取100天数据模拟分钟数据
    minute_data.index = pd.date_range('2023-01-01 09:30', periods=100, freq='1min')
    
    intraday_manager = LabelEngineeringManager()
    
    # 添加短期标签（适合日内交易）
    intraday_periods = [1, 5, 15, 30]  # 1分钟、5分钟、15分钟、30分钟
    for period in intraday_periods:
        intraday_manager.add_return_label(f'return_{period}min', periods=period)
        intraday_manager.add_direction_label(
            f'direction_{period}min', 
            periods=period, 
            neutral_zone=0.001  # 更小的中性区间
        )
    
    intraday_labels = intraday_manager.fit_transform(minute_data)
    print(f"日内标签数量: {len(intraday_labels.columns)}")
    
    # 场景2: 波动率标签
    print("\n场景2: 波动率标签构建")
    
    # 计算滚动波动率作为标签
    windows = [5, 10, 20]
    volatility_labels = pd.DataFrame(index=data.index)
    
    for window in windows:
        returns = data['close'].pct_change()
        vol = returns.rolling(window=window).std() * np.sqrt(252)  # 年化波动率
        volatility_labels[f'volatility_{window}d'] = vol
        
        # 波动率分位数标签
        vol_rank = vol.rolling(window=60).rank(pct=True)  # 60天滚动分位数
        volatility_labels[f'vol_rank_{window}d'] = vol_rank
    
    print(f"波动率标签统计:")
    print(volatility_labels.describe().round(4))
    
    # 场景3: 事件驱动标签
    print("\n场景3: 事件驱动标签构建")
    
    # 模拟大幅波动事件
    returns = data['close'].pct_change()
    
    # 定义事件
    large_moves = abs(returns) > returns.std() * 2  # 2倍标准差以上的波动
    gap_events = abs(returns) > 0.05  # 5%以上的跳空
    
    event_labels = pd.DataFrame(index=data.index)
    event_labels['large_move_event'] = large_moves.astype(int)
    event_labels['gap_event'] = gap_events.astype(int)
    
    # 事件后的收益率标签
    for period in [1, 3, 5]:
        event_labels[f'post_event_return_{period}d'] = returns.shift(-period)
    
    print(f"事件统计:")
    print(f"大幅波动事件: {large_moves.sum()} 次 ({large_moves.mean():.2%})")
    print(f"跳空事件: {gap_events.sum()} 次 ({gap_events.mean():.2%})")
    
    return intraday_labels, volatility_labels, event_labels


def main():
    """主函数：运行所有示例"""
    print("标签构建模块使用示例")
    print("="*60)
    
    try:
        # 运行所有示例
        returns_df = example_1_basic_return_labels()
        directions_df = example_2_direction_labels()
        excess_returns = example_3_excess_return_labels()
        labels, manager = example_4_comprehensive_label_engineering()
        smoothing_results = example_5_label_smoothing_and_noise_reduction()
        advanced_labels = example_6_advanced_applications()
        real_world_results = example_7_real_world_scenarios()
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        
        print(f"\n总结:")
        print(f"- 基础收益率标签: {len(returns_df.columns)} 个")
        print(f"- 方向标签: {len(directions_df.columns)} 个")
        print(f"- 综合标签: {len(labels.columns)} 个")
        print(f"- 高级标签: {len(advanced_labels.columns)} 个")
        
        print(f"\n标签构建模块功能验证:")
        print(f"✓ 收益率标签计算")
        print(f"✓ 方向标签生成")
        print(f"✓ 超额收益计算")
        print(f"✓ 标签平滑处理")
        print(f"✓ 噪声处理")
        print(f"✓ 多时间框架分析")
        print(f"✓ 真实场景应用")
        
    except Exception as e:
        print(f"\n运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()