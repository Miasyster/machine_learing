"""
机器学习回测流水线 - 简化版

完整的从数据准备、模型训练到策略回测的端到端流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import sys
import os
warnings.filterwarnings('ignore')
# 设置中文字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']  
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False  

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入训练框架
from src.training.supervised import ClassificationTrainer
from src.features.feature_engineering import create_feature_engineer, calculate_all_features

# 导入回测框架
from src.backtest import (
    BacktestConfig, Order, OrderType, OrderSide,
    PercentageCommissionModel, FixedSlippageModel
)
from src.backtest.engine import BarBacktestEngine
from src.backtest.analysis import BacktestAnalyzer


class MLStrategy:
    """基于机器学习的交易策略"""
    
    def __init__(self, model, feature_columns, prediction_threshold=0.3):
        self.model = model
        self.feature_columns = feature_columns
        self.prediction_threshold = prediction_threshold
        self.engine = None
        self.name = "ML_Strategy"
    
    def set_engine(self, engine):
        """设置回测引擎引用"""
        self.engine = engine
    
    def on_bar(self, current_time, bar_data=None):
        """策略主逻辑 - 每个bar调用一次"""
        if self.engine is None:
            return
        
        # 获取当前数据
        current_data = self.engine.get_current_bar_data("STOCK")
        if current_data is None:
            return
        
        # 获取历史数据用于特征计算 - 日线数据需要更长的历史窗口
        historical_data = self.engine.get_historical_data("STOCK", lookback=100)
        if historical_data is None or len(historical_data) < 50:
            return
        
        try:
            # 计算特征
            features = self._calculate_features(historical_data)
            if features is None:
                return
            
            # 生成预测
            signal = self._generate_signal(features)
            
            # 执行交易逻辑
            self._execute_trading_logic(signal, current_data['close'], current_time)
            
        except Exception as e:
            print(f"策略执行错误: {e}")
    
    def _calculate_features(self, data):
        """计算特征"""
        try:
            # 使用特征工程模块计算技术指标
            features = calculate_all_features(data)
            
            # 添加额外特征 - 日线数据使用更短的滚动窗口
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            features['volatility'] = features['returns'].rolling(10).std()  # 10日波动率
            features['price_position'] = (data['close'] - data['close'].rolling(10).min()) / \
                                        (data['close'].rolling(10).max() - data['close'].rolling(10).min())
            features['volume_sma'] = data['volume'].rolling(10).mean()  # 10日成交量均线
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            
            # 获取最新的特征值
            latest_features = features.iloc[-1]
            
            # 检查是否包含所需特征
            if not all(col in latest_features.index for col in self.feature_columns):
                return None
            
            # 提取所需特征并填充NaN
            feature_values = latest_features[self.feature_columns].fillna(0)
            
            return feature_values.values.reshape(1, -1)
            
        except Exception as e:
            print(f"特征计算错误: {e}")
            return None
    
    def _generate_signal(self, features):
        """生成交易信号"""
        try:
            # 初始化调试计数器
            if hasattr(self, '_signal_count'):
                self._signal_count += 1
            else:
                self._signal_count = 1
            
            if hasattr(self.model, 'predict_proba'):
                # 分类模型 - 使用概率预测
                probabilities = self.model.predict_proba(features)
                prediction = self.model.predict(features)[0]
                
                # 每500次预测输出一次调试信息
                if self._signal_count % 500 == 0:
                    print(f"信号生成调试 (第{self._signal_count}次):")
                    print(f"  预测类别: {prediction} (0=下跌, 1=持平, 2=上涨)")
                    print(f"  预测概率: {probabilities[0]}")
                    print(f"  阈值: {self.prediction_threshold}")
                
                if probabilities.shape[1] >= 3:
                    # 三分类：0=下跌, 1=持平, 2=上涨
                    sell_prob = probabilities[0, 0]  # 下跌概率
                    hold_prob = probabilities[0, 1]  # 持平概率
                    buy_prob = probabilities[0, 2]   # 上涨概率
                    
                    if self._signal_count % 100 == 0:
                        print(f"  下跌概率: {sell_prob:.4f}")
                        print(f"  持平概率: {hold_prob:.4f}")
                        print(f"  上涨概率: {buy_prob:.4f}")
                        print(f"  阈值: {self.prediction_threshold}")
                    
                    if buy_prob > self.prediction_threshold:
                        if self._signal_count % 100 == 0:
                            print(f"  → 生成买入信号")
                        return 1  # 买入
                    elif sell_prob > self.prediction_threshold:
                        if self._signal_count % 100 == 0:
                            print(f"  → 生成卖出信号")
                        return -1  # 卖出
                    else:
                        if self._signal_count % 100 == 0:
                            print(f"  → 持有")
                        return 0  # 持有
            else:
                # 回归模型
                prediction = self.model.predict(features)[0]
                
                # 每100次预测输出一次调试信息
                if self._signal_count % 100 == 0:
                    print(f"信号生成调试 (第{self._signal_count}次):")
                    print(f"  回归预测值: {prediction:.4f}")
                    print(f"  阈值: {self.prediction_threshold}")
                
                if prediction > self.prediction_threshold:
                    if self._signal_count % 100 == 0:
                        print(f"  → 生成买入信号")
                    return 1
                elif prediction < -self.prediction_threshold:
                    if self._signal_count % 100 == 0:
                        print(f"  → 生成卖出信号")
                    return -1
                else:
                    if self._signal_count % 100 == 0:
                        print(f"  → 无信号")
                    return 0
        except Exception as e:
            print(f"信号生成错误: {e}")
            return 0
        
        return 0
    
    def _execute_trading_logic(self, signal, current_price, current_time):
        """执行交易逻辑"""
        try:
            # 获取当前持仓
            position = self.engine.positions.get("STOCK")
            current_quantity = position.quantity if position else 0
            
            # 计算目标仓位 - 使用固定的仓位大小
            available_cash = self.engine.current_capital
            position_size = 0.5  # 固定使用50%的资金进行交易
            
            # 添加交易计数器
            if not hasattr(self, '_trade_count'):
                self._trade_count = 0
            
            if signal == 1 and current_quantity <= 0:  # 买入信号
                target_value = available_cash * position_size
                target_quantity = target_value / current_price  # 允许小数数量
                
                print(f"买入信号触发: 价格={current_price:.2f}, 可用资金={available_cash:.2f}, 目标数量={target_quantity:.6f}")
                
                if target_quantity > 0.000001:  # 最小交易数量
                    order = Order(
                        symbol="STOCK",
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=target_quantity,
                        timestamp=current_time
                    )
                    self.engine.place_order(order)
                    self._trade_count += 1
                    print(f"执行买入订单: 数量={target_quantity:.6f}, 交易次数={self._trade_count}")
                else:
                    print(f"买入信号但目标数量太小，跳过交易")
                    
            elif signal == -1 and current_quantity > 0:  # 卖出信号
                print(f"卖出信号触发: 价格={current_price:.2f}, 当前持仓={current_quantity}")
                order = Order(
                    symbol="STOCK",
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=current_quantity,
                    timestamp=current_time
                )
                self.engine.place_order(order)
                self._trade_count += 1
                print(f"执行卖出订单: 数量={current_quantity}, 交易次数={self._trade_count}")
            elif signal != 0:
                print(f"信号={signal}, 但条件不满足: 当前持仓={current_quantity}")
                
        except Exception as e:
            print(f"交易执行错误: {e}")


def load_btcusdt_data():
    """加载真实的BTCUSDT日线数据"""
    print("加载真实的BTCUSDT日线数据...")
    
    # 数据文件路径 - 使用日线数据
    data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'data', 'raw', 'klines', 'BTCUSDT_1d.csv')
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"BTCUSDT日线数据文件不存在: {data_file}")
    
    # 读取CSV数据
    df = pd.read_csv(data_file)
    print(f"原始数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 转换时间列
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    
    # 选择需要的列并重命名以匹配回测引擎的期望格式
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    df_clean = df[required_columns].copy()
    
    # 确保数据类型正确
    for col in required_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 删除任何包含NaN的行
    df_clean = df_clean.dropna()
    
    print(f"清理后的数据形状: {df_clean.shape}")
    print(f"数据时间范围: {df_clean.index.min()} 到 {df_clean.index.max()}")
    print(f"价格范围: ${df_clean['close'].min():.2f} - ${df_clean['close'].max():.2f}")
    
    return df_clean


def generate_sample_data(start_date='2020-01-01', end_date='2025-07-01'):
    """加载BTCUSDT数据（保持函数名兼容性）"""
    return load_btcusdt_data()


def prepare_training_data(data):
    """准备训练数据"""
    print("准备机器学习特征...")
    
    # 计算技术指标
    features = calculate_all_features(data)
    
    # 添加额外特征 - 日线数据使用更短的滚动窗口
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    features['volatility'] = features['returns'].rolling(10).std()  # 10日波动率
    features['price_position'] = (data['close'] - data['close'].rolling(10).min()) / \
                                (data['close'].rolling(10).max() - data['close'].rolling(10).min())
    features['volume_sma'] = data['volume'].rolling(10).mean()  # 10日成交量均线
    features['volume_ratio'] = data['volume'] / features['volume_sma']
    
    # 创建目标变量（未来收益率分类）- 日线数据预测未来5日收益
    future_returns = data['close'].shift(-5) / data['close'] - 1  # 预测未来5日收益
    
    # 使用分位数创建更平衡的目标
    q33 = future_returns.quantile(0.33)
    q67 = future_returns.quantile(0.67)
    
    target = pd.cut(future_returns, 
                   bins=[-np.inf, q33, q67, np.inf], 
                   labels=[0, 1, 2])
    
    print(f"目标分位数: 33%={q33:.4f}, 67%={q67:.4f}")
    
    # 删除包含NaN的行
    valid_idx = ~(features.isnull().any(axis=1) | target.isnull())
    features_clean = features[valid_idx]
    target_clean = target[valid_idx]
    
    # 获取特征列名
    feature_columns = features_clean.columns.tolist()
    
    print(f"特征数量: {features_clean.shape[1]}")
    print(f"样本数量: {len(features_clean)}")
    print(f"目标分布: {target_clean.value_counts().to_dict()}")
    
    return features_clean, target_clean, feature_columns


def train_model(features, target):
    """训练机器学习模型"""
    print("训练机器学习模型...")
    
    # 数据分割
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 训练随机森林模型
    trainer = ClassificationTrainer(
        models=['random_forest'],
        param_grids={
            'random_forest': {
                'n_estimators': [200],
                'max_depth': [15],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
                'class_weight': ['balanced'],  # 添加类别权重平衡
                'random_state': [42]
            }
        }
    )
    
    trainer.fit(X_train.values, y_train.values)
    model = trainer.best_model_
    
    # 评估模型
    train_score = model.score(X_train.values, y_train.values)
    test_score = model.score(X_test.values, y_test.values)
    
    print(f"训练准确率: {train_score:.4f}")
    print(f"测试准确率: {test_score:.4f}")
    
    return model


def run_backtest(data, model, feature_columns):
    """运行回测"""
    print("运行机器学习策略回测...")
    
    # 配置回测参数
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_model=PercentageCommissionModel(commission_rate=0.001),  # 0.1% 手续费
        slippage_model=FixedSlippageModel(slippage_bps=5.0),  # 0.05% 滑点
        max_position_size=0.5  # 最大单个持仓占比50%
    )
    
    # 创建回测引擎
    engine = BarBacktestEngine(config)
    
    # 添加数据
    engine.add_data(data, "STOCK")
    
    # 创建策略
    strategy = MLStrategy(model, feature_columns, prediction_threshold=0.1)
    engine.add_strategy(strategy)
    
    # 运行回测
    results = engine.run()
    
    print(f"回测完成!")
    print(f"总收益率: {results.total_return:.2%}")
    print(f"年化收益率: {results.annualized_return:.2%}")
    print(f"最大回撤: {results.max_drawdown:.2%}")
    print(f"夏普比率: {results.sharpe_ratio:.4f}")
    print(f"交易次数: {len(results.trades)}")
    
    return results


def create_visualization(results, data):
    """创建可视化图表"""
    print("生成可视化图表...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. 价格走势
    ax1 = axes[0, 0]
    ax1.plot(data.index, data['close'], label='价格', linewidth=1)
    ax1.set_title('价格走势', fontsize=12, fontweight='bold')
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 资产净值曲线
    ax2 = axes[0, 1]
    if hasattr(results, 'equity_curve') and len(results.equity_curve) > 0:
        ax2.plot(results.equity_curve.index, results.equity_curve.values, label='投资组合净值', linewidth=2)
        ax2.axhline(y=results.initial_capital, color='gray', linestyle='--', 
                   label=f'初始资金: ${results.initial_capital:,.0f}')
    else:
        # 如果没有净值数据，显示初始资本的水平线
        ax2.axhline(y=results.initial_capital, label='初始资本', linewidth=2, color='blue')
    ax2.set_title('投资组合净值曲线', fontsize=12, fontweight='bold')
    ax2.set_ylabel('净值 ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 收益分布（移到左侧）
    ax3 = axes[1, 0]
    if len(results.trades) > 0:
        # 计算每笔交易的收益率
        trade_returns = []
        for trade in results.trades:
            if hasattr(trade, 'pnl'):
                trade_returns.append(trade.pnl / results.initial_capital)
        
        if trade_returns:
            ax3.hist(trade_returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(np.mean(trade_returns), color='red', linestyle='--', 
                       label=f'均值: {np.mean(trade_returns):.2%}')
        else:
            ax3.text(0.5, 0.5, '无交易数据', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, '无交易数据', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_title('交易收益分布', fontsize=12, fontweight='bold')
    ax3.set_xlabel('收益率')
    ax3.set_ylabel('频次')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 回撤图（移到右侧）
    ax4 = axes[1, 1]
    if hasattr(results, 'drawdown_series') and results.drawdown_series is not None and len(results.drawdown_series) > 0:
        ax4.fill_between(results.drawdown_series.index, 
                        results.drawdown_series.values * 100, 0,
                        alpha=0.3, color='red', label='回撤')
        ax4.plot(results.drawdown_series.index, 
                results.drawdown_series.values * 100, 
                color='darkred', linewidth=1)
        ax4.axhline(y=results.max_drawdown * 100, color='red', linestyle='--', 
                   label=f'最大回撤: {results.max_drawdown:.2%}')
    else:
        # 如果没有回撤数据，计算简单回撤
        if hasattr(results, 'equity_curve') and len(results.equity_curve) > 0:
            equity_values = results.equity_curve.values
            peak = np.maximum.accumulate(equity_values)
            drawdown = (equity_values - peak) / peak
            ax4.fill_between(results.equity_curve.index, 
                            drawdown * 100, 0,
                            alpha=0.3, color='red', label='回撤')
            ax4.plot(results.equity_curve.index, 
                    drawdown * 100, 
                    color='darkred', linewidth=1)
            max_dd = np.min(drawdown)
            ax4.axhline(y=max_dd * 100, color='red', linestyle='--', 
                       label=f'最大回撤: {max_dd:.2%}')
        else:
            ax4.text(0.5, 0.5, '无回撤数据', ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_title('投资组合回撤', fontsize=12, fontweight='bold')
    ax4.set_ylabel('回撤 (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()  # 回撤图通常倒置Y轴
    
    # 5. 关键指标
    ax5 = axes[2, 0]
    ax5.axis('off')
    
    # 计算交易统计
    total_trades = len(results.trades)
    profitable_trades = 0
    losing_trades = 0
    
    for trade in results.trades:
        if hasattr(trade, 'pnl'):
            if trade.pnl > 0:
                profitable_trades += 1
            elif trade.pnl < 0:
                losing_trades += 1
    
    win_rate = profitable_trades / max(total_trades, 1) * 100
    
    metrics_text = f"""关键绩效指标

总收益率: {results.total_return:.2%}
年化收益率: {results.annualized_return:.2%}
夏普比率: {results.sharpe_ratio:.4f}
最大回撤: {results.max_drawdown:.2%}

交易统计
总交易次数: {total_trades}
盈利交易: {profitable_trades}
亏损交易: {losing_trades}
胜率: {win_rate:.1f}%
"""
    
    ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 6. 预留位置（可用于其他图表）
    ax6 = axes[2, 1]
    ax6.axis('off')
    ax6.text(0.5, 0.5, '预留图表位置', ha='center', va='center', transform=ax6.transAxes, 
             fontsize=12, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('ml_backtest_results.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'ml_backtest_results.png'")
    
    return fig


def main():
    """主函数"""
    print("=== 机器学习回测流水线 ===\n")
    
    try:
        # 1. 生成数据
        print("生成模拟市场数据...")
        market_data = generate_sample_data(start_date='2020-01-01', end_date='2025-07-01')
        print(f"生成了 {len(market_data)} 天的市场数据 (2020.01.01 - 2025.07.01)")
        print(f"价格范围: {market_data['close'].min():.2f} - {market_data['close'].max():.2f}")
        
        # 2. 准备训练数据
        print("准备机器学习特征...")
        features, target, feature_columns = prepare_training_data(market_data)
        print(f"特征数量: {len(feature_columns)}")
        print(f"样本数量: {len(features)}")
        print(f"目标分布: {dict(target.value_counts())}")
        
        # 检查特征质量
        print(f"特征统计:")
        print(f"  - 特征均值范围: {features.mean().min():.4f} - {features.mean().max():.4f}")
        print(f"  - 特征标准差范围: {features.std().min():.4f} - {features.std().max():.4f}")
        print(f"  - 缺失值数量: {features.isnull().sum().sum()}")
        
        # 3. 训练模型
        print("训练机器学习模型...")
        model = train_model(features, target)
        
        # 检查模型预测
        print("检查模型预测质量...")
        predictions = model.predict(features)
        pred_proba = model.predict_proba(features)
        print(f"预测分布: {dict(pd.Series(predictions).value_counts())}")
        print(f"预测概率统计:")
        for i, class_name in enumerate(['下跌', '持平', '上涨']):
            print(f"  - {class_name}类概率: 均值={pred_proba[:, i].mean():.4f}, 标准差={pred_proba[:, i].std():.4f}")
        
        # 4. 运行回测
        results = run_backtest(market_data, model, feature_columns)
        
        # 详细分析回测结果
        print(f"\n=== 回测结果分析 ===")
        print(f"总交易次数: {len(results.trades)}")
        print(f"总收益率: {results.total_return:.4f}")
        print(f"年化收益率: {results.annualized_return:.4f}")
        print(f"夏普比率: {results.sharpe_ratio:.4f}")
        print(f"最大回撤: {results.max_drawdown:.4f}")
        
        if len(results.trades) > 0:
            trade_pnls = [trade.pnl if hasattr(trade, 'pnl') else 0 for trade in results.trades]
            print(f"交易盈亏统计:")
            print(f"  - 平均盈亏: {np.mean(trade_pnls):.2f}")
            print(f"  - 最大盈利: {max(trade_pnls):.2f}")
            print(f"  - 最大亏损: {min(trade_pnls):.2f}")
        else:
            print("⚠️ 警告: 没有产生任何交易!")
        
        # 5. 生成可视化
        fig = create_visualization(results, market_data)
        
        print("\n=== 流程完成 ===")
        print("所有结果已生成，请查看 'ml_backtest_results.png' 文件")
        
        return results, fig
        
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    results, figure = main()