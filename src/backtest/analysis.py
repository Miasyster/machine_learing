"""
回测结果分析和可视化模块

提供全面的回测结果分析功能，包括：
- 性能指标计算
- 风险分析
- 交易分析
- 可视化图表
- 报告生成
"""

import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly未安装，将使用matplotlib进行可视化")

from .base import BacktestResult, Trade, Position


class BacktestAnalyzer:
    """回测结果分析器"""
    
    def __init__(self, result: BacktestResult):
        """
        初始化分析器
        
        Args:
            result: 回测结果
        """
        self.result = result
        self.trades_df = self._prepare_trades_dataframe()
        self.positions_df = self._prepare_positions_dataframe()
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 设置更全面的中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
        plt.rcParams['font.family'] = 'sans-serif'
    
    def _prepare_trades_dataframe(self) -> pd.DataFrame:
        """准备交易数据框"""
        if not self.result.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.result.trades:
            trades_data.append({
                'trade_id': trade.trade_id,
                'order_id': trade.order_id,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'timestamp': trade.timestamp,
                'commission': trade.commission,
                'value': trade.quantity * trade.price,
                'metadata': trade.metadata
            })
        
        df = pd.DataFrame(trades_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def _prepare_positions_dataframe(self) -> pd.DataFrame:
        """准备持仓数据框"""
        if not self.result.positions:
            return pd.DataFrame()
        
        positions_data = []
        for position in self.result.positions:
            positions_data.append({
                'symbol': position.symbol,
                'side': position.side.value,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'timestamp': position.timestamp
            })
        
        df = pd.DataFrame(positions_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def calculate_advanced_metrics(self) -> Dict[str, float]:
        """计算高级性能指标"""
        metrics = {}
        
        if self.result.equity_curve is None or self.result.equity_curve.empty:
            return metrics
        
        equity = self.result.equity_curve
        returns = self.result.daily_returns
        
        # 基础指标
        metrics['total_return'] = self.result.total_return
        metrics['annualized_return'] = self.result.annualized_return
        metrics['volatility'] = returns.std() * np.sqrt(252) if not returns.empty else 0
        metrics['sharpe_ratio'] = self.result.sharpe_ratio
        metrics['sortino_ratio'] = self.result.sortino_ratio
        metrics['max_drawdown'] = self.result.max_drawdown
        
        # 高级风险指标
        if not returns.empty:
            # VaR (Value at Risk)
            metrics['var_95'] = returns.quantile(0.05)
            metrics['var_99'] = returns.quantile(0.01)
            
            # CVaR (Conditional Value at Risk)
            metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
            metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
            
            # 偏度和峰度
            metrics['skewness'] = returns.skew()
            metrics['kurtosis'] = returns.kurtosis()
            
            # 下行偏差
            downside_returns = returns[returns < 0]
            metrics['downside_deviation'] = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
            
            # Calmar比率
            metrics['calmar_ratio'] = self.result.annualized_return / self.result.max_drawdown if self.result.max_drawdown > 0 else 0
            
            # 信息比率 (假设基准收益率为0)
            excess_returns = returns
            metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # 交易相关指标
        metrics['win_rate'] = self.result.win_rate
        metrics['profit_factor'] = self.result.profit_factor
        metrics['total_trades'] = self.result.total_trades
        metrics['avg_trade_return'] = self.result.avg_trade_return
        
        # 回撤相关指标
        if self.result.drawdown_series is not None and not self.result.drawdown_series.empty:
            drawdown = self.result.drawdown_series
            
            # 平均回撤
            drawdown_periods = drawdown[drawdown < 0]
            metrics['avg_drawdown'] = drawdown_periods.mean() if not drawdown_periods.empty else 0
            
            # 回撤持续时间
            drawdown_duration = self._calculate_drawdown_duration(drawdown)
            metrics['max_drawdown_duration'] = drawdown_duration['max_duration']
            metrics['avg_drawdown_duration'] = drawdown_duration['avg_duration']
        
        # 收益稳定性
        if not returns.empty and len(returns) >= 12:
            # 月度收益率
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if not monthly_returns.empty:
                metrics['monthly_win_rate'] = (monthly_returns > 0).mean()
                metrics['best_month'] = monthly_returns.max()
                metrics['worst_month'] = monthly_returns.min()
        
        return metrics
    
    def _calculate_drawdown_duration(self, drawdown_series: pd.Series) -> Dict[str, int]:
        """计算回撤持续时间"""
        durations = []
        current_duration = 0
        
        for dd in drawdown_series:
            if dd < 0:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # 处理最后一个回撤期
        if current_duration > 0:
            durations.append(current_duration)
        
        return {
            'max_duration': max(durations) if durations else 0,
            'avg_duration': np.mean(durations) if durations else 0
        }
    
    def analyze_trades(self) -> Dict[str, Any]:
        """分析交易表现"""
        if self.trades_df.empty:
            return {}
        
        analysis = {}
        
        # 按标的分析
        symbol_analysis = {}
        for symbol in self.trades_df['symbol'].unique():
            symbol_trades = self.trades_df[self.trades_df['symbol'] == symbol]
            
            # 计算该标的的PnL (简化计算)
            buy_trades = symbol_trades[symbol_trades['side'] == 'BUY']
            sell_trades = symbol_trades[symbol_trades['side'] == 'SELL']
            
            symbol_analysis[symbol] = {
                'total_trades': len(symbol_trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'total_volume': symbol_trades['quantity'].sum(),
                'total_value': symbol_trades['value'].sum(),
                'avg_trade_size': symbol_trades['quantity'].mean(),
                'total_commission': symbol_trades['commission'].sum()
            }
        
        analysis['by_symbol'] = symbol_analysis
        
        # 时间分析
        if 'timestamp' in self.trades_df.columns:
            self.trades_df['hour'] = self.trades_df['timestamp'].dt.hour
            self.trades_df['day_of_week'] = self.trades_df['timestamp'].dt.day_name()
            self.trades_df['month'] = self.trades_df['timestamp'].dt.month
            
            analysis['by_hour'] = self.trades_df.groupby('hour').size().to_dict()
            analysis['by_day_of_week'] = self.trades_df.groupby('day_of_week').size().to_dict()
            analysis['by_month'] = self.trades_df.groupby('month').size().to_dict()
        
        # 交易规模分析
        analysis['trade_size_stats'] = {
            'min_quantity': self.trades_df['quantity'].min(),
            'max_quantity': self.trades_df['quantity'].max(),
            'avg_quantity': self.trades_df['quantity'].mean(),
            'median_quantity': self.trades_df['quantity'].median(),
            'std_quantity': self.trades_df['quantity'].std()
        }
        
        # 价格分析
        analysis['price_stats'] = {
            'min_price': self.trades_df['price'].min(),
            'max_price': self.trades_df['price'].max(),
            'avg_price': self.trades_df['price'].mean(),
            'median_price': self.trades_df['price'].median(),
            'std_price': self.trades_df['price'].std()
        }
        
        return analysis
    
    def plot_equity_curve(self, save_path: Optional[str] = None, 
                         use_plotly: bool = True) -> Optional[Any]:
        """绘制权益曲线"""
        if self.result.equity_curve is None or self.result.equity_curve.empty:
            print("没有权益曲线数据")
            return None
        
        equity = self.result.equity_curve
        
        if use_plotly and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity.values,
                mode='lines',
                name='权益曲线',
                line=dict(color='blue', width=2)
            ))
            
            # 添加基准线
            fig.add_hline(
                y=self.result.initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="初始资金"
            )
            
            fig.update_layout(
                title='权益曲线',
                xaxis_title='时间',
                yaxis_title='组合价值',
                hovermode='x unified',
                template='plotly_white'
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        
        else:
            # 使用matplotlib
            plt.figure(figsize=(12, 6))
            plt.plot(equity.index, equity.values, linewidth=2, label='权益曲线')
            plt.axhline(y=self.result.initial_capital, color='gray', linestyle='--', alpha=0.7, label='初始资金')
            
            plt.title('权益曲线')
            plt.xlabel('时间')
            plt.ylabel('组合价值')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return plt.gcf()
    
    def plot_drawdown(self, save_path: Optional[str] = None, 
                     use_plotly: bool = True) -> Optional[Any]:
        """绘制回撤图"""
        if self.result.drawdown_series is None or self.result.drawdown_series.empty:
            print("没有回撤数据")
            return None
        
        drawdown = self.result.drawdown_series * 100  # 转换为百分比
        
        if use_plotly and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='回撤',
                fill='tonexty',
                line=dict(color='red', width=1),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            fig.add_hline(y=0, line_color="black", line_width=1)
            
            fig.update_layout(
                title='回撤分析',
                xaxis_title='时间',
                yaxis_title='回撤 (%)',
                hovermode='x unified',
                template='plotly_white'
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        
        else:
            # 使用matplotlib
            plt.figure(figsize=(12, 6))
            plt.fill_between(drawdown.index, drawdown.values, 0, 
                           color='red', alpha=0.3, label='回撤')
            plt.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            plt.axhline(y=0, color='black', linewidth=1)
            
            plt.title('回撤分析')
            plt.xlabel('时间')
            plt.ylabel('回撤 (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return plt.gcf()
    
    def plot_returns_distribution(self, save_path: Optional[str] = None, 
                                use_plotly: bool = True) -> Optional[Any]:
        """绘制收益率分布"""
        if self.result.daily_returns is None or self.result.daily_returns.empty:
            print("没有收益率数据")
            return None
        
        returns = self.result.daily_returns * 100  # 转换为百分比
        
        if use_plotly and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=returns.values,
                nbinsx=50,
                name='收益率分布',
                opacity=0.7
            ))
            
            # 添加正态分布拟合
            mean_return = returns.mean()
            std_return = returns.std()
            x_norm = np.linspace(returns.min(), returns.max(), 100)
            y_norm = len(returns) * (returns.max() - returns.min()) / 50 * \
                    (1 / (std_return * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * ((x_norm - mean_return) / std_return) ** 2)
            
            fig.add_trace(go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='正态分布拟合',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title='日收益率分布',
                xaxis_title='收益率 (%)',
                yaxis_title='频次',
                template='plotly_white'
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        
        else:
            # 使用matplotlib
            plt.figure(figsize=(10, 6))
            
            # 绘制直方图
            n, bins, patches = plt.hist(returns.values, bins=50, alpha=0.7, 
                                      density=True, label='收益率分布')
            
            # 添加正态分布拟合
            mean_return = returns.mean()
            std_return = returns.std()
            x_norm = np.linspace(returns.min(), returns.max(), 100)
            y_norm = (1 / (std_return * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * ((x_norm - mean_return) / std_return) ** 2)
            plt.plot(x_norm, y_norm, 'r-', linewidth=2, label='正态分布拟合')
            
            plt.title('日收益率分布')
            plt.xlabel('收益率 (%)')
            plt.ylabel('密度')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return plt.gcf()
    
    def plot_rolling_metrics(self, window: int = 252, save_path: Optional[str] = None,
                           use_plotly: bool = True) -> Optional[Any]:
        """绘制滚动指标"""
        if self.result.daily_returns is None or self.result.daily_returns.empty:
            print("没有收益率数据")
            return None
        
        returns = self.result.daily_returns
        
        # 计算滚动指标
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        if use_plotly and PLOTLY_AVAILABLE:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['滚动年化收益率', '滚动年化波动率', '滚动夏普比率'],
                vertical_spacing=0.08
            )
            
            # 滚动收益率
            fig.add_trace(go.Scatter(
                x=rolling_return.index,
                y=rolling_return.values * 100,
                mode='lines',
                name='年化收益率 (%)',
                line=dict(color='blue')
            ), row=1, col=1)
            
            # 滚动波动率
            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values * 100,
                mode='lines',
                name='年化波动率 (%)',
                line=dict(color='orange')
            ), row=2, col=1)
            
            # 滚动夏普比率
            fig.add_trace(go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='夏普比率',
                line=dict(color='green')
            ), row=3, col=1)
            
            fig.update_layout(
                title=f'滚动指标分析 (窗口: {window}天)',
                height=800,
                template='plotly_white',
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        
        else:
            # 使用matplotlib
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # 滚动收益率
            axes[0].plot(rolling_return.index, rolling_return.values * 100, 
                        color='blue', linewidth=1.5)
            axes[0].set_title('滚动年化收益率')
            axes[0].set_ylabel('收益率 (%)')
            axes[0].grid(True, alpha=0.3)
            
            # 滚动波动率
            axes[1].plot(rolling_vol.index, rolling_vol.values * 100, 
                        color='orange', linewidth=1.5)
            axes[1].set_title('滚动年化波动率')
            axes[1].set_ylabel('波动率 (%)')
            axes[1].grid(True, alpha=0.3)
            
            # 滚动夏普比率
            axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, 
                        color='green', linewidth=1.5)
            axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            axes[2].set_title('滚动夏普比率')
            axes[2].set_ylabel('夏普比率')
            axes[2].set_xlabel('时间')
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'滚动指标分析 (窗口: {window}天)')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return fig
    
    def plot_trade_analysis(self, save_path: Optional[str] = None,
                          use_plotly: bool = True) -> Optional[Any]:
        """绘制交易分析图"""
        if self.trades_df.empty:
            print("没有交易数据")
            return None
        
        trade_analysis = self.analyze_trades()
        
        if use_plotly and PLOTLY_AVAILABLE:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['按标的交易次数', '按小时交易分布', '交易规模分布', '价格分布'],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "histogram"}]]
            )
            
            # 按标的交易次数
            if 'by_symbol' in trade_analysis:
                symbols = list(trade_analysis['by_symbol'].keys())
                trade_counts = [trade_analysis['by_symbol'][s]['total_trades'] for s in symbols]
                
                fig.add_trace(go.Bar(
                    x=symbols,
                    y=trade_counts,
                    name='交易次数'
                ), row=1, col=1)
            
            # 按小时交易分布
            if 'by_hour' in trade_analysis:
                hours = list(trade_analysis['by_hour'].keys())
                hour_counts = list(trade_analysis['by_hour'].values())
                
                fig.add_trace(go.Bar(
                    x=hours,
                    y=hour_counts,
                    name='小时分布'
                ), row=1, col=2)
            
            # 交易规模分布
            fig.add_trace(go.Histogram(
                x=self.trades_df['quantity'],
                name='交易规模'
            ), row=2, col=1)
            
            # 价格分布
            fig.add_trace(go.Histogram(
                x=self.trades_df['price'],
                name='价格分布'
            ), row=2, col=2)
            
            fig.update_layout(
                title='交易分析',
                height=800,
                template='plotly_white',
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        
        else:
            # 使用matplotlib
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 按标的交易次数
            if 'by_symbol' in trade_analysis:
                symbols = list(trade_analysis['by_symbol'].keys())
                trade_counts = [trade_analysis['by_symbol'][s]['total_trades'] for s in symbols]
                
                axes[0, 0].bar(symbols, trade_counts)
                axes[0, 0].set_title('按标的交易次数')
                axes[0, 0].set_ylabel('交易次数')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 按小时交易分布
            if 'by_hour' in trade_analysis:
                hours = list(trade_analysis['by_hour'].keys())
                hour_counts = list(trade_analysis['by_hour'].values())
                
                axes[0, 1].bar(hours, hour_counts)
                axes[0, 1].set_title('按小时交易分布')
                axes[0, 1].set_xlabel('小时')
                axes[0, 1].set_ylabel('交易次数')
            
            # 交易规模分布
            axes[1, 0].hist(self.trades_df['quantity'], bins=30, alpha=0.7)
            axes[1, 0].set_title('交易规模分布')
            axes[1, 0].set_xlabel('交易数量')
            axes[1, 0].set_ylabel('频次')
            
            # 价格分布
            axes[1, 1].hist(self.trades_df['price'], bins=30, alpha=0.7)
            axes[1, 1].set_title('价格分布')
            axes[1, 1].set_xlabel('价格')
            axes[1, 1].set_ylabel('频次')
            
            plt.suptitle('交易分析')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            return fig
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """生成回测报告"""
        advanced_metrics = self.calculate_advanced_metrics()
        trade_analysis = self.analyze_trades()
        
        report = f"""
# 回测分析报告

## 基本信息
- 回测期间: {self.result.start_date.strftime('%Y-%m-%d')} 至 {self.result.end_date.strftime('%Y-%m-%d')}
- 初始资金: {self.result.initial_capital:,.2f}
- 最终资金: {self.result.final_capital:,.2f}
- 回测天数: {(self.result.end_date - self.result.start_date).days}

## 收益指标
- 总收益率: {self.result.total_return:.2%}
- 年化收益率: {self.result.annualized_return:.2%}
- 年化波动率: {advanced_metrics.get('volatility', 0):.2%}
- 夏普比率: {self.result.sharpe_ratio:.3f}
- Sortino比率: {self.result.sortino_ratio:.3f}
- Calmar比率: {advanced_metrics.get('calmar_ratio', 0):.3f}
- 信息比率: {advanced_metrics.get('information_ratio', 0):.3f}

## 风险指标
- 最大回撤: {self.result.max_drawdown:.2%}
- 平均回撤: {advanced_metrics.get('avg_drawdown', 0):.2%}
- 最大回撤持续时间: {advanced_metrics.get('max_drawdown_duration', 0)}天
- 平均回撤持续时间: {advanced_metrics.get('avg_drawdown_duration', 0):.1f}天
- VaR (95%): {advanced_metrics.get('var_95', 0):.2%}
- CVaR (95%): {advanced_metrics.get('cvar_95', 0):.2%}
- 下行偏差: {advanced_metrics.get('downside_deviation', 0):.2%}

## 交易统计
- 总交易次数: {self.result.total_trades}
- 盈利交易: {self.result.winning_trades}
- 亏损交易: {self.result.losing_trades}
- 胜率: {self.result.win_rate:.2%}
- 盈利因子: {self.result.profit_factor:.3f}
- 平均交易收益: {self.result.avg_trade_return:.2f}
- 平均盈利交易: {self.result.avg_winning_trade:.2f}
- 平均亏损交易: {self.result.avg_losing_trade:.2f}
- 最大连续盈利: {self.result.max_consecutive_wins}
- 最大连续亏损: {self.result.max_consecutive_losses}

## 收益率分布
- 偏度: {advanced_metrics.get('skewness', 0):.3f}
- 峰度: {advanced_metrics.get('kurtosis', 0):.3f}
- 最佳月份: {advanced_metrics.get('best_month', 0):.2%}
- 最差月份: {advanced_metrics.get('worst_month', 0):.2%}
- 月度胜率: {advanced_metrics.get('monthly_win_rate', 0):.2%}
"""
        
        if trade_analysis and 'by_symbol' in trade_analysis:
            report += "\n## 按标的分析\n"
            for symbol, stats in trade_analysis['by_symbol'].items():
                report += f"### {symbol}\n"
                report += f"- 交易次数: {stats['total_trades']}\n"
                report += f"- 总交易量: {stats['total_volume']:,.0f}\n"
                report += f"- 总交易额: {stats['total_value']:,.2f}\n"
                report += f"- 平均交易规模: {stats['avg_trade_size']:,.0f}\n"
                report += f"- 总手续费: {stats['total_commission']:,.2f}\n\n"
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def create_dashboard(self, save_dir: Optional[str] = None, 
                        use_plotly: bool = True) -> Dict[str, Any]:
        """创建完整的分析仪表板"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        dashboard = {}
        
        # 生成各种图表
        dashboard['equity_curve'] = self.plot_equity_curve(
            save_path=str(save_dir / 'equity_curve.html') if save_dir and use_plotly else None,
            use_plotly=use_plotly
        )
        
        dashboard['drawdown'] = self.plot_drawdown(
            save_path=str(save_dir / 'drawdown.html') if save_dir and use_plotly else None,
            use_plotly=use_plotly
        )
        
        dashboard['returns_distribution'] = self.plot_returns_distribution(
            save_path=str(save_dir / 'returns_distribution.html') if save_dir and use_plotly else None,
            use_plotly=use_plotly
        )
        
        dashboard['rolling_metrics'] = self.plot_rolling_metrics(
            save_path=str(save_dir / 'rolling_metrics.html') if save_dir and use_plotly else None,
            use_plotly=use_plotly
        )
        
        dashboard['trade_analysis'] = self.plot_trade_analysis(
            save_path=str(save_dir / 'trade_analysis.html') if save_dir and use_plotly else None,
            use_plotly=use_plotly
        )
        
        # 生成报告
        dashboard['report'] = self.generate_report(
            save_path=str(save_dir / 'backtest_report.md') if save_dir else None
        )
        
        # 计算高级指标
        dashboard['advanced_metrics'] = self.calculate_advanced_metrics()
        dashboard['trade_analysis_data'] = self.analyze_trades()
        
        return dashboard


# 便利函数
def analyze_backtest_result(result: BacktestResult, 
                          save_dir: Optional[str] = None,
                          use_plotly: bool = True) -> BacktestAnalyzer:
    """
    分析回测结果的便利函数
    
    Args:
        result: 回测结果
        save_dir: 保存目录
        use_plotly: 是否使用plotly
        
    Returns:
        分析器对象
    """
    analyzer = BacktestAnalyzer(result)
    
    if save_dir:
        analyzer.create_dashboard(save_dir, use_plotly)
    
    return analyzer


def compare_backtest_results(results: Dict[str, BacktestResult], 
                           save_path: Optional[str] = None) -> pd.DataFrame:
    """
    比较多个回测结果
    
    Args:
        results: 回测结果字典 {name: result}
        save_path: 保存路径
        
    Returns:
        比较结果DataFrame
    """
    comparison_data = []
    
    for name, result in results.items():
        analyzer = BacktestAnalyzer(result)
        metrics = analyzer.calculate_advanced_metrics()
        
        comparison_data.append({
            'Strategy': name,
            'Total Return': result.total_return,
            'Annualized Return': result.annualized_return,
            'Volatility': metrics.get('volatility', 0),
            'Sharpe Ratio': result.sharpe_ratio,
            'Sortino Ratio': result.sortino_ratio,
            'Max Drawdown': result.max_drawdown,
            'Calmar Ratio': metrics.get('calmar_ratio', 0),
            'Win Rate': result.win_rate,
            'Profit Factor': result.profit_factor,
            'Total Trades': result.total_trades
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if save_path:
        comparison_df.to_csv(save_path, index=False)
    
    return comparison_df