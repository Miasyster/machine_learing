"""
模型解释可视化工具
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .base import ExplanationResult

logger = logging.getLogger(__name__)


class ExplanationVisualizer:
    """模型解释可视化器"""
    
    def __init__(self,
                 style: str = 'seaborn',
                 figsize: Tuple[int, int] = (10, 6),
                 dpi: int = 100,
                 color_palette: str = 'viridis'):
        """
        初始化可视化器
        
        Args:
            style: 绘图风格
            figsize: 图形大小
            dpi: 图形分辨率
            color_palette: 颜色调色板
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = color_palette
        
        # 设置绘图风格
        plt.style.use('default')
        if style == 'seaborn':
            sns.set_style("whitegrid")
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_feature_importance(self,
                               result: ExplanationResult,
                               top_n: int = 20,
                               method: str = 'primary',
                               show_std: bool = True,
                               save_path: Optional[str] = None) -> Figure:
        """
        绘制特征重要性图
        
        Args:
            result: 解释结果
            top_n: 显示的特征数量
            method: 重要性方法
            show_std: 是否显示标准差
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        # 选择重要性数据
        if method == 'primary' and result.feature_importance is not None:
            importance = result.feature_importance
            std = result.feature_importance_std
        elif method == 'permutation' and result.permutation_importance is not None:
            importance = result.permutation_importance
            std = result.feature_importance_std
        else:
            raise ValueError(f"Method '{method}' not available in results")
        
        # 获取top_n特征
        top_indices = np.argsort(importance)[-top_n:]
        top_importance = importance[top_indices]
        top_features = [result.feature_names[i] for i in top_indices]
        top_std = std[top_indices] if std is not None and show_std else None
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 绘制水平条形图
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_importance, 
                      xerr=top_std if top_std is not None else None,
                      capsize=3, alpha=0.8,
                      color=plt.cm.get_cmap(self.color_palette)(np.linspace(0, 1, len(top_features))))
        
        # 设置标签和标题
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('特征重要性')
        ax.set_title(f'特征重要性排名 (Top {top_n}) - {method.title()}')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_importance)):
            ax.text(value + (top_std[i] if top_std is not None else 0) + 0.01 * max(top_importance),
                   bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}',
                   va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_shap_summary(self,
                         result: ExplanationResult,
                         plot_type: str = 'bar',
                         max_display: int = 20,
                         save_path: Optional[str] = None) -> Figure:
        """
        绘制SHAP摘要图
        
        Args:
            result: 解释结果
            plot_type: 图形类型 ('bar', 'beeswarm', 'violin')
            max_display: 最大显示特征数
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        if result.shap_values is None:
            raise ValueError("SHAP values not available in results")
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, creating custom summary plot")
            return self._plot_custom_shap_summary(result, max_display, save_path)
        
        # 使用SHAP的内置绘图功能
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        try:
            if plot_type == 'bar':
                shap.summary_plot(result.shap_values, 
                                feature_names=result.feature_names,
                                plot_type='bar',
                                max_display=max_display,
                                show=False)
            elif plot_type == 'beeswarm':
                shap.summary_plot(result.shap_values,
                                feature_names=result.feature_names,
                                max_display=max_display,
                                show=False)
            elif plot_type == 'violin':
                shap.summary_plot(result.shap_values,
                                feature_names=result.feature_names,
                                plot_type='violin',
                                max_display=max_display,
                                show=False)
            
            plt.title(f'SHAP 特征重要性摘要 ({plot_type.title()})')
            
        except Exception as e:
            logger.warning(f"SHAP plotting failed: {e}, using custom plot")
            plt.close(fig)
            return self._plot_custom_shap_summary(result, max_display, save_path)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_custom_shap_summary(self,
                                 result: ExplanationResult,
                                 max_display: int,
                                 save_path: Optional[str] = None) -> Figure:
        """自定义SHAP摘要图"""
        # 计算平均绝对SHAP值
        mean_abs_shap = np.abs(result.shap_values).mean(axis=0)
        
        # 获取top特征
        top_indices = np.argsort(mean_abs_shap)[-max_display:]
        top_features = [result.feature_names[i] for i in top_indices]
        top_values = mean_abs_shap[top_indices]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_values, alpha=0.8,
                      color=plt.cm.get_cmap(self.color_palette)(np.linspace(0, 1, len(top_features))))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('平均绝对SHAP值')
        ax.set_title(f'SHAP 特征重要性摘要 (Top {max_display})')
        
        # 添加数值标签
        for bar, value in zip(bars, top_values):
            ax.text(value + 0.01 * max(top_values),
                   bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}',
                   va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_shap_waterfall(self,
                           result: ExplanationResult,
                           instance_idx: int,
                           save_path: Optional[str] = None) -> Figure:
        """
        绘制SHAP瀑布图（单个实例）
        
        Args:
            result: 解释结果
            instance_idx: 实例索引
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        if result.shap_values is None:
            raise ValueError("SHAP values not available in results")
        
        if instance_idx >= len(result.shap_values):
            raise ValueError(f"Instance index {instance_idx} out of range")
        
        instance_shap = result.shap_values[instance_idx]
        expected_value = result.shap_expected_value if result.shap_expected_value is not None else 0
        
        # 创建瀑布图数据
        feature_contributions = list(zip(result.feature_names, instance_shap))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 限制显示的特征数量
        max_features = min(15, len(feature_contributions))
        top_contributions = feature_contributions[:max_features]
        
        # 计算累积值
        cumulative = [expected_value]
        for _, value in top_contributions:
            cumulative.append(cumulative[-1] + value)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        x_pos = np.arange(len(top_contributions) + 2)
        
        # 绘制基准值
        ax.bar(0, expected_value, color='gray', alpha=0.7, label='基准值')
        
        # 绘制特征贡献
        for i, (feature, value) in enumerate(top_contributions):
            color = 'green' if value > 0 else 'red'
            ax.bar(i + 1, value, bottom=cumulative[i], 
                  color=color, alpha=0.7)
            
            # 添加连接线
            if i < len(top_contributions) - 1:
                ax.plot([i + 1.4, i + 1.6], [cumulative[i + 1], cumulative[i + 1]], 
                       'k--', alpha=0.5)
        
        # 绘制最终预测
        final_value = cumulative[-1]
        ax.bar(len(top_contributions) + 1, final_value, color='blue', alpha=0.7, label='预测值')
        
        # 设置标签
        labels = ['基准值'] + [f'{name}\n({value:+.3f})' for name, value in top_contributions] + ['预测值']
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('预测值')
        ax.set_title(f'SHAP 瀑布图 - 实例 {instance_idx}')
        
        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_dependence(self,
                               X: np.ndarray,
                               shap_values: np.ndarray,
                               feature_idx: int,
                               feature_names: List[str],
                               interaction_idx: Optional[int] = None,
                               save_path: Optional[str] = None) -> Figure:
        """
        绘制特征依赖图
        
        Args:
            X: 输入数据
            shap_values: SHAP值
            feature_idx: 主要特征索引
            feature_names: 特征名称
            interaction_idx: 交互特征索引
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        x_values = X[:, feature_idx]
        y_values = shap_values[:, feature_idx]
        
        if interaction_idx is not None:
            # 使用交互特征进行颜色编码
            colors = X[:, interaction_idx]
            scatter = ax.scatter(x_values, y_values, c=colors, 
                               cmap=self.color_palette, alpha=0.6)
            plt.colorbar(scatter, label=feature_names[interaction_idx])
        else:
            ax.scatter(x_values, y_values, alpha=0.6, 
                      color=plt.cm.get_cmap(self.color_palette)(0.5))
        
        ax.set_xlabel(feature_names[feature_idx])
        ax.set_ylabel(f'SHAP值 ({feature_names[feature_idx]})')
        ax.set_title(f'特征依赖图 - {feature_names[feature_idx]}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_importance_comparison(self,
                                  result: ExplanationResult,
                                  methods: List[str] = None,
                                  top_n: int = 15,
                                  save_path: Optional[str] = None) -> Figure:
        """
        比较不同重要性方法的结果
        
        Args:
            result: 解释结果
            methods: 要比较的方法列表
            top_n: 显示的特征数量
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        if result.visualization_data is None or 'importance_methods' not in result.visualization_data:
            raise ValueError("Importance comparison data not available")
        
        importance_data = result.visualization_data['importance_methods']
        
        if methods is None:
            methods = [method for method, data in importance_data.items() if data is not None]
        
        # 准备数据
        comparison_data = {}
        for method in methods:
            if method in importance_data and importance_data[method] is not None:
                data = importance_data[method]
                if isinstance(data, dict) and 'importance' in data:
                    comparison_data[method] = data['importance']
                elif isinstance(data, np.ndarray):
                    comparison_data[method] = data
        
        if len(comparison_data) < 2:
            raise ValueError("Need at least 2 methods for comparison")
        
        # 获取所有方法中的top特征
        all_importance = np.zeros(len(result.feature_names))
        for scores in comparison_data.values():
            all_importance += scores / len(comparison_data)
        
        top_indices = np.argsort(all_importance)[-top_n:]
        top_features = [result.feature_names[i] for i in top_indices]
        
        # 创建比较数据框
        comparison_df = pd.DataFrame(index=top_features)
        for method, scores in comparison_data.items():
            comparison_df[method] = scores[top_indices]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        # 绘制分组条形图
        x = np.arange(len(top_features))
        width = 0.8 / len(comparison_data)
        
        for i, (method, scores) in enumerate(comparison_df.items()):
            offset = (i - len(comparison_data)/2 + 0.5) * width
            ax.bar(x + offset, scores, width, label=method, alpha=0.8)
        
        ax.set_xlabel('特征')
        ax.set_ylabel('重要性分数')
        ax.set_title(f'特征重要性方法比较 (Top {top_n})')
        ax.set_xticks(x)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_stability_analysis(self,
                               stability_data: Dict[str, Any],
                               top_n: int = 15,
                               save_path: Optional[str] = None) -> Figure:
        """
        绘制稳定性分析图
        
        Args:
            stability_data: 稳定性分析数据
            top_n: 显示的特征数量
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        if 'coefficient_of_variation' not in stability_data:
            raise ValueError("Coefficient of variation data not available")
        
        cv = stability_data['coefficient_of_variation']
        mean_importance = stability_data['mean_importance']
        feature_names = [item[0] for item in stability_data['stability_ranking']]
        
        # 获取最稳定的特征
        stable_indices = np.argsort(cv)[:top_n]
        stable_features = [feature_names[i] for i in stable_indices]
        stable_cv = cv[stable_indices]
        stable_importance = mean_importance[stable_indices]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # 左图：变异系数
        y_pos = np.arange(len(stable_features))
        bars1 = ax1.barh(y_pos, stable_cv, alpha=0.8, color='orange')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(stable_features)
        ax1.set_xlabel('变异系数 (越小越稳定)')
        ax1.set_title(f'特征稳定性排名 (Top {top_n})')
        
        # 添加数值标签
        for bar, value in zip(bars1, stable_cv):
            ax1.text(value + 0.01 * max(stable_cv),
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}',
                    va='center', ha='left', fontsize=9)
        
        # 右图：重要性 vs 稳定性散点图
        ax2.scatter(mean_importance, cv, alpha=0.6, s=50)
        
        # 标注最稳定的几个特征
        for i in stable_indices[:5]:
            ax2.annotate(feature_names[i], 
                        (mean_importance[i], cv[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        ax2.set_xlabel('平均重要性')
        ax2.set_ylabel('变异系数')
        ax2.set_title('重要性 vs 稳定性')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_explanation_dashboard(self,
                                   result: ExplanationResult,
                                   X: Optional[np.ndarray] = None,
                                   instance_idx: Optional[int] = None,
                                   save_path: Optional[str] = None) -> Figure:
        """
        创建综合解释仪表板
        
        Args:
            result: 解释结果
            X: 输入数据（用于依赖图）
            instance_idx: 实例索引（用于局部解释）
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        # 确定子图布局
        if result.shap_values is not None and result.feature_importance is not None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
            axes = axes.flatten()
        elif result.shap_values is not None or result.feature_importance is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
            axes = axes.flatten()
        else:
            raise ValueError("No explanation data available")
        
        plot_idx = 0
        
        # 特征重要性图
        if result.feature_importance is not None:
            ax = axes[plot_idx]
            top_indices = np.argsort(result.feature_importance)[-15:]
            top_importance = result.feature_importance[top_indices]
            top_features = [result.feature_names[i] for i in top_indices]
            
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_importance, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.set_xlabel('特征重要性')
            ax.set_title('特征重要性排名')
            plot_idx += 1
        
        # SHAP摘要图
        if result.shap_values is not None:
            ax = axes[plot_idx]
            mean_abs_shap = np.abs(result.shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-15:]
            top_shap = mean_abs_shap[top_indices]
            top_features = [result.feature_names[i] for i in top_indices]
            
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_shap, alpha=0.8, color='green')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.set_xlabel('平均绝对SHAP值')
            ax.set_title('SHAP特征重要性')
            plot_idx += 1
        
        # 实例解释（如果提供）
        if instance_idx is not None and result.shap_values is not None:
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                instance_shap = result.shap_values[instance_idx]
                
                # 获取top贡献特征
                feature_contributions = list(zip(result.feature_names, instance_shap))
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                top_contributions = feature_contributions[:10]
                
                features, values = zip(*top_contributions)
                colors = ['green' if v > 0 else 'red' for v in values]
                
                y_pos = np.arange(len(features))
                ax.barh(y_pos, values, color=colors, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('SHAP值')
                ax.set_title(f'实例 {instance_idx} 的特征贡献')
                plot_idx += 1
        
        # 方法比较（如果有多种方法）
        if (result.visualization_data and 
            'importance_methods' in result.visualization_data and
            plot_idx < len(axes)):
            
            ax = axes[plot_idx]
            importance_data = result.visualization_data['importance_methods']
            valid_methods = {k: v for k, v in importance_data.items() 
                           if v is not None and isinstance(v, (np.ndarray, dict))}
            
            if len(valid_methods) >= 2:
                # 简化的方法比较
                method_names = list(valid_methods.keys())[:2]
                method_scores = []
                
                for method in method_names:
                    data = valid_methods[method]
                    if isinstance(data, dict) and 'importance' in data:
                        scores = data['importance']
                    else:
                        scores = data
                    method_scores.append(scores)
                
                # 计算相关性
                if len(method_scores) == 2:
                    correlation = np.corrcoef(method_scores[0], method_scores[1])[0, 1]
                    ax.scatter(method_scores[0], method_scores[1], alpha=0.6)
                    ax.set_xlabel(f'{method_names[0]} 重要性')
                    ax.set_ylabel(f'{method_names[1]} 重要性')
                    ax.set_title(f'方法比较 (相关性: {correlation:.3f})')
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig