"""
评估可视化模块

提供各种评估结果的可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import warnings

from .base import EvaluationResult

logger = logging.getLogger(__name__)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 忽略matplotlib警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class EvaluationVisualizer:
    """评估可视化器"""
    
    def __init__(self, 
                 style: str = 'seaborn-v0_8',
                 figure_size: Tuple[int, int] = (10, 6),
                 dpi: int = 300,
                 color_palette: str = 'Set2'):
        """
        初始化可视化器
        
        Args:
            style: matplotlib样式
            figure_size: 图形大小
            dpi: 图像分辨率
            color_palette: 颜色调色板
        """
        self.style = style
        self.figure_size = figure_size
        self.dpi = dpi
        self.color_palette = color_palette
        
        # 设置样式
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
        
        sns.set_palette(color_palette)
        
        if logger.isEnabledFor(logging.INFO):
            logger.info("Initialized EvaluationVisualizer")
    
    def plot_model_comparison(self, 
                            results: List[EvaluationResult],
                            metrics: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> None:
        """
        绘制模型比较图
        
        Args:
            results: 评估结果列表
            metrics: 要比较的指标
            save_path: 保存路径
        """
        if not results:
            logger.warning("No results provided for comparison")
            return
        
        # 确定要比较的指标
        if metrics is None:
            # 使用第一个结果中的所有指标
            metrics = list(results[0].test_scores.keys())
        
        # 准备数据
        model_names = [result.model_name for result in results]
        comparison_data = []
        
        for metric in metrics:
            for result in results:
                if metric in result.test_scores:
                    comparison_data.append({
                        'Model': result.model_name,
                        'Metric': metric,
                        'Score': result.test_scores[metric]
                    })
        
        if not comparison_data:
            logger.warning("No valid data for comparison")
            return
        
        df = pd.DataFrame(comparison_data)
        
        # 创建子图
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            metric_data = df[df['Metric'] == metric]
            
            if not metric_data.empty:
                # 条形图
                sns.barplot(data=metric_data, x='Model', y='Score', ax=ax)
                ax.set_title(f'{metric}')
                ax.tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for j, (model, score) in enumerate(zip(metric_data['Model'], metric_data['Score'])):
                    ax.text(j, score + 0.01 * ax.get_ylim()[1], f'{score:.3f}', 
                           ha='center', va='bottom')
        
        # 隐藏多余的子图
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_radar(self, 
                             results: List[EvaluationResult],
                             metrics: Optional[List[str]] = None,
                             normalize: bool = True,
                             save_path: Optional[str] = None) -> None:
        """
        绘制性能雷达图
        
        Args:
            results: 评估结果列表
            metrics: 要显示的指标
            normalize: 是否标准化指标
            save_path: 保存路径
        """
        if not results:
            logger.warning("No results provided for radar plot")
            return
        
        # 确定指标
        if metrics is None:
            all_metrics = set()
            for result in results:
                all_metrics.update(result.test_scores.keys())
            metrics = list(all_metrics)[:8]  # 限制指标数量
        
        # 准备数据
        data = []
        for result in results:
            row = []
            for metric in metrics:
                score = result.test_scores.get(metric, 0)
                row.append(score)
            data.append(row)
        
        data = np.array(data)
        
        # 标准化
        if normalize:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data.T).T
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=self.figure_size, subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
        
        for i, (result, color) in enumerate(zip(results, colors)):
            values = data[i].tolist()
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=result.model_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1 if normalize else None)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Model Performance Radar Chart')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Radar plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, 
                           results: List[EvaluationResult],
                           metric: str = 'accuracy',
                           save_path: Optional[str] = None) -> None:
        """
        绘制学习曲线
        
        Args:
            results: 评估结果列表
            metric: 要显示的指标
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        for result in results:
            if hasattr(result, 'learning_curve') and result.learning_curve:
                curve_data = result.learning_curve.get(metric)
                if curve_data:
                    train_sizes = curve_data['train_sizes']
                    train_scores = curve_data['train_scores_mean']
                    val_scores = curve_data['val_scores_mean']
                    train_std = curve_data.get('train_scores_std', np.zeros_like(train_scores))
                    val_std = curve_data.get('val_scores_std', np.zeros_like(val_scores))
                    
                    # 训练曲线
                    ax.plot(train_sizes, train_scores, 'o-', 
                           label=f'{result.model_name} (Train)')
                    ax.fill_between(train_sizes, 
                                   train_scores - train_std,
                                   train_scores + train_std, 
                                   alpha=0.1)
                    
                    # 验证曲线
                    ax.plot(train_sizes, val_scores, 's--', 
                           label=f'{result.model_name} (Val)')
                    ax.fill_between(train_sizes, 
                                   val_scores - val_std,
                                   val_scores + val_std, 
                                   alpha=0.1)
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel(metric.title())
        ax.set_title(f'Learning Curves - {metric.title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def plot_cross_validation_scores(self, 
                                   results: List[EvaluationResult],
                                   metric: str = 'accuracy',
                                   save_path: Optional[str] = None) -> None:
        """
        绘制交叉验证分数分布
        
        Args:
            results: 评估结果列表
            metric: 要显示的指标
            save_path: 保存路径
        """
        # 准备数据
        cv_data = []
        for result in results:
            if result.cv_scores and metric in result.cv_scores:
                scores = result.cv_scores[metric]
                for score in scores:
                    cv_data.append({
                        'Model': result.model_name,
                        'Score': score
                    })
        
        if not cv_data:
            logger.warning(f"No cross-validation data for metric '{metric}'")
            return
        
        df = pd.DataFrame(cv_data)
        
        # 创建箱线图和小提琴图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 箱线图
        sns.boxplot(data=df, x='Model', y='Score', ax=ax1)
        ax1.set_title(f'Cross-Validation Scores - {metric.title()} (Box Plot)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 小提琴图
        sns.violinplot(data=df, x='Model', y='Score', ax=ax2)
        ax2.set_title(f'Cross-Validation Scores - {metric.title()} (Violin Plot)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"CV scores plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance_comparison(self, 
                                         results: List[EvaluationResult],
                                         top_k: int = 20,
                                         save_path: Optional[str] = None) -> None:
        """
        绘制特征重要性比较
        
        Args:
            results: 评估结果列表
            top_k: 显示前k个重要特征
            save_path: 保存路径
        """
        # 收集特征重要性数据
        importance_data = []
        
        for result in results:
            if result.feature_importance is not None:
                feature_names = result.feature_names or [f'Feature {i}' for i in range(len(result.feature_importance))]
                
                for i, (name, importance) in enumerate(zip(feature_names, result.feature_importance)):
                    importance_data.append({
                        'Model': result.model_name,
                        'Feature': name,
                        'Importance': importance,
                        'Rank': i
                    })
        
        if not importance_data:
            logger.warning("No feature importance data available")
            return
        
        df = pd.DataFrame(importance_data)
        
        # 获取所有模型中最重要的特征
        avg_importance = df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
        top_features = avg_importance.head(top_k).index.tolist()
        
        # 过滤数据
        df_filtered = df[df['Feature'].isin(top_features)]
        
        # 创建热图
        pivot_df = df_filtered.pivot(index='Feature', columns='Model', values='Importance')
        pivot_df = pivot_df.reindex(top_features)  # 按重要性排序
        
        plt.figure(figsize=(max(8, len(results) * 1.5), max(6, top_k * 0.3)))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Feature Importance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Features')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Feature importance comparison saved to {save_path}")
        
        plt.show()
    
    def plot_error_analysis(self, 
                          result: EvaluationResult,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          feature_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> None:
        """
        绘制错误分析图
        
        Args:
            result: 评估结果
            X_test: 测试特征
            y_test: 测试标签
            feature_names: 特征名称
            save_path: 保存路径
        """
        if result.test_predictions is None:
            logger.warning("No test predictions available for error analysis")
            return
        
        # 计算错误
        if result.task_type == 'classification':
            errors = (y_test != result.test_predictions)
            error_indices = np.where(errors)[0]
        else:  # regression
            residuals = np.abs(y_test - result.test_predictions)
            # 选择误差最大的样本
            error_indices = np.argsort(residuals)[-50:]  # 前50个最大误差
        
        if len(error_indices) == 0:
            logger.info("No errors found for analysis")
            return
        
        # 分析错误样本的特征分布
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(X_test.shape[1])]
        
        # 选择前10个最重要的特征进行分析
        if result.feature_importance is not None:
            top_features = np.argsort(result.feature_importance)[::-1][:10]
        else:
            top_features = list(range(min(10, X_test.shape[1])))
        
        n_features = len(top_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature_idx in enumerate(top_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # 正确预测的样本
            correct_indices = np.setdiff1d(range(len(y_test)), error_indices)
            
            # 绘制分布
            ax.hist(X_test[correct_indices, feature_idx], bins=20, alpha=0.7, 
                   label='Correct', density=True)
            ax.hist(X_test[error_indices, feature_idx], bins=20, alpha=0.7, 
                   label='Error', density=True)
            
            ax.set_xlabel(feature_names[feature_idx])
            ax.set_ylabel('Density')
            ax.set_title(f'Feature Distribution: {feature_names[feature_idx]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(f'Error Analysis - {result.model_name}', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Error analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_calibration_curve(self, 
                             results: List[EvaluationResult],
                             y_test: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        绘制校准曲线（仅适用于分类任务）
        
        Args:
            results: 评估结果列表
            y_test: 测试标签
            save_path: 保存路径
        """
        from sklearn.calibration import calibration_curve
        
        plt.figure(figsize=self.figure_size)
        
        for result in results:
            if (result.task_type == 'classification' and 
                hasattr(result, 'prediction_probabilities') and
                result.prediction_probabilities is not None):
                
                # 二分类情况
                if len(np.unique(y_test)) == 2:
                    y_prob = result.prediction_probabilities
                    if y_prob.ndim > 1:
                        y_prob = y_prob[:, 1]
                    
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_test, y_prob, n_bins=10
                    )
                    
                    plt.plot(mean_predicted_value, fraction_of_positives, 
                            marker='o', label=result.model_name)
        
        # 完美校准线
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {save_path}")
        
        plt.show()
    
    def create_evaluation_dashboard(self, 
                                  results: List[EvaluationResult],
                                  save_dir: Optional[str] = None) -> None:
        """
        创建评估仪表板
        
        Args:
            results: 评估结果列表
            save_dir: 保存目录
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型比较
        self.plot_model_comparison(
            results, 
            save_path=str(save_dir / 'model_comparison.png') if save_dir else None
        )
        
        # 性能雷达图
        self.plot_performance_radar(
            results,
            save_path=str(save_dir / 'performance_radar.png') if save_dir else None
        )
        
        # 交叉验证分数
        if any(result.cv_scores for result in results):
            # 选择第一个可用的指标
            available_metrics = []
            for result in results:
                if result.cv_scores:
                    available_metrics.extend(result.cv_scores.keys())
            
            if available_metrics:
                metric = available_metrics[0]
                self.plot_cross_validation_scores(
                    results, 
                    metric=metric,
                    save_path=str(save_dir / f'cv_scores_{metric}.png') if save_dir else None
                )
        
        # 特征重要性比较
        if any(result.feature_importance is not None for result in results):
            self.plot_feature_importance_comparison(
                results,
                save_path=str(save_dir / 'feature_importance_comparison.png') if save_dir else None
            )
        
        logger.info("Evaluation dashboard created successfully")


def create_summary_table(results: List[EvaluationResult], 
                        metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    创建结果汇总表
    
    Args:
        results: 评估结果列表
        metrics: 要包含的指标
        
    Returns:
        汇总表DataFrame
    """
    if not results:
        return pd.DataFrame()
    
    # 确定指标
    if metrics is None:
        all_metrics = set()
        for result in results:
            all_metrics.update(result.test_scores.keys())
        metrics = sorted(list(all_metrics))
    
    # 创建汇总数据
    summary_data = []
    
    for result in results:
        row = {'Model': result.model_name}
        
        # 测试集指标
        for metric in metrics:
            score = result.test_scores.get(metric, np.nan)
            row[f'Test_{metric}'] = score
        
        # 交叉验证指标
        if result.cv_scores:
            for metric in metrics:
                if metric in result.cv_scores:
                    cv_scores = result.cv_scores[metric]
                    row[f'CV_{metric}_mean'] = np.mean(cv_scores)
                    row[f'CV_{metric}_std'] = np.std(cv_scores)
        
        # 其他信息
        row['Evaluation_Time'] = getattr(result, 'evaluation_time', np.nan)
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def save_evaluation_report(results: List[EvaluationResult], 
                          output_path: str,
                          include_plots: bool = True) -> None:
    """
    保存评估报告
    
    Args:
        results: 评估结果列表
        output_path: 输出路径
        include_plots: 是否包含图表
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建汇总表
    summary_df = create_summary_table(results)
    summary_df.to_csv(output_path / 'evaluation_summary.csv', index=False)
    
    # 保存详细报告
    with open(output_path / 'detailed_report.txt', 'w', encoding='utf-8') as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            if hasattr(result, 'generate_detailed_report'):
                f.write(result.generate_detailed_report())
            else:
                f.write(f"Model: {result.model_name}\n")
                f.write(f"Task Type: {result.task_type}\n")
                f.write(f"Test Scores: {result.test_scores}\n")
            f.write("\n" + "-" * 50 + "\n\n")
    
    # 创建可视化
    if include_plots:
        visualizer = EvaluationVisualizer()
        visualizer.create_evaluation_dashboard(results, str(output_path / 'plots'))
    
    logger.info(f"Evaluation report saved to {output_path}")