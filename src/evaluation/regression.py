"""
回归任务评估器

专门用于回归模型的评估
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

from .base import BaseEvaluator, EvaluationResult, EvaluationConfig

logger = logging.getLogger(__name__)


class RegressionEvaluator(BaseEvaluator):
    """回归评估器"""
    
    def __init__(self, 
                 config: Optional[EvaluationConfig] = None,
                 **kwargs):
        """
        初始化回归评估器
        
        Args:
            config: 评估配置
            **kwargs: 其他参数
        """
        if config is None:
            config = EvaluationConfig()
            config.metrics = [
                'mse', 'rmse', 'mae', 'r2', 'mape', 
                'explained_variance', 'max_error', 'median_ae'
            ]
        
        super().__init__(config, **kwargs)
        
        # 回归特定配置
        self.residual_analysis = kwargs.get('residual_analysis', True)
        self.prediction_intervals = kwargs.get('prediction_intervals', True)
        
        if self.verbose:
            logger.info("Initialized RegressionEvaluator")
    
    def evaluate_model(self, 
                      model: BaseEstimator,
                      X: np.ndarray,
                      y: np.ndarray,
                      model_name: Optional[str] = None,
                      **kwargs) -> EvaluationResult:
        """
        评估回归模型
        
        Args:
            model: 回归模型
            X: 特征
            y: 目标值
            model_name: 模型名称
            **kwargs: 其他参数
            
        Returns:
            评估结果
        """
        start_time = time.time()
        
        if model_name is None:
            model_name = model.__class__.__name__
        
        # 准备数据
        if self.X_train_ is None:
            self.prepare_data(X, y)
        
        # 训练模型
        if not hasattr(model, 'predict'):
            model.fit(self.X_train_, self.y_train_)
        
        # 创建结果对象
        result = EvaluationResult(
            model_name=model_name,
            task_type='regression',
            model_params=model.get_params() if hasattr(model, 'get_params') else {}
        )
        
        # 预测
        train_pred = model.predict(self.X_train_)
        val_pred = model.predict(self.X_val_)
        test_pred = model.predict(self.X_test_)
        
        # 存储预测结果
        result.train_predictions = train_pred
        result.val_predictions = val_pred
        result.test_predictions = test_pred
        
        # 计算指标
        result.train_scores = self.calculate_regression_metrics(
            self.y_train_, train_pred
        )
        result.val_scores = self.calculate_regression_metrics(
            self.y_val_, val_pred
        )
        result.test_scores = self.calculate_regression_metrics(
            self.y_test_, test_pred
        )
        
        # 交叉验证
        result.cv_scores = self.cross_validate_model(model, X, y)
        
        # 特征重要性
        result.feature_importance = self.get_feature_importance(model)
        
        # 残差分析
        if self.residual_analysis:
            result.residual_analysis = self._perform_residual_analysis(
                self.y_test_, test_pred
            )
        
        # 置信区间
        self._calculate_confidence_intervals(result, model, X, y)
        
        # 预测区间
        if self.prediction_intervals:
            result.prediction_intervals = self._calculate_prediction_intervals(
                model, self.X_test_, self.y_test_, test_pred
            )
        
        # 生成可视化
        if self.config.save_plots:
            self._generate_regression_plots(result, model)
        
        result.evaluation_time = time.time() - start_time
        
        if self.verbose:
            logger.info(f"Evaluated {model_name} in {result.evaluation_time:.2f}s")
        
        return result
    
    def calculate_regression_metrics(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            指标字典
        """
        metrics = {}
        
        try:
            # 基础指标
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # 其他指标
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
            metrics['max_error'] = max_error(y_true, y_pred)
            metrics['median_ae'] = median_absolute_error(y_true, y_pred)
            
            # MAPE (避免除零)
            mask = y_true != 0
            if np.any(mask):
                metrics['mape'] = mean_absolute_percentage_error(
                    y_true[mask], y_pred[mask]
                )
            else:
                metrics['mape'] = np.inf
            
            # 自定义指标
            metrics['mean_residual'] = np.mean(y_true - y_pred)
            metrics['std_residual'] = np.std(y_true - y_pred)
            
            # 相对指标
            y_mean = np.mean(y_true)
            if y_mean != 0:
                metrics['relative_rmse'] = metrics['rmse'] / abs(y_mean)
                metrics['relative_mae'] = metrics['mae'] / abs(y_mean)
            
            # 百分位数误差
            abs_errors = np.abs(y_true - y_pred)
            metrics['mae_p50'] = np.percentile(abs_errors, 50)
            metrics['mae_p90'] = np.percentile(abs_errors, 90)
            metrics['mae_p95'] = np.percentile(abs_errors, 95)
            
        except Exception as e:
            logger.warning(f"Failed to calculate some regression metrics: {e}")
        
        return metrics
    
    def _perform_residual_analysis(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict[str, Any]:
        """
        执行残差分析
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            残差分析结果
        """
        residuals = y_true - y_pred
        
        analysis = {}
        
        try:
            # 基础统计
            analysis['mean'] = np.mean(residuals)
            analysis['std'] = np.std(residuals)
            analysis['min'] = np.min(residuals)
            analysis['max'] = np.max(residuals)
            analysis['median'] = np.median(residuals)
            
            # 分布检验
            # Shapiro-Wilk正态性检验
            if len(residuals) <= 5000:  # 样本量限制
                stat, p_value = stats.shapiro(residuals)
                analysis['shapiro_stat'] = stat
                analysis['shapiro_p_value'] = p_value
                analysis['is_normal'] = p_value > 0.05
            
            # Jarque-Bera正态性检验
            stat, p_value = stats.jarque_bera(residuals)
            analysis['jarque_bera_stat'] = stat
            analysis['jarque_bera_p_value'] = p_value
            
            # 异方差性检验 (Breusch-Pagan)
            # 简化版本：残差平方与预测值的相关性
            residuals_squared = residuals ** 2
            correlation, p_value = stats.pearsonr(y_pred, residuals_squared)
            analysis['heteroscedasticity_corr'] = correlation
            analysis['heteroscedasticity_p_value'] = p_value
            analysis['is_homoscedastic'] = abs(correlation) < 0.1
            
            # 自相关检验 (Durbin-Watson近似)
            diff_residuals = np.diff(residuals)
            dw_stat = np.sum(diff_residuals**2) / np.sum(residuals[1:]**2)
            analysis['durbin_watson'] = dw_stat
            analysis['has_autocorrelation'] = dw_stat < 1.5 or dw_stat > 2.5
            
            # 异常值检测
            q1, q3 = np.percentile(residuals, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = (residuals < lower_bound) | (residuals > upper_bound)
            analysis['outlier_count'] = np.sum(outliers)
            analysis['outlier_percentage'] = np.sum(outliers) / len(residuals) * 100
            analysis['outlier_indices'] = np.where(outliers)[0].tolist()
            
        except Exception as e:
            logger.warning(f"Failed to perform residual analysis: {e}")
        
        return analysis
    
    def _calculate_prediction_intervals(self, 
                                      model: BaseEstimator,
                                      X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      y_pred: np.ndarray,
                                      confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        计算预测区间
        
        Args:
            model: 模型
            X_test: 测试特征
            y_test: 测试目标值
            y_pred: 预测值
            confidence_level: 置信水平
            
        Returns:
            预测区间信息
        """
        intervals = {}
        
        try:
            # 计算残差标准差
            residuals = y_test - y_pred
            residual_std = np.std(residuals)
            
            # 计算置信区间
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            margin = z_score * residual_std
            
            intervals['lower_bound'] = y_pred - margin
            intervals['upper_bound'] = y_pred + margin
            intervals['width'] = 2 * margin
            intervals['confidence_level'] = confidence_level
            
            # 计算覆盖率
            coverage = np.mean(
                (y_test >= intervals['lower_bound']) & 
                (y_test <= intervals['upper_bound'])
            )
            intervals['actual_coverage'] = coverage
            intervals['expected_coverage'] = confidence_level
            
            # 区间质量指标
            intervals['mean_width'] = np.mean(intervals['width'])
            intervals['median_width'] = np.median(intervals['width'])
            
        except Exception as e:
            logger.warning(f"Failed to calculate prediction intervals: {e}")
        
        return intervals
    
    def _calculate_confidence_intervals(self, 
                                      result: EvaluationResult,
                                      model: BaseEstimator,
                                      X: np.ndarray,
                                      y: np.ndarray) -> None:
        """计算置信区间"""
        if not self.config.include_statistical_tests:
            return
        
        try:
            from .utils import bootstrap_confidence_interval
            
            # 为主要指标计算置信区间
            main_metrics = ['mse', 'rmse', 'mae', 'r2']
            
            for metric in main_metrics:
                if metric in result.test_scores:
                    ci = bootstrap_confidence_interval(
                        model, self.X_test_, self.y_test_,
                        metric=metric,
                        n_bootstrap=self.config.bootstrap_samples,
                        confidence_level=self.config.confidence_level
                    )
                    result.add_confidence_interval(metric, ci)
                    
        except Exception as e:
            logger.warning(f"Failed to calculate confidence intervals: {e}")
    
    def _generate_regression_plots(self, 
                                 result: EvaluationResult,
                                 model: BaseEstimator) -> None:
        """生成回归可视化图表"""
        if not self.config.output_dir:
            return
        
        import os
        
        plot_dir = os.path.join(self.config.output_dir, 'plots', result.model_name)
        os.makedirs(plot_dir, exist_ok=True)
        
        try:
            # 预测vs真实值散点图
            self._plot_predictions_vs_actual(result, plot_dir)
            
            # 残差图
            self._plot_residuals(result, plot_dir)
            
            # 残差分布
            self._plot_residual_distribution(result, plot_dir)
            
            # QQ图
            self._plot_qq_plot(result, plot_dir)
            
            # 特征重要性
            if result.feature_importance is not None:
                self._plot_feature_importance(result, plot_dir)
            
            # 预测区间
            if hasattr(result, 'prediction_intervals'):
                self._plot_prediction_intervals(result, plot_dir)
            
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    def _plot_predictions_vs_actual(self, result: EvaluationResult, plot_dir: str) -> None:
        """绘制预测vs真实值散点图"""
        plt.figure(figsize=self.config.figure_size)
        
        # 散点图
        plt.scatter(self.y_test_, result.test_predictions, alpha=0.6)
        
        # 完美预测线
        min_val = min(np.min(self.y_test_), np.min(result.test_predictions))
        max_val = max(np.max(self.y_test_), np.max(result.test_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # 添加R²信息
        r2 = result.test_scores.get('r2', 0)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs Actual - {result.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(plot_dir, f'predictions_vs_actual.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        result.add_plot('predictions_vs_actual', plot_path)
    
    def _plot_residuals(self, result: EvaluationResult, plot_dir: str) -> None:
        """绘制残差图"""
        residuals = self.y_test_ - result.test_predictions
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 残差vs预测值
        axes[0, 0].scatter(result.test_predictions, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 残差vs真实值
        axes[0, 1].scatter(self.y_test_, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Actual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 标准化残差
        std_residuals = residuals / np.std(residuals)
        axes[1, 0].scatter(result.test_predictions, std_residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].axhline(y=2, color='orange', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(y=-2, color='orange', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Standardized Residuals')
        axes[1, 0].set_title('Standardized Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 残差的绝对值
        axes[1, 1].scatter(result.test_predictions, np.abs(residuals), alpha=0.6)
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('|Residuals|')
        axes[1, 1].set_title('Absolute Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(plot_dir, f'residuals.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        result.add_plot('residuals', plot_path)
    
    def _plot_residual_distribution(self, result: EvaluationResult, plot_dir: str) -> None:
        """绘制残差分布"""
        residuals = self.y_test_ - result.test_predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 直方图
        axes[0].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
        
        # 拟合正态分布
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', 
                    label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        axes[0].set_xlabel('Residuals')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Residual Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 箱线图
        axes[1].boxplot(residuals, vert=True)
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Box Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(plot_dir, f'residual_distribution.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        result.add_plot('residual_distribution', plot_path)
    
    def _plot_qq_plot(self, result: EvaluationResult, plot_dir: str) -> None:
        """绘制QQ图"""
        residuals = self.y_test_ - result.test_predictions
        
        plt.figure(figsize=self.config.figure_size)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot - {result.model_name}')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(plot_dir, f'qq_plot.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        result.add_plot('qq_plot', plot_path)
    
    def _plot_feature_importance(self, result: EvaluationResult, plot_dir: str) -> None:
        """绘制特征重要性"""
        importance = result.feature_importance
        feature_names = result.feature_names or [f'Feature {i}' for i in range(len(importance))]
        
        # 选择前20个最重要的特征
        top_k = min(20, len(importance))
        indices = np.argsort(importance)[::-1][:top_k]
        
        plt.figure(figsize=self.config.figure_size)
        plt.barh(range(top_k), importance[indices])
        plt.yticks(range(top_k), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {result.model_name}')
        plt.gca().invert_yaxis()
        
        plot_path = os.path.join(plot_dir, f'feature_importance.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        result.add_plot('feature_importance', plot_path)
    
    def _plot_prediction_intervals(self, result: EvaluationResult, plot_dir: str) -> None:
        """绘制预测区间"""
        if not hasattr(result, 'prediction_intervals'):
            return
        
        intervals = result.prediction_intervals
        
        # 选择部分数据点进行可视化
        n_points = min(100, len(self.y_test_))
        indices = np.random.choice(len(self.y_test_), n_points, replace=False)
        indices = np.sort(indices)
        
        plt.figure(figsize=self.config.figure_size)
        
        # 排序以便更好的可视化
        sort_idx = np.argsort(result.test_predictions[indices])
        sorted_indices = indices[sort_idx]
        
        x_pos = range(len(sorted_indices))
        
        # 预测区间
        plt.fill_between(
            x_pos,
            intervals['lower_bound'][sorted_indices],
            intervals['upper_bound'][sorted_indices],
            alpha=0.3, label=f'{intervals["confidence_level"]*100:.0f}% Prediction Interval'
        )
        
        # 真实值和预测值
        plt.scatter(x_pos, self.y_test_[sorted_indices], 
                   color='red', alpha=0.7, label='Actual', s=30)
        plt.scatter(x_pos, result.test_predictions[sorted_indices], 
                   color='blue', alpha=0.7, label='Predicted', s=30)
        
        plt.xlabel('Sample Index (sorted by prediction)')
        plt.ylabel('Value')
        plt.title(f'Prediction Intervals - {result.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加覆盖率信息
        coverage = intervals.get('actual_coverage', 0)
        plt.text(0.05, 0.95, f'Coverage: {coverage:.1%}', 
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plot_path = os.path.join(plot_dir, f'prediction_intervals.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        result.add_plot('prediction_intervals', plot_path)
    
    def generate_detailed_report(self, result: EvaluationResult) -> str:
        """生成详细的回归报告"""
        report = []
        report.append(f"Regression Evaluation Report: {result.model_name}")
        report.append("=" * 60)
        report.append("")
        
        # 基础信息
        report.append(f"Task Type: {result.task_type}")
        report.append(f"Evaluation Time: {result.evaluation_time:.2f}s")
        report.append("")
        
        # 性能指标
        report.append("Performance Metrics:")
        report.append("-" * 30)
        
        for split, scores in [
            ("Training", result.train_scores),
            ("Validation", result.val_scores),
            ("Test", result.test_scores)
        ]:
            if scores:
                report.append(f"\n{split} Set:")
                for metric, score in scores.items():
                    if not np.isnan(score):
                        report.append(f"  {metric}: {score:.4f}")
        
        # 交叉验证结果
        if result.cv_scores:
            report.append("\nCross-Validation Results:")
            report.append("-" * 30)
            for metric, scores in result.cv_scores.items():
                stats = result.get_cv_score_stats(metric)
                report.append(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # 置信区间
        if result.confidence_intervals:
            report.append("\nConfidence Intervals:")
            report.append("-" * 30)
            for metric, (lower, upper) in result.confidence_intervals.items():
                report.append(f"{metric}: [{lower:.4f}, {upper:.4f}]")
        
        # 残差分析
        if hasattr(result, 'residual_analysis'):
            analysis = result.residual_analysis
            report.append("\nResidual Analysis:")
            report.append("-" * 30)
            report.append(f"Mean: {analysis.get('mean', 0):.4f}")
            report.append(f"Std: {analysis.get('std', 0):.4f}")
            report.append(f"Normal Distribution: {analysis.get('is_normal', 'Unknown')}")
            report.append(f"Homoscedastic: {analysis.get('is_homoscedastic', 'Unknown')}")
            report.append(f"Outliers: {analysis.get('outlier_count', 0)} ({analysis.get('outlier_percentage', 0):.1f}%)")
        
        # 预测区间
        if hasattr(result, 'prediction_intervals'):
            intervals = result.prediction_intervals
            report.append("\nPrediction Intervals:")
            report.append("-" * 30)
            report.append(f"Confidence Level: {intervals.get('confidence_level', 0)*100:.0f}%")
            report.append(f"Actual Coverage: {intervals.get('actual_coverage', 0)*100:.1f}%")
            report.append(f"Mean Width: {intervals.get('mean_width', 0):.4f}")
        
        return "\n".join(report)