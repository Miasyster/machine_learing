"""
分类任务评估器

专门用于分类模型的评估
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseEvaluator, EvaluationResult, EvaluationConfig

logger = logging.getLogger(__name__)


class ClassificationEvaluator(BaseEvaluator):
    """分类评估器"""
    
    def __init__(self, 
                 config: Optional[EvaluationConfig] = None,
                 **kwargs):
        """
        初始化分类评估器
        
        Args:
            config: 评估配置
            **kwargs: 其他参数
        """
        if config is None:
            config = EvaluationConfig()
            config.metrics = [
                'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                'roc_auc_ovr', 'average_precision'
            ]
        
        super().__init__(config, **kwargs)
        
        # 分类特定配置
        self.average_method = kwargs.get('average_method', 'macro')
        self.pos_label = kwargs.get('pos_label', 1)
        self.multi_class = kwargs.get('multi_class', 'ovr')
        
        if self.verbose:
            logger.info("Initialized ClassificationEvaluator")
    
    def evaluate_model(self, 
                      model: BaseEstimator,
                      X: np.ndarray,
                      y: np.ndarray,
                      model_name: Optional[str] = None,
                      **kwargs) -> EvaluationResult:
        """
        评估分类模型
        
        Args:
            model: 分类模型
            X: 特征
            y: 标签
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
            task_type='classification',
            model_params=model.get_params() if hasattr(model, 'get_params') else {}
        )
        
        # 预测
        train_pred = model.predict(self.X_train_)
        val_pred = model.predict(self.X_val_)
        test_pred = model.predict(self.X_test_)
        
        # 预测概率
        train_prob = None
        val_prob = None
        test_prob = None
        
        if hasattr(model, 'predict_proba'):
            train_prob = model.predict_proba(self.X_train_)
            val_prob = model.predict_proba(self.X_val_)
            test_prob = model.predict_proba(self.X_test_)
        elif hasattr(model, 'decision_function'):
            train_prob = model.decision_function(self.X_train_)
            val_prob = model.decision_function(self.X_val_)
            test_prob = model.decision_function(self.X_test_)
        
        # 存储预测结果
        result.train_predictions = train_pred
        result.val_predictions = val_pred
        result.test_predictions = test_pred
        result.prediction_probabilities = test_prob
        
        # 计算指标
        result.train_scores = self.calculate_classification_metrics(
            self.y_train_, train_pred, train_prob
        )
        result.val_scores = self.calculate_classification_metrics(
            self.y_val_, val_pred, val_prob
        )
        result.test_scores = self.calculate_classification_metrics(
            self.y_test_, test_pred, test_prob
        )
        
        # 交叉验证
        result.cv_scores = self.cross_validate_model(model, X, y)
        
        # 特征重要性
        result.feature_importance = self.get_feature_importance(model)
        
        # 置信区间
        self._calculate_confidence_intervals(result, model, X, y)
        
        # 生成可视化
        if self.config.save_plots:
            self._generate_classification_plots(result, model)
        
        result.evaluation_time = time.time() - start_time
        
        if self.verbose:
            logger.info(f"Evaluated {model_name} in {result.evaluation_time:.2f}s")
        
        return result
    
    def calculate_classification_metrics(self, 
                                       y_true: np.ndarray, 
                                       y_pred: np.ndarray,
                                       y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 多分类平均指标
        for avg in ['macro', 'micro', 'weighted']:
            try:
                metrics[f'precision_{avg}'] = precision_score(
                    y_true, y_pred, average=avg, zero_division=0
                )
                metrics[f'recall_{avg}'] = recall_score(
                    y_true, y_pred, average=avg, zero_division=0
                )
                metrics[f'f1_{avg}'] = f1_score(
                    y_true, y_pred, average=avg, zero_division=0
                )
            except Exception as e:
                logger.warning(f"Failed to calculate {avg} metrics: {e}")
        
        # 概率相关指标
        if y_prob is not None:
            try:
                # ROC AUC
                if len(np.unique(y_true)) == 2:
                    # 二分类
                    prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                    metrics['roc_auc'] = roc_auc_score(y_true, prob)
                    metrics['average_precision'] = average_precision_score(y_true, prob)
                else:
                    # 多分类
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )
                    metrics['roc_auc_ovo'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovo', average='macro'
                    )
                
                # 对数损失
                metrics['log_loss'] = log_loss(y_true, y_prob)
                
            except Exception as e:
                logger.warning(f"Failed to calculate probability metrics: {e}")
        
        # 类别特定指标
        try:
            unique_labels = np.unique(y_true)
            if len(unique_labels) <= 10:  # 避免太多类别
                for label in unique_labels:
                    precision = precision_score(y_true, y_pred, labels=[label], average=None, zero_division=0)
                    recall = recall_score(y_true, y_pred, labels=[label], average=None, zero_division=0)
                    f1 = f1_score(y_true, y_pred, labels=[label], average=None, zero_division=0)
                    
                    if len(precision) > 0:
                        metrics[f'precision_class_{label}'] = precision[0]
                        metrics[f'recall_class_{label}'] = recall[0]
                        metrics[f'f1_class_{label}'] = f1[0]
        except Exception as e:
            logger.warning(f"Failed to calculate class-specific metrics: {e}")
        
        return metrics
    
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
            main_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            
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
    
    def _generate_classification_plots(self, 
                                     result: EvaluationResult,
                                     model: BaseEstimator) -> None:
        """生成分类可视化图表"""
        if not self.config.output_dir:
            return
        
        import os
        
        plot_dir = os.path.join(self.config.output_dir, 'plots', result.model_name)
        os.makedirs(plot_dir, exist_ok=True)
        
        try:
            # 混淆矩阵
            self._plot_confusion_matrix(result, plot_dir)
            
            # ROC曲线
            if result.prediction_probabilities is not None:
                self._plot_roc_curve(result, plot_dir)
                self._plot_precision_recall_curve(result, plot_dir)
            
            # 特征重要性
            if result.feature_importance is not None:
                self._plot_feature_importance(result, plot_dir)
            
            # 分类报告热图
            self._plot_classification_report(result, plot_dir)
            
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    def _plot_confusion_matrix(self, result: EvaluationResult, plot_dir: str) -> None:
        """绘制混淆矩阵"""
        cm = confusion_matrix(self.y_test_, result.test_predictions)
        
        plt.figure(figsize=self.config.figure_size)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {result.model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plot_path = os.path.join(plot_dir, f'confusion_matrix.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        result.add_plot('confusion_matrix', plot_path)
    
    def _plot_roc_curve(self, result: EvaluationResult, plot_dir: str) -> None:
        """绘制ROC曲线"""
        y_prob = result.prediction_probabilities
        
        plt.figure(figsize=self.config.figure_size)
        
        unique_labels = np.unique(self.y_test_)
        
        if len(unique_labels) == 2:
            # 二分类
            prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            fpr, tpr, _ = roc_curve(self.y_test_, prob)
            auc = roc_auc_score(self.y_test_, prob)
            
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        else:
            # 多分类
            for i, label in enumerate(unique_labels):
                y_true_binary = (self.y_test_ == label).astype(int)
                y_prob_binary = y_prob[:, i]
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary)
                auc = roc_auc_score(y_true_binary, y_prob_binary)
                
                plt.plot(fpr, tpr, label=f'Class {label} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {result.model_name}')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(plot_dir, f'roc_curve.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        result.add_plot('roc_curve', plot_path)
    
    def _plot_precision_recall_curve(self, result: EvaluationResult, plot_dir: str) -> None:
        """绘制精确率-召回率曲线"""
        y_prob = result.prediction_probabilities
        
        plt.figure(figsize=self.config.figure_size)
        
        unique_labels = np.unique(self.y_test_)
        
        if len(unique_labels) == 2:
            # 二分类
            prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            precision, recall, _ = precision_recall_curve(self.y_test_, prob)
            ap = average_precision_score(self.y_test_, prob)
            
            plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
        else:
            # 多分类
            for i, label in enumerate(unique_labels):
                y_true_binary = (self.y_test_ == label).astype(int)
                y_prob_binary = y_prob[:, i]
                
                precision, recall, _ = precision_recall_curve(y_true_binary, y_prob_binary)
                ap = average_precision_score(y_true_binary, y_prob_binary)
                
                plt.plot(recall, precision, label=f'Class {label} (AP = {ap:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {result.model_name}')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(plot_dir, f'pr_curve.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        result.add_plot('pr_curve', plot_path)
    
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
    
    def _plot_classification_report(self, result: EvaluationResult, plot_dir: str) -> None:
        """绘制分类报告热图"""
        report = classification_report(
            self.y_test_, result.test_predictions, 
            output_dict=True, zero_division=0
        )
        
        # 转换为DataFrame
        df = pd.DataFrame(report).iloc[:-1, :].T  # 排除support行
        
        plt.figure(figsize=self.config.figure_size)
        sns.heatmap(df.iloc[:-3, :-1], annot=True, cmap='Blues', fmt='.3f')
        plt.title(f'Classification Report - {result.model_name}')
        
        plot_path = os.path.join(plot_dir, f'classification_report.{self.config.plot_format}')
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        result.add_plot('classification_report', plot_path)
    
    def generate_detailed_report(self, result: EvaluationResult) -> str:
        """生成详细的分类报告"""
        report = []
        report.append(f"Classification Evaluation Report: {result.model_name}")
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
        
        # 混淆矩阵
        if result.test_predictions is not None:
            report.append("\nConfusion Matrix:")
            report.append("-" * 30)
            cm = confusion_matrix(self.y_test_, result.test_predictions)
            report.append(str(cm))
        
        # 分类报告
        if result.test_predictions is not None:
            report.append("\nDetailed Classification Report:")
            report.append("-" * 30)
            class_report = classification_report(
                self.y_test_, result.test_predictions, zero_division=0
            )
            report.append(class_report)
        
        return "\n".join(report)