"""
训练工具函数模块

提供训练过程中的各种工具函数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import time
import pickle
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import validation_curve, learning_curve
import warnings

from .base import TrainingResult

logger = logging.getLogger(__name__)


def save_model(model: BaseEstimator, 
               filepath: Union[str, Path],
               format: str = 'joblib',
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    保存模型
    
    Args:
        model: 模型
        filepath: 文件路径
        format: 保存格式 ('joblib', 'pickle')
        metadata: 元数据
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'joblib':
        joblib.dump(model, filepath)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # 保存元数据
    if metadata:
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: Union[str, Path],
               format: str = 'joblib') -> BaseEstimator:
    """
    加载模型
    
    Args:
        filepath: 文件路径
        format: 保存格式 ('joblib', 'pickle')
        
    Returns:
        加载的模型
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    if format == 'joblib':
        model = joblib.load(filepath)
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Model loaded from {filepath}")
    return model


def load_model_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    加载模型元数据
    
    Args:
        filepath: 模型文件路径
        
    Returns:
        元数据字典
    """
    filepath = Path(filepath)
    metadata_path = filepath.with_suffix('.json')
    
    if not metadata_path.exists():
        return {}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def compare_models(results: Dict[str, TrainingResult],
                  metrics: Optional[List[str]] = None,
                  sort_by: str = 'accuracy') -> pd.DataFrame:
    """
    比较多个模型的性能
    
    Args:
        results: 训练结果字典
        metrics: 要比较的指标
        sort_by: 排序指标
        
    Returns:
        比较结果DataFrame
    """
    if not results:
        return pd.DataFrame()
    
    data = []
    
    for model_name, result in results.items():
        row = {'model': model_name}
        
        # 添加训练指标
        for metric, score in result.train_scores.items():
            row[f'train_{metric}'] = score
        
        # 添加验证指标
        for metric, score in result.val_scores.items():
            row[f'val_{metric}'] = score
        
        # 添加测试指标
        for metric, score in result.test_scores.items():
            row[f'test_{metric}'] = score
        
        # 添加交叉验证指标
        for metric in result.cv_scores:
            cv_stats = result.get_cv_score_stats(metric)
            if cv_stats:
                row[f'cv_{metric}_mean'] = cv_stats['mean']
                row[f'cv_{metric}_std'] = cv_stats['std']
        
        # 添加训练时间
        row['training_time'] = result.training_time
        
        # 添加最佳参数数量
        if result.best_params:
            row['n_params'] = len(result.best_params)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 排序
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    return df


def plot_training_history(result: TrainingResult,
                         metrics: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[Union[str, Path]] = None) -> None:
    """
    绘制训练历史
    
    Args:
        result: 训练结果
        metrics: 要绘制的指标
        figsize: 图形大小
        save_path: 保存路径
    """
    if not result.history:
        logger.warning("No training history available")
        return
    
    if metrics is None:
        metrics = list(result.history.keys())
    
    n_metrics = len(metrics)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in result.history:
            epochs = range(1, len(result.history[metric]) + 1)
            axes[i].plot(epochs, result.history[metric], label=f'Train {metric}')
            
            # 如果有验证历史
            val_metric = f'val_{metric}'
            if val_metric in result.history:
                axes[i].plot(epochs, result.history[val_metric], label=f'Val {metric}')
            
            axes[i].set_title(f'{metric.capitalize()} History')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_validation_curve(model: BaseEstimator,
                         X: np.ndarray,
                         y: np.ndarray,
                         param_name: str,
                         param_range: List,
                         cv: int = 5,
                         scoring: str = 'accuracy',
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[Union[str, Path]] = None) -> None:
    """
    绘制验证曲线
    
    Args:
        model: 模型
        X: 特征
        y: 标签
        param_name: 参数名称
        param_range: 参数范围
        cv: 交叉验证折数
        scoring: 评分指标
        figsize: 图形大小
        save_path: 保存路径
    """
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel(f'{scoring.capitalize()} Score')
    plt.title(f'Validation Curve for {param_name}')
    plt.legend(loc='best')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Validation curve plot saved to {save_path}")
    
    plt.show()


def plot_learning_curve(model: BaseEstimator,
                       X: np.ndarray,
                       y: np.ndarray,
                       train_sizes: Optional[np.ndarray] = None,
                       cv: int = 5,
                       scoring: str = 'accuracy',
                       figsize: Tuple[int, int] = (10, 6),
                       save_path: Optional[Union[str, Path]] = None) -> None:
    """
    绘制学习曲线
    
    Args:
        model: 模型
        X: 特征
        y: 标签
        train_sizes: 训练集大小
        cv: 交叉验证折数
        scoring: 评分指标
        figsize: 图形大小
        save_path: 保存路径
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel(f'{scoring.capitalize()} Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curve plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         labels: Optional[List[str]] = None,
                         normalize: bool = False,
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[Union[str, Path]] = None) -> None:
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 类别标签
        normalize: 是否归一化
        figsize: 图形大小
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(importance: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           top_k: int = 20,
                           figsize: Tuple[int, int] = (10, 8),
                           save_path: Optional[Union[str, Path]] = None) -> None:
    """
    绘制特征重要性
    
    Args:
        importance: 特征重要性
        feature_names: 特征名称
        top_k: 显示前k个特征
        figsize: 图形大小
        save_path: 保存路径
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance))]
    
    # 排序
    indices = np.argsort(importance)[::-1][:top_k]
    sorted_importance = importance[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(sorted_importance)), sorted_importance)
    plt.yticks(range(len(sorted_importance)), sorted_names)
    plt.xlabel('Importance')
    plt.title(f'Top {top_k} Feature Importance')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def generate_classification_report(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 target_names: Optional[List[str]] = None,
                                 output_dict: bool = False) -> Union[str, Dict]:
    """
    生成分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        target_names: 类别名称
        output_dict: 是否返回字典格式
        
    Returns:
        分类报告
    """
    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=output_dict
    )


def calculate_model_complexity(model: BaseEstimator) -> Dict[str, Any]:
    """
    计算模型复杂度
    
    Args:
        model: 模型
        
    Returns:
        复杂度信息
    """
    complexity = {}
    
    # 参数数量
    if hasattr(model, 'get_params'):
        params = model.get_params()
        complexity['n_parameters'] = len(params)
        complexity['parameters'] = params
    
    # 特定模型的复杂度指标
    if hasattr(model, 'n_features_in_'):
        complexity['n_features'] = model.n_features_in_
    
    if hasattr(model, 'tree_'):
        # 决策树
        complexity['tree_depth'] = model.tree_.max_depth
        complexity['n_nodes'] = model.tree_.node_count
        complexity['n_leaves'] = model.tree_.n_leaves
    
    if hasattr(model, 'support_vectors_'):
        # SVM
        complexity['n_support_vectors'] = len(model.support_vectors_)
    
    if hasattr(model, 'coef_'):
        # 线性模型
        coef = model.coef_
        if coef.ndim > 1:
            complexity['n_coefficients'] = coef.size
        else:
            complexity['n_coefficients'] = len(coef)
        complexity['l1_norm'] = np.sum(np.abs(coef))
        complexity['l2_norm'] = np.sqrt(np.sum(coef ** 2))
    
    return complexity


def estimate_memory_usage(model: BaseEstimator) -> Dict[str, float]:
    """
    估计模型内存使用量
    
    Args:
        model: 模型
        
    Returns:
        内存使用信息（MB）
    """
    import sys
    
    memory_info = {}
    
    # 模型大小
    model_size = sys.getsizeof(model) / (1024 * 1024)  # MB
    memory_info['model_size_mb'] = model_size
    
    # 特定组件大小
    if hasattr(model, 'tree_'):
        tree_size = sys.getsizeof(model.tree_) / (1024 * 1024)
        memory_info['tree_size_mb'] = tree_size
    
    if hasattr(model, 'support_vectors_'):
        sv_size = sys.getsizeof(model.support_vectors_) / (1024 * 1024)
        memory_info['support_vectors_size_mb'] = sv_size
    
    if hasattr(model, 'coef_'):
        coef_size = sys.getsizeof(model.coef_) / (1024 * 1024)
        memory_info['coefficients_size_mb'] = coef_size
    
    return memory_info


def benchmark_prediction_time(model: BaseEstimator,
                            X: np.ndarray,
                            n_runs: int = 100) -> Dict[str, float]:
    """
    基准测试预测时间
    
    Args:
        model: 模型
        X: 测试数据
        n_runs: 运行次数
        
    Returns:
        时间统计信息
    """
    times = []
    
    for _ in range(n_runs):
        start_time = time.time()
        model.predict(X)
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'predictions_per_second': len(X) / np.mean(times)
    }


def create_training_summary(results: Dict[str, TrainingResult],
                          save_path: Optional[Union[str, Path]] = None) -> str:
    """
    创建训练总结报告
    
    Args:
        results: 训练结果字典
        save_path: 保存路径
        
    Returns:
        总结报告文本
    """
    summary = []
    summary.append("=" * 60)
    summary.append("TRAINING SUMMARY REPORT")
    summary.append("=" * 60)
    summary.append("")
    
    # 总体统计
    summary.append(f"Total models trained: {len(results)}")
    summary.append(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # 最佳模型
    if results:
        best_model = max(results.items(), 
                        key=lambda x: x[1].val_scores.get('accuracy', 0))
        summary.append(f"Best model: {best_model[0]}")
        summary.append(f"Best accuracy: {best_model[1].val_scores.get('accuracy', 'N/A'):.4f}")
        summary.append("")
    
    # 详细结果
    summary.append("DETAILED RESULTS:")
    summary.append("-" * 40)
    
    for model_name, result in results.items():
        summary.append(f"\nModel: {model_name}")
        summary.append(f"Training time: {result.training_time:.2f}s")
        
        # 训练分数
        if result.train_scores:
            summary.append("Training scores:")
            for metric, score in result.train_scores.items():
                summary.append(f"  {metric}: {score:.4f}")
        
        # 验证分数
        if result.val_scores:
            summary.append("Validation scores:")
            for metric, score in result.val_scores.items():
                summary.append(f"  {metric}: {score:.4f}")
        
        # 最佳参数
        if result.best_params:
            summary.append("Best parameters:")
            for param, value in result.best_params.items():
                summary.append(f"  {param}: {value}")
    
    summary.append("")
    summary.append("=" * 60)
    
    report_text = "\n".join(summary)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Training summary saved to {save_path}")
    
    return report_text


def setup_logging(log_level: str = 'INFO',
                 log_file: Optional[Union[str, Path]] = None) -> None:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    # 抑制一些警告
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)