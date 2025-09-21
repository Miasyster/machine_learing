"""
评估工具函数模块

提供评估过程中需要的各种工具函数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import time
import pickle
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(model: BaseEstimator,
                                X: np.ndarray,
                                y: np.ndarray,
                                metric: str = 'accuracy',
                                n_bootstrap: int = 1000,
                                confidence_level: float = 0.95,
                                random_state: Optional[int] = None) -> Tuple[float, float]:
    """
    使用Bootstrap方法计算置信区间
    
    Args:
        model: 已训练的模型
        X: 特征数据
        y: 目标数据
        metric: 评估指标
        n_bootstrap: Bootstrap样本数
        confidence_level: 置信水平
        random_state: 随机种子
        
    Returns:
        置信区间 (lower, upper)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    bootstrap_scores = []
    
    # 选择评估函数
    metric_func = get_metric_function(metric)
    
    for _ in range(n_bootstrap):
        # Bootstrap采样
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # 预测并计算指标
        try:
            y_pred = model.predict(X_boot)
            score = metric_func(y_boot, y_pred)
            bootstrap_scores.append(score)
        except Exception as e:
            logger.warning(f"Bootstrap iteration failed: {e}")
            continue
    
    if not bootstrap_scores:
        logger.error("All bootstrap iterations failed")
        return (np.nan, np.nan)
    
    # 计算置信区间
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(bootstrap_scores, lower_percentile)
    upper = np.percentile(bootstrap_scores, upper_percentile)
    
    return (lower, upper)


def get_metric_function(metric: str) -> Callable:
    """
    获取指标计算函数
    
    Args:
        metric: 指标名称
        
    Returns:
        指标计算函数
    """
    metric_functions = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score
    }
    
    if metric not in metric_functions:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return metric_functions[metric]


def cross_validate_with_groups(model: BaseEstimator,
                              X: np.ndarray,
                              y: np.ndarray,
                              groups: Optional[np.ndarray] = None,
                              cv: int = 5,
                              scoring: Union[str, List[str]] = 'accuracy',
                              random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    带分组的交叉验证
    
    Args:
        model: 模型
        X: 特征数据
        y: 目标数据
        groups: 分组标识
        cv: 交叉验证折数
        scoring: 评估指标
        random_state: 随机种子
        
    Returns:
        交叉验证分数
    """
    from sklearn.model_selection import GroupKFold, cross_validate
    
    # 选择交叉验证策略
    if groups is not None:
        cv_strategy = GroupKFold(n_splits=cv)
    else:
        # 根据任务类型选择策略
        if len(np.unique(y)) < 20:  # 分类任务
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:  # 回归任务
            cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # 执行交叉验证
    if isinstance(scoring, str):
        scoring = [scoring]
    
    cv_results = cross_validate(
        model, X, y, 
        groups=groups,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    return cv_results


def calculate_learning_curve(model: BaseEstimator,
                           X: np.ndarray,
                           y: np.ndarray,
                           train_sizes: Optional[np.ndarray] = None,
                           cv: int = 5,
                           scoring: str = 'accuracy',
                           random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    计算学习曲线
    
    Args:
        model: 模型
        X: 特征数据
        y: 目标数据
        train_sizes: 训练集大小
        cv: 交叉验证折数
        scoring: 评估指标
        random_state: 随机种子
        
    Returns:
        学习曲线数据
    """
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    # 选择交叉验证策略
    if len(np.unique(y)) < 20:  # 分类任务
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:  # 回归任务
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state
    )
    
    return {
        'train_sizes': train_sizes_abs,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'train_scores_mean': np.mean(train_scores, axis=1),
        'train_scores_std': np.std(train_scores, axis=1),
        'val_scores_mean': np.mean(val_scores, axis=1),
        'val_scores_std': np.std(val_scores, axis=1)
    }


def calculate_validation_curve(model: BaseEstimator,
                             X: np.ndarray,
                             y: np.ndarray,
                             param_name: str,
                             param_range: np.ndarray,
                             cv: int = 5,
                             scoring: str = 'accuracy',
                             random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    计算验证曲线
    
    Args:
        model: 模型
        X: 特征数据
        y: 目标数据
        param_name: 参数名称
        param_range: 参数范围
        cv: 交叉验证折数
        scoring: 评估指标
        random_state: 随机种子
        
    Returns:
        验证曲线数据
    """
    from sklearn.model_selection import validation_curve
    
    # 选择交叉验证策略
    if len(np.unique(y)) < 20:  # 分类任务
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:  # 回归任务
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1
    )
    
    return {
        'param_range': param_range,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'train_scores_mean': np.mean(train_scores, axis=1),
        'train_scores_std': np.std(train_scores, axis=1),
        'val_scores_mean': np.mean(val_scores, axis=1),
        'val_scores_std': np.std(val_scores, axis=1)
    }


def evaluate_model_stability(model: BaseEstimator,
                           X: np.ndarray,
                           y: np.ndarray,
                           n_runs: int = 10,
                           test_size: float = 0.2,
                           scoring: str = 'accuracy',
                           random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    评估模型稳定性
    
    Args:
        model: 模型
        X: 特征数据
        y: 目标数据
        n_runs: 运行次数
        test_size: 测试集比例
        scoring: 评估指标
        random_state: 随机种子
        
    Returns:
        稳定性评估结果
    """
    from sklearn.model_selection import train_test_split
    
    scores = []
    metric_func = get_metric_function(scoring)
    
    for i in range(n_runs):
        # 随机分割数据
        seed = random_state + i if random_state is not None else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        
        # 训练和评估
        model_copy = clone(model)
        model_copy.fit(X_train, y_train)
        y_pred = model_copy.predict(X_test)
        score = metric_func(y_test, y_pred)
        scores.append(score)
    
    scores = np.array(scores)
    
    return {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'median': np.median(scores),
        'q25': np.percentile(scores, 25),
        'q75': np.percentile(scores, 75),
        'coefficient_of_variation': np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else np.inf
    }


def benchmark_prediction_time(model: BaseEstimator,
                            X: np.ndarray,
                            n_runs: int = 100) -> Dict[str, float]:
    """
    基准测试预测时间
    
    Args:
        model: 已训练的模型
        X: 特征数据
        n_runs: 运行次数
        
    Returns:
        时间统计
    """
    times = []
    
    for _ in range(n_runs):
        start_time = time.time()
        _ = model.predict(X)
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


def calculate_model_complexity(model: BaseEstimator) -> Dict[str, Any]:
    """
    计算模型复杂度
    
    Args:
        model: 模型
        
    Returns:
        复杂度指标
    """
    complexity = {}
    
    # 参数数量
    try:
        params = model.get_params()
        complexity['n_parameters'] = len(params)
        complexity['parameters'] = params
    except:
        complexity['n_parameters'] = None
    
    # 特定模型的复杂度指标
    model_name = model.__class__.__name__
    
    if hasattr(model, 'tree_'):
        # 决策树
        complexity['tree_depth'] = model.tree_.max_depth
        complexity['n_nodes'] = model.tree_.node_count
        complexity['n_leaves'] = model.tree_.n_leaves
    
    elif hasattr(model, 'estimators_'):
        # 集成方法
        complexity['n_estimators'] = len(model.estimators_)
        if hasattr(model.estimators_[0], 'tree_'):
            depths = [est.tree_.max_depth for est in model.estimators_]
            complexity['avg_tree_depth'] = np.mean(depths)
            complexity['max_tree_depth'] = np.max(depths)
    
    elif hasattr(model, 'coef_'):
        # 线性模型
        if model.coef_.ndim == 1:
            complexity['n_features'] = len(model.coef_)
            complexity['n_nonzero_coef'] = np.sum(model.coef_ != 0)
        else:
            complexity['n_features'] = model.coef_.shape[1]
            complexity['n_nonzero_coef'] = np.sum(model.coef_ != 0)
    
    elif hasattr(model, 'support_vectors_'):
        # SVM
        complexity['n_support_vectors'] = len(model.support_vectors_)
        complexity['support_vector_ratio'] = len(model.support_vectors_) / model.n_support_.sum()
    
    return complexity


def save_evaluation_results(results: List[Any],
                          output_path: str,
                          format: str = 'pickle') -> None:
    """
    保存评估结果
    
    Args:
        results: 评估结果列表
        output_path: 输出路径
        format: 保存格式 ('pickle', 'joblib')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'joblib':
        joblib.dump(results, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Evaluation results saved to {output_path}")


def load_evaluation_results(input_path: str,
                          format: str = 'pickle') -> List[Any]:
    """
    加载评估结果
    
    Args:
        input_path: 输入路径
        format: 文件格式 ('pickle', 'joblib')
        
    Returns:
        评估结果列表
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    if format == 'pickle':
        with open(input_path, 'rb') as f:
            results = pickle.load(f)
    elif format == 'joblib':
        results = joblib.load(input_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Evaluation results loaded from {input_path}")
    return results


def create_evaluation_summary(results: List[Any],
                            metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    创建评估摘要表
    
    Args:
        results: 评估结果列表
        metrics: 要包含的指标
        
    Returns:
        摘要DataFrame
    """
    summary_data = []
    
    for result in results:
        if hasattr(result, 'model_name') and hasattr(result, 'test_scores'):
            row = {'Model': result.model_name}
            
            # 添加测试分数
            for metric, score in result.test_scores.items():
                if metrics is None or metric in metrics:
                    row[f'Test_{metric}'] = score
            
            # 添加交叉验证分数
            if hasattr(result, 'cv_scores') and result.cv_scores:
                for metric, scores in result.cv_scores.items():
                    if metrics is None or metric in metrics:
                        row[f'CV_{metric}_mean'] = np.mean(scores)
                        row[f'CV_{metric}_std'] = np.std(scores)
            
            # 添加其他信息
            if hasattr(result, 'evaluation_time'):
                row['Evaluation_Time'] = result.evaluation_time
            
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def detect_overfitting(train_scores: np.ndarray,
                      val_scores: np.ndarray,
                      threshold: float = 0.1) -> Dict[str, Any]:
    """
    检测过拟合
    
    Args:
        train_scores: 训练分数
        val_scores: 验证分数
        threshold: 过拟合阈值
        
    Returns:
        过拟合检测结果
    """
    train_mean = np.mean(train_scores)
    val_mean = np.mean(val_scores)
    
    # 计算差异
    score_gap = train_mean - val_mean
    relative_gap = score_gap / train_mean if train_mean != 0 else np.inf
    
    # 判断过拟合
    is_overfitting = relative_gap > threshold
    
    return {
        'is_overfitting': is_overfitting,
        'train_mean': train_mean,
        'val_mean': val_mean,
        'score_gap': score_gap,
        'relative_gap': relative_gap,
        'threshold': threshold,
        'severity': 'high' if relative_gap > 2 * threshold else 'moderate' if relative_gap > threshold else 'low'
    }


def calculate_feature_stability(X: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              n_bootstrap: int = 100,
                              random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    计算特征稳定性
    
    Args:
        X: 特征数据
        feature_names: 特征名称
        n_bootstrap: Bootstrap样本数
        random_state: 随机种子
        
    Returns:
        特征稳定性指标
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # 计算每个特征的统计量
    feature_stats = {}
    
    for i, name in enumerate(feature_names):
        feature_data = X[:, i]
        
        # Bootstrap采样计算统计量的稳定性
        bootstrap_means = []
        bootstrap_stds = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_data = feature_data[indices]
            
            bootstrap_means.append(np.mean(sample_data))
            bootstrap_stds.append(np.std(sample_data))
        
        feature_stats[name] = {
            'mean_stability': np.std(bootstrap_means),
            'std_stability': np.std(bootstrap_stds),
            'original_mean': np.mean(feature_data),
            'original_std': np.std(feature_data)
        }
    
    return feature_stats


def generate_evaluation_config(task_type: str = 'classification',
                             cv_folds: int = 5,
                             test_size: float = 0.2,
                             random_state: int = 42,
                             **kwargs) -> Dict[str, Any]:
    """
    生成评估配置
    
    Args:
        task_type: 任务类型
        cv_folds: 交叉验证折数
        test_size: 测试集比例
        random_state: 随机种子
        **kwargs: 其他配置
        
    Returns:
        评估配置字典
    """
    config = {
        'task_type': task_type,
        'cv_folds': cv_folds,
        'test_size': test_size,
        'random_state': random_state,
        'include_statistical_tests': kwargs.get('include_statistical_tests', True),
        'bootstrap_samples': kwargs.get('bootstrap_samples', 1000),
        'confidence_level': kwargs.get('confidence_level', 0.95),
        'save_plots': kwargs.get('save_plots', True),
        'plot_format': kwargs.get('plot_format', 'png'),
        'figure_size': kwargs.get('figure_size', (10, 6)),
        'dpi': kwargs.get('dpi', 300)
    }
    
    # 根据任务类型设置默认指标
    if task_type == 'classification':
        config['default_metrics'] = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro'
        ]
    elif task_type == 'regression':
        config['default_metrics'] = [
            'mse', 'rmse', 'mae', 'r2'
        ]
    else:
        config['default_metrics'] = []
    
    return config