"""
评估系统基础类

定义评估器接口和基础数据结构
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """评估配置"""
    
    # 基础配置
    metrics: List[str] = field(default_factory=lambda: ['accuracy'])
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # 可视化配置
    plot_style: str = 'seaborn'
    figure_size: Tuple[int, int] = (10, 8)
    save_plots: bool = True
    plot_format: str = 'png'
    dpi: int = 300
    
    # 报告配置
    include_plots: bool = True
    include_statistical_tests: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # 性能配置
    n_jobs: int = 1
    verbose: bool = True
    
    # 输出配置
    output_dir: Optional[str] = None
    save_results: bool = True
    
    def __post_init__(self):
        """后处理初始化"""
        if self.output_dir:
            import os
            os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class EvaluationResult:
    """评估结果"""
    
    # 基础信息
    model_name: str
    task_type: str
    evaluation_time: float = 0.0
    
    # 性能指标
    train_scores: Dict[str, float] = field(default_factory=dict)
    val_scores: Dict[str, float] = field(default_factory=dict)
    test_scores: Dict[str, float] = field(default_factory=dict)
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    
    # 预测结果
    train_predictions: Optional[np.ndarray] = None
    val_predictions: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None
    prediction_probabilities: Optional[np.ndarray] = None
    
    # 模型信息
    model_params: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    
    # 统计信息
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    statistical_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 可视化
    plots: Dict[str, str] = field(default_factory=dict)  # plot_name -> file_path
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_cv_score_stats(self, metric: str) -> Dict[str, float]:
        """获取交叉验证分数统计"""
        if metric not in self.cv_scores:
            return {}
        
        scores = np.array(self.cv_scores[metric])
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores)
        }
    
    def get_best_score(self, metric: str, score_type: str = 'cv') -> float:
        """获取最佳分数"""
        if score_type == 'cv':
            stats = self.get_cv_score_stats(metric)
            return stats.get('mean', 0.0)
        elif score_type == 'val':
            return self.val_scores.get(metric, 0.0)
        elif score_type == 'test':
            return self.test_scores.get(metric, 0.0)
        elif score_type == 'train':
            return self.train_scores.get(metric, 0.0)
        else:
            return 0.0
    
    def add_confidence_interval(self, metric: str, ci: Tuple[float, float]):
        """添加置信区间"""
        self.confidence_intervals[metric] = ci
    
    def add_statistical_test(self, test_name: str, result: Dict[str, Any]):
        """添加统计检验结果"""
        self.statistical_tests[test_name] = result
    
    def add_plot(self, plot_name: str, file_path: str):
        """添加图表"""
        self.plots[plot_name] = file_path
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        
        # 基础信息
        result['model_name'] = self.model_name
        result['task_type'] = self.task_type
        result['evaluation_time'] = self.evaluation_time
        
        # 性能指标
        result['train_scores'] = self.train_scores
        result['val_scores'] = self.val_scores
        result['test_scores'] = self.test_scores
        result['cv_scores'] = self.cv_scores
        
        # 统计信息
        result['confidence_intervals'] = self.confidence_intervals
        result['statistical_tests'] = self.statistical_tests
        
        # 模型信息
        result['model_params'] = self.model_params
        if self.feature_importance is not None:
            result['feature_importance'] = self.feature_importance.tolist()
        result['feature_names'] = self.feature_names
        
        # 可视化
        result['plots'] = self.plots
        
        # 元数据
        result['metadata'] = self.metadata
        
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame（用于比较）"""
        data = {'model': self.model_name}
        
        # 添加分数
        for metric, score in self.train_scores.items():
            data[f'train_{metric}'] = score
        
        for metric, score in self.val_scores.items():
            data[f'val_{metric}'] = score
        
        for metric, score in self.test_scores.items():
            data[f'test_{metric}'] = score
        
        # 添加交叉验证统计
        for metric in self.cv_scores:
            stats = self.get_cv_score_stats(metric)
            data[f'cv_{metric}_mean'] = stats.get('mean', np.nan)
            data[f'cv_{metric}_std'] = stats.get('std', np.nan)
        
        # 添加置信区间
        for metric, (lower, upper) in self.confidence_intervals.items():
            data[f'{metric}_ci_lower'] = lower
            data[f'{metric}_ci_upper'] = upper
        
        data['evaluation_time'] = self.evaluation_time
        
        return pd.DataFrame([data])


class BaseEvaluator(ABC):
    """基础评估器"""
    
    def __init__(self, 
                 config: Optional[EvaluationConfig] = None,
                 **kwargs):
        """
        初始化评估器
        
        Args:
            config: 评估配置
            **kwargs: 其他参数
        """
        self.config = config or EvaluationConfig()
        self.verbose = kwargs.get('verbose', self.config.verbose)
        
        # 评估结果
        self.results_: Dict[str, EvaluationResult] = {}
        
        # 数据
        self.X_train_: Optional[np.ndarray] = None
        self.X_val_: Optional[np.ndarray] = None
        self.X_test_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.y_val_: Optional[np.ndarray] = None
        self.y_test_: Optional[np.ndarray] = None
        
        if self.verbose:
            logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def evaluate_model(self, 
                      model: BaseEstimator,
                      X: np.ndarray,
                      y: np.ndarray,
                      model_name: Optional[str] = None,
                      **kwargs) -> EvaluationResult:
        """
        评估单个模型
        
        Args:
            model: 模型
            X: 特征
            y: 标签
            model_name: 模型名称
            **kwargs: 其他参数
            
        Returns:
            评估结果
        """
        pass
    
    def evaluate_multiple_models(self,
                                models: Dict[str, BaseEstimator],
                                X: np.ndarray,
                                y: np.ndarray,
                                **kwargs) -> Dict[str, EvaluationResult]:
        """
        评估多个模型
        
        Args:
            models: 模型字典
            X: 特征
            y: 标签
            **kwargs: 其他参数
            
        Returns:
            评估结果字典
        """
        results = {}
        
        for model_name, model in models.items():
            if self.verbose:
                logger.info(f"Evaluating model: {model_name}")
            
            try:
                result = self.evaluate_model(model, X, y, model_name, **kwargs)
                results[model_name] = result
                self.results_[model_name] = result
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        return results
    
    def prepare_data(self, 
                    X: np.ndarray, 
                    y: np.ndarray,
                    test_size: Optional[float] = None,
                    val_size: Optional[float] = None) -> None:
        """
        准备数据集
        
        Args:
            X: 特征
            y: 标签
            test_size: 测试集比例
            val_size: 验证集比例
        """
        from sklearn.model_selection import train_test_split
        
        test_size = test_size or self.config.test_size
        val_size = val_size or 0.2
        
        # 分割训练集和测试集
        X_temp, self.X_test_, y_temp, self.y_test_ = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.config.random_state,
            stratify=y if len(np.unique(y)) < 10 else None
        )
        
        # 分割训练集和验证集
        self.X_train_, self.X_val_, self.y_train_, self.y_val_ = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.random_state,
            stratify=y_temp if len(np.unique(y_temp)) < 10 else None
        )
        
        if self.verbose:
            logger.info(f"Data prepared: Train={len(self.X_train_)}, "
                      f"Val={len(self.X_val_)}, Test={len(self.X_test_)}")
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            
        Returns:
            指标字典
        """
        metrics = {}
        
        for metric_name in self.config.metrics:
            try:
                scorer = get_scorer(metric_name)
                if metric_name in ['roc_auc', 'average_precision'] and y_prob is not None:
                    # 需要概率的指标
                    if y_prob.ndim > 1 and y_prob.shape[1] > 2:
                        # 多分类，使用ovr策略
                        score = scorer._score_func(y_true, y_prob, multi_class='ovr')
                    else:
                        # 二分类或回归
                        prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                        score = scorer._score_func(y_true, prob)
                else:
                    # 使用预测标签
                    score = scorer._score_func(y_true, y_pred)
                
                metrics[metric_name] = score
                
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_name}: {e}")
                metrics[metric_name] = np.nan
        
        return metrics
    
    def cross_validate_model(self, 
                           model: BaseEstimator,
                           X: np.ndarray,
                           y: np.ndarray) -> Dict[str, List[float]]:
        """
        交叉验证模型
        
        Args:
            model: 模型
            X: 特征
            y: 标签
            
        Returns:
            交叉验证分数
        """
        from sklearn.model_selection import cross_validate
        
        # 构建评分器
        scoring = {}
        for metric in self.config.metrics:
            try:
                scorer = get_scorer(metric)
                scoring[metric] = scorer
            except:
                continue
        
        if not scoring:
            scoring = ['accuracy']
        
        # 执行交叉验证
        cv_results = cross_validate(
            model, X, y,
            cv=self.config.cv_folds,
            scoring=scoring,
            n_jobs=self.config.n_jobs,
            return_train_score=True
        )
        
        # 整理结果
        cv_scores = {}
        for metric in scoring:
            test_key = f'test_{metric}' if isinstance(scoring, dict) else 'test_score'
            if test_key in cv_results:
                cv_scores[metric] = cv_results[test_key].tolist()
        
        return cv_scores
    
    def get_feature_importance(self, model: BaseEstimator) -> Optional[np.ndarray]:
        """获取特征重要性"""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                return np.mean(np.abs(coef), axis=0)
            else:
                return np.abs(coef)
        else:
            return None
    
    def save_results(self, filepath: str) -> None:
        """保存评估结果"""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results_, f)
        
        if self.verbose:
            logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """加载评估结果"""
        import pickle
        
        with open(filepath, 'rb') as f:
            self.results_ = pickle.load(f)
        
        if self.verbose:
            logger.info(f"Results loaded from {filepath}")
    
    def get_results_summary(self) -> pd.DataFrame:
        """获取结果摘要"""
        if not self.results_:
            return pd.DataFrame()
        
        dfs = []
        for result in self.results_.values():
            dfs.append(result.to_dataframe())
        
        return pd.concat(dfs, ignore_index=True)
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, EvaluationResult]:
        """
        获取最佳模型
        
        Args:
            metric: 评估指标
            
        Returns:
            最佳模型名称和结果
        """
        if not self.results_:
            raise ValueError("No evaluation results available")
        
        best_score = -np.inf
        best_model = None
        best_result = None
        
        for model_name, result in self.results_.items():
            score = result.get_best_score(metric, 'cv')
            if score > best_score:
                best_score = score
                best_model = model_name
                best_result = result
        
        return best_model, best_result