"""
训练框架基础类

定义训练器的基础接口和数据结构
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
import pickle
import json
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # 基础配置
    random_state: Optional[int] = 42
    n_jobs: int = 1
    verbose: bool = True
    
    # 数据分割
    test_size: float = 0.2
    validation_size: float = 0.2
    stratify: bool = True
    
    # 交叉验证
    cv_folds: int = 5
    cv_strategy: str = 'kfold'  # 'kfold', 'stratified_kfold', 'time_series'
    
    # 早停策略
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 0.001
    
    # 模型保存
    save_best_model: bool = True
    save_all_models: bool = False
    model_save_path: Optional[str] = None
    
    # 超参数优化
    hyperparameter_optimization: bool = False
    optimization_method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian'
    optimization_trials: int = 100
    
    # 集成学习
    ensemble_methods: List[str] = field(default_factory=lambda: ['voting'])
    ensemble_weights: Optional[List[float]] = None
    
    # 其他配置
    metrics: List[str] = field(default_factory=lambda: ['accuracy'])
    scoring: str = 'accuracy'
    refit: Union[str, bool] = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
            'test_size': self.test_size,
            'validation_size': self.validation_size,
            'stratify': self.stratify,
            'cv_folds': self.cv_folds,
            'cv_strategy': self.cv_strategy,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'save_best_model': self.save_best_model,
            'save_all_models': self.save_all_models,
            'model_save_path': self.model_save_path,
            'hyperparameter_optimization': self.hyperparameter_optimization,
            'optimization_method': self.optimization_method,
            'optimization_trials': self.optimization_trials,
            'ensemble_methods': self.ensemble_methods,
            'ensemble_weights': self.ensemble_weights,
            'metrics': self.metrics,
            'scoring': self.scoring,
            'refit': self.refit
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """从字典创建配置"""
        return cls(**config_dict)


@dataclass
class TrainingResult:
    """训练结果类"""
    
    # 基础信息
    model_name: str
    task_type: str
    training_time: float
    
    # 模型和预测
    best_model: Optional[BaseEstimator] = None
    all_models: List[BaseEstimator] = field(default_factory=list)
    train_predictions: Optional[np.ndarray] = None
    val_predictions: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None
    
    # 性能指标
    train_scores: Dict[str, float] = field(default_factory=dict)
    val_scores: Dict[str, float] = field(default_factory=dict)
    test_scores: Dict[str, float] = field(default_factory=dict)
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    
    # 超参数
    best_params: Dict[str, Any] = field(default_factory=dict)
    all_params: List[Dict[str, Any]] = field(default_factory=list)
    
    # 训练历史
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    validation_history: Dict[str, List[float]] = field(default_factory=dict)
    
    # 特征重要性
    feature_importance: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    
    # 其他信息
    config: Optional[TrainingConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_best_score(self, metric: str = 'accuracy', dataset: str = 'val') -> float:
        """获取最佳分数"""
        if dataset == 'train':
            return self.train_scores.get(metric, 0.0)
        elif dataset == 'val':
            return self.val_scores.get(metric, 0.0)
        elif dataset == 'test':
            return self.test_scores.get(metric, 0.0)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def get_cv_score_stats(self, metric: str = 'accuracy') -> Dict[str, float]:
        """获取交叉验证分数统计"""
        if metric not in self.cv_scores:
            return {}
        
        scores = self.cv_scores[metric]
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores)
        }
    
    def save(self, save_path: str) -> None:
        """保存训练结果"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        if self.best_model is not None:
            model_path = save_path / f"{self.model_name}_best_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
        
        # 保存结果数据
        result_data = {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'training_time': self.training_time,
            'train_scores': self.train_scores,
            'val_scores': self.val_scores,
            'test_scores': self.test_scores,
            'cv_scores': self.cv_scores,
            'best_params': self.best_params,
            'all_params': self.all_params,
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'feature_names': self.feature_names,
            'config': self.config.to_dict() if self.config else None,
            'metadata': self.metadata
        }
        
        result_path = save_path / f"{self.model_name}_training_result.json"
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        logger.info(f"Training result saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str, model_name: str) -> 'TrainingResult':
        """加载训练结果"""
        load_path = Path(load_path)
        
        # 加载结果数据
        result_path = load_path / f"{model_name}_training_result.json"
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        # 加载模型
        model_path = load_path / f"{model_name}_best_model.pkl"
        best_model = None
        if model_path.exists():
            with open(model_path, 'rb') as f:
                best_model = pickle.load(f)
        
        # 重建配置
        config = None
        if result_data.get('config'):
            config = TrainingConfig.from_dict(result_data['config'])
        
        # 创建结果对象
        result = cls(
            model_name=result_data['model_name'],
            task_type=result_data['task_type'],
            training_time=result_data['training_time'],
            best_model=best_model,
            train_scores=result_data.get('train_scores', {}),
            val_scores=result_data.get('val_scores', {}),
            test_scores=result_data.get('test_scores', {}),
            cv_scores=result_data.get('cv_scores', {}),
            best_params=result_data.get('best_params', {}),
            all_params=result_data.get('all_params', []),
            training_history=result_data.get('training_history', {}),
            validation_history=result_data.get('validation_history', {}),
            feature_importance=np.array(result_data['feature_importance']) if result_data.get('feature_importance') else None,
            feature_names=result_data.get('feature_names'),
            config=config,
            metadata=result_data.get('metadata', {})
        )
        
        return result


class BaseTrainer(ABC):
    """训练器基类"""
    
    def __init__(self, 
                 config: Optional[TrainingConfig] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        初始化训练器
        
        Args:
            config: 训练配置
            random_state: 随机种子
            verbose: 是否显示详细信息
        """
        self.config = config or TrainingConfig()
        if random_state is not None:
            self.config.random_state = random_state
        if verbose is not None:
            self.config.verbose = verbose
        
        self.verbose = self.config.verbose
        self.random_state = self.config.random_state
        
        # 训练状态
        self.is_fitted_ = False
        self.training_results_ = {}
        self.best_model_ = None
        self.best_score_ = -np.inf
        
        # 回调函数
        self.callbacks = []
        
        if self.verbose:
            logger.info(f"Initialized {self.__class__.__name__} with config: {self.config.to_dict()}")
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseTrainer':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的训练器
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 预测特征
            
        Returns:
            预测结果
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        预测概率（如果支持）
        
        Args:
            X: 预测特征
            
        Returns:
            预测概率
        """
        if hasattr(self.best_model_, 'predict_proba'):
            return self.best_model_.predict_proba(X)
        return None
    
    def score(self, X: np.ndarray, y: np.ndarray, metric: str = 'auto') -> float:
        """
        评估模型
        
        Args:
            X: 特征
            y: 标签
            metric: 评估指标
            
        Returns:
            评估分数
        """
        if not self.is_fitted_:
            raise ValueError("Trainer must be fitted before scoring")
        
        predictions = self.predict(X)
        
        if metric == 'auto':
            metric = 'accuracy' if self._is_classification() else 'neg_mean_squared_error'
        
        if metric == 'accuracy':
            return accuracy_score(y, predictions)
        elif metric == 'neg_mean_squared_error':
            return -mean_squared_error(y, predictions)
        else:
            # 使用sklearn的评分函数
            from sklearn.metrics import get_scorer
            scorer = get_scorer(metric)
            return scorer(self.best_model_, X, y)
    
    def _is_classification(self) -> bool:
        """判断是否为分类任务"""
        return hasattr(self.best_model_, 'predict_proba') or \
               hasattr(self.best_model_, 'decision_function')
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """分割数据"""
        if self.config.test_size > 0:
            stratify = y if self.config.stratify and self._is_classification() else None
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify
            )
            
            if self.config.validation_size > 0:
                val_size = self.config.validation_size / (1 - self.config.test_size)
                stratify_temp = y_temp if self.config.stratify and self._is_classification() else None
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=val_size,
                    random_state=self.config.random_state,
                    stratify=stratify_temp
                )
                
                return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                return X_temp, X_test, y_temp, y_test
        else:
            if self.config.validation_size > 0:
                stratify = y if self.config.stratify and self._is_classification() else None
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=self.config.validation_size,
                    random_state=self.config.random_state,
                    stratify=stratify
                )
                
                return X_train, X_val, y_train, y_val
            else:
                return X, y
    
    def _evaluate_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
        predictions = model.predict(X)
        scores = {}
        
        for metric in self.config.metrics:
            try:
                if metric == 'accuracy':
                    scores[metric] = accuracy_score(y, predictions)
                elif metric == 'mse':
                    scores[metric] = mean_squared_error(y, predictions)
                elif metric == 'rmse':
                    scores[metric] = np.sqrt(mean_squared_error(y, predictions))
                else:
                    from sklearn.metrics import get_scorer
                    scorer = get_scorer(metric)
                    scores[metric] = scorer(model, X, y)
            except Exception as e:
                logger.warning(f"Failed to compute metric {metric}: {e}")
                scores[metric] = 0.0
        
        return scores
    
    def add_callback(self, callback: Callable) -> None:
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def _call_callbacks(self, event: str, **kwargs) -> None:
        """调用回调函数"""
        for callback in self.callbacks:
            try:
                callback(event, **kwargs)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")
    
    def get_training_results(self) -> Dict[str, TrainingResult]:
        """获取训练结果"""
        return self.training_results_
    
    def get_best_model(self) -> Optional[BaseEstimator]:
        """获取最佳模型"""
        return self.best_model_
    
    def get_best_score(self) -> float:
        """获取最佳分数"""
        return self.best_score_
    
    def save_results(self, save_path: str) -> None:
        """保存训练结果"""
        for model_name, result in self.training_results_.items():
            result.save(save_path)
    
    def load_results(self, load_path: str, model_names: List[str]) -> None:
        """加载训练结果"""
        for model_name in model_names:
            try:
                result = TrainingResult.load(load_path, model_name)
                self.training_results_[model_name] = result
                
                # 更新最佳模型
                if result.best_model and result.get_best_score() > self.best_score_:
                    self.best_model_ = result.best_model
                    self.best_score_ = result.get_best_score()
                    
            except Exception as e:
                logger.warning(f"Failed to load result for {model_name}: {e}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"