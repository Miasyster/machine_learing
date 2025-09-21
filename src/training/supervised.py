"""
监督学习训练器

实现分类和回归任务的训练器
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Type
import logging
import time
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from .base import BaseTrainer, TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)


class SupervisedTrainer(BaseTrainer):
    """监督学习训练器"""
    
    # 预定义的分类器
    CLASSIFIERS = {
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression,
        'svm': SVC,
        'decision_tree': DecisionTreeClassifier,
        'knn': KNeighborsClassifier,
        'naive_bayes': GaussianNB,
        'mlp': MLPClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'ada_boost': AdaBoostClassifier,
        'extra_trees': ExtraTreesClassifier
    }
    
    # 预定义的回归器
    REGRESSORS = {
        'random_forest': RandomForestRegressor,
        'linear_regression': LinearRegression,
        'svm': SVR,
        'decision_tree': DecisionTreeRegressor,
        'knn': KNeighborsRegressor,
        'mlp': MLPRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'ada_boost': AdaBoostRegressor,
        'extra_trees': ExtraTreesRegressor
    }
    
    # 默认超参数网格
    DEFAULT_PARAM_GRIDS = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'logistic_regression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'svm': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        },
        'decision_tree': {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'mlp': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
    
    def __init__(self, 
                 task_type: str = 'classification',
                 models: Optional[Union[List[str], Dict[str, BaseEstimator]]] = None,
                 param_grids: Optional[Dict[str, Dict[str, List]]] = None,
                 config: Optional[TrainingConfig] = None,
                 **kwargs):
        """
        初始化监督学习训练器
        
        Args:
            task_type: 任务类型 ('classification' 或 'regression')
            models: 要训练的模型列表或字典
            param_grids: 超参数网格
            config: 训练配置
            **kwargs: 其他参数
        """
        super().__init__(config, **kwargs)
        
        self.task_type = task_type
        self.param_grids = param_grids or {}
        
        # 设置模型
        if models is None:
            if task_type == 'classification':
                self.models = {name: cls() for name, cls in self.CLASSIFIERS.items()}
            else:
                self.models = {name: cls() for name, cls in self.REGRESSORS.items()}
        elif isinstance(models, list):
            if task_type == 'classification':
                self.models = {name: self.CLASSIFIERS[name]() for name in models if name in self.CLASSIFIERS}
            else:
                self.models = {name: self.REGRESSORS[name]() for name in models if name in self.REGRESSORS}
        else:
            self.models = models
        
        # 设置默认超参数网格
        for model_name in self.models.keys():
            if model_name not in self.param_grids and model_name in self.DEFAULT_PARAM_GRIDS:
                self.param_grids[model_name] = self.DEFAULT_PARAM_GRIDS[model_name]
        
        if self.verbose:
            logger.info(f"Initialized SupervisedTrainer for {task_type} with models: {list(self.models.keys())}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SupervisedTrainer':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的训练器
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"Starting training with {len(self.models)} models")
        
        # 分割数据
        data_splits = self._split_data(X, y)
        if len(data_splits) == 6:  # train, val, test
            X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        elif len(data_splits) == 4:  # train, val 或 train, test
            if self.config.validation_size > 0:
                X_train, X_val, y_train, y_val = data_splits
                X_test = y_test = None
            else:
                X_train, X_test, y_train, y_test = data_splits
                X_val = y_val = None
        else:  # 只有训练集
            X_train, y_train = data_splits
            X_val = X_test = y_val = y_test = None
        
        # 训练每个模型
        for model_name, model in self.models.items():
            if self.verbose:
                logger.info(f"Training {model_name}...")
            
            model_start_time = time.time()
            
            try:
                # 超参数优化
                if self.config.hyperparameter_optimization and model_name in self.param_grids:
                    best_model = self._optimize_hyperparameters(
                        model, X_train, y_train, model_name
                    )
                else:
                    # 设置随机种子
                    if hasattr(model, 'random_state'):
                        model.set_params(random_state=self.config.random_state)
                    
                    # 训练模型
                    best_model = model.fit(X_train, y_train)
                
                # 评估模型
                train_scores = self._evaluate_model(best_model, X_train, y_train)
                val_scores = self._evaluate_model(best_model, X_val, y_val) if X_val is not None else {}
                test_scores = self._evaluate_model(best_model, X_test, y_test) if X_test is not None else {}
                
                # 交叉验证
                cv_scores = self._cross_validate(best_model, X_train, y_train)
                
                # 获取特征重要性
                feature_importance = self._get_feature_importance(best_model)
                
                # 创建训练结果
                model_training_time = time.time() - model_start_time
                result = TrainingResult(
                    model_name=model_name,
                    task_type=self.task_type,
                    training_time=model_training_time,
                    best_model=best_model,
                    train_scores=train_scores,
                    val_scores=val_scores,
                    test_scores=test_scores,
                    cv_scores=cv_scores,
                    best_params=best_model.get_params() if hasattr(best_model, 'get_params') else {},
                    feature_importance=feature_importance,
                    config=self.config
                )
                
                # 生成预测
                result.train_predictions = best_model.predict(X_train)
                if X_val is not None:
                    result.val_predictions = best_model.predict(X_val)
                if X_test is not None:
                    result.test_predictions = best_model.predict(X_test)
                
                self.training_results_[model_name] = result
                
                # 更新最佳模型
                primary_metric = self.config.metrics[0] if self.config.metrics else 'accuracy'
                current_score = val_scores.get(primary_metric, train_scores.get(primary_metric, 0))
                
                if current_score > self.best_score_:
                    self.best_score_ = current_score
                    self.best_model_ = best_model
                
                if self.verbose:
                    logger.info(f"Completed {model_name} in {model_training_time:.2f}s, "
                              f"score: {current_score:.4f}")
                
                # 调用回调
                self._call_callbacks('model_trained', 
                                   model_name=model_name, 
                                   result=result)
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        total_time = time.time() - start_time
        self.is_fitted_ = True
        
        if self.verbose:
            logger.info(f"Training completed in {total_time:.2f}s. "
                      f"Best model: {self._get_best_model_name()}")
        
        # 调用回调
        self._call_callbacks('training_completed', 
                           total_time=total_time,
                           best_score=self.best_score_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用最佳模型进行预测
        
        Args:
            X: 预测特征
            
        Returns:
            预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Trainer must be fitted before prediction")
        
        if self.best_model_ is None:
            raise ValueError("No best model available")
        
        return self.best_model_.predict(X)
    
    def predict_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        使用所有模型进行预测
        
        Args:
            X: 预测特征
            
        Returns:
            所有模型的预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Trainer must be fitted before prediction")
        
        predictions = {}
        for model_name, result in self.training_results_.items():
            if result.best_model is not None:
                predictions[model_name] = result.best_model.predict(X)
        
        return predictions
    
    def _optimize_hyperparameters(self, 
                                model: BaseEstimator, 
                                X: np.ndarray, 
                                y: np.ndarray,
                                model_name: str) -> BaseEstimator:
        """优化超参数"""
        param_grid = self.param_grids[model_name]
        
        if self.config.optimization_method == 'grid_search':
            search = GridSearchCV(
                model,
                param_grid,
                cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=self.config.n_jobs,
                verbose=1 if self.verbose else 0,
                refit=self.config.refit
            )
        elif self.config.optimization_method == 'random_search':
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=self.config.optimization_trials,
                cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=self.config.n_jobs,
                verbose=1 if self.verbose else 0,
                random_state=self.config.random_state,
                refit=self.config.refit
            )
        else:
            # 使用贝叶斯优化（如果可用）
            try:
                from ..optimization import BayesianOptimizer
                optimizer = BayesianOptimizer()
                # 这里需要实现贝叶斯优化的接口
                # 暂时回退到网格搜索
                search = GridSearchCV(
                    model,
                    param_grid,
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring,
                    n_jobs=self.config.n_jobs,
                    verbose=1 if self.verbose else 0,
                    refit=self.config.refit
                )
            except ImportError:
                search = GridSearchCV(
                    model,
                    param_grid,
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring,
                    n_jobs=self.config.n_jobs,
                    verbose=1 if self.verbose else 0,
                    refit=self.config.refit
                )
        
        search.fit(X, y)
        
        if self.verbose:
            logger.info(f"Best parameters for {model_name}: {search.best_params_}")
            logger.info(f"Best score for {model_name}: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def _cross_validate(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """交叉验证"""
        cv_scores = {}
        
        for metric in self.config.metrics:
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=self.config.cv_folds,
                    scoring=metric,
                    n_jobs=self.config.n_jobs
                )
                cv_scores[metric] = scores.tolist()
            except Exception as e:
                logger.warning(f"Failed to compute CV score for {metric}: {e}")
                cv_scores[metric] = []
        
        return cv_scores
    
    def _get_feature_importance(self, model: BaseEstimator) -> Optional[np.ndarray]:
        """获取特征重要性"""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 对于线性模型，使用系数的绝对值
            coef = model.coef_
            if coef.ndim > 1:
                return np.mean(np.abs(coef), axis=0)
            else:
                return np.abs(coef)
        else:
            return None
    
    def _get_best_model_name(self) -> str:
        """获取最佳模型名称"""
        for name, result in self.training_results_.items():
            if result.best_model is self.best_model_:
                return name
        return "Unknown"
    
    def get_model_comparison(self) -> Dict[str, Dict[str, float]]:
        """获取模型比较结果"""
        comparison = {}
        
        for model_name, result in self.training_results_.items():
            comparison[model_name] = {}
            
            # 添加各种分数
            for metric in self.config.metrics:
                if metric in result.train_scores:
                    comparison[model_name][f'train_{metric}'] = result.train_scores[metric]
                if metric in result.val_scores:
                    comparison[model_name][f'val_{metric}'] = result.val_scores[metric]
                if metric in result.test_scores:
                    comparison[model_name][f'test_{metric}'] = result.test_scores[metric]
                
                # 添加交叉验证统计
                if metric in result.cv_scores:
                    cv_stats = result.get_cv_score_stats(metric)
                    for stat_name, stat_value in cv_stats.items():
                        comparison[model_name][f'cv_{metric}_{stat_name}'] = stat_value
            
            # 添加训练时间
            comparison[model_name]['training_time'] = result.training_time
        
        return comparison
    
    def add_model(self, name: str, model: BaseEstimator, param_grid: Optional[Dict] = None) -> None:
        """添加新模型"""
        self.models[name] = model
        if param_grid:
            self.param_grids[name] = param_grid
    
    def remove_model(self, name: str) -> None:
        """移除模型"""
        if name in self.models:
            del self.models[name]
        if name in self.param_grids:
            del self.param_grids[name]
        if name in self.training_results_:
            del self.training_results_[name]


class ClassificationTrainer(SupervisedTrainer):
    """分类任务训练器"""
    
    def __init__(self, **kwargs):
        kwargs['task_type'] = 'classification'
        super().__init__(**kwargs)


class RegressionTrainer(SupervisedTrainer):
    """回归任务训练器"""
    
    def __init__(self, **kwargs):
        kwargs['task_type'] = 'regression'
        super().__init__(**kwargs)