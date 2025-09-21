"""
模型选择模块

提供自动模型选择、超参数优化和模型比较功能
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import time
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import get_scorer
import pandas as pd

from .base import BaseTrainer, TrainingConfig, TrainingResult
from .supervised import SupervisedTrainer
from .unsupervised import UnsupervisedTrainer
from .ensemble_trainer import EnsembleTrainer

logger = logging.getLogger(__name__)


class AutoMLTrainer(BaseTrainer):
    """自动机器学习训练器"""
    
    def __init__(self, 
                 task_type: str = 'classification',
                 time_budget: Optional[float] = None,
                 include_ensemble: bool = True,
                 config: Optional[TrainingConfig] = None,
                 **kwargs):
        """
        初始化AutoML训练器
        
        Args:
            task_type: 任务类型
            time_budget: 时间预算（秒）
            include_ensemble: 是否包含集成方法
            config: 训练配置
            **kwargs: 其他参数
        """
        super().__init__(config, **kwargs)
        
        self.task_type = task_type
        self.time_budget = time_budget
        self.include_ensemble = include_ensemble
        
        # 训练器
        self.trainers = {}
        self._setup_trainers()
        
        # 模型选择策略
        self.selection_strategy = kwargs.get('selection_strategy', 'best_cv_score')
        self.selection_metric = kwargs.get('selection_metric', 'accuracy')
        
        if self.verbose:
            logger.info(f"Initialized AutoMLTrainer for {task_type}")
    
    def _setup_trainers(self):
        """设置训练器"""
        if self.task_type in ['classification', 'regression']:
            self.trainers['supervised'] = SupervisedTrainer(
                task_type=self.task_type,
                config=self.config
            )
            
            if self.include_ensemble:
                self.trainers['ensemble'] = EnsembleTrainer(
                    base_trainer=self.trainers['supervised'],
                    config=self.config
                )
        
        elif self.task_type in ['clustering', 'dimensionality_reduction', 'anomaly_detection']:
            self.trainers['unsupervised'] = UnsupervisedTrainer(
                task_type=self.task_type,
                config=self.config
            )
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'AutoMLTrainer':
        """
        自动训练和选择最佳模型
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的训练器
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("Starting AutoML training...")
        
        # 训练所有训练器
        for trainer_name, trainer in self.trainers.items():
            if self.time_budget and (time.time() - start_time) > self.time_budget:
                logger.warning(f"Time budget exceeded, stopping at {trainer_name}")
                break
            
            if self.verbose:
                logger.info(f"Training with {trainer_name} trainer...")
            
            try:
                trainer.fit(X, y, **kwargs)
                
                # 合并结果
                for model_name, result in trainer.get_training_results().items():
                    full_name = f"{trainer_name}_{model_name}"
                    self.training_results_[full_name] = result
                
            except Exception as e:
                logger.error(f"Failed to train with {trainer_name}: {e}")
                continue
        
        # 选择最佳模型
        self._select_best_model()
        
        total_time = time.time() - start_time
        self.is_fitted_ = True
        
        if self.verbose:
            logger.info(f"AutoML training completed in {total_time:.2f}s. "
                      f"Best model: {self._get_best_model_name()}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted_:
            raise ValueError("Trainer must be fitted before prediction")
        
        if self.best_model_ is None:
            raise ValueError("No best model available")
        
        return self.best_model_.predict(X)
    
    def _select_best_model(self):
        """选择最佳模型"""
        if not self.training_results_:
            return
        
        best_score = -np.inf
        best_model = None
        
        for model_name, result in self.training_results_.items():
            if result.best_model is None:
                continue
            
            # 获取评估分数
            if self.selection_strategy == 'best_cv_score':
                cv_stats = result.get_cv_score_stats(self.selection_metric)
                score = cv_stats.get('mean', -np.inf)
            elif self.selection_strategy == 'best_val_score':
                score = result.val_scores.get(self.selection_metric, -np.inf)
            elif self.selection_strategy == 'best_test_score':
                score = result.test_scores.get(self.selection_metric, -np.inf)
            else:
                score = result.train_scores.get(self.selection_metric, -np.inf)
            
            if score > best_score:
                best_score = score
                best_model = result.best_model
        
        self.best_model_ = best_model
        self.best_score_ = best_score
    
    def _get_best_model_name(self) -> str:
        """获取最佳模型名称"""
        for name, result in self.training_results_.items():
            if result.best_model is self.best_model_:
                return name
        return "Unknown"
    
    def get_leaderboard(self) -> pd.DataFrame:
        """获取模型排行榜"""
        data = []
        
        for model_name, result in self.training_results_.items():
            row = {'model': model_name}
            
            # 添加分数
            for metric in self.config.metrics:
                if metric in result.train_scores:
                    row[f'train_{metric}'] = result.train_scores[metric]
                if metric in result.val_scores:
                    row[f'val_{metric}'] = result.val_scores[metric]
                if metric in result.test_scores:
                    row[f'test_{metric}'] = result.test_scores[metric]
                
                # 添加交叉验证分数
                cv_stats = result.get_cv_score_stats(metric)
                if cv_stats:
                    row[f'cv_{metric}_mean'] = cv_stats.get('mean', np.nan)
                    row[f'cv_{metric}_std'] = cv_stats.get('std', np.nan)
            
            # 添加训练时间
            row['training_time'] = result.training_time
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 按选择指标排序
        sort_column = f'cv_{self.selection_metric}_mean'
        if sort_column not in df.columns:
            sort_column = f'val_{self.selection_metric}'
            if sort_column not in df.columns:
                sort_column = f'train_{self.selection_metric}'
        
        if sort_column in df.columns:
            df = df.sort_values(sort_column, ascending=False)
        
        return df


class ModelSelector:
    """模型选择器"""
    
    def __init__(self, 
                 scoring: str = 'accuracy',
                 cv: int = 5,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        初始化模型选择器
        
        Args:
            scoring: 评分指标
            cv: 交叉验证折数
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.results_ = {}
        
        if self.verbose:
            logger.info(f"Initialized ModelSelector with scoring={scoring}, cv={cv}")
    
    def compare_models(self, 
                      models: Dict[str, BaseEstimator], 
                      X: np.ndarray, 
                      y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        比较多个模型
        
        Args:
            models: 模型字典
            X: 特征
            y: 标签
            
        Returns:
            比较结果
        """
        results = {}
        
        for model_name, model in models.items():
            if self.verbose:
                logger.info(f"Evaluating {model_name}...")
            
            try:
                # 交叉验证
                cv_scores = cross_val_score(
                    model, X, y,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs
                )
                
                results[model_name] = {
                    'mean_score': np.mean(cv_scores),
                    'std_score': np.std(cv_scores),
                    'min_score': np.min(cv_scores),
                    'max_score': np.max(cv_scores),
                    'scores': cv_scores.tolist()
                }
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                results[model_name] = {
                    'mean_score': np.nan,
                    'std_score': np.nan,
                    'min_score': np.nan,
                    'max_score': np.nan,
                    'scores': []
                }
        
        self.results_ = results
        return results
    
    def select_best_model(self, 
                         models: Dict[str, BaseEstimator], 
                         X: np.ndarray, 
                         y: np.ndarray) -> Tuple[str, BaseEstimator]:
        """
        选择最佳模型
        
        Args:
            models: 模型字典
            X: 特征
            y: 标签
            
        Returns:
            最佳模型名称和模型实例
        """
        results = self.compare_models(models, X, y)
        
        best_model_name = max(results.keys(), key=lambda k: results[k]['mean_score'])
        best_model = models[best_model_name]
        
        if self.verbose:
            logger.info(f"Best model: {best_model_name} "
                      f"(score: {results[best_model_name]['mean_score']:.4f})")
        
        return best_model_name, best_model
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """获取结果DataFrame"""
        if not self.results_:
            return pd.DataFrame()
        
        data = []
        for model_name, result in self.results_.items():
            row = {'model': model_name}
            row.update(result)
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('mean_score', ascending=False)
        return df


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, 
                 method: str = 'grid_search',
                 scoring: str = 'accuracy',
                 cv: int = 5,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        初始化超参数优化器
        
        Args:
            method: 优化方法 ('grid_search', 'random_search', 'bayesian')
            scoring: 评分指标
            cv: 交叉验证折数
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        self.method = method
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        
        if self.verbose:
            logger.info(f"Initialized HyperparameterOptimizer with method={method}")
    
    def optimize(self, 
                model: BaseEstimator, 
                param_grid: Dict[str, List], 
                X: np.ndarray, 
                y: np.ndarray,
                **kwargs) -> BaseEstimator:
        """
        优化超参数
        
        Args:
            model: 模型
            param_grid: 参数网格
            X: 特征
            y: 标签
            **kwargs: 其他参数
            
        Returns:
            优化后的模型
        """
        if self.method == 'grid_search':
            return self._grid_search(model, param_grid, X, y, **kwargs)
        elif self.method == 'random_search':
            return self._random_search(model, param_grid, X, y, **kwargs)
        elif self.method == 'bayesian':
            return self._bayesian_optimization(model, param_grid, X, y, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
    
    def _grid_search(self, model, param_grid, X, y, **kwargs):
        """网格搜索"""
        from sklearn.model_selection import GridSearchCV
        
        search = GridSearchCV(
            model,
            param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0,
            **kwargs
        )
        
        search.fit(X, y)
        
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        
        if self.verbose:
            logger.info(f"Best parameters: {self.best_params_}")
            logger.info(f"Best score: {self.best_score_:.4f}")
        
        return self.best_estimator_
    
    def _random_search(self, model, param_grid, X, y, **kwargs):
        """随机搜索"""
        from sklearn.model_selection import RandomizedSearchCV
        
        n_iter = kwargs.pop('n_iter', 100)
        
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0,
            **kwargs
        )
        
        search.fit(X, y)
        
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        
        if self.verbose:
            logger.info(f"Best parameters: {self.best_params_}")
            logger.info(f"Best score: {self.best_score_:.4f}")
        
        return self.best_estimator_
    
    def _bayesian_optimization(self, model, param_grid, X, y, **kwargs):
        """贝叶斯优化"""
        try:
            from ..optimization import BayesianOptimizer
            
            optimizer = BayesianOptimizer()
            # 这里需要实现贝叶斯优化的接口
            # 暂时回退到随机搜索
            return self._random_search(model, param_grid, X, y, **kwargs)
            
        except ImportError:
            logger.warning("Bayesian optimization not available, falling back to random search")
            return self._random_search(model, param_grid, X, y, **kwargs)


class ModelAnalyzer:
    """模型分析器"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化模型分析器
        
        Args:
            verbose: 是否显示详细信息
        """
        self.verbose = verbose
        
        if self.verbose:
            logger.info("Initialized ModelAnalyzer")
    
    def validation_curve_analysis(self, 
                                 model: BaseEstimator,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 param_name: str,
                                 param_range: List,
                                 cv: int = 5,
                                 scoring: str = 'accuracy') -> Dict[str, np.ndarray]:
        """
        验证曲线分析
        
        Args:
            model: 模型
            X: 特征
            y: 标签
            param_name: 参数名称
            param_range: 参数范围
            cv: 交叉验证折数
            scoring: 评分指标
            
        Returns:
            验证曲线结果
        """
        train_scores, val_scores = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        return {
            'param_range': np.array(param_range),
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }
    
    def learning_curve_analysis(self, 
                               model: BaseEstimator,
                               X: np.ndarray,
                               y: np.ndarray,
                               train_sizes: Optional[np.ndarray] = None,
                               cv: int = 5,
                               scoring: str = 'accuracy') -> Dict[str, np.ndarray]:
        """
        学习曲线分析
        
        Args:
            model: 模型
            X: 特征
            y: 标签
            train_sizes: 训练集大小
            cv: 交叉验证折数
            scoring: 评分指标
            
        Returns:
            学习曲线结果
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
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }
    
    def feature_importance_analysis(self, 
                                  model: BaseEstimator,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        特征重要性分析
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称
            
        Returns:
            特征重要性结果
        """
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                importance = np.mean(np.abs(coef), axis=0)
            else:
                importance = np.abs(coef)
        
        if importance is None:
            return {}
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # 排序
        indices = np.argsort(importance)[::-1]
        
        return {
            'importance': importance,
            'feature_names': feature_names,
            'sorted_indices': indices,
            'sorted_importance': importance[indices],
            'sorted_feature_names': [feature_names[i] for i in indices]
        }