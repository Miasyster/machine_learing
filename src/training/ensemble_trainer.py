"""
集成学习训练器

结合多个模型进行集成训练
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
from sklearn.base import BaseEstimator

from .base import BaseTrainer, TrainingConfig, TrainingResult
from .supervised import SupervisedTrainer
from ..ensemble import (
    VotingEnsemble, WeightedVotingEnsemble,
    AveragingEnsemble, DynamicAveragingEnsemble, AdaptiveAveragingEnsemble,
    StackingEnsemble, MultiLevelStackingEnsemble, DynamicStackingEnsemble,
    BlendingEnsemble, DynamicBlendingEnsemble, AdaptiveBlendingEnsemble,
    DynamicModelSelectionEnsemble, OnlineEnsembleLearning, AdaptiveEnsembleStrategy
)

logger = logging.getLogger(__name__)


class EnsembleTrainer(BaseTrainer):
    """集成学习训练器"""
    
    # 预定义的集成方法
    ENSEMBLE_METHODS = {
        'voting': VotingEnsemble,
        'weighted_voting': WeightedVotingEnsemble,
        'averaging': AveragingEnsemble,
        'dynamic_averaging': DynamicAveragingEnsemble,
        'adaptive_averaging': AdaptiveAveragingEnsemble,
        'stacking': StackingEnsemble,
        'multilevel_stacking': MultiLevelStackingEnsemble,
        'dynamic_stacking': DynamicStackingEnsemble,
        'blending': BlendingEnsemble,
        'dynamic_blending': DynamicBlendingEnsemble,
        'adaptive_blending': AdaptiveBlendingEnsemble,
        'dynamic_selection': DynamicModelSelectionEnsemble,
        'online_learning': OnlineEnsembleLearning,
        'adaptive_strategy': AdaptiveEnsembleStrategy
    }
    
    def __init__(self, 
                 base_trainer: Optional[SupervisedTrainer] = None,
                 ensemble_methods: Optional[List[str]] = None,
                 ensemble_configs: Optional[Dict[str, Dict]] = None,
                 config: Optional[TrainingConfig] = None,
                 **kwargs):
        """
        初始化集成学习训练器
        
        Args:
            base_trainer: 基础训练器（用于训练基模型）
            ensemble_methods: 集成方法列表
            ensemble_configs: 集成方法配置
            config: 训练配置
            **kwargs: 其他参数
        """
        super().__init__(config, **kwargs)
        
        # 基础训练器
        self.base_trainer = base_trainer or SupervisedTrainer()
        
        # 集成方法
        self.ensemble_methods = ensemble_methods or ['voting', 'averaging', 'stacking']
        self.ensemble_configs = ensemble_configs or {}
        
        # 集成器实例
        self.ensemble_models = {}
        
        # 基模型
        self.base_models = {}
        
        if self.verbose:
            logger.info(f"Initialized EnsembleTrainer with methods: {self.ensemble_methods}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'EnsembleTrainer':
        """
        训练集成模型
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的训练器
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"Starting ensemble training with {len(self.ensemble_methods)} methods")
        
        # 首先训练基模型
        if self.verbose:
            logger.info("Training base models...")
        
        base_start_time = time.time()
        self.base_trainer.fit(X, y, **kwargs)
        base_training_time = time.time() - base_start_time
        
        # 获取基模型
        self.base_models = {}
        for model_name, result in self.base_trainer.get_training_results().items():
            if result.best_model is not None:
                self.base_models[model_name] = result.best_model
        
        if len(self.base_models) < 2:
            raise ValueError("Need at least 2 base models for ensemble learning")
        
        if self.verbose:
            logger.info(f"Trained {len(self.base_models)} base models in {base_training_time:.2f}s")
        
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
        
        # 训练每个集成方法
        for method_name in self.ensemble_methods:
            if self.verbose:
                logger.info(f"Training ensemble method: {method_name}")
            
            method_start_time = time.time()
            
            try:
                # 创建集成器
                ensemble_class = self.ENSEMBLE_METHODS[method_name]
                ensemble_config = self.ensemble_configs.get(method_name, {})
                
                # 设置随机种子
                if 'random_state' not in ensemble_config:
                    ensemble_config['random_state'] = self.config.random_state
                
                ensemble_model = ensemble_class(**ensemble_config)
                
                # 训练集成模型
                base_models_list = list(self.base_models.values())
                ensemble_model.fit(base_models_list, X_train, y_train)
                
                # 评估集成模型
                train_scores = self._evaluate_model(ensemble_model, X_train, y_train)
                val_scores = self._evaluate_model(ensemble_model, X_val, y_val) if X_val is not None else {}
                test_scores = self._evaluate_model(ensemble_model, X_test, y_test) if X_test is not None else {}
                
                # 交叉验证
                cv_scores = self._cross_validate_ensemble(ensemble_model, base_models_list, X_train, y_train)
                
                # 创建训练结果
                method_training_time = time.time() - method_start_time
                result = TrainingResult(
                    model_name=method_name,
                    task_type='ensemble',
                    training_time=method_training_time,
                    best_model=ensemble_model,
                    train_scores=train_scores,
                    val_scores=val_scores,
                    test_scores=test_scores,
                    cv_scores=cv_scores,
                    best_params=ensemble_model.get_params() if hasattr(ensemble_model, 'get_params') else {},
                    config=self.config
                )
                
                # 生成预测
                result.train_predictions = ensemble_model.predict(X_train)
                if X_val is not None:
                    result.val_predictions = ensemble_model.predict(X_val)
                if X_test is not None:
                    result.test_predictions = ensemble_model.predict(X_test)
                
                # 添加集成特定的元数据
                result.metadata['base_models'] = list(self.base_models.keys())
                result.metadata['ensemble_method'] = method_name
                result.metadata['base_training_time'] = base_training_time
                
                # 获取模型权重（如果可用）
                if hasattr(ensemble_model, 'weights_'):
                    result.metadata['model_weights'] = ensemble_model.weights_
                
                # 获取模型多样性信息
                if hasattr(ensemble_model, 'diversity_'):
                    result.metadata['model_diversity'] = ensemble_model.diversity_
                
                self.training_results_[method_name] = result
                self.ensemble_models[method_name] = ensemble_model
                
                # 更新最佳模型
                primary_metric = self.config.metrics[0] if self.config.metrics else 'accuracy'
                current_score = val_scores.get(primary_metric, train_scores.get(primary_metric, 0))
                
                if current_score > self.best_score_:
                    self.best_score_ = current_score
                    self.best_model_ = ensemble_model
                
                if self.verbose:
                    logger.info(f"Completed {method_name} in {method_training_time:.2f}s, "
                              f"score: {current_score:.4f}")
                
                # 调用回调
                self._call_callbacks('ensemble_trained', 
                                   method_name=method_name, 
                                   result=result)
                
            except Exception as e:
                logger.error(f"Failed to train ensemble method {method_name}: {e}")
                continue
        
        total_time = time.time() - start_time
        self.is_fitted_ = True
        
        if self.verbose:
            logger.info(f"Ensemble training completed in {total_time:.2f}s. "
                      f"Best method: {self._get_best_method_name()}")
        
        # 调用回调
        self._call_callbacks('training_completed', 
                           total_time=total_time,
                           best_score=self.best_score_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用最佳集成模型进行预测
        
        Args:
            X: 预测特征
            
        Returns:
            预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Trainer must be fitted before prediction")
        
        if self.best_model_ is None:
            raise ValueError("No best ensemble model available")
        
        return self.best_model_.predict(X)
    
    def predict_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        使用所有集成方法进行预测
        
        Args:
            X: 预测特征
            
        Returns:
            所有集成方法的预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Trainer must be fitted before prediction")
        
        predictions = {}
        for method_name, ensemble_model in self.ensemble_models.items():
            predictions[method_name] = ensemble_model.predict(X)
        
        return predictions
    
    def predict_base_models(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        使用基模型进行预测
        
        Args:
            X: 预测特征
            
        Returns:
            基模型的预测结果
        """
        if not self.base_models:
            raise ValueError("No base models available")
        
        predictions = {}
        for model_name, model in self.base_models.items():
            predictions[model_name] = model.predict(X)
        
        return predictions
    
    def _cross_validate_ensemble(self, 
                                ensemble_model, 
                                base_models: List[BaseEstimator], 
                                X: np.ndarray, 
                                y: np.ndarray) -> Dict[str, List[float]]:
        """集成模型交叉验证"""
        cv_scores = {}
        
        try:
            from sklearn.model_selection import cross_val_score
            
            # 创建一个包装器来处理集成模型的交叉验证
            class EnsembleWrapper:
                def __init__(self, ensemble_class, base_models, **kwargs):
                    self.ensemble_class = ensemble_class
                    self.base_models = base_models
                    self.kwargs = kwargs
                
                def fit(self, X, y):
                    self.ensemble_ = self.ensemble_class(**self.kwargs)
                    self.ensemble_.fit(self.base_models, X, y)
                    return self
                
                def predict(self, X):
                    return self.ensemble_.predict(X)
                
                def score(self, X, y):
                    predictions = self.predict(X)
                    from sklearn.metrics import accuracy_score
                    return accuracy_score(y, predictions)
            
            wrapper = EnsembleWrapper(
                ensemble_model.__class__, 
                base_models, 
                **ensemble_model.get_params()
            )
            
            for metric in self.config.metrics:
                try:
                    scores = cross_val_score(
                        wrapper, X, y,
                        cv=self.config.cv_folds,
                        scoring=metric,
                        n_jobs=self.config.n_jobs
                    )
                    cv_scores[metric] = scores.tolist()
                except Exception as e:
                    logger.warning(f"Failed to compute CV score for {metric}: {e}")
                    cv_scores[metric] = []
        
        except Exception as e:
            logger.warning(f"Failed to perform cross-validation: {e}")
        
        return cv_scores
    
    def _get_best_method_name(self) -> str:
        """获取最佳集成方法名称"""
        for name, result in self.training_results_.items():
            if result.best_model is self.best_model_:
                return name
        return "Unknown"
    
    def get_ensemble_comparison(self) -> Dict[str, Dict[str, float]]:
        """获取集成方法比较结果"""
        comparison = {}
        
        for method_name, result in self.training_results_.items():
            comparison[method_name] = {}
            
            # 添加各种分数
            for metric in self.config.metrics:
                if metric in result.train_scores:
                    comparison[method_name][f'train_{metric}'] = result.train_scores[metric]
                if metric in result.val_scores:
                    comparison[method_name][f'val_{metric}'] = result.val_scores[metric]
                if metric in result.test_scores:
                    comparison[method_name][f'test_{metric}'] = result.test_scores[metric]
                
                # 添加交叉验证统计
                if metric in result.cv_scores:
                    cv_stats = result.get_cv_score_stats(metric)
                    for stat_name, stat_value in cv_stats.items():
                        comparison[method_name][f'cv_{metric}_{stat_name}'] = stat_value
            
            # 添加训练时间
            comparison[method_name]['training_time'] = result.training_time
            comparison[method_name]['base_training_time'] = result.metadata.get('base_training_time', 0)
            
            # 添加集成特定信息
            comparison[method_name]['n_base_models'] = len(result.metadata.get('base_models', []))
            
            if 'model_diversity' in result.metadata:
                comparison[method_name]['model_diversity'] = result.metadata['model_diversity']
        
        return comparison
    
    def get_base_model_comparison(self) -> Dict[str, Dict[str, float]]:
        """获取基模型比较结果"""
        return self.base_trainer.get_model_comparison()
    
    def add_ensemble_method(self, name: str, method_class, config: Optional[Dict] = None) -> None:
        """添加新的集成方法"""
        self.ENSEMBLE_METHODS[name] = method_class
        if name not in self.ensemble_methods:
            self.ensemble_methods.append(name)
        if config:
            self.ensemble_configs[name] = config
    
    def remove_ensemble_method(self, name: str) -> None:
        """移除集成方法"""
        if name in self.ensemble_methods:
            self.ensemble_methods.remove(name)
        if name in self.ensemble_configs:
            del self.ensemble_configs[name]
        if name in self.ensemble_models:
            del self.ensemble_models[name]
        if name in self.training_results_:
            del self.training_results_[name]
    
    def get_model_weights(self, method_name: str) -> Optional[np.ndarray]:
        """获取指定集成方法的模型权重"""
        if method_name in self.training_results_:
            return self.training_results_[method_name].metadata.get('model_weights')
        return None
    
    def get_model_diversity(self, method_name: str) -> Optional[float]:
        """获取指定集成方法的模型多样性"""
        if method_name in self.training_results_:
            return self.training_results_[method_name].metadata.get('model_diversity')
        return None
    
    def update_ensemble_weights(self, method_name: str, new_weights: np.ndarray) -> None:
        """更新集成方法的权重"""
        if method_name in self.ensemble_models:
            ensemble_model = self.ensemble_models[method_name]
            if hasattr(ensemble_model, 'update_weights'):
                ensemble_model.update_weights(new_weights)
                # 更新结果中的权重信息
                if method_name in self.training_results_:
                    self.training_results_[method_name].metadata['model_weights'] = new_weights
    
    def get_ensemble_model(self, method_name: str):
        """获取指定的集成模型"""
        return self.ensemble_models.get(method_name)
    
    def get_base_models(self) -> Dict[str, BaseEstimator]:
        """获取基模型"""
        return self.base_models.copy()
    
    def __repr__(self) -> str:
        return f"EnsembleTrainer(methods={self.ensemble_methods}, base_trainer={self.base_trainer})"