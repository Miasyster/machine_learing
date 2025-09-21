"""
投票集成方法

实现硬投票和软投票集成策略
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from sklearn.base import BaseEstimator
from scipy import stats

from .base import BaseEnsemble, EnsembleResult

logger = logging.getLogger(__name__)


class VotingEnsemble(BaseEnsemble):
    """投票集成器"""
    
    def __init__(self,
                 models: List[BaseEstimator],
                 voting: str = 'hard',
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化投票集成器
        
        Args:
            models: 基础模型列表
            voting: 投票方式 ('hard', 'soft')
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(models, model_names, random_state, n_jobs, verbose)
        
        if voting not in ['hard', 'soft']:
            raise ValueError("voting must be 'hard' or 'soft'")
        
        self.voting = voting
        
        # 验证软投票的要求
        if voting == 'soft':
            for i, model in enumerate(models):
                if not hasattr(model, 'predict_proba'):
                    raise ValueError(f"Model {i} ({self.model_names[i]}) does not support probability prediction for soft voting")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'VotingEnsemble':
        """
        训练投票集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        if self.verbose:
            logger.info(f"Training voting ensemble with {self.n_models_} models using {self.voting} voting")
        
        # 训练各个模型
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            if self.verbose:
                logger.info(f"Training model {i+1}/{self.n_models_}: {name}")
            
            try:
                model.fit(X, y, **kwargs)
            except Exception as e:
                logger.error(f"Failed to train model {name}: {e}")
                raise
        
        self.is_fitted_ = True
        
        # 评估各个模型
        self.evaluate_individual_models(X, y)
        
        if self.verbose:
            logger.info("Voting ensemble training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """
        进行投票预测
        
        Args:
            X: 预测特征
            
        Returns:
            集成预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # 获取各模型预测
        individual_predictions = self.get_individual_predictions(X)
        individual_probabilities = None
        
        if self.voting == 'hard':
            predictions = self._hard_voting(individual_predictions)
            prediction_probabilities = None
        else:  # soft voting
            individual_probabilities = self.get_individual_probabilities(X)
            predictions, prediction_probabilities = self._soft_voting(individual_probabilities)
        
        return EnsembleResult(
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            individual_predictions=individual_predictions,
            individual_probabilities=individual_probabilities,
            model_scores=self.model_scores_
        )
    
    def _hard_voting(self, individual_predictions: List[np.ndarray]) -> np.ndarray:
        """
        硬投票
        
        Args:
            individual_predictions: 各模型预测结果
            
        Returns:
            投票结果
        """
        predictions_array = np.array(individual_predictions).T  # shape: (n_samples, n_models)
        
        if self._is_classifier():
            # 分类任务：使用众数
            final_predictions = []
            for i in range(predictions_array.shape[0]):
                sample_preds = predictions_array[i, :]
                mode_result = stats.mode(sample_preds, keepdims=True)
                final_predictions.append(mode_result.mode[0])
            
            return np.array(final_predictions)
        else:
            # 回归任务：使用平均值
            return np.mean(predictions_array, axis=1)
    
    def _soft_voting(self, individual_probabilities: List[np.ndarray]) -> tuple:
        """
        软投票
        
        Args:
            individual_probabilities: 各模型概率预测结果
            
        Returns:
            (预测结果, 预测概率)
        """
        # 过滤掉None值
        valid_probabilities = [prob for prob in individual_probabilities if prob is not None]
        
        if not valid_probabilities:
            raise ValueError("No models support probability prediction for soft voting")
        
        # 平均概率
        avg_probabilities = np.mean(valid_probabilities, axis=0)
        
        # 预测类别
        predictions = np.argmax(avg_probabilities, axis=1)
        
        return predictions, avg_probabilities


class WeightedVotingEnsemble(VotingEnsemble):
    """加权投票集成器"""
    
    def __init__(self,
                 models: List[BaseEstimator],
                 weights: Optional[np.ndarray] = None,
                 voting: str = 'hard',
                 weight_optimization: str = 'none',
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化加权投票集成器
        
        Args:
            models: 基础模型列表
            weights: 模型权重
            voting: 投票方式 ('hard', 'soft')
            weight_optimization: 权重优化方法 ('none', 'accuracy', 'cross_val', 'differential_evolution')
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(models, voting, model_names, random_state, n_jobs, verbose)
        
        self.weights = weights
        self.weight_optimization = weight_optimization
        self.optimized_weights_ = None
        
        # 验证权重
        if weights is not None:
            weights = np.array(weights)
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if np.any(weights < 0):
                raise ValueError("Weights must be non-negative")
            # 归一化权重
            self.weights = weights / np.sum(weights)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'WeightedVotingEnsemble':
        """
        训练加权投票集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        # 先训练基础模型
        super().fit(X, y, **kwargs)
        
        # 优化权重
        if self.weight_optimization != 'none':
            self.optimized_weights_ = self._optimize_weights(X, y)
            if self.verbose:
                logger.info(f"Optimized weights: {self.optimized_weights_}")
        else:
            self.optimized_weights_ = self.weights
        
        return self
    
    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        优化模型权重
        
        Args:
            X: 特征
            y: 标签
            
        Returns:
            优化后的权重
        """
        if self.weight_optimization == 'accuracy':
            return self._optimize_weights_by_accuracy(X, y)
        elif self.weight_optimization == 'cross_val':
            return self._optimize_weights_by_cross_validation(X, y)
        elif self.weight_optimization == 'differential_evolution':
            return self._optimize_weights_by_differential_evolution(X, y)
        else:
            raise ValueError(f"Unknown weight optimization method: {self.weight_optimization}")
    
    def _optimize_weights_by_accuracy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """基于准确率优化权重"""
        scores = []
        for model in self.models:
            score = model.score(X, y)
            scores.append(max(score, 0))  # 确保非负
        
        scores = np.array(scores)
        if np.sum(scores) == 0:
            return np.ones(len(self.models)) / len(self.models)
        
        return scores / np.sum(scores)
    
    def _optimize_weights_by_cross_validation(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> np.ndarray:
        """基于交叉验证优化权重"""
        from sklearn.model_selection import cross_val_score
        
        scores = []
        for model in self.models:
            cv_scores = cross_val_score(model, X, y, cv=cv, n_jobs=self.n_jobs)
            scores.append(max(np.mean(cv_scores), 0))
        
        scores = np.array(scores)
        if np.sum(scores) == 0:
            return np.ones(len(self.models)) / len(self.models)
        
        return scores / np.sum(scores)
    
    def _optimize_weights_by_differential_evolution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """使用差分进化算法优化权重"""
        try:
            from scipy.optimize import differential_evolution
        except ImportError:
            logger.warning("scipy not available, falling back to accuracy-based weights")
            return self._optimize_weights_by_accuracy(X, y)
        
        def objective(weights):
            # 归一化权重
            weights = weights / np.sum(weights)
            
            # 计算加权预测
            individual_predictions = self.get_individual_predictions(X)
            
            if self.voting == 'hard':
                if self._is_classifier():
                    # 分类任务的硬投票
                    weighted_predictions = self._weighted_hard_voting(individual_predictions, weights)
                else:
                    # 回归任务
                    predictions_array = np.array(individual_predictions).T
                    weighted_predictions = np.average(predictions_array, axis=1, weights=weights)
            else:
                # 软投票
                individual_probabilities = self.get_individual_probabilities(X)
                valid_probabilities = [prob for prob in individual_probabilities if prob is not None]
                valid_weights = weights[:len(valid_probabilities)]
                valid_weights = valid_weights / np.sum(valid_weights)
                
                avg_probabilities = np.average(valid_probabilities, axis=0, weights=valid_weights)
                weighted_predictions = np.argmax(avg_probabilities, axis=1)
            
            # 计算负准确率（因为要最小化）
            if self._is_classifier():
                from sklearn.metrics import accuracy_score
                return -accuracy_score(y, weighted_predictions)
            else:
                from sklearn.metrics import mean_squared_error
                return mean_squared_error(y, weighted_predictions)
        
        # 设置边界
        bounds = [(0.01, 1.0) for _ in range(len(self.models))]
        
        # 运行优化
        result = differential_evolution(
            objective,
            bounds,
            seed=self.random_state,
            maxiter=100,
            popsize=15
        )
        
        optimized_weights = result.x
        return optimized_weights / np.sum(optimized_weights)
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """
        进行加权投票预测
        
        Args:
            X: 预测特征
            
        Returns:
            集成预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # 获取各模型预测
        individual_predictions = self.get_individual_predictions(X)
        individual_probabilities = None
        
        # 使用的权重
        weights = self.optimized_weights_ if self.optimized_weights_ is not None else self.weights
        if weights is None:
            weights = np.ones(len(self.models)) / len(self.models)
        
        if self.voting == 'hard':
            predictions = self._weighted_hard_voting(individual_predictions, weights)
            prediction_probabilities = None
        else:  # soft voting
            individual_probabilities = self.get_individual_probabilities(X)
            predictions, prediction_probabilities = self._weighted_soft_voting(individual_probabilities, weights)
        
        return EnsembleResult(
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            individual_predictions=individual_predictions,
            individual_probabilities=individual_probabilities,
            weights=weights,
            model_scores=self.model_scores_
        )
    
    def _weighted_hard_voting(self, individual_predictions: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """
        加权硬投票
        
        Args:
            individual_predictions: 各模型预测结果
            weights: 模型权重
            
        Returns:
            投票结果
        """
        predictions_array = np.array(individual_predictions).T  # shape: (n_samples, n_models)
        
        if self._is_classifier():
            # 分类任务：加权投票
            final_predictions = []
            
            for i in range(predictions_array.shape[0]):
                sample_preds = predictions_array[i, :]
                unique_preds = np.unique(sample_preds)
                
                # 计算每个类别的加权票数
                vote_counts = {}
                for pred, weight in zip(sample_preds, weights):
                    if pred in vote_counts:
                        vote_counts[pred] += weight
                    else:
                        vote_counts[pred] = weight
                
                # 选择票数最多的类别
                best_pred = max(vote_counts, key=vote_counts.get)
                final_predictions.append(best_pred)
            
            return np.array(final_predictions)
        else:
            # 回归任务：加权平均
            return np.average(predictions_array, axis=1, weights=weights)
    
    def _weighted_soft_voting(self, individual_probabilities: List[np.ndarray], weights: np.ndarray) -> tuple:
        """
        加权软投票
        
        Args:
            individual_probabilities: 各模型概率预测结果
            weights: 模型权重
            
        Returns:
            (预测结果, 预测概率)
        """
        # 过滤掉None值
        valid_probabilities = []
        valid_weights = []
        
        for prob, weight in zip(individual_probabilities, weights):
            if prob is not None:
                valid_probabilities.append(prob)
                valid_weights.append(weight)
        
        if not valid_probabilities:
            raise ValueError("No models support probability prediction for soft voting")
        
        # 归一化权重
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / np.sum(valid_weights)
        
        # 加权平均概率
        avg_probabilities = np.average(valid_probabilities, axis=0, weights=valid_weights)
        
        # 预测类别
        predictions = np.argmax(avg_probabilities, axis=1)
        
        return predictions, avg_probabilities
    
    def get_weights(self) -> np.ndarray:
        """
        获取当前使用的权重
        
        Returns:
            权重数组
        """
        if self.optimized_weights_ is not None:
            return self.optimized_weights_
        elif self.weights is not None:
            return self.weights
        else:
            return np.ones(len(self.models)) / len(self.models)