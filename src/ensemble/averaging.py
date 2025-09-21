"""
平均集成方法

实现简单平均、加权平均和动态平均集成策略
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, accuracy_score

from .base import BaseEnsemble, EnsembleResult

logger = logging.getLogger(__name__)


class AveragingEnsemble(BaseEnsemble):
    """平均集成器"""
    
    def __init__(self,
                 models: List[BaseEstimator],
                 averaging_method: str = 'simple',
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化平均集成器
        
        Args:
            models: 基础模型列表
            averaging_method: 平均方法 ('simple', 'weighted', 'geometric', 'harmonic')
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(models, model_names, random_state, n_jobs, verbose)
        
        if averaging_method not in ['simple', 'weighted', 'geometric', 'harmonic']:
            raise ValueError("averaging_method must be one of: 'simple', 'weighted', 'geometric', 'harmonic'")
        
        self.averaging_method = averaging_method
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'AveragingEnsemble':
        """
        训练平均集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        if self.verbose:
            logger.info(f"Training averaging ensemble with {self.n_models_} models using {self.averaging_method} averaging")
        
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
            logger.info("Averaging ensemble training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """
        进行平均预测
        
        Args:
            X: 预测特征
            
        Returns:
            集成预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # 获取各模型预测
        individual_predictions = self.get_individual_predictions(X)
        individual_probabilities = self.get_individual_probabilities(X)
        
        # 进行平均
        if self.averaging_method == 'simple':
            predictions, prediction_probabilities = self._simple_averaging(
                individual_predictions, individual_probabilities
            )
        elif self.averaging_method == 'weighted':
            predictions, prediction_probabilities = self._weighted_averaging(
                individual_predictions, individual_probabilities
            )
        elif self.averaging_method == 'geometric':
            predictions, prediction_probabilities = self._geometric_averaging(
                individual_predictions, individual_probabilities
            )
        elif self.averaging_method == 'harmonic':
            predictions, prediction_probabilities = self._harmonic_averaging(
                individual_predictions, individual_probabilities
            )
        
        return EnsembleResult(
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            individual_predictions=individual_predictions,
            individual_probabilities=individual_probabilities,
            model_scores=self.model_scores_
        )
    
    def _simple_averaging(self, individual_predictions: List[np.ndarray], 
                         individual_probabilities: List[np.ndarray]) -> tuple:
        """简单平均"""
        predictions_array = np.array(individual_predictions)
        predictions = np.mean(predictions_array, axis=0)
        
        prediction_probabilities = None
        if any(prob is not None for prob in individual_probabilities):
            valid_probabilities = [prob for prob in individual_probabilities if prob is not None]
            if valid_probabilities:
                prediction_probabilities = np.mean(valid_probabilities, axis=0)
        
        return predictions, prediction_probabilities
    
    def _weighted_averaging(self, individual_predictions: List[np.ndarray], 
                           individual_probabilities: List[np.ndarray]) -> tuple:
        """加权平均（基于模型性能）"""
        # 使用模型得分作为权重
        weights = np.array([score for score in self.model_scores_.values()])
        weights = weights / np.sum(weights)  # 归一化
        
        predictions_array = np.array(individual_predictions)
        predictions = np.average(predictions_array, axis=0, weights=weights)
        
        prediction_probabilities = None
        if any(prob is not None for prob in individual_probabilities):
            valid_probabilities = [prob for prob in individual_probabilities if prob is not None]
            if valid_probabilities:
                # 对应的权重
                valid_weights = weights[:len(valid_probabilities)]
                valid_weights = valid_weights / np.sum(valid_weights)
                prediction_probabilities = np.average(valid_probabilities, axis=0, weights=valid_weights)
        
        return predictions, prediction_probabilities
    
    def _geometric_averaging(self, individual_predictions: List[np.ndarray], 
                            individual_probabilities: List[np.ndarray]) -> tuple:
        """几何平均"""
        predictions_array = np.array(individual_predictions)
        
        # 确保所有预测都是正数（对于几何平均）
        if np.any(predictions_array <= 0):
            logger.warning("Geometric averaging requires positive values, adding small constant")
            predictions_array = np.abs(predictions_array) + 1e-8
        
        predictions = np.exp(np.mean(np.log(predictions_array), axis=0))
        
        prediction_probabilities = None
        if any(prob is not None for prob in individual_probabilities):
            valid_probabilities = [prob for prob in individual_probabilities if prob is not None]
            if valid_probabilities:
                prob_array = np.array(valid_probabilities)
                prob_array = np.maximum(prob_array, 1e-8)  # 避免log(0)
                prediction_probabilities = np.exp(np.mean(np.log(prob_array), axis=0))
                # 重新归一化
                prediction_probabilities = prediction_probabilities / np.sum(prediction_probabilities, axis=1, keepdims=True)
        
        return predictions, prediction_probabilities
    
    def _harmonic_averaging(self, individual_predictions: List[np.ndarray], 
                           individual_probabilities: List[np.ndarray]) -> tuple:
        """调和平均"""
        predictions_array = np.array(individual_predictions)
        
        # 确保所有预测都是正数（对于调和平均）
        if np.any(predictions_array <= 0):
            logger.warning("Harmonic averaging requires positive values, adding small constant")
            predictions_array = np.abs(predictions_array) + 1e-8
        
        predictions = len(individual_predictions) / np.sum(1.0 / predictions_array, axis=0)
        
        prediction_probabilities = None
        if any(prob is not None for prob in individual_probabilities):
            valid_probabilities = [prob for prob in individual_probabilities if prob is not None]
            if valid_probabilities:
                prob_array = np.array(valid_probabilities)
                prob_array = np.maximum(prob_array, 1e-8)  # 避免除零
                prediction_probabilities = len(valid_probabilities) / np.sum(1.0 / prob_array, axis=0)
                # 重新归一化
                prediction_probabilities = prediction_probabilities / np.sum(prediction_probabilities, axis=1, keepdims=True)
        
        return predictions, prediction_probabilities


class DynamicAveragingEnsemble(AveragingEnsemble):
    """动态平均集成器"""
    
    def __init__(self,
                 models: List[BaseEstimator],
                 weight_function: Optional[Callable] = None,
                 adaptation_rate: float = 0.1,
                 window_size: int = 100,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化动态平均集成器
        
        Args:
            models: 基础模型列表
            weight_function: 权重计算函数
            adaptation_rate: 适应率
            window_size: 滑动窗口大小
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(models, 'weighted', model_names, random_state, n_jobs, verbose)
        
        self.weight_function = weight_function or self._default_weight_function
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        
        # 动态权重相关
        self.dynamic_weights_ = None
        self.prediction_history_ = []
        self.error_history_ = []
        self.current_weights_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'DynamicAveragingEnsemble':
        """
        训练动态平均集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        # 先训练基础模型
        super().fit(X, y, **kwargs)
        
        # 初始化动态权重
        self.dynamic_weights_ = np.ones(self.n_models_) / self.n_models_
        self.current_weights_ = self.dynamic_weights_.copy()
        
        # 在训练集上初始化权重
        self._initialize_dynamic_weights(X, y)
        
        return self
    
    def _initialize_dynamic_weights(self, X: np.ndarray, y: np.ndarray):
        """在训练集上初始化动态权重"""
        individual_predictions = self.get_individual_predictions(X)
        
        # 计算各模型的初始误差
        initial_errors = []
        for pred in individual_predictions:
            if self._is_classifier():
                error = 1 - accuracy_score(y, pred)
            else:
                error = mean_squared_error(y, pred)
            initial_errors.append(error)
        
        # 基于误差计算初始权重
        initial_errors = np.array(initial_errors)
        if np.sum(initial_errors) > 0:
            # 误差越小，权重越大
            weights = 1 / (initial_errors + 1e-8)
            self.dynamic_weights_ = weights / np.sum(weights)
        
        self.current_weights_ = self.dynamic_weights_.copy()
    
    def predict(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> EnsembleResult:
        """
        进行动态平均预测
        
        Args:
            X: 预测特征
            y_true: 真实标签（用于动态调整权重）
            
        Returns:
            集成预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # 获取各模型预测
        individual_predictions = self.get_individual_predictions(X)
        individual_probabilities = self.get_individual_probabilities(X)
        
        # 使用当前权重进行加权平均
        predictions, prediction_probabilities = self._weighted_averaging_with_weights(
            individual_predictions, individual_probabilities, self.current_weights_
        )
        
        # 如果提供了真实标签，更新权重
        if y_true is not None:
            self._update_dynamic_weights(individual_predictions, predictions, y_true)
        
        return EnsembleResult(
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            individual_predictions=individual_predictions,
            individual_probabilities=individual_probabilities,
            weights=self.current_weights_.copy(),
            model_scores=self.model_scores_
        )
    
    def _weighted_averaging_with_weights(self, individual_predictions: List[np.ndarray], 
                                        individual_probabilities: List[np.ndarray],
                                        weights: np.ndarray) -> tuple:
        """使用指定权重进行加权平均"""
        predictions_array = np.array(individual_predictions)
        predictions = np.average(predictions_array, axis=0, weights=weights)
        
        prediction_probabilities = None
        if any(prob is not None for prob in individual_probabilities):
            valid_probabilities = [prob for prob in individual_probabilities if prob is not None]
            if valid_probabilities:
                valid_weights = weights[:len(valid_probabilities)]
                valid_weights = valid_weights / np.sum(valid_weights)
                prediction_probabilities = np.average(valid_probabilities, axis=0, weights=valid_weights)
        
        return predictions, prediction_probabilities
    
    def _update_dynamic_weights(self, individual_predictions: List[np.ndarray], 
                               ensemble_predictions: np.ndarray, y_true: np.ndarray):
        """更新动态权重"""
        # 计算各模型的当前误差
        current_errors = []
        for pred in individual_predictions:
            if self._is_classifier():
                error = 1 - accuracy_score(y_true, pred)
            else:
                error = mean_squared_error(y_true, pred)
            current_errors.append(error)
        
        # 计算集成模型的误差
        if self._is_classifier():
            ensemble_error = 1 - accuracy_score(y_true, ensemble_predictions)
        else:
            ensemble_error = mean_squared_error(y_true, ensemble_predictions)
        
        # 更新误差历史
        self.error_history_.append({
            'individual_errors': current_errors,
            'ensemble_error': ensemble_error
        })
        
        # 保持窗口大小
        if len(self.error_history_) > self.window_size:
            self.error_history_.pop(0)
        
        # 使用权重函数计算新权重
        new_weights = self.weight_function(current_errors, self.current_weights_)
        
        # 应用适应率
        self.current_weights_ = (1 - self.adaptation_rate) * self.current_weights_ + \
                               self.adaptation_rate * new_weights
        
        # 归一化权重
        self.current_weights_ = self.current_weights_ / np.sum(self.current_weights_)
    
    def _default_weight_function(self, errors: List[float], current_weights: np.ndarray) -> np.ndarray:
        """默认权重函数"""
        errors = np.array(errors)
        
        # 误差越小，权重越大
        if np.sum(errors) > 0:
            weights = 1 / (errors + 1e-8)
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(errors)) / len(errors)
        
        return weights
    
    def get_weight_history(self) -> List[np.ndarray]:
        """获取权重变化历史"""
        return getattr(self, 'weight_history_', [])
    
    def get_error_history(self) -> List[Dict]:
        """获取误差历史"""
        return self.error_history_


class AdaptiveAveragingEnsemble(DynamicAveragingEnsemble):
    """自适应平均集成器"""
    
    def __init__(self,
                 models: List[BaseEstimator],
                 adaptation_strategy: str = 'performance_based',
                 performance_metric: str = 'auto',
                 adaptation_threshold: float = 0.05,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化自适应平均集成器
        
        Args:
            models: 基础模型列表
            adaptation_strategy: 适应策略 ('performance_based', 'diversity_based', 'hybrid')
            performance_metric: 性能指标 ('auto', 'accuracy', 'mse', 'mae')
            adaptation_threshold: 适应阈值
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(models, None, 0.1, 100, model_names, random_state, n_jobs, verbose)
        
        self.adaptation_strategy = adaptation_strategy
        self.performance_metric = performance_metric
        self.adaptation_threshold = adaptation_threshold
        
        # 设置权重函数
        if adaptation_strategy == 'performance_based':
            self.weight_function = self._performance_based_weights
        elif adaptation_strategy == 'diversity_based':
            self.weight_function = self._diversity_based_weights
        elif adaptation_strategy == 'hybrid':
            self.weight_function = self._hybrid_weights
        else:
            raise ValueError(f"Unknown adaptation strategy: {adaptation_strategy}")
    
    def _performance_based_weights(self, errors: List[float], current_weights: np.ndarray) -> np.ndarray:
        """基于性能的权重调整"""
        errors = np.array(errors)
        
        # 计算性能得分（误差的倒数）
        performance_scores = 1 / (errors + 1e-8)
        
        # 归一化
        weights = performance_scores / np.sum(performance_scores)
        
        return weights
    
    def _diversity_based_weights(self, errors: List[float], current_weights: np.ndarray) -> np.ndarray:
        """基于多样性的权重调整"""
        # 计算模型间的多样性
        if len(self.prediction_history_) < 2:
            return current_weights
        
        # 获取最近的预测
        recent_predictions = self.prediction_history_[-self.window_size:]
        
        # 计算多样性矩阵
        diversity_scores = self._calculate_diversity_scores(recent_predictions)
        
        # 结合性能和多样性
        errors = np.array(errors)
        performance_scores = 1 / (errors + 1e-8)
        
        # 加权组合
        combined_scores = 0.7 * performance_scores + 0.3 * diversity_scores
        weights = combined_scores / np.sum(combined_scores)
        
        return weights
    
    def _hybrid_weights(self, errors: List[float], current_weights: np.ndarray) -> np.ndarray:
        """混合策略权重调整"""
        # 结合性能和多样性
        performance_weights = self._performance_based_weights(errors, current_weights)
        diversity_weights = self._diversity_based_weights(errors, current_weights)
        
        # 动态调整组合比例
        if len(self.error_history_) > 10:
            recent_errors = [h['ensemble_error'] for h in self.error_history_[-10:]]
            error_trend = np.mean(np.diff(recent_errors))
            
            if error_trend > 0:  # 误差增加，更注重多样性
                alpha = 0.3
            else:  # 误差减少，更注重性能
                alpha = 0.7
        else:
            alpha = 0.5
        
        weights = alpha * performance_weights + (1 - alpha) * diversity_weights
        return weights / np.sum(weights)
    
    def _calculate_diversity_scores(self, prediction_history: List[List[np.ndarray]]) -> np.ndarray:
        """计算多样性得分"""
        if not prediction_history:
            return np.ones(self.n_models_) / self.n_models_
        
        # 计算预测的方差作为多样性指标
        diversity_scores = []
        
        for i in range(self.n_models_):
            model_predictions = []
            for predictions in prediction_history:
                if i < len(predictions):
                    model_predictions.append(predictions[i])
            
            if model_predictions:
                # 计算该模型预测的方差
                pred_array = np.concatenate(model_predictions)
                diversity = np.var(pred_array)
                diversity_scores.append(diversity)
            else:
                diversity_scores.append(0)
        
        diversity_scores = np.array(diversity_scores)
        if np.sum(diversity_scores) > 0:
            return diversity_scores / np.sum(diversity_scores)
        else:
            return np.ones(self.n_models_) / self.n_models_