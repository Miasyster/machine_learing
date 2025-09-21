"""
动态集成方法

实现动态模型选择、在线集成学习和自适应集成策略
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import deque
import time

from .base import BaseEnsemble, EnsembleResult

logger = logging.getLogger(__name__)


class DynamicModelSelectionEnsemble(BaseEnsemble):
    """动态模型选择集成器"""
    
    def __init__(self,
                 models: List[BaseEstimator],
                 selection_strategy: str = 'performance_based',
                 window_size: int = 100,
                 adaptation_threshold: float = 0.05,
                 min_samples_for_adaptation: int = 50,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化动态模型选择集成器
        
        Args:
            models: 候选模型列表
            selection_strategy: 选择策略 ('performance_based', 'diversity_based', 'confidence_based')
            window_size: 滑动窗口大小
            adaptation_threshold: 适应阈值
            min_samples_for_adaptation: 适应所需的最小样本数
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(models, model_names, random_state, n_jobs, verbose)
        
        self.selection_strategy = selection_strategy
        self.window_size = window_size
        self.adaptation_threshold = adaptation_threshold
        self.min_samples_for_adaptation = min_samples_for_adaptation
        
        # 动态选择相关
        self.current_model_index_ = 0
        self.performance_window_ = deque(maxlen=window_size)
        self.prediction_history_ = deque(maxlen=window_size)
        self.model_performances_ = {}
        self.adaptation_history_ = []
        self.sample_count_ = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'DynamicModelSelectionEnsemble':
        """
        训练动态模型选择集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        if self.verbose:
            logger.info(f"Training dynamic model selection ensemble with {self.n_models_} models")
        
        # 训练所有模型
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            if self.verbose:
                logger.info(f"Training model {i+1}/{self.n_models_}: {name}")
            
            try:
                model.fit(X, y, **kwargs)
            except Exception as e:
                logger.error(f"Failed to train model {name}: {e}")
                raise
        
        self.is_fitted_ = True
        
        # 评估各个模型并选择初始模型
        self.evaluate_individual_models(X, y)
        self._initialize_model_selection(X, y)
        
        if self.verbose:
            logger.info(f"Initial selected model: {self.model_names[self.current_model_index_]}")
        
        return self
    
    def _initialize_model_selection(self, X: np.ndarray, y: np.ndarray):
        """初始化模型选择"""
        # 基于训练集性能选择初始模型
        best_score = -np.inf
        best_index = 0
        
        for i, model in enumerate(self.models):
            score = model.score(X, y)
            if score > best_score:
                best_score = score
                best_index = i
        
        self.current_model_index_ = best_index
        
        # 初始化性能记录
        for i, name in enumerate(self.model_names):
            self.model_performances_[name] = deque(maxlen=self.window_size)
    
    def predict(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> EnsembleResult:
        """
        进行动态模型选择预测
        
        Args:
            X: 预测特征
            y_true: 真实标签（用于动态调整）
            
        Returns:
            集成预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # 使用当前选择的模型进行预测
        current_model = self.models[self.current_model_index_]
        predictions = current_model.predict(X)
        
        # 获取概率预测（如果支持）
        prediction_probabilities = None
        if hasattr(current_model, 'predict_proba'):
            try:
                prediction_probabilities = current_model.predict_proba(X)
            except:
                pass
        
        # 获取所有模型的预测（用于分析）
        individual_predictions = self.get_individual_predictions(X)
        individual_probabilities = self.get_individual_probabilities(X)
        
        # 如果提供了真实标签，更新性能并可能切换模型
        if y_true is not None:
            self._update_performance_and_adapt(X, y_true, predictions, individual_predictions)
        
        return EnsembleResult(
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            individual_predictions=individual_predictions,
            individual_probabilities=individual_probabilities,
            selected_model_index=self.current_model_index_,
            selected_model_name=self.model_names[self.current_model_index_],
            model_scores=self.model_scores_
        )
    
    def _update_performance_and_adapt(self, X: np.ndarray, y_true: np.ndarray, 
                                     current_predictions: np.ndarray, 
                                     individual_predictions: List[np.ndarray]):
        """更新性能并适应模型选择"""
        self.sample_count_ += len(y_true)
        
        # 计算当前模型的性能
        if self._is_classifier():
            current_performance = accuracy_score(y_true, current_predictions)
        else:
            current_performance = -mean_squared_error(y_true, current_predictions)
        
        # 更新性能窗口
        self.performance_window_.append(current_performance)
        current_model_name = self.model_names[self.current_model_index_]
        self.model_performances_[current_model_name].append(current_performance)
        
        # 记录预测历史
        self.prediction_history_.append({
            'predictions': current_predictions,
            'true_labels': y_true,
            'model_index': self.current_model_index_,
            'performance': current_performance
        })
        
        # 检查是否需要适应
        if (self.sample_count_ >= self.min_samples_for_adaptation and 
            len(self.performance_window_) >= self.window_size // 2):
            
            self._adapt_model_selection(X, y_true, individual_predictions)
    
    def _adapt_model_selection(self, X: np.ndarray, y_true: np.ndarray, 
                              individual_predictions: List[np.ndarray]):
        """适应模型选择"""
        if self.selection_strategy == 'performance_based':
            new_model_index = self._performance_based_selection(y_true, individual_predictions)
        elif self.selection_strategy == 'diversity_based':
            new_model_index = self._diversity_based_selection(individual_predictions)
        elif self.selection_strategy == 'confidence_based':
            new_model_index = self._confidence_based_selection(X, individual_predictions)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        # 检查是否需要切换模型
        if new_model_index != self.current_model_index_:
            # 计算性能改进
            current_avg_performance = np.mean(list(self.performance_window_))
            
            # 估算新模型的性能
            if self._is_classifier():
                new_performance = accuracy_score(y_true, individual_predictions[new_model_index])
            else:
                new_performance = -mean_squared_error(y_true, individual_predictions[new_model_index])
            
            # 如果改进超过阈值，则切换
            if new_performance - current_avg_performance > self.adaptation_threshold:
                old_model_name = self.model_names[self.current_model_index_]
                new_model_name = self.model_names[new_model_index]
                
                if self.verbose:
                    logger.info(f"Switching model from {old_model_name} to {new_model_name} "
                               f"(improvement: {new_performance - current_avg_performance:.4f})")
                
                self.current_model_index_ = new_model_index
                
                # 记录适应历史
                self.adaptation_history_.append({
                    'sample_count': self.sample_count_,
                    'old_model': old_model_name,
                    'new_model': new_model_name,
                    'old_performance': current_avg_performance,
                    'new_performance': new_performance,
                    'improvement': new_performance - current_avg_performance
                })
    
    def _performance_based_selection(self, y_true: np.ndarray, 
                                   individual_predictions: List[np.ndarray]) -> int:
        """基于性能的模型选择"""
        best_performance = -np.inf
        best_index = self.current_model_index_
        
        for i, pred in enumerate(individual_predictions):
            if self._is_classifier():
                performance = accuracy_score(y_true, pred)
            else:
                performance = -mean_squared_error(y_true, pred)
            
            if performance > best_performance:
                best_performance = performance
                best_index = i
        
        return best_index
    
    def _diversity_based_selection(self, individual_predictions: List[np.ndarray]) -> int:
        """基于多样性的模型选择"""
        # 计算与其他模型的多样性
        diversity_scores = []
        
        for i, pred_i in enumerate(individual_predictions):
            diversity = 0
            for j, pred_j in enumerate(individual_predictions):
                if i != j:
                    # 计算预测差异
                    if self._is_classifier():
                        diversity += np.mean(pred_i != pred_j)
                    else:
                        diversity += np.mean(np.abs(pred_i - pred_j))
            
            diversity_scores.append(diversity / (len(individual_predictions) - 1))
        
        # 选择多样性最高的模型
        return np.argmax(diversity_scores)
    
    def _confidence_based_selection(self, X: np.ndarray, 
                                   individual_predictions: List[np.ndarray]) -> int:
        """基于置信度的模型选择"""
        confidence_scores = []
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'predict_proba'):
                try:
                    probas = model.predict_proba(X)
                    # 使用最大概率作为置信度
                    confidence = np.mean(np.max(probas, axis=1))
                    confidence_scores.append(confidence)
                except:
                    confidence_scores.append(0.5)  # 默认置信度
            else:
                # 对于回归模型，使用预测的一致性作为置信度指标
                if hasattr(model, 'predict'):
                    # 这里简化为使用当前性能作为置信度
                    model_name = self.model_names[i]
                    if model_name in self.model_performances_ and self.model_performances_[model_name]:
                        confidence = np.mean(list(self.model_performances_[model_name]))
                        confidence_scores.append(max(confidence, 0))
                    else:
                        confidence_scores.append(0.5)
                else:
                    confidence_scores.append(0.5)
        
        # 选择置信度最高的模型
        return np.argmax(confidence_scores)
    
    def get_adaptation_history(self) -> List[Dict]:
        """获取适应历史"""
        return self.adaptation_history_
    
    def get_current_model_info(self) -> Dict:
        """获取当前模型信息"""
        return {
            'index': self.current_model_index_,
            'name': self.model_names[self.current_model_index_],
            'recent_performance': list(self.performance_window_)[-10:] if self.performance_window_ else []
        }


class OnlineEnsembleLearning(BaseEnsemble):
    """在线集成学习器"""
    
    def __init__(self,
                 models: List[BaseEstimator],
                 learning_rate: float = 0.1,
                 forgetting_factor: float = 0.95,
                 update_frequency: int = 10,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化在线集成学习器
        
        Args:
            models: 基础模型列表
            learning_rate: 学习率
            forgetting_factor: 遗忘因子
            update_frequency: 更新频率
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(models, model_names, random_state, n_jobs, verbose)
        
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor
        self.update_frequency = update_frequency
        
        # 在线学习相关
        self.weights_ = None
        self.sample_count_ = 0
        self.update_count_ = 0
        self.performance_history_ = []
        self.weight_history_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'OnlineEnsembleLearning':
        """
        训练在线集成学习器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        if self.verbose:
            logger.info(f"Training online ensemble learning with {self.n_models_} models")
        
        # 训练所有模型
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            if self.verbose:
                logger.info(f"Training model {i+1}/{self.n_models_}: {name}")
            
            try:
                model.fit(X, y, **kwargs)
            except Exception as e:
                logger.error(f"Failed to train model {name}: {e}")
                raise
        
        # 初始化权重
        self.weights_ = np.ones(self.n_models_) / self.n_models_
        
        self.is_fitted_ = True
        
        # 评估各个模型
        self.evaluate_individual_models(X, y)
        
        if self.verbose:
            logger.info("Online ensemble learning training completed")
        
        return self
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'OnlineEnsembleLearning':
        """
        在线更新模型
        
        Args:
            X: 新的训练特征
            y: 新的训练标签
            
        Returns:
            更新后的集成器
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before partial fitting")
        
        self.sample_count_ += len(y)
        
        # 获取当前预测
        individual_predictions = self.get_individual_predictions(X)
        
        # 计算各模型的损失
        losses = []
        for pred in individual_predictions:
            if self._is_classifier():
                loss = 1 - accuracy_score(y, pred)
            else:
                loss = mean_squared_error(y, pred)
            losses.append(loss)
        
        losses = np.array(losses)
        
        # 更新权重
        self._update_weights(losses)
        
        # 记录性能
        ensemble_pred = self._weighted_prediction(individual_predictions)
        if self._is_classifier():
            ensemble_performance = accuracy_score(y, ensemble_pred)
        else:
            ensemble_performance = -mean_squared_error(y, ensemble_pred)
        
        self.performance_history_.append({
            'sample_count': self.sample_count_,
            'individual_losses': losses.tolist(),
            'ensemble_performance': ensemble_performance,
            'weights': self.weights_.copy()
        })
        
        # 定期更新模型（如果支持）
        if self.sample_count_ % self.update_frequency == 0:
            self._update_models(X, y)
        
        return self
    
    def _update_weights(self, losses: np.ndarray):
        """更新模型权重"""
        # 使用指数加权平均更新权重
        # 损失越小，权重越大
        new_weights = np.exp(-self.learning_rate * losses)
        new_weights = new_weights / np.sum(new_weights)
        
        # 应用遗忘因子
        self.weights_ = self.forgetting_factor * self.weights_ + \
                       (1 - self.forgetting_factor) * new_weights
        
        # 归一化
        self.weights_ = self.weights_ / np.sum(self.weights_)
        
        self.weight_history_.append(self.weights_.copy())
    
    def _update_models(self, X: np.ndarray, y: np.ndarray):
        """更新支持在线学习的模型"""
        for i, model in enumerate(self.models):
            if hasattr(model, 'partial_fit'):
                try:
                    model.partial_fit(X, y)
                    if self.verbose:
                        logger.debug(f"Updated model {self.model_names[i]} with partial_fit")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Failed to update model {self.model_names[i]}: {e}")
        
        self.update_count_ += 1
    
    def _weighted_prediction(self, individual_predictions: List[np.ndarray]) -> np.ndarray:
        """加权预测"""
        predictions_array = np.array(individual_predictions)
        return np.average(predictions_array, axis=0, weights=self.weights_)
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """
        进行在线集成预测
        
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
        
        # 加权预测
        predictions = self._weighted_prediction(individual_predictions)
        
        # 加权概率预测
        prediction_probabilities = None
        if any(prob is not None for prob in individual_probabilities):
            valid_probabilities = [prob for prob in individual_probabilities if prob is not None]
            if valid_probabilities:
                valid_weights = self.weights_[:len(valid_probabilities)]
                valid_weights = valid_weights / np.sum(valid_weights)
                prediction_probabilities = np.average(valid_probabilities, axis=0, weights=valid_weights)
        
        return EnsembleResult(
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            individual_predictions=individual_predictions,
            individual_probabilities=individual_probabilities,
            weights=self.weights_.copy(),
            model_scores=self.model_scores_
        )
    
    def get_weight_history(self) -> List[np.ndarray]:
        """获取权重变化历史"""
        return self.weight_history_
    
    def get_performance_history(self) -> List[Dict]:
        """获取性能历史"""
        return self.performance_history_


class AdaptiveEnsembleStrategy(BaseEnsemble):
    """自适应集成策略"""
    
    def __init__(self,
                 models: List[BaseEstimator],
                 strategies: List[str] = None,
                 strategy_weights: Optional[np.ndarray] = None,
                 adaptation_method: str = 'performance_based',
                 evaluation_window: int = 100,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化自适应集成策略
        
        Args:
            models: 基础模型列表
            strategies: 集成策略列表 ['voting', 'averaging', 'stacking', 'blending']
            strategy_weights: 策略权重
            adaptation_method: 适应方法 ('performance_based', 'diversity_based', 'hybrid')
            evaluation_window: 评估窗口大小
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(models, model_names, random_state, n_jobs, verbose)
        
        if strategies is None:
            strategies = ['voting', 'averaging']
        
        self.strategies = strategies
        self.strategy_weights = strategy_weights
        self.adaptation_method = adaptation_method
        self.evaluation_window = evaluation_window
        
        # 自适应策略相关
        self.strategy_ensembles_ = {}
        self.strategy_performances_ = {strategy: deque(maxlen=evaluation_window) 
                                     for strategy in strategies}
        self.current_strategy_weights_ = None
        self.adaptation_history_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'AdaptiveEnsembleStrategy':
        """
        训练自适应集成策略
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        if self.verbose:
            logger.info(f"Training adaptive ensemble strategy with {len(self.strategies)} strategies")
        
        # 训练各种策略的集成器
        for strategy in self.strategies:
            if self.verbose:
                logger.info(f"Training {strategy} ensemble")
            
            if strategy == 'voting':
                from .voting import VotingEnsemble
                ensemble = VotingEnsemble(
                    models=self.models,
                    voting='soft' if self._is_classifier() else 'hard',
                    model_names=self.model_names,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=False
                )
            elif strategy == 'averaging':
                from .averaging import AveragingEnsemble
                ensemble = AveragingEnsemble(
                    models=self.models,
                    averaging_method='weighted',
                    model_names=self.model_names,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=False
                )
            elif strategy == 'stacking':
                from .stacking import StackingEnsemble
                ensemble = StackingEnsemble(
                    base_models=self.models,
                    model_names=self.model_names,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=False
                )
            elif strategy == 'blending':
                from .blending import BlendingEnsemble
                ensemble = BlendingEnsemble(
                    base_models=self.models,
                    model_names=self.model_names,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=False
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            ensemble.fit(X, y, **kwargs)
            self.strategy_ensembles_[strategy] = ensemble
        
        # 初始化策略权重
        if self.strategy_weights is None:
            self.current_strategy_weights_ = np.ones(len(self.strategies)) / len(self.strategies)
        else:
            self.current_strategy_weights_ = np.array(self.strategy_weights)
        
        self.is_fitted_ = True
        
        # 评估各个模型
        self.evaluate_individual_models(X, y)
        
        if self.verbose:
            logger.info("Adaptive ensemble strategy training completed")
        
        return self
    
    def predict(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> EnsembleResult:
        """
        进行自适应集成预测
        
        Args:
            X: 预测特征
            y_true: 真实标签（用于策略适应）
            
        Returns:
            集成预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # 获取各策略的预测
        strategy_predictions = {}
        strategy_probabilities = {}
        
        for strategy, ensemble in self.strategy_ensembles_.items():
            result = ensemble.predict(X)
            strategy_predictions[strategy] = result.predictions
            strategy_probabilities[strategy] = result.prediction_probabilities
        
        # 加权组合策略预测
        final_predictions, final_probabilities = self._combine_strategy_predictions(
            strategy_predictions, strategy_probabilities
        )
        
        # 如果提供了真实标签，更新策略权重
        if y_true is not None:
            self._update_strategy_weights(strategy_predictions, y_true)
        
        # 获取个体模型预测（使用第一个策略的结果）
        first_strategy = list(self.strategy_ensembles_.keys())[0]
        first_result = self.strategy_ensembles_[first_strategy].predict(X)
        
        return EnsembleResult(
            predictions=final_predictions,
            prediction_probabilities=final_probabilities,
            individual_predictions=first_result.individual_predictions,
            individual_probabilities=first_result.individual_probabilities,
            strategy_predictions=strategy_predictions,
            strategy_weights=self.current_strategy_weights_.copy(),
            model_scores=self.model_scores_
        )
    
    def _combine_strategy_predictions(self, strategy_predictions: Dict[str, np.ndarray],
                                    strategy_probabilities: Dict[str, Optional[np.ndarray]]) -> tuple:
        """组合策略预测"""
        # 加权平均预测
        predictions_list = []
        weights_list = []
        
        for i, strategy in enumerate(self.strategies):
            if strategy in strategy_predictions:
                predictions_list.append(strategy_predictions[strategy])
                weights_list.append(self.current_strategy_weights_[i])
        
        if predictions_list:
            predictions_array = np.array(predictions_list)
            weights_array = np.array(weights_list)
            weights_array = weights_array / np.sum(weights_array)  # 归一化
            
            final_predictions = np.average(predictions_array, axis=0, weights=weights_array)
        else:
            final_predictions = np.zeros(len(X))
        
        # 加权平均概率
        final_probabilities = None
        prob_list = []
        prob_weights = []
        
        for i, strategy in enumerate(self.strategies):
            if (strategy in strategy_probabilities and 
                strategy_probabilities[strategy] is not None):
                prob_list.append(strategy_probabilities[strategy])
                prob_weights.append(self.current_strategy_weights_[i])
        
        if prob_list:
            prob_array = np.array(prob_list)
            prob_weights_array = np.array(prob_weights)
            prob_weights_array = prob_weights_array / np.sum(prob_weights_array)
            
            final_probabilities = np.average(prob_array, axis=0, weights=prob_weights_array)
        
        return final_predictions, final_probabilities
    
    def _update_strategy_weights(self, strategy_predictions: Dict[str, np.ndarray], 
                                y_true: np.ndarray):
        """更新策略权重"""
        # 计算各策略的性能
        strategy_performances = {}
        
        for strategy, predictions in strategy_predictions.items():
            if self._is_classifier():
                performance = accuracy_score(y_true, predictions)
            else:
                performance = -mean_squared_error(y_true, predictions)
            
            strategy_performances[strategy] = performance
            self.strategy_performances_[strategy].append(performance)
        
        # 根据适应方法更新权重
        if self.adaptation_method == 'performance_based':
            new_weights = self._performance_based_weights(strategy_performances)
        elif self.adaptation_method == 'diversity_based':
            new_weights = self._diversity_based_weights(strategy_predictions)
        elif self.adaptation_method == 'hybrid':
            new_weights = self._hybrid_weights(strategy_performances, strategy_predictions)
        else:
            raise ValueError(f"Unknown adaptation method: {self.adaptation_method}")
        
        # 记录权重变化
        old_weights = self.current_strategy_weights_.copy()
        self.current_strategy_weights_ = new_weights
        
        # 记录适应历史
        self.adaptation_history_.append({
            'timestamp': time.time(),
            'old_weights': old_weights,
            'new_weights': new_weights,
            'strategy_performances': strategy_performances
        })
    
    def _performance_based_weights(self, strategy_performances: Dict[str, float]) -> np.ndarray:
        """基于性能的权重更新"""
        performances = []
        for strategy in self.strategies:
            if strategy in strategy_performances:
                performances.append(max(strategy_performances[strategy], 0))
            else:
                performances.append(0)
        
        performances = np.array(performances)
        if np.sum(performances) > 0:
            return performances / np.sum(performances)
        else:
            return np.ones(len(self.strategies)) / len(self.strategies)
    
    def _diversity_based_weights(self, strategy_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """基于多样性的权重更新"""
        # 计算策略间的多样性
        diversity_scores = []
        predictions_list = [strategy_predictions[strategy] for strategy in self.strategies 
                           if strategy in strategy_predictions]
        
        for i, strategy in enumerate(self.strategies):
            if strategy not in strategy_predictions:
                diversity_scores.append(0)
                continue
            
            diversity = 0
            pred_i = strategy_predictions[strategy]
            
            for j, pred_j in enumerate(predictions_list):
                if i != j:
                    if self._is_classifier():
                        diversity += np.mean(pred_i != pred_j)
                    else:
                        diversity += np.mean(np.abs(pred_i - pred_j))
            
            if len(predictions_list) > 1:
                diversity /= (len(predictions_list) - 1)
            
            diversity_scores.append(diversity)
        
        diversity_scores = np.array(diversity_scores)
        if np.sum(diversity_scores) > 0:
            return diversity_scores / np.sum(diversity_scores)
        else:
            return np.ones(len(self.strategies)) / len(self.strategies)
    
    def _hybrid_weights(self, strategy_performances: Dict[str, float], 
                       strategy_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """混合权重更新"""
        performance_weights = self._performance_based_weights(strategy_performances)
        diversity_weights = self._diversity_based_weights(strategy_predictions)
        
        # 动态调整组合比例
        if len(self.adaptation_history_) > 5:
            recent_performances = [h['strategy_performances'] for h in self.adaptation_history_[-5:]]
            # 如果性能变化较大，更注重性能；否则更注重多样性
            performance_variance = np.var([list(p.values()) for p in recent_performances])
            if performance_variance > 0.01:
                alpha = 0.7  # 更注重性能
            else:
                alpha = 0.3  # 更注重多样性
        else:
            alpha = 0.5
        
        combined_weights = alpha * performance_weights + (1 - alpha) * diversity_weights
        return combined_weights / np.sum(combined_weights)
    
    def get_strategy_performances(self) -> Dict[str, List[float]]:
        """获取策略性能历史"""
        return {strategy: list(performances) 
                for strategy, performances in self.strategy_performances_.items()}
    
    def get_adaptation_history(self) -> List[Dict]:
        """获取适应历史"""
        return self.adaptation_history_
    
    def get_current_strategy_weights(self) -> Dict[str, float]:
        """获取当前策略权重"""
        return dict(zip(self.strategies, self.current_strategy_weights_))