"""
Blending混合集成方法

实现标准Blending、动态Blending和自适应Blending策略
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

from .base import BaseEnsemble, EnsembleResult

logger = logging.getLogger(__name__)


class BlendingEnsemble(BaseEnsemble):
    """Blending混合集成器"""
    
    def __init__(self,
                 base_models: List[BaseEstimator],
                 meta_model: Optional[BaseEstimator] = None,
                 holdout_size: float = 0.2,
                 use_probas: bool = True,
                 use_features_in_secondary: bool = False,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化Blending集成器
        
        Args:
            base_models: 基础模型列表
            meta_model: 元学习器
            holdout_size: 保留集大小比例
            use_probas: 是否使用概率预测作为元特征
            use_features_in_secondary: 是否在元学习器中使用原始特征
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(base_models, model_names, random_state, n_jobs, verbose)
        
        self.base_models = base_models
        self.meta_model = meta_model
        self.holdout_size = holdout_size
        self.use_probas = use_probas
        self.use_features_in_secondary = use_features_in_secondary
        
        # 训练后的模型
        self.fitted_base_models_ = None
        self.fitted_meta_model_ = None
        self.meta_features_shape_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BlendingEnsemble':
        """
        训练Blending集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
            
        Raises:
            ValueError: 当输入数据无效时
            RuntimeError: 当训练过程中出现错误时
        """
        # 输入验证
        X, y = self._validate_input(X, y)
        
        if len(self.base_models) == 0:
            raise ValueError("At least one base model must be provided")
        
        if not 0 < self.holdout_size < 1:
            raise ValueError("holdout_size must be between 0 and 1")
        
        if self.verbose:
            logger.info(f"Training blending ensemble with {self.n_models_} base models")
            logger.info(f"Input shape: {X.shape}, target shape: {y.shape}")
        
        try:
            # 设置默认元学习器
            if self.meta_model is None:
                if self._is_classifier():
                    self.meta_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
                else:
                    self.meta_model = LinearRegression()
            
            # 分割数据：训练集和保留集
            try:
                if self._is_classifier():
                    X_train, X_holdout, y_train, y_holdout = train_test_split(
                        X, y, test_size=self.holdout_size, 
                        random_state=self.random_state, stratify=y
                    )
                else:
                    X_train, X_holdout, y_train, y_holdout = train_test_split(
                        X, y, test_size=self.holdout_size, 
                        random_state=self.random_state
                    )
            except Exception as e:
                logger.error(f"Failed to split data: {str(e)}")
                raise RuntimeError(f"Data splitting failed: {str(e)}") from e
            
            if self.verbose:
                logger.info(f"Split data: train={X_train.shape[0]}, holdout={X_holdout.shape[0]}")
            
            # 验证分割后的数据大小
            if X_train.shape[0] == 0 or X_holdout.shape[0] == 0:
                raise RuntimeError("Data split resulted in empty train or holdout set")
            
            # 训练基础模型
            if self.verbose:
                logger.info("Training base models on training set")
            
            self.fitted_base_models_ = []
            failed_models = []
            
            for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
                try:
                    if self.verbose:
                        logger.info(f"Training base model {i+1}/{self.n_models_}: {name}")
                    
                    fitted_model = clone(model)
                    fitted_model.fit(X_train, y_train, **kwargs)
                    self.fitted_base_models_.append(fitted_model)
                    
                except Exception as e:
                    logger.error(f"Failed to train base model {name}: {str(e)}")
                    failed_models.append((i, name, str(e)))
                    # 添加占位符以保持索引一致性
                    self.fitted_base_models_.append(None)
            
            # 检查是否有足够的成功训练的模型
            successful_models = sum(1 for model in self.fitted_base_models_ if model is not None)
            if successful_models == 0:
                raise RuntimeError("All base models failed to train")
            elif failed_models:
                logger.warning(f"{len(failed_models)} base models failed to train: {[name for _, name, _ in failed_models]}")
            
            # 在保留集上生成元特征
            if self.verbose:
                logger.info("Generating meta-features on holdout set")
            
            meta_features = self._generate_meta_features_on_holdout(X_holdout)
            
            # 验证元特征
            if meta_features.shape[0] != X_holdout.shape[0]:
                raise RuntimeError(f"Meta-features shape mismatch: expected {X_holdout.shape[0]} samples, got {meta_features.shape[0]}")
            
            # 如果使用原始特征
            if self.use_features_in_secondary:
                if self.verbose:
                    logger.info("Combining meta-features with original features")
                meta_features = np.hstack([meta_features, X_holdout])
            
            # 训练元学习器
            if self.verbose:
                logger.info("Training meta-learner on holdout set")
            
            self.fitted_meta_model_ = clone(self.meta_model)
            self.fitted_meta_model_.fit(meta_features, y_holdout)
            
            self.meta_features_shape_ = meta_features.shape[1]
            self.is_fitted_ = True
            
            # 评估各个模型（在整个训练集上）
            try:
                self.evaluate_individual_models(X, y)
            except Exception as e:
                logger.warning(f"Model evaluation failed: {str(e)}")
                # 继续执行，不让评估失败影响训练
            
            if self.verbose:
                logger.info("Blending ensemble training completed successfully")
                if failed_models:
                    logger.info(f"Successfully trained {successful_models}/{self.n_models_} base models")
            
            return self
            
        except Exception as e:
            logger.error(f"Blending ensemble training failed: {str(e)}")
            raise RuntimeError(f"Failed to train blending ensemble: {str(e)}") from e
    
    def _generate_meta_features_on_holdout(self, X_holdout: np.ndarray) -> np.ndarray:
        """
        在保留集上生成元特征
        
        Args:
            X_holdout: 保留集特征
            
        Returns:
            元特征矩阵
            
        Raises:
            RuntimeError: 当元特征生成失败时
            ValueError: 当输入数据无效时
        """
        if X_holdout is None or X_holdout.size == 0:
            raise ValueError("Holdout data cannot be None or empty")
        
        meta_features_list = []
        failed_models = []
        
        for i, (model, name) in enumerate(zip(self.fitted_base_models_, self.model_names)):
            if self.verbose:
                logger.info(f"Generating meta-features for model {i+1}/{self.n_models_}: {name}")
            
            try:
                if self.use_probas and hasattr(model, 'predict_proba') and self._is_classifier():
                    # 使用概率预测
                    meta_pred = model.predict_proba(X_holdout)
                else:
                    # 使用普通预测
                    meta_pred = model.predict(X_holdout)
                    if meta_pred.ndim == 1:
                        meta_pred = meta_pred.reshape(-1, 1)
                
                # 验证预测结果
                if meta_pred is None or meta_pred.size == 0:
                    raise ValueError(f"Model {name} returned empty predictions")
                
                if meta_pred.shape[0] != X_holdout.shape[0]:
                    raise ValueError(f"Model {name} prediction shape mismatch: expected {X_holdout.shape[0]} samples, got {meta_pred.shape[0]}")
                
                # 检查无效值
                if np.any(np.isnan(meta_pred)) or np.any(np.isinf(meta_pred)):
                    logger.warning(f"Model {name} produced invalid predictions, filling with zeros")
                    meta_pred = np.nan_to_num(meta_pred, nan=0.0, posinf=0.0, neginf=0.0)
                
                meta_features_list.append(meta_pred)
                
            except Exception as e:
                logger.error(f"Failed to generate meta-features for model {name}: {str(e)}")
                failed_models.append((i, name, str(e)))
                
                # 创建默认预测（全零）
                if self.use_probas and hasattr(model, 'predict_proba') and self._is_classifier():
                    n_classes = getattr(model, 'classes_', [0, 1])
                    default_pred = np.zeros((X_holdout.shape[0], len(n_classes)))
                    if len(n_classes) > 1:
                        default_pred[:, 0] = 1.0  # 默认预测第一个类别
                else:
                    default_pred = np.zeros((X_holdout.shape[0], 1))
                
                meta_features_list.append(default_pred)
        
        if not meta_features_list:
            raise RuntimeError("No valid meta-features could be generated from any base model")
        
        # 合并元特征
        try:
            meta_features = np.hstack(meta_features_list)
        except Exception as e:
            raise RuntimeError(f"Failed to combine meta-features: {str(e)}") from e
        
        if self.verbose:
            logger.info(f"Generated meta-features shape: {meta_features.shape}")
            if failed_models:
                logger.warning(f"Failed to generate meta-features for {len(failed_models)} models: {[name for _, name, _ in failed_models]}")
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """
        进行Blending预测
        
        Args:
            X: 预测特征
            
        Returns:
            集成预测结果
            
        Raises:
            ValueError: 当模型未训练或输入数据无效时
            RuntimeError: 当预测过程失败时
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        if X is None or X.size == 0:
            raise ValueError("Input data cannot be None or empty")
        
        try:
            # 获取基础模型预测作为元特征
            meta_features = self._get_meta_features_for_prediction(X)
            
            # 验证元特征形状
            if meta_features.shape[0] != X.shape[0]:
                raise RuntimeError(f"Meta-features shape mismatch: expected {X.shape[0]} samples, got {meta_features.shape[0]}")
            
            # 如果使用原始特征
            if self.use_features_in_secondary:
                meta_features = np.hstack([meta_features, X])
            
            # 元学习器预测
            if self.verbose:
                logger.info("Making meta-model predictions")
            
            predictions = self.fitted_meta_model_.predict(meta_features)
            
            # 获取概率预测（如果支持）
            prediction_probabilities = None
            if hasattr(self.fitted_meta_model_, 'predict_proba'):
                try:
                    prediction_probabilities = self.fitted_meta_model_.predict_proba(meta_features)
                except Exception as e:
                    logger.warning(f"Failed to get prediction probabilities: {str(e)}")
            
            # 获取各基础模型的预测
            individual_predictions = self.get_individual_predictions(X)
            individual_probabilities = self.get_individual_probabilities(X)
            
        except Exception as e:
            logger.error(f"Blending prediction failed: {str(e)}")
            raise RuntimeError(f"Failed to make blending predictions: {str(e)}") from e
        
        return EnsembleResult(
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            individual_predictions=individual_predictions,
            individual_probabilities=individual_probabilities,
            model_scores=self.model_scores_
        )
    
    def _get_meta_features_for_prediction(self, X: np.ndarray) -> np.ndarray:
        """
        为预测生成元特征
        
        Args:
            X: 输入特征
            
        Returns:
            元特征矩阵
            
        Raises:
            RuntimeError: 当元特征生成失败时
        """
        if not hasattr(self, 'fitted_base_models_') or not self.fitted_base_models_:
            raise RuntimeError("No fitted base models available for prediction")
        
        meta_features_list = []
        failed_models = []
        
        for i, (model, name) in enumerate(zip(self.fitted_base_models_, self.model_names)):
            try:
                if self.use_probas and hasattr(model, 'predict_proba') and self._is_classifier():
                    meta_pred = model.predict_proba(X)
                else:
                    meta_pred = model.predict(X)
                    if meta_pred.ndim == 1:
                        meta_pred = meta_pred.reshape(-1, 1)
                
                # 验证预测结果
                if meta_pred is None or meta_pred.size == 0:
                    raise ValueError(f"Model {name} returned empty predictions")
                
                if meta_pred.shape[0] != X.shape[0]:
                    raise ValueError(f"Model {name} prediction shape mismatch")
                
                # 处理无效值
                if np.any(np.isnan(meta_pred)) or np.any(np.isinf(meta_pred)):
                    logger.warning(f"Model {name} produced invalid predictions during prediction")
                    meta_pred = np.nan_to_num(meta_pred, nan=0.0, posinf=0.0, neginf=0.0)
                
                meta_features_list.append(meta_pred)
                
            except Exception as e:
                logger.error(f"Failed to get predictions from model {name}: {str(e)}")
                failed_models.append((i, name, str(e)))
                
                # 创建默认预测
                if self.use_probas and hasattr(model, 'predict_proba') and self._is_classifier():
                    n_classes = getattr(model, 'classes_', [0, 1])
                    default_pred = np.zeros((X.shape[0], len(n_classes)))
                    if len(n_classes) > 1:
                        default_pred[:, 0] = 1.0
                else:
                    default_pred = np.zeros((X.shape[0], 1))
                
                meta_features_list.append(default_pred)
        
        if not meta_features_list:
            raise RuntimeError("No valid predictions could be obtained from any base model")
        
        try:
            meta_features = np.hstack(meta_features_list)
        except Exception as e:
            raise RuntimeError(f"Failed to combine meta-features for prediction: {str(e)}") from e
        
        if failed_models and self.verbose:
            logger.warning(f"Failed predictions from {len(failed_models)} models: {[name for _, name, _ in failed_models]}")
        
        return meta_features
    
    def get_individual_predictions(self, X: np.ndarray) -> List[np.ndarray]:
        """
        获取各个基础模型的预测结果
        
        Args:
            X: 输入特征
            
        Returns:
            各模型的预测结果列表
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for i, (model, name) in enumerate(zip(self.fitted_base_models_, self.model_names)):
            try:
                pred = model.predict(X)
                if pred is None or pred.size == 0:
                    logger.warning(f"Model {name} returned empty predictions, using zeros")
                    pred = np.zeros(X.shape[0])
                
                # 处理无效值
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    logger.warning(f"Model {name} produced invalid predictions, filling with zeros")
                    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to get predictions from model {name}: {str(e)}")
                # 使用默认预测
                predictions.append(np.zeros(X.shape[0]))
        
        return predictions
    
    def get_individual_probabilities(self, X: np.ndarray) -> List[Optional[np.ndarray]]:
        """
        获取各个基础模型的概率预测
        
        Args:
            X: 输入特征
            
        Returns:
            各模型的概率预测结果列表
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        probabilities = []
        for i, (model, name) in enumerate(zip(self.fitted_base_models_, self.model_names)):
            if hasattr(model, 'predict_proba'):
                try:
                    prob = model.predict_proba(X)
                    if prob is None or prob.size == 0:
                        logger.warning(f"Model {name} returned empty probabilities")
                        probabilities.append(None)
                    else:
                        # 处理无效值
                        if np.any(np.isnan(prob)) or np.any(np.isinf(prob)):
                            logger.warning(f"Model {name} produced invalid probabilities, normalizing")
                            prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
                            # 重新归一化概率
                            prob_sum = prob.sum(axis=1, keepdims=True)
                            prob_sum[prob_sum == 0] = 1.0  # 避免除零
                            prob = prob / prob_sum
                        
                        probabilities.append(prob)
                except Exception as e:
                    logger.error(f"Failed to get probabilities from model {name}: {str(e)}")
                    probabilities.append(None)
            else:
                probabilities.append(None)
        
        return probabilities


class DynamicBlendingEnsemble(BlendingEnsemble):
    """动态Blending集成器"""
    
    def __init__(self,
                 base_models: List[BaseEstimator],
                 meta_model_pool: List[BaseEstimator],
                 selection_strategy: str = 'performance',
                 adaptation_frequency: int = 100,
                 holdout_size: float = 0.2,
                 use_probas: bool = True,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化动态Blending集成器
        
        Args:
            base_models: 基础模型列表
            meta_model_pool: 元学习器池
            selection_strategy: 选择策略 ('performance', 'diversity', 'adaptive')
            adaptation_frequency: 适应频率
            holdout_size: 保留集大小比例
            use_probas: 是否使用概率预测作为元特征
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        # 初始化时使用第一个元学习器
        super().__init__(
            base_models, meta_model_pool[0], holdout_size, use_probas, False,
            model_names, random_state, n_jobs, verbose
        )
        
        self.meta_model_pool = meta_model_pool
        self.selection_strategy = selection_strategy
        self.adaptation_frequency = adaptation_frequency
        
        # 动态选择相关
        self.current_meta_model_index_ = 0
        self.meta_model_performances_ = []
        self.prediction_count_ = 0
        self.adaptation_history_ = []
        self.holdout_meta_features_ = None
        self.holdout_targets_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'DynamicBlendingEnsemble':
        """
        训练动态Blending集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        if self.verbose:
            logger.info("Training dynamic blending ensemble")
        
        # 分割数据
        if self._is_classifier():
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y, test_size=self.holdout_size, 
                random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y, test_size=self.holdout_size, 
                random_state=self.random_state
            )
        
        # 训练基础模型
        self.fitted_base_models_ = []
        for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
            if self.verbose:
                logger.info(f"Training base model {i+1}/{self.n_models_}: {name}")
            
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train, **kwargs)
            self.fitted_base_models_.append(fitted_model)
        
        # 生成保留集上的元特征
        self.holdout_meta_features_ = self._generate_meta_features_on_holdout(X_holdout)
        self.holdout_targets_ = y_holdout
        
        # 评估所有元学习器
        self._evaluate_meta_models()
        
        # 选择最佳元学习器
        self.current_meta_model_index_ = self._select_best_meta_model()
        self.meta_model = self.meta_model_pool[self.current_meta_model_index_]
        
        # 训练选定的元学习器
        self.fitted_meta_model_ = clone(self.meta_model)
        self.fitted_meta_model_.fit(self.holdout_meta_features_, self.holdout_targets_)
        
        self.meta_features_shape_ = self.holdout_meta_features_.shape[1]
        self.is_fitted_ = True
        
        # 评估各个模型
        self.evaluate_individual_models(X, y)
        
        if self.verbose:
            logger.info(f"Selected meta-model: {type(self.meta_model).__name__}")
        
        return self
    
    def _evaluate_meta_models(self):
        """评估所有元学习器"""
        from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
        
        # 设置交叉验证
        if self._is_classifier():
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        self.meta_model_performances_ = []
        
        for i, meta_model in enumerate(self.meta_model_pool):
            if self.verbose:
                logger.info(f"Evaluating meta-model {i+1}/{len(self.meta_model_pool)}: {type(meta_model).__name__}")
            
            # 交叉验证评估
            scores = cross_val_score(
                meta_model, self.holdout_meta_features_, self.holdout_targets_, 
                cv=cv, n_jobs=self.n_jobs
            )
            
            performance = {
                'model_index': i,
                'model_name': type(meta_model).__name__,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores
            }
            
            self.meta_model_performances_.append(performance)
    
    def _select_best_meta_model(self) -> int:
        """选择最佳元学习器"""
        if self.selection_strategy == 'performance':
            # 选择性能最好的
            best_idx = max(range(len(self.meta_model_performances_)), 
                          key=lambda i: self.meta_model_performances_[i]['mean_score'])
        elif self.selection_strategy == 'diversity':
            # 选择多样性最好的（这里简化为选择标准差最大的）
            best_idx = max(range(len(self.meta_model_performances_)), 
                          key=lambda i: self.meta_model_performances_[i]['std_score'])
        elif self.selection_strategy == 'adaptive':
            # 自适应选择（结合性能和稳定性）
            scores = []
            for perf in self.meta_model_performances_:
                # 结合平均性能和稳定性
                score = perf['mean_score'] - 0.1 * perf['std_score']
                scores.append(score)
            best_idx = np.argmax(scores)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        return best_idx
    
    def predict(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> EnsembleResult:
        """
        进行动态Blending预测
        
        Args:
            X: 预测特征
            y_true: 真实标签（用于动态调整）
            
        Returns:
            集成预测结果
        """
        # 检查是否需要重新选择元学习器
        if (self.prediction_count_ > 0 and 
            self.prediction_count_ % self.adaptation_frequency == 0 and 
            y_true is not None):
            self._adapt_meta_model(X, y_true)
        
        # 进行预测
        result = super().predict(X)
        
        self.prediction_count_ += len(X)
        
        return result
    
    def _adapt_meta_model(self, X: np.ndarray, y_true: np.ndarray):
        """动态调整元学习器"""
        if self.verbose:
            logger.info("Adapting meta-model selection")
        
        # 获取当前元特征
        meta_features = self._get_meta_features_for_prediction(X)
        
        # 评估当前元学习器的性能
        current_predictions = self.fitted_meta_model_.predict(meta_features)
        
        if self._is_classifier():
            current_performance = accuracy_score(y_true, current_predictions)
        else:
            current_performance = -mean_squared_error(y_true, current_predictions)
        
        # 测试其他元学习器
        best_performance = current_performance
        best_model_index = self.current_meta_model_index_
        
        for i, meta_model in enumerate(self.meta_model_pool):
            if i == self.current_meta_model_index_:
                continue
            
            # 快速训练和评估
            temp_model = clone(meta_model)
            temp_model.fit(meta_features, y_true)
            temp_predictions = temp_model.predict(meta_features)
            
            if self._is_classifier():
                temp_performance = accuracy_score(y_true, temp_predictions)
            else:
                temp_performance = -mean_squared_error(y_true, temp_predictions)
            
            if temp_performance > best_performance:
                best_performance = temp_performance
                best_model_index = i
        
        # 如果找到更好的模型，切换
        if best_model_index != self.current_meta_model_index_:
            if self.verbose:
                logger.info(f"Switching meta-model from {type(self.fitted_meta_model_).__name__} "
                           f"to {type(self.meta_model_pool[best_model_index]).__name__}")
            
            self.current_meta_model_index_ = best_model_index
            self.fitted_meta_model_ = clone(self.meta_model_pool[best_model_index])
            self.fitted_meta_model_.fit(meta_features, y_true)
            
            # 记录适应历史
            self.adaptation_history_.append({
                'prediction_count': self.prediction_count_,
                'old_model': type(self.meta_model).__name__,
                'new_model': type(self.meta_model_pool[best_model_index]).__name__,
                'performance_improvement': best_performance - current_performance
            })


class AdaptiveBlendingEnsemble(BlendingEnsemble):
    """自适应Blending集成器"""
    
    def __init__(self,
                 base_models: List[BaseEstimator],
                 meta_model: Optional[BaseEstimator] = None,
                 weight_function: Optional[Callable] = None,
                 adaptation_rate: float = 0.1,
                 holdout_size: float = 0.2,
                 use_probas: bool = True,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化自适应Blending集成器
        
        Args:
            base_models: 基础模型列表
            meta_model: 元学习器
            weight_function: 权重函数
            adaptation_rate: 适应率
            holdout_size: 保留集大小比例
            use_probas: 是否使用概率预测作为元特征
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(
            base_models, meta_model, holdout_size, use_probas, False,
            model_names, random_state, n_jobs, verbose
        )
        
        self.weight_function = weight_function or self._default_weight_function
        self.adaptation_rate = adaptation_rate
        
        # 自适应权重相关
        self.adaptive_weights_ = None
        self.performance_history_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'AdaptiveBlendingEnsemble':
        """
        训练自适应Blending集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        # 先训练基础Blending
        super().fit(X, y, **kwargs)
        
        # 初始化自适应权重
        self.adaptive_weights_ = np.ones(self.n_models_) / self.n_models_
        
        # 在保留集上初始化权重
        self._initialize_adaptive_weights()
        
        return self
    
    def _initialize_adaptive_weights(self):
        """初始化自适应权重"""
        if self.holdout_meta_features_ is None or self.holdout_targets_ is None:
            return
        
        # 计算各基础模型在保留集上的性能
        individual_performances = []
        
        # 从元特征中提取各模型的预测
        start_idx = 0
        for i, model in enumerate(self.fitted_base_models_):
            if self.use_probas and hasattr(model, 'predict_proba') and self._is_classifier():
                n_classes = len(np.unique(self.holdout_targets_))
                end_idx = start_idx + n_classes
                model_probs = self.holdout_meta_features_[:, start_idx:end_idx]
                model_preds = np.argmax(model_probs, axis=1)
            else:
                end_idx = start_idx + 1
                model_preds = self.holdout_meta_features_[:, start_idx:end_idx].ravel()
            
            # 计算性能
            if self._is_classifier():
                performance = accuracy_score(self.holdout_targets_, model_preds)
            else:
                performance = -mean_squared_error(self.holdout_targets_, model_preds)
            
            individual_performances.append(max(performance, 0))
            start_idx = end_idx
        
        # 基于性能计算初始权重
        individual_performances = np.array(individual_performances)
        if np.sum(individual_performances) > 0:
            self.adaptive_weights_ = individual_performances / np.sum(individual_performances)
    
    def predict(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> EnsembleResult:
        """
        进行自适应Blending预测
        
        Args:
            X: 预测特征
            y_true: 真实标签（用于权重调整）
            
        Returns:
            集成预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # 获取基础模型预测
        individual_predictions = self.get_individual_predictions(X)
        individual_probabilities = self.get_individual_probabilities(X)
        
        # 使用自适应权重进行加权预测
        predictions, prediction_probabilities = self._adaptive_weighted_prediction(
            individual_predictions, individual_probabilities
        )
        
        # 如果提供了真实标签，更新权重
        if y_true is not None:
            self._update_adaptive_weights(individual_predictions, predictions, y_true)
        
        return EnsembleResult(
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            individual_predictions=individual_predictions,
            individual_probabilities=individual_probabilities,
            weights=self.adaptive_weights_.copy(),
            model_scores=self.model_scores_
        )
    
    def _adaptive_weighted_prediction(self, individual_predictions: List[np.ndarray], 
                                     individual_probabilities: List[np.ndarray]) -> tuple:
        """自适应加权预测"""
        # 加权预测
        predictions_array = np.array(individual_predictions)
        predictions = np.average(predictions_array, axis=0, weights=self.adaptive_weights_)
        
        prediction_probabilities = None
        if any(prob is not None for prob in individual_probabilities):
            valid_probabilities = [prob for prob in individual_probabilities if prob is not None]
            if valid_probabilities:
                valid_weights = self.adaptive_weights_[:len(valid_probabilities)]
                valid_weights = valid_weights / np.sum(valid_weights)
                prediction_probabilities = np.average(valid_probabilities, axis=0, weights=valid_weights)
        
        return predictions, prediction_probabilities
    
    def _update_adaptive_weights(self, individual_predictions: List[np.ndarray], 
                                ensemble_predictions: np.ndarray, y_true: np.ndarray):
        """更新自适应权重"""
        # 计算各模型的当前性能
        current_performances = []
        for pred in individual_predictions:
            if self._is_classifier():
                performance = accuracy_score(y_true, pred)
            else:
                performance = -mean_squared_error(y_true, pred)
            current_performances.append(max(performance, 0))
        
        # 计算集成模型的性能
        if self._is_classifier():
            ensemble_performance = accuracy_score(y_true, ensemble_predictions)
        else:
            ensemble_performance = -mean_squared_error(y_true, ensemble_predictions)
        
        # 记录性能历史
        self.performance_history_.append({
            'individual_performances': current_performances,
            'ensemble_performance': ensemble_performance,
            'weights': self.adaptive_weights_.copy()
        })
        
        # 使用权重函数计算新权重
        new_weights = self.weight_function(current_performances, self.adaptive_weights_)
        
        # 应用适应率
        self.adaptive_weights_ = (1 - self.adaptation_rate) * self.adaptive_weights_ + \
                                self.adaptation_rate * new_weights
        
        # 归一化权重
        self.adaptive_weights_ = self.adaptive_weights_ / np.sum(self.adaptive_weights_)
    
    def _default_weight_function(self, performances: List[float], current_weights: np.ndarray) -> np.ndarray:
        """默认权重函数"""
        performances = np.array(performances)
        
        # 性能越好，权重越大
        if np.sum(performances) > 0:
            weights = performances / np.sum(performances)
        else:
            weights = np.ones(len(performances)) / len(performances)
        
        return weights
    
    def get_performance_history(self) -> List[Dict]:
        """获取性能历史"""
        return self.performance_history_
    
    def get_current_weights(self) -> np.ndarray:
        """获取当前权重"""
        return self.adaptive_weights_.copy() if self.adaptive_weights_ is not None else None