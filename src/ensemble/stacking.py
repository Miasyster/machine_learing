"""
Stacking堆叠集成方法

实现标准Stacking、多层Stacking和动态Stacking策略
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import BaseEnsemble, EnsembleResult

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEnsemble):
    """Stacking堆叠集成器"""
    
    def __init__(self,
                 base_models: List[BaseEstimator],
                 meta_model: Optional[BaseEstimator] = None,
                 cv: Union[int, object] = 5,
                 use_probas: bool = True,
                 use_features_in_secondary: bool = False,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化Stacking集成器
        
        Args:
            base_models: 基础模型列表
            meta_model: 元学习器
            cv: 交叉验证折数或交叉验证对象
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
        self.cv = cv
        self.use_probas = use_probas
        self.use_features_in_secondary = use_features_in_secondary
        
        # 训练后的模型
        self.fitted_base_models_ = None
        self.fitted_meta_model_ = None
        self.meta_features_shape_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'StackingEnsemble':
        """
        训练Stacking集成器
        
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
        
        if self.verbose:
            logger.info(f"Training stacking ensemble with {self.n_models_} base models")
            logger.info(f"Input shape: {X.shape}, target shape: {y.shape}")
        
        try:
            # 设置默认元学习器
            if self.meta_model is None:
                if self._is_classifier():
                    self.meta_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
                else:
                    self.meta_model = LinearRegression()
            
            # 生成元特征
            if self.verbose:
                logger.info("Generating meta-features using cross-validation")
            
            meta_features = self._generate_meta_features(X, y)
            
            # 验证元特征
            if meta_features.shape[0] != X.shape[0]:
                raise RuntimeError(f"Meta-features shape mismatch: expected {X.shape[0]} samples, got {meta_features.shape[0]}")
            
            # 如果使用原始特征
            if self.use_features_in_secondary:
                if self.verbose:
                    logger.info("Combining meta-features with original features")
                meta_features = np.hstack([meta_features, X])
            
            # 训练所有基础模型
            if self.verbose:
                logger.info("Training base models on full dataset")
            
            self.fitted_base_models_ = []
            failed_models = []
            
            for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
                try:
                    if self.verbose:
                        logger.info(f"Training base model {i+1}/{self.n_models_}: {name}")
                    
                    fitted_model = clone(model)
                    fitted_model.fit(X, y, **kwargs)
                    self.fitted_base_models_.append(fitted_model)
                    
                except Exception as e:
                    logger.error(f"Failed to train base model {name}: {str(e)}")
                    failed_models.append((i, name, str(e)))
                    # 添加一个占位符以保持索引一致性
                    self.fitted_base_models_.append(None)
            
            # 检查是否有足够的成功训练的模型
            successful_models = sum(1 for model in self.fitted_base_models_ if model is not None)
            if successful_models == 0:
                raise RuntimeError("All base models failed to train")
            elif failed_models:
                logger.warning(f"{len(failed_models)} base models failed to train: {[name for _, name, _ in failed_models]}")
            
            # 训练元学习器
            if self.verbose:
                logger.info("Training meta-learner")
            
            self.fitted_meta_model_ = clone(self.meta_model)
            self.fitted_meta_model_.fit(meta_features, y)
            
            self.meta_features_shape_ = meta_features.shape[1]
            self.is_fitted_ = True
            
            # 评估各个模型
            try:
                self.evaluate_individual_models(X, y)
            except Exception as e:
                logger.warning(f"Model evaluation failed: {str(e)}")
                # 继续执行，不让评估失败影响训练
            
            if self.verbose:
                logger.info("Stacking ensemble training completed successfully")
                if failed_models:
                    logger.info(f"Successfully trained {successful_models}/{self.n_models_} base models")
            
            return self
            
        except Exception as e:
            logger.error(f"Stacking ensemble training failed: {str(e)}")
            raise RuntimeError(f"Failed to train stacking ensemble: {str(e)}") from e
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        生成元特征
        
        Args:
            X: 训练特征
            y: 训练标签
            
        Returns:
            元特征矩阵
            
        Raises:
            RuntimeError: 当元特征生成失败时
        """
        try:
            # 设置交叉验证
            if isinstance(self.cv, int):
                if self.cv < 2:
                    raise ValueError("Cross-validation folds must be at least 2")
                if self.cv > len(y):
                    logger.warning(f"CV folds ({self.cv}) > samples ({len(y)}), using leave-one-out")
                    self.cv = min(self.cv, len(y))
                
                if self._is_classifier():
                    cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                else:
                    cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            else:
                cv = self.cv
            
            meta_features_list = []
            failed_models = []
            
            for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
                try:
                    if self.verbose:
                        logger.info(f"Generating meta-features for model {i+1}/{self.n_models_}: {name}")
                    
                    if self.use_probas and hasattr(model, 'predict_proba') and self._is_classifier():
                        # 使用概率预测
                        meta_pred = cross_val_predict(
                            model, X, y, cv=cv, method='predict_proba', n_jobs=self.n_jobs
                        )
                        # 验证概率预测的形状
                        if meta_pred.ndim == 1:
                            meta_pred = meta_pred.reshape(-1, 1)
                    else:
                        # 使用普通预测
                        meta_pred = cross_val_predict(
                            model, X, y, cv=cv, method='predict', n_jobs=self.n_jobs
                        )
                        if meta_pred.ndim == 1:
                            meta_pred = meta_pred.reshape(-1, 1)
                    
                    # 验证预测结果
                    if meta_pred.shape[0] != X.shape[0]:
                        raise RuntimeError(f"Meta-prediction shape mismatch for model {name}")
                    
                    # 检查是否有无效值
                    if np.any(np.isnan(meta_pred)) or np.any(np.isinf(meta_pred)):
                        logger.warning(f"Model {name} produced invalid predictions, filling with zeros")
                        meta_pred = np.nan_to_num(meta_pred, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    meta_features_list.append(meta_pred)
                    
                except Exception as e:
                    logger.error(f"Failed to generate meta-features for model {name}: {str(e)}")
                    failed_models.append((i, name, str(e)))
                    # 添加零矩阵作为占位符
                    if self.use_probas and hasattr(model, 'predict_proba') and self._is_classifier():
                        # 估计类别数量
                        n_classes = len(np.unique(y)) if hasattr(y, '__len__') else 2
                        placeholder = np.zeros((X.shape[0], n_classes))
                    else:
                        placeholder = np.zeros((X.shape[0], 1))
                    meta_features_list.append(placeholder)
            
            if len(meta_features_list) == 0:
                raise RuntimeError("No meta-features could be generated from any base model")
            
            # 合并元特征
            meta_features = np.hstack(meta_features_list)
            
            if self.verbose:
                logger.info(f"Generated meta-features shape: {meta_features.shape}")
                if failed_models:
                    logger.warning(f"Failed to generate meta-features for {len(failed_models)} models")
            
            return meta_features
            
        except Exception as e:
            logger.error(f"Meta-feature generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate meta-features: {str(e)}") from e
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """
        进行Stacking预测
        
        Args:
            X: 预测特征
            
        Returns:
            集成预测结果
            
        Raises:
            ValueError: 当模型未训练或输入无效时
            RuntimeError: 当预测过程中出现错误时
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # 输入验证
        X = self._validate_prediction_input(X)
        
        try:
            # 获取基础模型预测作为元特征
            meta_features = self._get_meta_features_for_prediction(X)
            
            # 验证元特征形状
            expected_meta_features = self.meta_features_shape_
            if self.use_features_in_secondary:
                expected_meta_features -= X.shape[1]  # 减去原始特征的维度
            
            if meta_features.shape[1] != expected_meta_features:
                logger.warning(f"Meta-features shape mismatch: expected {expected_meta_features}, got {meta_features.shape[1]}")
            
            # 如果使用原始特征
            if self.use_features_in_secondary:
                meta_features = np.hstack([meta_features, X])
            
            # 验证最终元特征形状
            if meta_features.shape[1] != self.meta_features_shape_:
                raise RuntimeError(f"Final meta-features shape mismatch: expected {self.meta_features_shape_}, got {meta_features.shape[1]}")
            
            # 元学习器预测
            predictions = self.fitted_meta_model_.predict(meta_features)
            
            # 获取概率预测（如果支持）
            prediction_probabilities = None
            if hasattr(self.fitted_meta_model_, 'predict_proba'):
                try:
                    prediction_probabilities = self.fitted_meta_model_.predict_proba(meta_features)
                except Exception as e:
                    logger.warning(f"Failed to get probability predictions from meta-model: {str(e)}")
            
            # 获取各基础模型的预测
            individual_predictions = self.get_individual_predictions(X)
            individual_probabilities = self.get_individual_probabilities(X)
            
            return EnsembleResult(
                predictions=predictions,
                prediction_probabilities=prediction_probabilities,
                individual_predictions=individual_predictions,
                individual_probabilities=individual_probabilities,
                model_scores=getattr(self, 'model_scores_', None)
            )
            
        except Exception as e:
            logger.error(f"Stacking prediction failed: {str(e)}")
            raise RuntimeError(f"Failed to make stacking predictions: {str(e)}") from e
    
    def _get_meta_features_for_prediction(self, X: np.ndarray) -> np.ndarray:
        """
        获取预测时的元特征
        
        Args:
            X: 预测特征
            
        Returns:
            元特征矩阵
            
        Raises:
            RuntimeError: 当无法获取元特征时
        """
        meta_features_list = []
        failed_models = []
        
        for i, model in enumerate(self.fitted_base_models_):
            try:
                if model is None:
                    # 这是一个训练失败的模型，使用零预测
                    if self.use_probas and self._is_classifier():
                        # 估计类别数量（从其他成功的模型推断）
                        n_classes = 2  # 默认二分类
                        for other_model in self.fitted_base_models_:
                            if other_model is not None and hasattr(other_model, 'classes_'):
                                n_classes = len(other_model.classes_)
                                break
                        meta_pred = np.zeros((X.shape[0], n_classes))
                    else:
                        meta_pred = np.zeros((X.shape[0], 1))
                else:
                    if self.use_probas and hasattr(model, 'predict_proba') and self._is_classifier():
                        # 使用概率预测
                        meta_pred = model.predict_proba(X)
                    else:
                        # 使用普通预测
                        meta_pred = model.predict(X)
                        if meta_pred.ndim == 1:
                            meta_pred = meta_pred.reshape(-1, 1)
                
                # 检查预测结果的有效性
                if np.any(np.isnan(meta_pred)) or np.any(np.isinf(meta_pred)):
                    logger.warning(f"Model {i} produced invalid predictions, filling with zeros")
                    meta_pred = np.nan_to_num(meta_pred, nan=0.0, posinf=0.0, neginf=0.0)
                
                meta_features_list.append(meta_pred)
                
            except Exception as e:
                logger.error(f"Failed to get predictions from model {i}: {str(e)}")
                failed_models.append((i, str(e)))
                
                # 添加零预测作为占位符
                if self.use_probas and self._is_classifier():
                    # 尝试从其他模型推断类别数量
                    n_classes = 2
                    for other_model in self.fitted_base_models_:
                        if other_model is not None and hasattr(other_model, 'classes_'):
                            n_classes = len(other_model.classes_)
                            break
                    meta_pred = np.zeros((X.shape[0], n_classes))
                else:
                    meta_pred = np.zeros((X.shape[0], 1))
                
                meta_features_list.append(meta_pred)
        
        if len(meta_features_list) == 0:
            raise RuntimeError("No meta-features could be generated from any fitted model")
        
        try:
            meta_features = np.hstack(meta_features_list)
            
            if failed_models and self.verbose:
                logger.warning(f"Failed to get predictions from {len(failed_models)} models during prediction")
            
            return meta_features
            
        except Exception as e:
            raise RuntimeError(f"Failed to combine meta-features: {str(e)}") from e
    
    def get_individual_predictions(self, X: np.ndarray) -> List[np.ndarray]:
        """
        获取各基础模型的预测
        
        Args:
            X: 预测特征
            
        Returns:
            各模型的预测结果列表
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for i, model in enumerate(self.fitted_base_models_):
            try:
                if model is None:
                    # 训练失败的模型，返回零预测
                    pred = np.zeros(X.shape[0])
                else:
                    pred = model.predict(X)
                    
                # 检查预测结果的有效性
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    logger.warning(f"Model {i} produced invalid predictions, filling with zeros")
                    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    
                predictions.append(pred)
                
            except Exception as e:
                logger.error(f"Failed to get predictions from model {i}: {str(e)}")
                # 添加零预测作为占位符
                predictions.append(np.zeros(X.shape[0]))
        
        return predictions
    
    def get_individual_probabilities(self, X: np.ndarray) -> List[Optional[np.ndarray]]:
        """
        获取各基础模型的概率预测
        
        Args:
            X: 预测特征
            
        Returns:
            各模型的概率预测结果列表
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        probabilities = []
        for i, model in enumerate(self.fitted_base_models_):
            try:
                if model is None or not hasattr(model, 'predict_proba'):
                    probabilities.append(None)
                else:
                    prob = model.predict_proba(X)
                    
                    # 检查概率预测的有效性
                    if np.any(np.isnan(prob)) or np.any(np.isinf(prob)):
                        logger.warning(f"Model {i} produced invalid probability predictions")
                        prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
                        # 重新归一化概率
                        prob = prob / (prob.sum(axis=1, keepdims=True) + 1e-10)
                    
                    probabilities.append(prob)
                    
            except Exception as e:
                logger.error(f"Failed to get probability predictions from model {i}: {str(e)}")
                probabilities.append(None)
        
        return probabilities


class MultiLevelStackingEnsemble(BaseEnsemble):
    """多层Stacking集成器"""
    
    def __init__(self,
                 base_models: List[BaseEstimator],
                 meta_models: List[BaseEstimator],
                 cv: Union[int, object] = 5,
                 use_probas: bool = True,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化多层Stacking集成器
        
        Args:
            base_models: 基础模型列表
            meta_models: 元学习器列表（每层一个）
            cv: 交叉验证折数或交叉验证对象
            use_probas: 是否使用概率预测作为元特征
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(base_models, model_names, random_state, n_jobs, verbose)
        
        self.base_models = base_models
        self.meta_models = meta_models
        self.cv = cv
        self.use_probas = use_probas
        
        # 训练后的模型
        self.fitted_models_by_level_ = []
        self.n_levels_ = len(meta_models) + 1  # 基础层 + 元学习层数
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'MultiLevelStackingEnsemble':
        """
        训练多层Stacking集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        if self.verbose:
            logger.info(f"Training multi-level stacking ensemble with {self.n_levels_} levels")
        
        current_features = X
        current_targets = y
        
        # 第一层：基础模型
        if self.verbose:
            logger.info("Training level 1 (base models)")
        
        level_1_ensemble = StackingEnsemble(
            base_models=self.base_models,
            meta_model=self.meta_models[0],
            cv=self.cv,
            use_probas=self.use_probas,
            model_names=self.model_names,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        level_1_ensemble.fit(current_features, current_targets, **kwargs)
        self.fitted_models_by_level_.append(level_1_ensemble)
        
        # 更新特征为第一层的元特征
        current_features = level_1_ensemble._generate_meta_features(current_features, current_targets)
        
        # 后续层：元学习器
        for level in range(1, len(self.meta_models)):
            if self.verbose:
                logger.info(f"Training level {level + 1}")
            
            # 创建单模型的Stacking（实际上是简单的模型训练）
            meta_model = clone(self.meta_models[level])
            meta_model.fit(current_features, current_targets)
            self.fitted_models_by_level_.append(meta_model)
            
            # 如果不是最后一层，生成下一层的特征
            if level < len(self.meta_models) - 1:
                if hasattr(meta_model, 'predict_proba') and self.use_probas and self._is_classifier():
                    # 设置交叉验证
                    if isinstance(self.cv, int):
                        if self._is_classifier():
                            cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                        else:
                            cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                    else:
                        cv = self.cv
                    
                    current_features = cross_val_predict(
                        meta_model, current_features, current_targets, 
                        cv=cv, method='predict_proba', n_jobs=self.n_jobs
                    )
                else:
                    current_features = cross_val_predict(
                        meta_model, current_features, current_targets, 
                        cv=cv, method='predict', n_jobs=self.n_jobs
                    ).reshape(-1, 1)
        
        self.is_fitted_ = True
        
        if self.verbose:
            logger.info("Multi-level stacking ensemble training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """
        进行多层Stacking预测
        
        Args:
            X: 预测特征
            
        Returns:
            集成预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        current_features = X
        
        # 第一层预测
        level_1_result = self.fitted_models_by_level_[0].predict(current_features)
        current_features = level_1_result.meta_features
        
        # 后续层预测
        for level in range(1, len(self.fitted_models_by_level_)):
            meta_model = self.fitted_models_by_level_[level]
            
            if level == len(self.fitted_models_by_level_) - 1:
                # 最后一层，生成最终预测
                predictions = meta_model.predict(current_features)
                
                prediction_probabilities = None
                if hasattr(meta_model, 'predict_proba'):
                    try:
                        prediction_probabilities = meta_model.predict_proba(current_features)
                    except:
                        pass
                
                return EnsembleResult(
                    predictions=predictions,
                    prediction_probabilities=prediction_probabilities,
                    individual_predictions=level_1_result.individual_predictions,
                    individual_probabilities=level_1_result.individual_probabilities,
                    model_scores=getattr(self, 'model_scores_', {})
                )
            else:
                # 中间层，生成下一层的特征
                if hasattr(meta_model, 'predict_proba') and self.use_probas and self._is_classifier():
                    current_features = meta_model.predict_proba(current_features)
                else:
                    pred = meta_model.predict(current_features)
                    current_features = pred.reshape(-1, 1)


class DynamicStackingEnsemble(StackingEnsemble):
    """动态Stacking集成器"""
    
    def __init__(self,
                 base_models: List[BaseEstimator],
                 meta_model_pool: List[BaseEstimator],
                 selection_strategy: str = 'performance',
                 adaptation_frequency: int = 100,
                 cv: Union[int, object] = 5,
                 use_probas: bool = True,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化动态Stacking集成器
        
        Args:
            base_models: 基础模型列表
            meta_model_pool: 元学习器池
            selection_strategy: 选择策略 ('performance', 'diversity', 'adaptive')
            adaptation_frequency: 适应频率
            cv: 交叉验证折数或交叉验证对象
            use_probas: 是否使用概率预测作为元特征
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        # 初始化时使用第一个元学习器
        super().__init__(
            base_models, meta_model_pool[0], cv, use_probas, False,
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
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'DynamicStackingEnsemble':
        """
        训练动态Stacking集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成器
        """
        if self.verbose:
            logger.info("Training dynamic stacking ensemble")
        
        # 评估所有元学习器
        self._evaluate_meta_models(X, y, **kwargs)
        
        # 选择最佳元学习器
        self.current_meta_model_index_ = self._select_best_meta_model()
        self.meta_model = self.meta_model_pool[self.current_meta_model_index_]
        
        # 训练选定的模型
        super().fit(X, y, **kwargs)
        
        if self.verbose:
            logger.info(f"Selected meta-model: {type(self.meta_model).__name__}")
        
        return self
    
    def _evaluate_meta_models(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """评估所有元学习器"""
        from sklearn.model_selection import cross_val_score
        
        # 生成元特征
        meta_features = self._generate_meta_features(X, y)
        
        self.meta_model_performances_ = []
        
        for i, meta_model in enumerate(self.meta_model_pool):
            if self.verbose:
                logger.info(f"Evaluating meta-model {i+1}/{len(self.meta_model_pool)}: {type(meta_model).__name__}")
            
            # 交叉验证评估
            scores = cross_val_score(
                meta_model, meta_features, y, 
                cv=self.cv, n_jobs=self.n_jobs
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
        进行动态Stacking预测
        
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
            from sklearn.metrics import accuracy_score
            current_performance = accuracy_score(y_true, current_predictions)
        else:
            from sklearn.metrics import mean_squared_error
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
    
    def get_adaptation_history(self) -> List[Dict]:
        """获取适应历史"""
        return self.adaptation_history_
    
    def get_meta_model_performances(self) -> List[Dict]:
        """获取元学习器性能评估结果"""
        return self.meta_model_performances_