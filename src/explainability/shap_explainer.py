"""
SHAP (SHapley Additive exPlanations) 解释器
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from sklearn.base import BaseEstimator

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from .base import BaseExplainer, ExplanationResult

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """SHAP解释器"""
    
    def __init__(self,
                 model: BaseEstimator,
                 feature_names: Optional[List[str]] = None,
                 explainer_type: str = 'auto',
                 background_data: Optional[np.ndarray] = None,
                 n_background_samples: int = 100,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        """
        初始化SHAP解释器
        
        Args:
            model: 要解释的模型
            feature_names: 特征名称列表
            explainer_type: SHAP解释器类型 ('auto', 'tree', 'linear', 'kernel', 'deep', 'permutation')
            background_data: 背景数据集
            n_background_samples: 背景样本数量
            random_state: 随机种子
            verbose: 是否显示详细信息
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAPExplainer. Install it with: pip install shap")
        
        super().__init__(model, feature_names, random_state, verbose)
        
        self.explainer_type = explainer_type
        self.background_data = background_data
        self.n_background_samples = n_background_samples
        
        # SHAP解释器实例
        self.shap_explainer = None
        self.background_data_ = None
        
    def _prepare_background_data(self, X: np.ndarray) -> np.ndarray:
        """准备背景数据"""
        if self.background_data is not None:
            background = self.background_data
        else:
            # 使用训练数据的子集作为背景
            if len(X) > self.n_background_samples:
                np.random.seed(self.random_state)
                indices = np.random.choice(len(X), self.n_background_samples, replace=False)
                background = X[indices]
            else:
                background = X
        
        return background
    
    def _create_predict_wrapper(self):
        """创建预测包装器，处理EnsembleResult对象"""
        def predict_wrapper(X):
            try:
                # 尝试直接调用predict
                result = self.model.predict(X)
                
                # 如果返回EnsembleResult对象，提取predictions
                if hasattr(result, 'predictions'):
                    return result.predictions
                else:
                    return result
            except Exception:
                # 如果失败，尝试predict_proba
                try:
                    proba = self.model.predict_proba(X)
                    if proba.shape[1] == 2:
                        # 二分类，返回正类概率
                        return proba[:, 1]
                    else:
                        # 多分类，返回概率矩阵
                        return proba
                except Exception:
                    raise ValueError("Model prediction failed")
        
        return predict_wrapper
    
    def _create_shap_explainer(self, X: np.ndarray) -> None:
        """创建SHAP解释器"""
        explainer_type = self.explainer_type
        
        if explainer_type == 'auto':
            explainer_type = self._auto_select_explainer()
        
        if self.verbose:
            logger.info(f"Creating SHAP explainer of type: {explainer_type}")
        
        # 创建预测包装器
        predict_func = self._create_predict_wrapper()
        
        try:
            if explainer_type == 'tree':
                # 对于树模型，尝试直接使用模型
                try:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                except Exception:
                    # 如果失败，使用KernelExplainer
                    logger.warning("TreeExplainer failed, falling back to KernelExplainer")
                    self.shap_explainer = shap.KernelExplainer(predict_func, self.background_data_)
                
            elif explainer_type == 'linear':
                try:
                    self.shap_explainer = shap.LinearExplainer(self.model, self.background_data_)
                except Exception:
                    logger.warning("LinearExplainer failed, falling back to KernelExplainer")
                    self.shap_explainer = shap.KernelExplainer(predict_func, self.background_data_)
                
            elif explainer_type == 'kernel':
                self.shap_explainer = shap.KernelExplainer(predict_func, self.background_data_)
                
            elif explainer_type == 'permutation':
                self.shap_explainer = shap.PermutationExplainer(predict_func, self.background_data_)
                
            elif explainer_type == 'deep':
                # 对于深度学习模型
                self.shap_explainer = shap.DeepExplainer(self.model, self.background_data_)
                
            else:
                # 默认使用Kernel解释器
                logger.warning(f"Unknown explainer type {explainer_type}, using KernelExplainer")
                self.shap_explainer = shap.KernelExplainer(self.model.predict, self.background_data_)
                
        except Exception as e:
            logger.warning(f"Failed to create {explainer_type} explainer: {e}")
            logger.info("Falling back to KernelExplainer")
            self.shap_explainer = shap.KernelExplainer(self.model.predict, self.background_data_)
    
    def _auto_select_explainer(self) -> str:
        """自动选择SHAP解释器类型"""
        model_class = self.model.__class__.__name__
        
        # 基于模型类型选择解释器
        tree_models = [
            'RandomForestClassifier', 'RandomForestRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'DecisionTreeClassifier', 'DecisionTreeRegressor'
        ]
        
        linear_models = [
            'LinearRegression', 'LogisticRegression',
            'Ridge', 'Lasso', 'ElasticNet'
        ]
        
        if model_class in tree_models:
            return 'tree'
        elif model_class in linear_models:
            return 'linear'
        else:
            return 'kernel'
    
    def explain(self, X: np.ndarray, **kwargs) -> ExplanationResult:
        """
        解释模型预测
        
        Args:
            X: 输入数据
            **kwargs: 其他参数
            
        Returns:
            解释结果
        """
        X = self._validate_input(X)
        
        # 准备背景数据
        if self.background_data_ is None:
            self.background_data_ = self._prepare_background_data(X)
        
        # 创建SHAP解释器
        if self.shap_explainer is None:
            self._create_shap_explainer(X)
        
        if self.verbose:
            logger.info(f"Computing SHAP values for {len(X)} samples")
        
        try:
            # 计算SHAP值
            shap_values = self.shap_explainer.shap_values(X)
            
            # 处理多类分类的情况
            if isinstance(shap_values, list):
                # 对于多类分类，取第一类的SHAP值或者计算平均值
                if len(shap_values) == 2:
                    # 二分类，取正类的SHAP值
                    shap_values = shap_values[1]
                else:
                    # 多分类，计算绝对值的平均
                    shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            
            # 获取基准值
            if hasattr(self.shap_explainer, 'expected_value'):
                expected_value = self.shap_explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
                    expected_value = expected_value[1] if len(expected_value) == 2 else np.mean(expected_value)
            else:
                expected_value = None
            
            # 创建解释结果
            result = ExplanationResult(
                feature_names=self.feature_names,
                model_type=self.model_type,
                explanation_method='shap',
                shap_values=shap_values,
                shap_expected_value=expected_value,
                metadata={
                    'explainer_type': self.explainer_type,
                    'n_samples': len(X),
                    'n_features': len(self.feature_names),
                    'background_samples': len(self.background_data_)
                }
            )
            
            self.is_fitted_ = True
            return result
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            raise
    
    def explain_instance(self, X: np.ndarray, instance_idx: int, **kwargs) -> Dict[str, Any]:
        """
        解释单个实例的预测
        
        Args:
            X: 输入数据
            instance_idx: 实例索引
            **kwargs: 其他参数
            
        Returns:
            单个实例的解释结果
        """
        if instance_idx >= len(X):
            raise ValueError(f"Instance index {instance_idx} out of range for data with {len(X)} samples")
        
        # 获取完整的解释结果
        full_result = self.explain(X, **kwargs)
        
        # 提取单个实例的SHAP值
        instance_shap = full_result.shap_values[instance_idx]
        
        # 创建特征贡献排序
        feature_contributions = list(zip(self.feature_names, instance_shap))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 获取实例的预测值
        prediction = self.model.predict(X[instance_idx:instance_idx+1])[0]
        
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(X[instance_idx:instance_idx+1])[0]
        else:
            prediction_proba = None
        
        return {
            'instance_idx': instance_idx,
            'prediction': prediction,
            'prediction_proba': prediction_proba,
            'shap_values': instance_shap,
            'expected_value': full_result.shap_expected_value,
            'feature_contributions': feature_contributions,
            'top_positive_features': [(name, val) for name, val in feature_contributions if val > 0][:5],
            'top_negative_features': [(name, val) for name, val in feature_contributions if val < 0][:5],
            'feature_values': X[instance_idx].tolist(),
            'explanation_sum': np.sum(instance_shap),
            'metadata': {
                'model_type': self.model_type,
                'explainer_type': self.explainer_type
            }
        }
    
    def get_global_importance(self, X: np.ndarray) -> np.ndarray:
        """
        获取全局特征重要性（基于SHAP值的绝对值平均）
        
        Args:
            X: 输入数据
            
        Returns:
            全局特征重要性数组
        """
        result = self.explain(X)
        return np.abs(result.shap_values).mean(axis=0)
    
    def get_feature_interactions(self, X: np.ndarray, max_interactions: int = 10) -> List[Tuple[str, str, float]]:
        """
        获取特征交互效应（需要SHAP的交互值支持）
        
        Args:
            X: 输入数据
            max_interactions: 最大交互数量
            
        Returns:
            特征交互列表
        """
        try:
            if hasattr(self.shap_explainer, 'shap_interaction_values'):
                interaction_values = self.shap_explainer.shap_interaction_values(X)
                
                # 计算交互强度
                interactions = []
                n_features = len(self.feature_names)
                
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        interaction_strength = np.abs(interaction_values[:, i, j]).mean()
                        interactions.append((
                            self.feature_names[i],
                            self.feature_names[j],
                            interaction_strength
                        ))
                
                # 按交互强度排序
                interactions.sort(key=lambda x: x[2], reverse=True)
                return interactions[:max_interactions]
            else:
                logger.warning("Current SHAP explainer does not support interaction values")
                return []
                
        except Exception as e:
            logger.warning(f"Error computing feature interactions: {e}")
            return []
    
    def get_dependence_data(self, X: np.ndarray, feature_name: str) -> Dict[str, Any]:
        """
        获取特征依赖数据（用于绘制依赖图）
        
        Args:
            X: 输入数据
            feature_name: 特征名称
            
        Returns:
            依赖数据字典
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in feature names")
        
        feature_idx = self.feature_names.index(feature_name)
        result = self.explain(X)
        
        return {
            'feature_name': feature_name,
            'feature_values': X[:, feature_idx],
            'shap_values': result.shap_values[:, feature_idx],
            'feature_idx': feature_idx,
            'correlation_features': self._find_correlation_features(X, feature_idx)
        }
    
    def _find_correlation_features(self, X: np.ndarray, target_feature_idx: int) -> List[Tuple[str, float]]:
        """找到与目标特征相关性最高的特征"""
        target_values = X[:, target_feature_idx]
        correlations = []
        
        for i, feature_name in enumerate(self.feature_names):
            if i != target_feature_idx:
                correlation = np.corrcoef(target_values, X[:, i])[0, 1]
                if not np.isnan(correlation):
                    correlations.append((feature_name, abs(correlation)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        return correlations[:5]