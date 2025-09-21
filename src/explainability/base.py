"""
模型解释性基础类
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """解释结果数据类"""
    
    # 基础信息
    feature_names: List[str]
    model_type: str
    explanation_method: str
    
    # SHAP相关
    shap_values: Optional[np.ndarray] = None
    shap_base_value: Optional[Union[float, np.ndarray]] = None
    shap_expected_value: Optional[Union[float, np.ndarray]] = None
    
    # 特征重要性
    feature_importance: Optional[np.ndarray] = None
    feature_importance_std: Optional[np.ndarray] = None
    permutation_importance: Optional[np.ndarray] = None
    
    # 局部解释
    local_explanations: Optional[List[Dict[str, Any]]] = None
    
    # 全局解释
    global_explanations: Optional[Dict[str, Any]] = None
    
    # 可视化数据
    visualization_data: Optional[Dict[str, Any]] = None
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None
    
    def get_top_features(self, n_features: int = 10, method: str = 'importance') -> List[Tuple[str, float]]:
        """
        获取最重要的特征
        
        Args:
            n_features: 返回的特征数量
            method: 重要性计算方法 ('importance', 'shap', 'permutation')
            
        Returns:
            特征名称和重要性分数的列表
        """
        if method == 'importance' and self.feature_importance is not None:
            importance_scores = self.feature_importance
        elif method == 'shap' and self.shap_values is not None:
            # 使用SHAP值的绝对值平均作为重要性
            importance_scores = np.abs(self.shap_values).mean(axis=0)
        elif method == 'permutation' and self.permutation_importance is not None:
            importance_scores = self.permutation_importance
        else:
            raise ValueError(f"Method '{method}' not available or no data found")
        
        # 获取排序后的索引
        sorted_indices = np.argsort(importance_scores)[::-1][:n_features]
        
        return [(self.feature_names[i], importance_scores[i]) for i in sorted_indices]
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        获取特征重要性摘要
        
        Returns:
            特征重要性摘要字典
        """
        summary = {
            'total_features': len(self.feature_names),
            'explanation_method': self.explanation_method,
            'model_type': self.model_type
        }
        
        if self.feature_importance is not None:
            summary['feature_importance_available'] = True
            summary['top_feature_by_importance'] = self.feature_names[np.argmax(self.feature_importance)]
            summary['importance_range'] = {
                'min': float(np.min(self.feature_importance)),
                'max': float(np.max(self.feature_importance)),
                'mean': float(np.mean(self.feature_importance))
            }
        
        if self.shap_values is not None:
            summary['shap_values_available'] = True
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            summary['top_feature_by_shap'] = self.feature_names[np.argmax(mean_abs_shap)]
            summary['shap_range'] = {
                'min': float(np.min(mean_abs_shap)),
                'max': float(np.max(mean_abs_shap)),
                'mean': float(np.mean(mean_abs_shap))
            }
        
        return summary


class BaseExplainer(ABC):
    """模型解释器基础类"""
    
    def __init__(self,
                 model: BaseEstimator,
                 feature_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        """
        初始化解释器
        
        Args:
            model: 要解释的模型
            feature_names: 特征名称列表
            random_state: 随机种子
            verbose: 是否显示详细信息
        """
        self.model = model
        self.feature_names = feature_names
        self.random_state = random_state
        self.verbose = verbose
        
        # 验证模型是否已训练
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a predict method")
        
        # 获取模型类型
        self.model_type = self._get_model_type()
        
        # 状态变量
        self.is_fitted_ = False
        
    def _get_model_type(self) -> str:
        """获取模型类型"""
        model_class = self.model.__class__.__name__
        
        # 常见模型类型映射
        model_type_mapping = {
            'RandomForestClassifier': 'tree_based',
            'RandomForestRegressor': 'tree_based',
            'GradientBoostingClassifier': 'tree_based',
            'GradientBoostingRegressor': 'tree_based',
            'XGBClassifier': 'tree_based',
            'XGBRegressor': 'tree_based',
            'LGBMClassifier': 'tree_based',
            'LGBMRegressor': 'tree_based',
            'LogisticRegression': 'linear',
            'LinearRegression': 'linear',
            'Ridge': 'linear',
            'Lasso': 'linear',
            'SVC': 'kernel',
            'SVR': 'kernel',
            'MLPClassifier': 'neural_network',
            'MLPRegressor': 'neural_network'
        }
        
        return model_type_mapping.get(model_class, 'unknown')
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """验证输入数据"""
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            X = np.array(X)
        
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X = X.values
        
        if X.ndim != 2:
            raise ValueError("Input must be 2-dimensional")
        
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        elif len(self.feature_names) != X.shape[1]:
            raise ValueError("Number of feature names must match number of features")
        
        return X
    
    @abstractmethod
    def explain(self, X: np.ndarray, **kwargs) -> ExplanationResult:
        """
        解释模型预测
        
        Args:
            X: 输入数据
            **kwargs: 其他参数
            
        Returns:
            解释结果
        """
        pass
    
    @abstractmethod
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
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_class': self.model.__class__.__name__,
            'model_type': self.model_type,
            'feature_count': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted_,
            'explainer_class': self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__}, features={len(self.feature_names) if self.feature_names else 'unknown'})"