"""
模型集成基础类

定义集成学习的基础接口和数据结构
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error

# 延迟导入解释性模块，避免循环导入
try:
    from ..explainability import SHAPExplainer, FeatureImportanceExplainer, ExplanationVisualizer
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """集成结果数据类"""
    predictions: np.ndarray
    prediction_probabilities: Optional[np.ndarray] = None
    individual_predictions: Optional[List[np.ndarray]] = None
    individual_probabilities: Optional[List[np.ndarray]] = None
    weights: Optional[np.ndarray] = None
    model_scores: Optional[Dict[str, float]] = None
    ensemble_score: Optional[float] = None
    diversity_metrics: Optional[Dict[str, float]] = None
    
    def get_prediction_confidence(self) -> np.ndarray:
        """
        获取预测置信度
        
        Returns:
            预测置信度数组
        """
        if self.prediction_probabilities is not None:
            # 分类任务：使用最大概率作为置信度
            return np.max(self.prediction_probabilities, axis=1)
        elif self.individual_predictions is not None:
            # 回归任务：使用预测方差的倒数作为置信度
            individual_preds = np.array(self.individual_predictions).T
            pred_variance = np.var(individual_preds, axis=1)
            return 1.0 / (1.0 + pred_variance)
        else:
            return np.ones(len(self.predictions))
    
    def get_prediction_uncertainty(self) -> np.ndarray:
        """
        获取预测不确定性
        
        Returns:
            预测不确定性数组
        """
        if self.individual_predictions is not None:
            individual_preds = np.array(self.individual_predictions).T
            return np.std(individual_preds, axis=1)
        else:
            return np.zeros(len(self.predictions))
    
    def get_model_agreement(self) -> np.ndarray:
        """
        获取模型一致性
        
        Returns:
            模型一致性数组
        """
        if self.individual_predictions is not None:
            individual_preds = np.array(self.individual_predictions).T
            
            if self.prediction_probabilities is not None:
                # 分类任务：计算预测类别的一致性
                pred_classes = np.argmax(individual_preds, axis=2) if individual_preds.ndim == 3 else individual_preds
                mode_predictions = []
                agreements = []
                
                for i in range(len(pred_classes)):
                    unique, counts = np.unique(pred_classes[i], return_counts=True)
                    mode_pred = unique[np.argmax(counts)]
                    agreement = np.max(counts) / len(pred_classes[i])
                    mode_predictions.append(mode_pred)
                    agreements.append(agreement)
                
                return np.array(agreements)
            else:
                # 回归任务：使用标准差的倒数
                std_devs = np.std(individual_preds, axis=1)
                return 1.0 / (1.0 + std_devs)
        else:
            return np.ones(len(self.predictions))


class BaseEnsemble(ABC, BaseEstimator):
    """集成学习基础类"""
    
    def __init__(self,
                 models: List[BaseEstimator],
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化集成模型
        
        Args:
            models: 基础模型列表
            model_names: 模型名称列表
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        self.models = models
        self.model_names = model_names or [f"model_{i}" for i in range(len(models))]
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # 验证输入
        if len(self.models) != len(self.model_names):
            raise ValueError("Number of models must match number of model names")
        
        # 状态变量
        self.is_fitted_ = False
        self.n_models_ = len(self.models)
        self.feature_importances_ = None
        self.model_scores_ = {}
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseEnsemble':
        """
        训练集成模型
        
        Args:
            X: 训练特征
            y: 训练标签
            **kwargs: 其他参数
            
        Returns:
            训练后的集成模型
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """
        进行预测
        
        Args:
            X: 预测特征
            
        Returns:
            集成预测结果
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别概率（仅适用于分类任务）
        
        Args:
            X: 预测特征
            
        Returns:
            类别概率数组
        """
        result = self.predict(X)
        if result.prediction_probabilities is not None:
            return result.prediction_probabilities
        else:
            raise NotImplementedError("Probability prediction not available for this ensemble")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算模型得分
        
        Args:
            X: 特征
            y: 真实标签
            
        Returns:
            模型得分
        """
        result = self.predict(X)
        
        if self._is_classifier():
            return accuracy_score(y, result.predictions)
        else:
            return -mean_squared_error(y, result.predictions)
    
    def _is_classifier(self) -> bool:
        """判断是否为分类器"""
        return any(hasattr(model, 'predict_proba') or 
                  isinstance(model, ClassifierMixin) for model in self.models)
    
    def _is_regressor(self) -> bool:
        """检查是否为回归任务"""
        return any(isinstance(model, RegressorMixin) for model in self.models)
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        验证输入数据
        
        Args:
            X: 特征数据
            y: 目标变量
            
        Returns:
            验证后的X和y
            
        Raises:
            ValueError: 当输入数据无效时
        """
        # 转换为numpy数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # 检查数据形状
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got {y.ndim}D")
        
        # 检查样本数量一致性
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of samples. "
                           f"Got X: {X.shape[0]}, y: {y.shape[0]}")
        
        # 检查是否有空数据
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature")
        
        # 检查是否有无效值
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values")
        if np.any(np.isnan(y)):
            raise ValueError("y contains NaN values")
        if np.any(np.isinf(y)):
            raise ValueError("y contains infinite values")
        
        return X, y
    
    def _validate_prediction_input(self, X: np.ndarray) -> np.ndarray:
        """
        验证预测输入数据
        
        Args:
            X: 特征数据
            
        Returns:
            验证后的X
            
        Raises:
            ValueError: 当输入数据无效时
        """
        # 转换为numpy数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # 检查数据形状
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim}D")
        
        # 检查是否有空数据
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature")
        
        # 检查是否有无效值
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values")
        
        return X
    
    def get_individual_predictions(self, X: np.ndarray) -> List[np.ndarray]:
        """
        获取各个模型的预测结果
        
        Args:
            X: 预测特征
            
        Returns:
            各模型预测结果列表
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        return predictions
    
    def get_individual_probabilities(self, X: np.ndarray) -> List[np.ndarray]:
        """
        获取各个模型的概率预测结果
        
        Args:
            X: 预测特征
            
        Returns:
            各模型概率预测结果列表
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        probabilities = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)
                probabilities.append(prob)
            else:
                # 对于不支持概率预测的模型，返回None
                probabilities.append(None)
        
        return probabilities
    
    def evaluate_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估各个模型的性能
        
        Args:
            X: 特征
            y: 真实标签
            
        Returns:
            各模型性能字典
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before evaluation")
        
        scores = {}
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            try:
                score = model.score(X, y)
                scores[name] = score
            except Exception as e:
                logger.warning(f"Failed to evaluate model {name}: {e}")
                scores[name] = float('-inf')
        
        self.model_scores_ = scores
        return scores
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        获取特征重要性
        
        Returns:
            特征重要性数组
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before getting feature importances")
        
        importances = []
        weights = []
        
        for model in self.models:
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
                weights.append(1.0)
            elif hasattr(model, 'coef_'):
                # 对于线性模型，使用系数的绝对值
                coef = model.coef_
                if coef.ndim > 1:
                    coef = np.mean(np.abs(coef), axis=0)
                else:
                    coef = np.abs(coef)
                importances.append(coef)
                weights.append(1.0)
        
        if importances:
            # 加权平均特征重要性
            importances = np.array(importances)
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            self.feature_importances_ = np.average(importances, axis=0, weights=weights)
            return self.feature_importances_
        else:
            return None
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        交叉验证评估
        
        Args:
            X: 特征
            y: 标签
            cv: 交叉验证折数
            
        Returns:
            交叉验证结果
        """
        scores = cross_val_score(self, X, y, cv=cv, n_jobs=self.n_jobs)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'scores': scores.tolist()
        }
    
    def get_model_diversity(self, X: np.ndarray) -> Dict[str, float]:
        """
        计算模型多样性指标
        
        Args:
            X: 特征
            
        Returns:
            多样性指标字典
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before calculating diversity")
        
        predictions = self.get_individual_predictions(X)
        predictions = np.array(predictions).T  # shape: (n_samples, n_models)
        
        diversity_metrics = {}
        
        # 计算预测方差（适用于回归）
        if self._is_regressor():
            pred_variance = np.var(predictions, axis=1)
            diversity_metrics['prediction_variance'] = np.mean(pred_variance)
            diversity_metrics['avg_pairwise_difference'] = self._calculate_pairwise_difference(predictions)
        
        # 计算不一致性（适用于分类）
        if self._is_classifier():
            diversity_metrics['disagreement_rate'] = self._calculate_disagreement_rate(predictions)
            diversity_metrics['entropy'] = self._calculate_prediction_entropy(predictions)
        
        # 计算相关性
        correlations = []
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                corr = np.corrcoef(predictions[:, i], predictions[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            diversity_metrics['avg_correlation'] = np.mean(correlations)
            diversity_metrics['max_correlation'] = np.max(correlations)
            diversity_metrics['min_correlation'] = np.min(correlations)
        
        return diversity_metrics
    
    def _calculate_pairwise_difference(self, predictions: np.ndarray) -> float:
        """计算成对差异"""
        n_samples, n_models = predictions.shape
        total_diff = 0
        count = 0
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                diff = np.mean(np.abs(predictions[:, i] - predictions[:, j]))
                total_diff += diff
                count += 1
        
        return total_diff / count if count > 0 else 0
    
    def _calculate_disagreement_rate(self, predictions: np.ndarray) -> float:
        """计算不一致率"""
        n_samples, n_models = predictions.shape
        disagreements = 0
        
        for i in range(n_samples):
            sample_preds = predictions[i, :]
            unique_preds = np.unique(sample_preds)
            if len(unique_preds) > 1:
                disagreements += 1
        
        return disagreements / n_samples
    
    def _calculate_prediction_entropy(self, predictions: np.ndarray) -> float:
        """计算预测熵"""
        n_samples, n_models = predictions.shape
        entropies = []
        
        for i in range(n_samples):
            sample_preds = predictions[i, :]
            unique, counts = np.unique(sample_preds, return_counts=True)
            probs = counts / n_models
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            entropies.append(entropy)
        
        return np.mean(entropies)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'n_models': self.n_models_,
            'model_names': self.model_names,
            'model_types': [type(model).__name__ for model in self.models],
            'is_fitted': self.is_fitted_,
            'is_classifier': self._is_classifier(),
            'is_regressor': self._is_regressor()
        }
        
        if self.model_scores_:
            info['model_scores'] = self.model_scores_
        
        if self.feature_importances_ is not None:
            info['has_feature_importances'] = True
            info['n_features'] = len(self.feature_importances_)
        
        return info
    
    def explain_predictions(self,
                           X: np.ndarray,
                           method: str = 'shap',
                           feature_names: Optional[List[str]] = None,
                           background_data: Optional[np.ndarray] = None,
                           **kwargs) -> Any:
        """
        解释模型预测
        
        Args:
            X: 输入数据
            method: 解释方法 ('shap', 'feature_importance', 'both')
            feature_names: 特征名称
            background_data: 背景数据（用于SHAP）
            **kwargs: 其他参数
            
        Returns:
            解释结果
        """
        if not EXPLAINABILITY_AVAILABLE:
            raise ImportError("Explainability module not available. Please install required dependencies.")
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before explanation")
        
        # 验证解释方法
        valid_methods = ['shap', 'feature_importance', 'both']
        if method not in valid_methods:
            raise ValueError(f"Unknown explanation method: {method}. Valid methods are: {valid_methods}")
        
        X = self._validate_prediction_input(X)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        results = {}
        
        if method in ['shap', 'both']:
            try:
                shap_explainer = SHAPExplainer(
                    model=self,
                    background_data=background_data if background_data is not None else X[:100],
                    feature_names=feature_names
                )
                results['shap'] = shap_explainer.explain(X, **kwargs)
                logger.info("SHAP explanation completed")
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                results['shap'] = None
        
        if method in ['feature_importance', 'both']:
            try:
                fi_explainer = FeatureImportanceExplainer(
                    model=self,
                    feature_names=feature_names
                )
                results['feature_importance'] = fi_explainer.explain(X, **kwargs)
                logger.info("Feature importance explanation completed")
            except Exception as e:
                logger.warning(f"Feature importance explanation failed: {e}")
                results['feature_importance'] = None
        
        if method == 'shap':
            return results['shap']
        elif method == 'feature_importance':
            return results['feature_importance']
        else:
            return results
    
    def visualize_explanations(self,
                              explanation_result: Any,
                              plot_type: str = 'feature_importance',
                              save_path: Optional[str] = None,
                              **kwargs) -> Any:
        """
        可视化解释结果
        
        Args:
            explanation_result: 解释结果
            plot_type: 图形类型
            save_path: 保存路径
            **kwargs: 其他参数
            
        Returns:
            matplotlib Figure对象
        """
        if not EXPLAINABILITY_AVAILABLE:
            raise ImportError("Explainability module not available. Please install required dependencies.")
        
        # 分离可视化器构造参数和绘图参数
        viz_kwargs = {k: v for k, v in kwargs.items() 
                     if k in ['style', 'figsize', 'dpi', 'color_palette']}
        plot_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['style', 'figsize', 'dpi', 'color_palette']}
        
        visualizer = ExplanationVisualizer(**viz_kwargs)
        
        if plot_type == 'feature_importance':
            return visualizer.plot_feature_importance(explanation_result, save_path=save_path, **plot_kwargs)
        elif plot_type == 'shap_summary':
            return visualizer.plot_shap_summary(explanation_result, save_path=save_path, **plot_kwargs)
        elif plot_type == 'dashboard':
            return visualizer.create_explanation_dashboard(explanation_result, save_path=save_path, **plot_kwargs)
        elif plot_type == 'importance_comparison':
            return visualizer.plot_importance_comparison(explanation_result, save_path=save_path, **plot_kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(n_models={self.n_models_}, fitted={self.is_fitted_})"
    
    def __len__(self) -> int:
        """返回模型数量"""
        return self.n_models_