"""
特征重要性分析器
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings

from .base import BaseExplainer, ExplanationResult

logger = logging.getLogger(__name__)


class FeatureImportanceExplainer(BaseExplainer):
    """特征重要性解释器"""
    
    def __init__(self,
                 model: BaseEstimator,
                 feature_names: Optional[List[str]] = None,
                 importance_methods: List[str] = None,
                 n_repeats: int = 10,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化特征重要性解释器
        
        Args:
            model: 要解释的模型
            feature_names: 特征名称列表
            importance_methods: 重要性计算方法列表
            n_repeats: 排列重要性的重复次数
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(model, feature_names, random_state, verbose)
        
        if importance_methods is None:
            importance_methods = ['built_in', 'permutation']
        
        self.importance_methods = importance_methods
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        
        # 支持的重要性方法
        self.supported_methods = {
            'built_in': self._get_builtin_importance,
            'permutation': self._get_permutation_importance,
            'drop_column': self._get_drop_column_importance,
            'correlation': self._get_correlation_importance,
            'mutual_info': self._get_mutual_info_importance
        }
    
    def _is_classifier(self) -> bool:
        """检查模型是否为分类器"""
        # 检查是否有classes_属性（分类器特有）
        if hasattr(self.model, 'classes_'):
            return True
        
        # 检查模型类型
        classifier_types = [
            'classifier', 'classification'
        ]
        
        model_name = self.model.__class__.__name__.lower()
        return any(cls_type in model_name for cls_type in classifier_types)
        
        # 验证方法
        for method in self.importance_methods:
            if method not in self.supported_methods:
                raise ValueError(f"Unsupported importance method: {method}")
    
    def explain(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> ExplanationResult:
        """
        解释特征重要性
        
        Args:
            X: 输入特征
            y: 目标变量（某些方法需要）
            **kwargs: 其他参数
            
        Returns:
            ExplanationResult: 解释结果
        """
        # 检查是否需要目标变量
        methods_requiring_y = ['permutation', 'drop_column', 'mutual_info', 'correlation']
        if any(method in self.importance_methods for method in methods_requiring_y) and y is None:
            raise ValueError("Target variable y is required for permutation importance")
        
        if self.verbose:
            logger.info(f"Computing feature importance using methods: {self.importance_methods}")
        
        # 存储不同方法的重要性结果
        importance_results = {}
        
        for method in self.importance_methods:
            try:
                if self.verbose:
                    logger.info(f"Computing {method} importance")
                
                importance_func = self.supported_methods[method]
                importance_scores = importance_func(X, y, **kwargs)
                importance_results[method] = importance_scores
                
            except Exception as e:
                logger.warning(f"Failed to compute {method} importance: {e}")
                importance_results[method] = None
        
        # 选择主要的重要性分数（优先级：built_in > permutation > others）
        primary_importance = None
        primary_method = None
        
        for method in ['built_in', 'permutation', 'drop_column', 'mutual_info', 'correlation']:
            if method in importance_results and importance_results[method] is not None:
                if isinstance(importance_results[method], dict) and 'importance' in importance_results[method]:
                    primary_importance = importance_results[method]['importance']
                else:
                    primary_importance = importance_results[method]
                primary_method = method
                break
        
        # 创建解释结果
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0]) if X else 0
        feature_names = self.feature_names or [f'feature_{i}' for i in range(n_features)]
        
        result = ExplanationResult(
            feature_names=feature_names,
            model_type=self.model_type,
            explanation_method='feature_importance',
            feature_importance=primary_importance,
            metadata={
                'primary_method': primary_method,
                'all_methods': list(importance_results.keys()),
                'n_samples': len(X),
                'n_features': n_features
            }
        )
        
        # 添加排列重要性的标准差（如果可用）
        if 'permutation' in importance_results and importance_results['permutation'] is not None:
            perm_result = importance_results['permutation']
            if isinstance(perm_result, dict):
                result.permutation_importance = perm_result['importance']
                result.feature_importance_std = perm_result.get('std')
        
        # 添加所有方法的结果到可视化数据
        result.visualization_data = {
            'importance_methods': importance_results,
            'method_comparison': self._compare_methods(importance_results)
        }
        
        self.is_fitted_ = True
        return result
    
    def _get_builtin_importance(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> Optional[np.ndarray]:
        """获取模型内置的特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # 对于线性模型，使用系数的绝对值
            coef = self.model.coef_
            if coef.ndim > 1:
                # 多类分类的情况
                return np.abs(coef).mean(axis=0)
            else:
                return np.abs(coef)
        else:
            logger.warning("Model does not have built-in feature importance")
            return None
    
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

    def _get_permutation_importance(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """计算排列重要性"""
        if y is None:
            raise ValueError("Target variable y is required for permutation importance")
        
        # 确定评分方法
        if self._is_classifier():
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        # 创建一个包装模型来处理EnsembleResult
        class ModelWrapper:
            def __init__(self, model, predict_func):
                self.model = model
                self.predict_func = predict_func
                
            def predict(self, X):
                return self.predict_func(X)
                
            def fit(self, X, y):
                # 包装器不需要重新训练
                return self
                
            def score(self, X, y):
                predictions = self.predict(X)
                if hasattr(self.model, 'classes_') or (hasattr(self.model, '_is_classifier') and self.model._is_classifier()):
                    from sklearn.metrics import accuracy_score
                    return accuracy_score(y, predictions)
                else:
                    from sklearn.metrics import r2_score
                    return r2_score(y, predictions)
        
        predict_func = self._create_predict_wrapper()
        wrapped_model = ModelWrapper(self.model, predict_func)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            perm_importance = permutation_importance(
                wrapped_model, X, y,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                scoring=scoring
            )
        
        return {
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std,
            'all_scores': perm_importance.importances
        }
    
    def _get_drop_column_importance(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """计算删除列重要性"""
        if y is None:
            raise ValueError("Target variable y is required for drop column importance")
        
        # 获取基准分数
        baseline_score = self._get_model_score(X, y)
        
        importance_scores = []
        
        for i in range(X.shape[1]):
            # 创建删除第i列的数据
            X_dropped = np.delete(X, i, axis=1)
            
            # 训练新模型（使用相同参数）
            try:
                from sklearn.base import clone
                temp_model = clone(self.model)
                temp_model.fit(X_dropped, y)
                
                # 计算分数
                dropped_score = self._get_model_score(X_dropped, y, temp_model)
                
                # 重要性 = 基准分数 - 删除列后的分数
                importance = baseline_score - dropped_score
                importance_scores.append(importance)
                
            except Exception as e:
                logger.warning(f"Error computing drop column importance for feature {i}: {e}")
                importance_scores.append(0.0)
        
        return np.array(importance_scores)
    
    def _get_correlation_importance(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """计算相关性重要性"""
        if y is None:
            raise ValueError("Target variable y is required for correlation importance")
        
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
        
        return np.array(correlations)
    
    def _get_mutual_info_importance(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """计算互信息重要性"""
        if y is None:
            raise ValueError("Target variable y is required for mutual info importance")
        
        try:
            if self._is_classifier():
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            else:
                from sklearn.feature_selection import mutual_info_regression
                mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
            
            return mi_scores
            
        except ImportError:
            logger.warning("Scikit-learn feature_selection module not available")
            return np.zeros(X.shape[1])
    
    def _get_model_score(self, X: np.ndarray, y: np.ndarray, model: BaseEstimator = None) -> float:
        """获取模型分数"""
        if model is None:
            model = self.model
        
        predictions = model.predict(X)
        
        if self._is_classifier():
            return accuracy_score(y, predictions)
        else:
            return r2_score(y, predictions)
    
    def _compare_methods(self, importance_results: Dict[str, Any]) -> Dict[str, Any]:
        """比较不同重要性方法的结果"""
        valid_results = {}
        
        # 提取有效的重要性分数
        for method, result in importance_results.items():
            if result is not None:
                if isinstance(result, dict) and 'importance' in result:
                    valid_results[method] = result['importance']
                elif isinstance(result, np.ndarray):
                    valid_results[method] = result
        
        if len(valid_results) < 2:
            return {'message': 'Not enough methods for comparison'}
        
        # 计算方法间的相关性
        correlations = {}
        methods = list(valid_results.keys())
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                corr = np.corrcoef(valid_results[method1], valid_results[method2])[0, 1]
                correlations[f"{method1}_vs_{method2}"] = corr if not np.isnan(corr) else 0.0
        
        # 找到每个方法中最重要的特征
        top_features = {}
        for method, scores in valid_results.items():
            top_idx = np.argmax(scores)
            top_features[method] = {
                'feature': self.feature_names[top_idx],
                'score': scores[top_idx],
                'rank': 1
            }
        
        return {
            'correlations': correlations,
            'top_features_by_method': top_features,
            'agreement_score': np.mean(list(correlations.values()))
        }
    
    def explain_instance(self, X: np.ndarray, instance_idx: int, **kwargs) -> Dict[str, Any]:
        """
        解释单个实例（对于特征重要性，返回该实例的特征值和全局重要性）
        
        Args:
            X: 输入数据
            instance_idx: 实例索引
            **kwargs: 其他参数
            
        Returns:
            单个实例的解释结果
        """
        if instance_idx >= len(X):
            raise ValueError(f"Instance index {instance_idx} out of range")
        
        # 获取全局重要性
        y = kwargs.get('y')
        global_result = self.explain(X, y, **kwargs)
        
        instance_values = X[instance_idx]
        
        # 计算特征贡献（特征值 × 重要性）
        if global_result.feature_importance is not None:
            feature_contributions = instance_values * global_result.feature_importance
            
            # 排序特征贡献
            contributions_with_names = list(zip(self.feature_names, feature_contributions, instance_values))
            contributions_with_names.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            contributions_with_names = []
        
        # 获取预测
        prediction = self.model.predict(X[instance_idx:instance_idx+1])[0]
        
        return {
            'instance_idx': instance_idx,
            'prediction': prediction,
            'feature_values': instance_values.tolist(),
            'feature_importance': global_result.feature_importance.tolist() if global_result.feature_importance is not None else None,
            'feature_contributions': feature_contributions.tolist() if global_result.feature_importance is not None else None,
            'ranked_contributions': contributions_with_names,
            'top_contributing_features': contributions_with_names[:5],
            'metadata': {
                'model_type': self.model_type,
                'importance_methods': self.importance_methods
            }
        }
    
    def get_feature_ranking(self, X: np.ndarray, y: np.ndarray = None, method: str = 'primary') -> List[Tuple[str, float, int]]:
        """
        获取特征排名
        
        Args:
            X: 输入数据
            y: 目标变量
            method: 排名方法
            
        Returns:
            特征排名列表 (特征名, 重要性分数, 排名)
        """
        result = self.explain(X, y)
        
        if method == 'primary' and result.feature_importance is not None:
            importance_scores = result.feature_importance
        elif method == 'permutation' and result.permutation_importance is not None:
            importance_scores = result.permutation_importance
        else:
            raise ValueError(f"Method '{method}' not available")
        
        # 创建排名
        sorted_indices = np.argsort(importance_scores)[::-1]
        
        ranking = []
        for rank, idx in enumerate(sorted_indices, 1):
            ranking.append((
                self.feature_names[idx],
                importance_scores[idx],
                rank
            ))
        
        return ranking
    
    def get_stability_analysis(self, X: np.ndarray, y: np.ndarray, n_bootstrap: int = 10) -> Dict[str, Any]:
        """
        分析特征重要性的稳定性
        
        Args:
            X: 输入数据
            y: 目标变量
            n_bootstrap: 自助采样次数
            
        Returns:
            稳定性分析结果
        """
        if 'permutation' not in self.importance_methods:
            logger.warning("Permutation importance not enabled, adding it for stability analysis")
            original_methods = self.importance_methods.copy()
            self.importance_methods.append('permutation')
        else:
            original_methods = None
        
        bootstrap_results = []
        
        for i in range(n_bootstrap):
            # 自助采样
            np.random.seed(self.random_state + i if self.random_state else None)
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # 计算重要性
            try:
                result = self.explain(X_boot, y_boot)
                if result.feature_importance is not None:
                    bootstrap_results.append(result.feature_importance)
            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {e}")
        
        # 恢复原始方法列表
        if original_methods is not None:
            self.importance_methods = original_methods
        
        if not bootstrap_results:
            return {'error': 'No successful bootstrap iterations'}
        
        bootstrap_results = np.array(bootstrap_results)
        
        # 计算统计量
        mean_importance = np.mean(bootstrap_results, axis=0)
        std_importance = np.std(bootstrap_results, axis=0)
        cv_importance = std_importance / (mean_importance + 1e-8)  # 变异系数
        
        # 特征稳定性排名
        stability_ranking = list(zip(self.feature_names, cv_importance))
        stability_ranking.sort(key=lambda x: x[1])  # 变异系数越小越稳定
        
        return {
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'coefficient_of_variation': cv_importance,
            'stability_ranking': stability_ranking,
            'most_stable_features': stability_ranking[:5],
            'least_stable_features': stability_ranking[-5:],
            'n_bootstrap': len(bootstrap_results)
        }