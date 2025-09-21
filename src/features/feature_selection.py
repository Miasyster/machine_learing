"""
特征筛选模块

实现多种特征选择方法：
1. 相关性过滤
2. 基于模型的重要性筛选
3. LASSO特征选择
4. SHAP特征重要性分析

作者: AI Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings
from abc import ABC, abstractmethod

# 机器学习库
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, RFECV, SelectFromModel
)
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

# 可选导入
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class FeatureSelector(ABC):
    """特征选择器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.selected_features_ = None
        self.feature_scores_ = None
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'FeatureSelector':
        """拟合特征选择器"""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用特征选择"""
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
        """拟合并转换"""
        return self.fit(X, y, **kwargs).transform(X)
    
    def get_selected_features(self) -> List[str]:
        """获取选中的特征名称"""
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted first")
        return self.selected_features_
    
    def get_feature_scores(self) -> Dict[str, float]:
        """获取特征评分"""
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted first")
        return self.feature_scores_


class CorrelationSelector(FeatureSelector):
    """相关性过滤器"""
    
    def __init__(self, threshold: float = 0.95, method: str = 'pearson'):
        """
        初始化相关性过滤器
        
        Args:
            threshold: 相关性阈值，超过此值的特征对将被过滤
            method: 相关性计算方法 ('pearson', 'spearman', 'kendall')
        """
        super().__init__("CorrelationFilter")
        self.threshold = threshold
        self.method = method
        self.correlation_matrix_ = None
        self.dropped_features_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'CorrelationSelector':
        """
        拟合相关性过滤器
        
        Args:
            X: 特征数据
            y: 目标变量（可选，用于保留与目标相关性更高的特征）
        """
        # 检查输入数据
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        
        # 计算特征间相关性矩阵
        self.correlation_matrix_ = X.corr(method=self.method)
        
        # 找到高相关性的特征对
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix_.columns)):
            for j in range(i+1, len(self.correlation_matrix_.columns)):
                if abs(self.correlation_matrix_.iloc[i, j]) > self.threshold:
                    feat1 = self.correlation_matrix_.columns[i]
                    feat2 = self.correlation_matrix_.columns[j]
                    corr_val = self.correlation_matrix_.iloc[i, j]
                    high_corr_pairs.append((feat1, feat2, corr_val))
        
        # 决定保留哪个特征
        self.dropped_features_ = set()
        
        for feat1, feat2, corr_val in high_corr_pairs:
            if feat1 in self.dropped_features_ or feat2 in self.dropped_features_:
                continue
                
            # 如果有目标变量，保留与目标相关性更高的特征
            if y is not None:
                corr_with_target_1 = abs(X[feat1].corr(y))
                corr_with_target_2 = abs(X[feat2].corr(y))
                
                if corr_with_target_1 >= corr_with_target_2:
                    self.dropped_features_.add(feat2)
                else:
                    self.dropped_features_.add(feat1)
            else:
                # 否则随机选择一个删除（这里选择第二个）
                self.dropped_features_.add(feat2)
        
        # 设置选中的特征
        self.selected_features_ = [col for col in X.columns if col not in self.dropped_features_]
        
        # 计算特征评分（与目标的相关性）
        if y is not None:
            self.feature_scores_ = {
                col: abs(X[col].corr(y)) for col in self.selected_features_
            }
        else:
            # 使用平均相关性作为评分
            self.feature_scores_ = {
                col: abs(self.correlation_matrix_[col]).mean() 
                for col in self.selected_features_
            }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用相关性过滤"""
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted first")
        
        # 检查特征是否存在
        missing_features = [f for f in self.selected_features_ if f not in X.columns]
        if missing_features:
            raise ValueError(f"Features not found in X: {missing_features}")
        
        return X[self.selected_features_]


class ModelBasedSelector(FeatureSelector):
    """基于模型的特征重要性选择器"""
    
    def __init__(self, 
                 model: Optional[Any] = None,
                 n_features: Optional[int] = None,
                 threshold: Optional[float] = None,
                 cv_folds: int = 5):
        """
        初始化模型基础选择器
        
        Args:
            model: 用于特征选择的模型
            n_features: 选择的特征数量
            threshold: 重要性阈值
            cv_folds: 交叉验证折数
        """
        super().__init__("ModelBasedSelector")
        self.model = model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.n_features = n_features
        self.threshold = threshold
        self.cv_folds = cv_folds
        self.feature_importances_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ModelBasedSelector':
        """拟合模型并计算特征重要性"""
        
        # 训练模型
        self.model.fit(X, y)
        
        # 获取特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute")
        
        # 创建特征重要性字典
        self.feature_importances_ = dict(zip(X.columns, importances))
        
        # 根据重要性选择特征
        if self.n_features is not None:
            # 选择前n个重要特征
            sorted_features = sorted(self.feature_importances_.items(), 
                                   key=lambda x: x[1], reverse=True)
            self.selected_features_ = [feat for feat, _ in sorted_features[:self.n_features]]
        elif self.threshold is not None:
            # 选择重要性超过阈值的特征
            self.selected_features_ = [
                feat for feat, imp in self.feature_importances_.items() 
                if imp >= self.threshold
            ]
        else:
            # 选择重要性大于平均值的特征
            mean_importance = np.mean(list(self.feature_importances_.values()))
            self.selected_features_ = [
                feat for feat, imp in self.feature_importances_.items() 
                if imp >= mean_importance
            ]
        
        # 设置特征评分
        self.feature_scores_ = {
            feat: self.feature_importances_[feat] 
            for feat in self.selected_features_
        }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用特征选择"""
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted first")
        
        # 检查特征是否存在
        missing_features = [f for f in self.selected_features_ if f not in X.columns]
        if missing_features:
            raise ValueError(f"Features not found in X: {missing_features}")
        
        return X[self.selected_features_]


class LassoSelector(FeatureSelector):
    """LASSO特征选择器"""
    
    def __init__(self, 
                 alpha: Optional[float] = None,
                 cv_folds: int = 5,
                 max_iter: int = 1000,
                 normalize: bool = True):
        """
        初始化LASSO选择器
        
        Args:
            alpha: 正则化参数，None时使用交叉验证选择
            cv_folds: 交叉验证折数
            max_iter: 最大迭代次数
            normalize: 是否标准化特征
        """
        super().__init__("LassoSelector")
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.normalize = normalize
        self.lasso_model_ = None
        self.scaler_ = None
        self.best_alpha_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LassoSelector':
        """拟合LASSO模型"""
        
        # 标准化特征
        if self.normalize:
            self.scaler_ = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler_.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X.copy()
        
        # 选择alpha参数
        if self.alpha is None:
            # 使用交叉验证选择最佳alpha
            lasso_cv = LassoCV(cv=self.cv_folds, max_iter=self.max_iter, random_state=42)
            lasso_cv.fit(X_scaled, y)
            self.best_alpha_ = lasso_cv.alpha_
            self.lasso_model_ = Lasso(alpha=self.best_alpha_, max_iter=self.max_iter)
        else:
            self.best_alpha_ = self.alpha
            self.lasso_model_ = Lasso(alpha=self.alpha, max_iter=self.max_iter)
        
        # 训练LASSO模型
        self.lasso_model_.fit(X_scaled, y)
        
        # 选择非零系数的特征
        non_zero_coefs = self.lasso_model_.coef_ != 0
        self.selected_features_ = X.columns[non_zero_coefs].tolist()
        
        # 设置特征评分（系数的绝对值）
        self.feature_scores_ = {
            feat: abs(coef) for feat, coef in zip(X.columns, self.lasso_model_.coef_)
            if coef != 0
        }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用LASSO特征选择"""
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted first")
        
        # 检查特征是否存在
        missing_features = [f for f in self.selected_features_ if f not in X.columns]
        if missing_features:
            raise ValueError(f"Features not found in X: {missing_features}")
        
        return X[self.selected_features_]


class SHAPSelector(FeatureSelector):
    """SHAP特征重要性选择器"""
    
    def __init__(self, 
                 model: Optional[Any] = None,
                 n_features: Optional[int] = None,
                 threshold: Optional[float] = None,
                 sample_size: int = 1000):
        """
        初始化SHAP选择器
        
        Args:
            model: 用于SHAP分析的模型
            n_features: 选择的特征数量
            threshold: SHAP值阈值
            sample_size: 用于SHAP分析的样本数量
        """
        super().__init__("SHAPSelector")
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAPSelector. Install with: pip install shap")
        
        self.model = model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.n_features = n_features
        self.threshold = threshold
        self.sample_size = sample_size
        self.shap_values_ = None
        self.explainer_ = None
        self.shap_importance_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'SHAPSelector':
        """拟合模型并计算SHAP值"""
        
        # 训练模型
        self.model.fit(X, y)
        
        # 采样数据用于SHAP分析
        if len(X) > self.sample_size:
            sample_idx = np.random.choice(len(X), self.sample_size, replace=False)
            X_sample = X.iloc[sample_idx]
        else:
            X_sample = X
        
        # 创建SHAP解释器
        try:
            # 尝试使用TreeExplainer（适用于树模型）
            if hasattr(self.model, 'estimators_') or 'Forest' in str(type(self.model)):
                self.explainer_ = shap.TreeExplainer(self.model)
            else:
                # 使用通用解释器
                self.explainer_ = shap.Explainer(self.model, X_sample)
        except Exception:
            # 回退到KernelExplainer
            self.explainer_ = shap.KernelExplainer(self.model.predict, X_sample)
        
        # 计算SHAP值
        self.shap_values_ = self.explainer_.shap_values(X_sample)
        
        # 计算特征重要性（SHAP值的平均绝对值）
        if isinstance(self.shap_values_, list):
            # 多分类情况，取第一个类别的SHAP值
            shap_importance = np.mean(np.abs(self.shap_values_[0]), axis=0)
        else:
            shap_importance = np.mean(np.abs(self.shap_values_), axis=0)
        
        self.shap_importance_ = dict(zip(X.columns, shap_importance))
        
        # 根据SHAP重要性选择特征
        if self.n_features is not None:
            # 选择前n个重要特征
            sorted_features = sorted(self.shap_importance_.items(), 
                                   key=lambda x: x[1], reverse=True)
            self.selected_features_ = [feat for feat, _ in sorted_features[:self.n_features]]
        elif self.threshold is not None:
            # 选择重要性超过阈值的特征
            self.selected_features_ = [
                feat for feat, imp in self.shap_importance_.items() 
                if imp >= self.threshold
            ]
        else:
            # 选择重要性大于平均值的特征
            mean_importance = np.mean(list(self.shap_importance_.values()))
            self.selected_features_ = [
                feat for feat, imp in self.shap_importance_.items() 
                if imp >= mean_importance
            ]
        
        # 设置特征评分
        self.feature_scores_ = {
            feat: self.shap_importance_[feat] 
            for feat in self.selected_features_
        }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用SHAP特征选择"""
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted first")
        
        # 检查特征是否存在
        missing_features = [f for f in self.selected_features_ if f not in X.columns]
        if missing_features:
            raise ValueError(f"Features not found in X: {missing_features}")
        
        return X[self.selected_features_]
    
    def plot_shap_summary(self, X: pd.DataFrame, max_display: int = 20):
        """绘制SHAP摘要图"""
        if not self.is_fitted_:
            raise ValueError("Selector must be fitted first")
        
        # 采样数据用于绘图
        if len(X) > self.sample_size:
            sample_idx = np.random.choice(len(X), self.sample_size, replace=False)
            X_sample = X.iloc[sample_idx]
        else:
            X_sample = X
        
        # 计算SHAP值
        shap_values = self.explainer_.shap_values(X_sample)
        
        # 绘制摘要图
        shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
        return shap_values


class FeatureSelectionManager:
    """特征选择管理器"""
    
    def __init__(self):
        self.selectors = {}
        self.results = {}
    
    def add_selector(self, name: str, selector: FeatureSelector):
        """添加特征选择器"""
        self.selectors[name] = selector
    
    def run_selection(self, 
                     X: pd.DataFrame, 
                     y: pd.Series,
                     selector_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """运行特征选择"""
        
        if selector_names is None:
            selector_names = list(self.selectors.keys())
        
        results = {}
        
        for name in selector_names:
            if name not in self.selectors:
                print(f"Warning: Selector '{name}' not found")
                continue
            
            print(f"Running {name}...")
            selector = self.selectors[name]
            
            try:
                # 运行特征选择
                selected_X = selector.fit_transform(X, y)
                
                # 保存结果
                results[name] = {
                    'selector': selector,
                    'selected_features': selector.get_selected_features(),
                    'feature_scores': selector.get_feature_scores(),
                    'n_features_selected': len(selector.get_selected_features()),
                    'n_features_original': len(X.columns)
                }
                
                print(f"  - Selected {len(selector.get_selected_features())} out of {len(X.columns)} features")
                
            except Exception as e:
                print(f"  - Error: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def get_feature_intersection(self, selector_names: Optional[List[str]] = None) -> List[str]:
        """获取多个选择器的特征交集"""
        if selector_names is None:
            selector_names = list(self.results.keys())
        
        if not selector_names:
            return []
        
        # 获取第一个选择器的特征
        feature_sets = []
        for name in selector_names:
            if name in self.results and 'selected_features' in self.results[name]:
                feature_sets.append(set(self.results[name]['selected_features']))
        
        if not feature_sets:
            return []
        
        # 计算交集
        intersection = feature_sets[0]
        for feature_set in feature_sets[1:]:
            intersection = intersection.intersection(feature_set)
        
        return list(intersection)
    
    def get_feature_union(self, selector_names: Optional[List[str]] = None) -> List[str]:
        """获取多个选择器的特征并集"""
        if selector_names is None:
            selector_names = list(self.results.keys())
        
        union = set()
        for name in selector_names:
            if name in self.results and 'selected_features' in self.results[name]:
                union = union.union(set(self.results[name]['selected_features']))
        
        return list(union)
    
    def create_feature_ranking(self, method: str = 'average') -> Dict[str, float]:
        """创建特征排名"""
        all_features = set()
        for result in self.results.values():
            if 'selected_features' in result:
                all_features.update(result['selected_features'])
        
        feature_rankings = {}
        
        for feature in all_features:
            scores = []
            for result in self.results.values():
                if 'feature_scores' in result and feature in result['feature_scores']:
                    scores.append(result['feature_scores'][feature])
            
            if scores:
                if method == 'average':
                    feature_rankings[feature] = np.mean(scores)
                elif method == 'max':
                    feature_rankings[feature] = np.max(scores)
                elif method == 'min':
                    feature_rankings[feature] = np.min(scores)
        
        return dict(sorted(feature_rankings.items(), key=lambda x: x[1], reverse=True))
    
    def print_summary(self):
        """打印选择结果摘要"""
        print("\n" + "="*60)
        print("特征选择结果摘要")
        print("="*60)
        
        for name, result in self.results.items():
            if 'error' in result:
                print(f"\n{name}: 错误 - {result['error']}")
                continue
            
            print(f"\n{name}:")
            print(f"  - 原始特征数: {result['n_features_original']}")
            print(f"  - 选中特征数: {result['n_features_selected']}")
            print(f"  - 特征保留率: {result['n_features_selected']/result['n_features_original']:.2%}")
            
            # 显示前5个重要特征
            if 'feature_scores' in result:
                top_features = sorted(result['feature_scores'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                print(f"  - 前5个重要特征:")
                for feat, score in top_features:
                    print(f"    * {feat}: {score:.4f}")


# 便捷函数
def correlation_filter(X: pd.DataFrame, 
                      y: pd.Series = None, 
                      threshold: float = 0.95,
                      method: str = 'pearson') -> pd.DataFrame:
    """相关性过滤便捷函数"""
    selector = CorrelationSelector(threshold=threshold, method=method)
    return selector.fit_transform(X, y)


def model_based_selection(X: pd.DataFrame, 
                         y: pd.Series,
                         model: Optional[Any] = None,
                         n_features: Optional[int] = None) -> pd.DataFrame:
    """模型基础选择便捷函数"""
    selector = ModelBasedSelector(model=model, n_features=n_features)
    return selector.fit_transform(X, y)


def lasso_selection(X: pd.DataFrame, 
                   y: pd.Series,
                   alpha: Optional[float] = None) -> pd.DataFrame:
    """LASSO选择便捷函数"""
    selector = LassoSelector(alpha=alpha)
    return selector.fit_transform(X, y)


def shap_selection(X: pd.DataFrame, 
                  y: pd.Series,
                  model: Optional[Any] = None,
                  n_features: Optional[int] = None) -> pd.DataFrame:
    """SHAP选择便捷函数"""
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required. Install with: pip install shap")
    
    selector = SHAPSelector(model=model, n_features=n_features)
    return selector.fit_transform(X, y)


if __name__ == "__main__":
    # 示例用法
    print("特征选择模块已加载")
    
    # 生成示例数据
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    # 创建一些相关特征
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    
    # 添加一些高相关性特征
    X['feature_20'] = X['feature_0'] + np.random.randn(n_samples) * 0.1
    X['feature_21'] = X['feature_1'] + np.random.randn(n_samples) * 0.1
    
    # 创建目标变量（与前几个特征相关）
    y = (X['feature_0'] * 2 + X['feature_1'] * 1.5 + X['feature_2'] * 1 + 
         np.random.randn(n_samples) * 0.5)
    
    print(f"\n原始数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 创建特征选择管理器
    manager = FeatureSelectionManager()
    
    # 添加不同的选择器
    manager.add_selector('correlation', CorrelationSelector(threshold=0.8))
    manager.add_selector('random_forest', ModelBasedSelector(n_features=10))
    manager.add_selector('lasso', LassoSelector())
    
    # 运行特征选择
    results = manager.run_selection(X, y)
    
    # 打印摘要
    manager.print_summary()
    
    # 获取特征交集和并集
    intersection = manager.get_feature_intersection()
    union = manager.get_feature_union()
    
    print(f"\n特征交集 ({len(intersection)} 个): {intersection}")
    print(f"特征并集 ({len(union)} 个): {union}")
    
    # 创建特征排名
    ranking = manager.create_feature_ranking()
    print(f"\n综合特征排名 (前10个):")
    for i, (feat, score) in enumerate(list(ranking.items())[:10]):
        print(f"  {i+1}. {feat}: {score:.4f}")
    
    print("\n特征选择模块测试完成")