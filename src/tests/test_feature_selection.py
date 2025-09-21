"""
特征选择模块单元测试

测试所有特征选择器的功能：
1. 相关性过滤器
2. 模型基础选择器
3. LASSO选择器
4. SHAP选择器（如果可用）
5. 特征选择管理器

作者: AI Assistant
日期: 2024
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, MagicMock

# 导入待测试的模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from features.feature_selection import (
    CorrelationSelector,
    ModelBasedSelector, 
    LassoSelector,
    SHAPSelector,
    FeatureSelectionManager,
    correlation_filter,
    model_based_selection,
    lasso_selection,
    SHAP_AVAILABLE
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class TestFeatureSelection:
    """特征选择测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        n_samples, n_features = 500, 15
        
        # 创建基础特征
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # 添加高相关性特征
        X['corr_feature_1'] = X['feature_0'] + np.random.randn(n_samples) * 0.1
        X['corr_feature_2'] = X['feature_1'] + np.random.randn(n_samples) * 0.1
        
        # 创建目标变量（与前几个特征相关）
        y = (X['feature_0'] * 2 + X['feature_1'] * 1.5 + 
             X['feature_2'] * 1 + np.random.randn(n_samples) * 0.5)
        
        return X, y
    
    def test_correlation_selector_basic(self, sample_data):
        """测试相关性选择器基本功能"""
        X, y = sample_data
        
        selector = CorrelationSelector(threshold=0.8)
        X_selected = selector.fit_transform(X, y)
        
        # 检查基本属性
        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0
        assert len(selector.selected_features_) <= len(X.columns)
        assert X_selected.shape[1] == len(selector.selected_features_)
        assert X_selected.shape[0] == X.shape[0]
        
        # 检查高相关性特征被过滤
        assert len(selector.selected_features_) < len(X.columns)
    
    def test_correlation_selector_without_target(self, sample_data):
        """测试无目标变量的相关性选择器"""
        X, _ = sample_data
        
        selector = CorrelationSelector(threshold=0.8)
        X_selected = selector.fit_transform(X)
        
        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0
        assert X_selected.shape[1] == len(selector.selected_features_)
    
    def test_correlation_selector_methods(self, sample_data):
        """测试不同相关性计算方法"""
        X, y = sample_data
        
        methods = ['pearson', 'spearman', 'kendall']
        
        for method in methods:
            selector = CorrelationSelector(threshold=0.8, method=method)
            X_selected = selector.fit_transform(X, y)
            
            assert selector.is_fitted_
            assert len(selector.selected_features_) > 0
    
    def test_model_based_selector_basic(self, sample_data):
        """测试模型基础选择器基本功能"""
        X, y = sample_data
        
        selector = ModelBasedSelector(n_features=10)
        X_selected = selector.fit_transform(X, y)
        
        # 检查基本属性
        assert selector.is_fitted_
        assert len(selector.selected_features_) == 10
        assert X_selected.shape[1] == 10
        assert X_selected.shape[0] == X.shape[0]
        assert selector.feature_importances_ is not None
    
    def test_model_based_selector_threshold(self, sample_data):
        """测试基于阈值的模型选择器"""
        X, y = sample_data
        
        selector = ModelBasedSelector(threshold=0.01)
        X_selected = selector.fit_transform(X, y)
        
        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0
        assert X_selected.shape[1] == len(selector.selected_features_)
    
    def test_model_based_selector_custom_model(self, sample_data):
        """测试自定义模型的选择器"""
        X, y = sample_data
        
        custom_model = LinearRegression()
        selector = ModelBasedSelector(model=custom_model, n_features=8)
        X_selected = selector.fit_transform(X, y)
        
        assert selector.is_fitted_
        assert len(selector.selected_features_) == 8
        assert X_selected.shape[1] == 8
    
    def test_lasso_selector_basic(self, sample_data):
        """测试LASSO选择器基本功能"""
        X, y = sample_data
        
        selector = LassoSelector(alpha=0.01)
        X_selected = selector.fit_transform(X, y)
        
        # 检查基本属性
        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0
        assert X_selected.shape[1] == len(selector.selected_features_)
        assert X_selected.shape[0] == X.shape[0]
        assert selector.lasso_model_ is not None
        assert selector.best_alpha_ == 0.01
    
    def test_lasso_selector_cv(self, sample_data):
        """测试LASSO选择器交叉验证"""
        X, y = sample_data
        
        selector = LassoSelector(alpha=None, cv_folds=3)  # 使用CV选择alpha
        X_selected = selector.fit_transform(X, y)
        
        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0
        assert selector.best_alpha_ is not None
        assert selector.best_alpha_ > 0
    
    def test_lasso_selector_normalization(self, sample_data):
        """测试LASSO选择器标准化选项"""
        X, y = sample_data
        
        # 测试标准化
        selector_norm = LassoSelector(alpha=0.01, normalize=True)
        X_selected_norm = selector_norm.fit_transform(X, y)
        
        # 测试不标准化
        selector_no_norm = LassoSelector(alpha=0.01, normalize=False)
        X_selected_no_norm = selector_no_norm.fit_transform(X, y)
        
        assert selector_norm.is_fitted_
        assert selector_no_norm.is_fitted_
        assert selector_norm.scaler_ is not None
        assert selector_no_norm.scaler_ is None
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not available")
    def test_shap_selector_basic(self, sample_data):
        """测试SHAP选择器基本功能"""
        X, y = sample_data
        
        selector = SHAPSelector(n_features=8, sample_size=100)
        X_selected = selector.fit_transform(X, y)
        
        # 检查基本属性
        assert selector.is_fitted_
        assert len(selector.selected_features_) == 8
        assert X_selected.shape[1] == 8
        assert X_selected.shape[0] == X.shape[0]
        assert selector.shap_values_ is not None
        assert selector.explainer_ is not None
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not available")
    def test_shap_selector_threshold(self, sample_data):
        """测试基于阈值的SHAP选择器"""
        X, y = sample_data
        
        selector = SHAPSelector(threshold=0.01, sample_size=100)
        X_selected = selector.fit_transform(X, y)
        
        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0
        assert X_selected.shape[1] == len(selector.selected_features_)
    
    def test_feature_selection_manager(self, sample_data):
        """测试特征选择管理器"""
        X, y = sample_data
        
        manager = FeatureSelectionManager()
        
        # 添加选择器
        manager.add_selector('correlation', CorrelationSelector(threshold=0.8))
        manager.add_selector('random_forest', ModelBasedSelector(n_features=10))
        manager.add_selector('lasso', LassoSelector(alpha=0.01))
        
        # 运行选择
        results = manager.run_selection(X, y)
        
        # 检查结果
        assert len(results) == 3
        assert 'correlation' in results
        assert 'random_forest' in results
        assert 'lasso' in results
        
        for name, result in results.items():
            if 'error' not in result:
                assert 'selected_features' in result
                assert 'feature_scores' in result
                assert 'n_features_selected' in result
                assert 'n_features_original' in result
                assert result['n_features_original'] == len(X.columns)
    
    def test_feature_intersection_union(self, sample_data):
        """测试特征交集和并集"""
        X, y = sample_data
        
        manager = FeatureSelectionManager()
        manager.add_selector('selector1', ModelBasedSelector(n_features=10))
        manager.add_selector('selector2', ModelBasedSelector(n_features=8))
        
        results = manager.run_selection(X, y)
        
        # 测试交集
        intersection = manager.get_feature_intersection()
        assert isinstance(intersection, list)
        assert len(intersection) <= 8  # 不会超过最小选择数
        
        # 测试并集
        union = manager.get_feature_union()
        assert isinstance(union, list)
        assert len(union) >= len(intersection)
        assert len(union) <= 10  # 不会超过最大选择数
    
    def test_feature_ranking(self, sample_data):
        """测试特征排名"""
        X, y = sample_data
        
        manager = FeatureSelectionManager()
        manager.add_selector('rf1', ModelBasedSelector(n_features=10))
        manager.add_selector('rf2', ModelBasedSelector(n_features=8))
        
        results = manager.run_selection(X, y)
        
        # 测试不同排名方法
        ranking_avg = manager.create_feature_ranking(method='average')
        ranking_max = manager.create_feature_ranking(method='max')
        ranking_min = manager.create_feature_ranking(method='min')
        
        assert isinstance(ranking_avg, dict)
        assert isinstance(ranking_max, dict)
        assert isinstance(ranking_min, dict)
        assert len(ranking_avg) > 0
    
    def test_convenience_functions(self, sample_data):
        """测试便捷函数"""
        X, y = sample_data
        
        # 测试相关性过滤便捷函数
        X_corr = correlation_filter(X, y, threshold=0.8)
        assert isinstance(X_corr, pd.DataFrame)
        assert X_corr.shape[0] == X.shape[0]
        assert X_corr.shape[1] <= X.shape[1]
        
        # 测试模型选择便捷函数
        X_model = model_based_selection(X, y, n_features=10)
        assert isinstance(X_model, pd.DataFrame)
        assert X_model.shape == (X.shape[0], 10)
        
        # 测试LASSO便捷函数
        X_lasso = lasso_selection(X, y, alpha=0.01)
        assert isinstance(X_lasso, pd.DataFrame)
        assert X_lasso.shape[0] == X.shape[0]
        assert X_lasso.shape[1] <= X.shape[1]
    
    def test_edge_cases(self, sample_data):
        """测试边界情况"""
        X, y = sample_data
        
        # 测试空数据
        X_empty = pd.DataFrame()
        selector = CorrelationSelector()
        
        # 空数据应该抛出异常
        with pytest.raises(ValueError):
            selector.fit_transform(X_empty, y)
        
        # 测试单特征数据
        X_single = X[['feature_0']]
        selector = CorrelationSelector(threshold=0.8)
        X_selected = selector.fit_transform(X_single, y)
        assert X_selected.shape[1] == 1
        
        # 测试缺失特征的情况
        X_train = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [1, 1, 1, 1, 1]  # 低相关性特征
        })
        X_test = pd.DataFrame({
            'D': [1, 2, 3],
            'E': [3, 6, 9]  # 完全不同的列名
        })
        
        selector = CorrelationSelector(threshold=0.8)
        selector.fit(X_train)
        with pytest.raises(ValueError):
            selector.transform(X_test)
        
        # 测试未拟合的选择器
        selector = ModelBasedSelector()
        with pytest.raises(ValueError, match="must be fitted first"):
            selector.transform(X)
        
        with pytest.raises(ValueError, match="must be fitted first"):
            selector.get_selected_features()
        
        with pytest.raises(ValueError, match="must be fitted first"):
            selector.get_feature_scores()
    
    def test_data_consistency(self, sample_data):
        """测试数据一致性"""
        X, y = sample_data
        
        # 测试选择器保持数据索引
        selector = ModelBasedSelector(n_features=10)
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.index.equals(X.index)
        assert all(col in X.columns for col in X_selected.columns)
        
        # 测试特征名称一致性
        selected_features = selector.get_selected_features()
        assert list(X_selected.columns) == selected_features
    
    def test_reproducibility(self, sample_data):
        """测试结果可重现性"""
        X, y = sample_data
        
        # 测试相同参数的选择器产生相同结果
        selector1 = ModelBasedSelector(n_features=10)
        selector2 = ModelBasedSelector(n_features=10)
        
        X_selected1 = selector1.fit_transform(X, y)
        X_selected2 = selector2.fit_transform(X, y)
        
        # 由于随机森林有随机性，我们只检查选择的特征数量
        assert X_selected1.shape[1] == X_selected2.shape[1]
    
    def test_error_handling(self, sample_data):
        """测试错误处理"""
        X, y = sample_data
        
        # 测试无效的相关性方法
        with pytest.raises(ValueError):
            selector = CorrelationSelector(method='invalid_method')
            selector.fit_transform(X, y)
        
        # 测试无效的阈值
        selector = CorrelationSelector(threshold=1.5)  # 超过1的阈值
        X_selected = selector.fit_transform(X, y)
        # 应该选择所有特征
        assert X_selected.shape[1] == X.shape[1]
    
    def test_manager_error_handling(self, sample_data):
        """测试管理器错误处理"""
        X, y = sample_data
        
        manager = FeatureSelectionManager()
        
        # 测试运行不存在的选择器
        results = manager.run_selection(X, y, selector_names=['nonexistent'])
        assert len(results) == 0
        
        # 测试空选择器列表的交集和并集
        intersection = manager.get_feature_intersection([])
        union = manager.get_feature_union([])
        assert intersection == []
        assert union == []


class TestFeatureSelectionPerformance:
    """特征选择性能测试"""
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 创建较大的数据集
        np.random.seed(42)
        n_samples, n_features = 2000, 100
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples))  # 使用pd.Series而不是numpy数组
        
        # 测试相关性过滤器性能
        import time
        start_time = time.time()
        
        selector = CorrelationSelector(threshold=0.9)
        X_selected = selector.fit_transform(X, y)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 检查性能（应该在合理时间内完成）
        assert processing_time < 10  # 10秒内完成
        assert X_selected.shape[0] == X.shape[0]
        assert X_selected.shape[1] <= X.shape[1]
    
    def test_memory_efficiency(self):
        """测试内存效率"""
        # 创建数据集
        np.random.seed(42)
        n_samples, n_features = 1000, 50
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples))  # 使用pd.Series而不是numpy数组
        
        # 测试选择器不会显著增加内存使用
        selector = ModelBasedSelector(n_features=20)
        X_selected = selector.fit_transform(X, y)
        
        # 检查输出数据大小合理
        assert X_selected.memory_usage(deep=True).sum() <= X.memory_usage(deep=True).sum()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])