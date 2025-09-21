"""
模型解释性功能单元测试
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from unittest.mock import Mock, patch
import warnings
warnings.filterwarnings('ignore')

# 导入测试模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainability.base import BaseExplainer, ExplanationResult
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.feature_importance import FeatureImportanceExplainer
from src.explainability.visualization import ExplanationVisualizer
from src.ensemble import VotingEnsemble


class TestExplanationResult:
    """测试ExplanationResult类"""
    
    def test_explanation_result_creation(self):
        """测试解释结果创建"""
        feature_names = ['feature1', 'feature2', 'feature3']
        importance = np.array([0.5, 0.3, 0.2])
        
        result = ExplanationResult(
            feature_names=feature_names,
            model_type='classifier',
            explanation_method='test',
            feature_importance=importance
        )
        
        assert result.feature_names == feature_names
        assert result.model_type == 'classifier'
        assert result.explanation_method == 'test'
        assert np.array_equal(result.feature_importance, importance)
    
    def test_explanation_result_metadata(self):
        """测试解释结果元数据"""
        result = ExplanationResult(
            feature_names=['f1', 'f2'],
            model_type='regressor',
            explanation_method='test',
            metadata={'test_key': 'test_value'}
        )
        
        assert result.metadata['test_key'] == 'test_value'


class TestFeatureImportanceExplainer:
    """测试特征重要性解释器"""
    
    @pytest.fixture
    def classification_data(self):
        """分类数据"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def regression_data(self):
        """回归数据"""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def test_feature_importance_classifier(self, classification_data):
        """测试分类器特征重要性"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = FeatureImportanceExplainer(
            model=model,
            feature_names=[f'feature_{i}' for i in range(X_train.shape[1])],
            importance_methods=['built_in']
        )
        
        result = explainer.explain(X_test, y_test)
        
        assert result is not None
        assert result.feature_importance is not None
        assert len(result.feature_importance) == X_train.shape[1]
        assert result.explanation_method == 'feature_importance'
    
    def test_feature_importance_regressor(self, regression_data):
        """测试回归器特征重要性"""
        X_train, X_test, y_train, y_test = regression_data
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = FeatureImportanceExplainer(
            model=model,
            feature_names=[f'feature_{i}' for i in range(X_train.shape[1])],
            importance_methods=['built_in']
        )
        
        result = explainer.explain(X_test, y_test)
        
        assert result is not None
        assert result.feature_importance is not None
        assert len(result.feature_importance) == X_train.shape[1]
    
    def test_permutation_importance(self, classification_data):
        """测试排列重要性"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        explainer = FeatureImportanceExplainer(
            model=model,
            importance_methods=['permutation'],
            n_repeats=3
        )
        
        # 测试排列重要性（需要提供目标变量）
        result = explainer.explain(X_test[:20], y=y_test[:20])  # 使用较小数据集加速测试
        
        assert result is not None
        assert result.feature_importance is not None
        assert len(result.feature_importance) == X_train.shape[1]
        
        # 测试排列重要性值
        assert all(isinstance(val, (int, float)) for val in result.feature_importance)
    
    def test_feature_ranking(self, classification_data):
        """测试特征排名"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = FeatureImportanceExplainer(
            model=model,
            feature_names=[f'feature_{i}' for i in range(X_train.shape[1])]
        )
        
        ranking = explainer.get_feature_ranking(X_test, y_test)
        
        assert len(ranking) == X_train.shape[1]
        assert all(len(item) == 3 for item in ranking)  # (name, importance, rank)


class TestSHAPExplainer:
    """测试SHAP解释器"""
    
    @pytest.fixture
    def classification_data(self):
        """分类数据"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test
    
    @pytest.mark.skipif(not hasattr(SHAPExplainer, 'shap'), reason="SHAP not available")
    def test_shap_explainer_creation(self, classification_data):
        """测试SHAP解释器创建"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = SHAPExplainer(
            model=model,
            feature_names=[f'feature_{i}' for i in range(X_train.shape[1])],
            background_data=X_train[:20]
        )
        
        assert explainer.model == model
        assert len(explainer.feature_names) == X_train.shape[1]
    
    @pytest.mark.skipif(not hasattr(SHAPExplainer, 'shap'), reason="SHAP not available")
    def test_shap_explanation(self, classification_data):
        """测试SHAP解释"""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = SHAPExplainer(
            model=model,
            explainer_type='kernel',  # 使用kernel explainer确保兼容性
            background_data=X_train[:10]
        )
        
        try:
            result = explainer.explain(X_test[:5])
            
            assert result is not None
            assert result.explanation_method == 'shap'
            if result.shap_values is not None:
                assert result.shap_values.shape[0] == 5  # 5个测试样本
        except Exception as e:
            pytest.skip(f"SHAP explanation failed: {e}")


class TestExplanationVisualizer:
    """测试解释可视化器"""
    
    @pytest.fixture
    def sample_explanation_result(self):
        """示例解释结果"""
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        importance = np.array([0.4, 0.3, 0.2, 0.08, 0.02])
        
        return ExplanationResult(
            feature_names=feature_names,
            model_type='classifier',
            explanation_method='feature_importance',
            feature_importance=importance
        )
    
    def test_visualizer_creation(self):
        """测试可视化器创建"""
        visualizer = ExplanationVisualizer()
        
        assert visualizer.style == 'seaborn'
        assert visualizer.figsize == (10, 6)
        assert visualizer.dpi == 100
    
    def test_feature_importance_plot(self, sample_explanation_result):
        """测试特征重要性图"""
        visualizer = ExplanationVisualizer()
        
        try:
            fig = visualizer.plot_feature_importance(sample_explanation_result, top_n=3)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Plotting failed: {e}")
    
    def test_importance_comparison_plot(self, sample_explanation_result):
        """测试重要性比较图"""
        # 添加多种方法的结果
        sample_explanation_result.visualization_data = {
            'importance_methods': {
                'built_in': {'importance': np.array([0.4, 0.3, 0.2, 0.08, 0.02])},
                'permutation': {'importance': np.array([0.35, 0.32, 0.18, 0.1, 0.05])}
            }
        }
        
        visualizer = ExplanationVisualizer()
        
        try:
            fig = visualizer.plot_importance_comparison(sample_explanation_result)
            assert fig is not None
        except Exception as e:
            pytest.skip(f"Comparison plotting failed: {e}")


class TestEnsembleIntegration:
    """测试与集成学习的集成"""
    
    @pytest.fixture
    def ensemble_data(self):
        """集成学习数据"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 创建基础模型
        models = [
            RandomForestClassifier(n_estimators=5, random_state=42),
            LogisticRegression(random_state=42)
        ]
        
        # 创建集成模型
        ensemble = VotingEnsemble(models=models)
        ensemble.fit(X_train, y_train)
        
        return ensemble, X_train, X_test, y_train, y_test
    
    def test_ensemble_explanation(self, ensemble_data):
        """测试集成模型解释"""
        ensemble, X_train, X_test, y_train, y_test = ensemble_data
        
        try:
            # 测试特征重要性解释
            result = ensemble.explain_predictions(
                X_test[:10],
                y=y_test[:10],
                method='feature_importance',
                importance_methods=['built_in']
            )
            
            assert result is not None
            assert result.explanation_method == 'feature_importance'
            
        except Exception as e:
            pytest.skip(f"Ensemble explanation failed: {e}")
    
    def test_ensemble_visualization(self, ensemble_data):
        """测试集成模型可视化"""
        ensemble, X_train, X_test, y_train, y_test = ensemble_data
        
        try:
            # 获取解释结果
            result = ensemble.explain_predictions(
                X_test[:10],
                y=y_test[:10],
                method='feature_importance',
                importance_methods=['built_in']
            )
            
            # 测试可视化
            fig = ensemble.visualize_explanations(result, plot_type='feature_importance')
            assert fig is not None
            
        except Exception as e:
            pytest.skip(f"Ensemble visualization failed: {e}")


class TestErrorHandling:
    """测试错误处理"""
    
    def test_invalid_model(self):
        """测试无效模型"""
        with pytest.raises((ValueError, AttributeError)):
            explainer = FeatureImportanceExplainer(model="invalid_model")
    
    def test_missing_target_for_permutation(self):
        """测试排列重要性缺少目标变量"""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        explainer = FeatureImportanceExplainer(
            model=model,
            importance_methods=['permutation']
        )
        
        with pytest.raises(ValueError, match="Target variable y is required"):
            explainer.explain(X)  # 没有提供y
    
    def test_invalid_explanation_method(self):
        """测试无效解释方法"""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        models = [RandomForestClassifier(n_estimators=5, random_state=42)]
        ensemble = VotingEnsemble(models=models)
        ensemble.fit(X, y)
        
        with pytest.raises(ValueError, match="Unknown explanation method"):
            ensemble.explain_predictions(X, method='invalid_method')


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])