"""
模型解释性示例

演示如何使用SHAP和特征重要性分析来解释集成模型的预测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 导入我们的集成学习框架
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ensemble import VotingEnsemble, StackingEnsemble
from src.explainability import SHAPExplainer, FeatureImportanceExplainer, ExplanationVisualizer


def create_sample_data():
    """创建示例数据"""
    print("=== 创建示例数据 ===")
    
    # 使用乳腺癌数据集
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names.tolist()
    
    print(f"数据集大小: {X.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names


def create_ensemble_models():
    """创建集成模型"""
    print("\n=== 创建集成模型 ===")
    
    # 基础模型
    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        LogisticRegression(random_state=42, max_iter=1000),
        SVC(probability=True, random_state=42)
    ]
    
    model_names = ['RandomForest', 'LogisticRegression', 'SVM']
    
    # 创建投票集成
    voting_ensemble = VotingEnsemble(
        models=base_models,
        model_names=model_names,
        voting='soft'
    )
    
    # 创建堆叠集成
    stacking_ensemble = StackingEnsemble(
        base_models=base_models,
        model_names=model_names,
        meta_model=LogisticRegression(random_state=42)
    )
    
    return voting_ensemble, stacking_ensemble


def demonstrate_basic_explanation():
    """演示基础解释功能"""
    print("\n=== 基础模型解释演示 ===")
    
    # 准备数据
    X_train, X_test, y_train, y_test, feature_names = create_sample_data()
    
    # 创建并训练模型
    voting_ensemble, stacking_ensemble = create_ensemble_models()
    
    print("训练投票集成模型...")
    voting_ensemble.fit(X_train, y_train)
    
    print("训练堆叠集成模型...")
    stacking_ensemble.fit(X_train, y_train)
    
    # 评估模型
    voting_score = voting_ensemble.score(X_test, y_test)
    stacking_score = stacking_ensemble.score(X_test, y_test)
    
    print(f"投票集成准确率: {voting_score:.4f}")
    print(f"堆叠集成准确率: {stacking_score:.4f}")
    
    # 选择性能更好的模型进行解释
    best_model = voting_ensemble if voting_score >= stacking_score else stacking_ensemble
    model_name = "投票集成" if voting_score >= stacking_score else "堆叠集成"
    
    print(f"\n使用 {model_name} 进行解释分析...")
    
    return best_model, X_train, X_test, y_train, y_test, feature_names


def demonstrate_feature_importance():
    """演示特征重要性分析"""
    print("\n=== 特征重要性分析 ===")
    
    model, X_train, X_test, y_train, y_test, feature_names = demonstrate_basic_explanation()
    
    # 使用集成模型的内置解释功能
    try:
        print("计算特征重要性...")
        explanation_result = model.explain_predictions(
            X_test[:100],  # 使用前100个测试样本
            y=y_test[:100],
            method='feature_importance',
            feature_names=feature_names
        )
        
        print("特征重要性计算完成!")
        
        # 显示top特征
        if explanation_result and explanation_result.feature_importance is not None:
            top_indices = np.argsort(explanation_result.feature_importance)[-10:]
            print("\nTop 10 重要特征:")
            for i, idx in enumerate(reversed(top_indices)):
                print(f"{i+1:2d}. {feature_names[idx]:25s}: {explanation_result.feature_importance[idx]:.4f}")
        
        # 可视化特征重要性
        print("\n创建特征重要性可视化...")
        fig = model.visualize_explanations(
            explanation_result,
            plot_type='feature_importance',
            top_n=15,
            save_path='feature_importance.png'
        )
        plt.show()
        
        return explanation_result
        
    except Exception as e:
        print(f"特征重要性分析失败: {e}")
        return None


def demonstrate_shap_explanation():
    """演示SHAP解释"""
    print("\n=== SHAP解释分析 ===")
    
    model, X_train, X_test, y_train, y_test, feature_names = demonstrate_basic_explanation()
    
    try:
        print("计算SHAP值...")
        explanation_result = model.explain_predictions(
            X_test[:50],  # 使用前50个测试样本
            y=y_test[:50],
            method='shap',
            feature_names=feature_names,
            background_data=X_train[:100]  # 使用训练数据作为背景
        )
        
        print("SHAP值计算完成!")
        
        # 可视化SHAP摘要
        if explanation_result and explanation_result.shap_values is not None:
            print("\n创建SHAP摘要图...")
            fig = model.visualize_explanations(
                explanation_result,
                plot_type='shap_summary',
                save_path='shap_summary.png'
            )
            plt.show()
        
        return explanation_result
        
    except Exception as e:
        print(f"SHAP解释分析失败: {e}")
        print("这可能是因为缺少SHAP库或模型不兼容")
        return None


def demonstrate_comprehensive_explanation():
    """演示综合解释分析"""
    print("\n=== 综合解释分析 ===")
    
    model, X_train, X_test, y_train, y_test, feature_names = demonstrate_basic_explanation()
    
    try:
        print("执行综合解释分析...")
        explanation_results = model.explain_predictions(
            X_test[:30],  # 使用前30个测试样本
            y=y_test[:30],
            method='both',  # 同时使用SHAP和特征重要性
            feature_names=feature_names,
            background_data=X_train[:100]
        )
        
        print("综合解释分析完成!")
        
        # 创建综合仪表板
        if explanation_results:
            print("\n创建解释仪表板...")
            
            # 如果有SHAP结果，使用SHAP结果创建仪表板
            if explanation_results.get('shap') is not None:
                fig = model.visualize_explanations(
                    explanation_results['shap'],
                    plot_type='dashboard',
                    X=X_test[:30],
                    instance_idx=0,
                    save_path='explanation_dashboard.png'
                )
            # 否则使用特征重要性结果
            elif explanation_results.get('feature_importance') is not None:
                fig = model.visualize_explanations(
                    explanation_results['feature_importance'],
                    plot_type='feature_importance',
                    save_path='explanation_dashboard.png'
                )
            
            plt.show()
        
        return explanation_results
        
    except Exception as e:
        print(f"综合解释分析失败: {e}")
        return None


def demonstrate_individual_explainers():
    """演示单独使用解释器"""
    print("\n=== 单独使用解释器 ===")
    
    model, X_train, X_test, y_train, y_test, feature_names = demonstrate_basic_explanation()
    
    # 特征重要性解释器
    print("\n1. 特征重要性解释器")
    try:
        fi_explainer = FeatureImportanceExplainer(
            model=model,
            feature_names=feature_names
        )
        
        fi_result = fi_explainer.explain(X_test[:50], y=y_test[:50])
        
        if fi_result and fi_result.feature_importance is not None:
            print("特征重要性分析完成")
            
            # 稳定性分析
            stability_data = fi_explainer.analyze_stability(X_test[:100], n_iterations=10)
            print(f"最稳定的特征: {stability_data['stability_ranking'][:5]}")
            
            # 可视化
            visualizer = ExplanationVisualizer()
            fig = visualizer.plot_feature_importance(fi_result, top_n=10)
            plt.title("单独特征重要性分析")
            plt.show()
    
    except Exception as e:
        print(f"特征重要性解释器失败: {e}")
    
    # SHAP解释器
    print("\n2. SHAP解释器")
    try:
        shap_explainer = SHAPExplainer(
            model=model,
            background_data=X_train[:100],
            feature_names=feature_names
        )
        
        shap_result = shap_explainer.explain(X_test[:30])
        
        if shap_result and shap_result.shap_values is not None:
            print("SHAP分析完成")
            
            # 可视化
            visualizer = ExplanationVisualizer()
            fig = visualizer.plot_shap_summary(shap_result, max_display=10)
            plt.title("单独SHAP分析")
            plt.show()
            
            # 单个实例解释
            fig = visualizer.plot_shap_waterfall(shap_result, instance_idx=0)
            plt.title("单个实例SHAP瀑布图")
            plt.show()
    
    except Exception as e:
        print(f"SHAP解释器失败: {e}")


def main():
    """主函数"""
    print("模型解释性功能演示")
    print("=" * 50)
    
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        # 1. 特征重要性分析
        print("\n1. 特征重要性分析演示")
        fi_result = demonstrate_feature_importance()
        
        # 2. SHAP解释分析
        print("\n2. SHAP解释分析演示")
        shap_result = demonstrate_shap_explanation()
        
        # 3. 综合解释分析
        print("\n3. 综合解释分析演示")
        comprehensive_result = demonstrate_comprehensive_explanation()
        
        # 4. 单独使用解释器
        print("\n4. 单独使用解释器演示")
        demonstrate_individual_explainers()
        
        print("\n" + "=" * 50)
        print("模型解释性演示完成!")
        print("\n主要功能:")
        print("✓ 特征重要性分析 (内置、排列、删除列)")
        print("✓ SHAP值计算和可视化")
        print("✓ 多种可视化图表")
        print("✓ 稳定性分析")
        print("✓ 综合解释仪表板")
        print("✓ 与集成学习框架无缝集成")
        
        if fi_result:
            print("\n✓ 特征重要性分析成功")
        if shap_result:
            print("✓ SHAP解释分析成功")
        if comprehensive_result:
            print("✓ 综合解释分析成功")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        print("请检查依赖库是否正确安装")


if __name__ == "__main__":
    main()