"""
Stacking和Blending集成方法使用示例

本示例展示如何使用我们实现的集成方法进行机器学习任务
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_boston, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 导入我们的集成方法
from src.ensemble.stacking import StackingEnsemble, MultiLevelStackingEnsemble, DynamicStackingEnsemble
from src.ensemble.blending import BlendingEnsemble, DynamicBlendingEnsemble, AdaptiveBlendingEnsemble
from src.ensemble.calibration import ModelCalibrator, CalibratedEnsemble, TemperatureScaling


def example_1_basic_stacking():
    """示例1: 基础Stacking分类"""
    print("="*60)
    print("示例1: 基础Stacking分类 - 乳腺癌数据集")
    print("="*60)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 定义基础模型
    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        SVC(probability=True, random_state=42),
        KNeighborsClassifier(n_neighbors=5)
    ]
    
    model_names = ['RandomForest', 'GradientBoosting', 'SVM', 'KNN']
    
    # 创建Stacking集成
    stacking = StackingEnsemble(
        base_models=base_models,
        meta_model=LogisticRegression(random_state=42),
        cv=5,
        use_probas=True,
        model_names=model_names,
        verbose=True
    )
    
    # 训练模型
    print("\\n训练Stacking模型...")
    stacking.fit(X_train, y_train)
    
    # 预测
    result = stacking.predict(X_test)
    accuracy = accuracy_score(y_test, result.predictions)
    
    print(f"\\nStacking集成准确率: {accuracy:.4f}")
    
    # 获取各个基础模型的预测
    individual_preds = stacking.get_individual_predictions(X_test)
    individual_probs = stacking.get_individual_probabilities(X_test)
    
    print("\\n各基础模型准确率:")
    for i, (pred, name) in enumerate(zip(individual_preds, model_names)):
        acc = accuracy_score(y_test, pred)
        print(f"  {name}: {acc:.4f}")
    
    # 显示分类报告
    print("\\n分类报告:")
    print(classification_report(y_test, result.predictions, target_names=data.target_names))
    
    return stacking, result


def example_2_multi_level_stacking():
    """示例2: 多层Stacking"""
    print("\\n" + "="*60)
    print("示例2: 多层Stacking - 红酒数据集")
    print("="*60)
    
    # 加载数据
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"类别数量: {len(np.unique(y))}")
    
    # 定义基础模型
    base_models = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        GradientBoostingClassifier(n_estimators=50, random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    # 定义多层元模型
    meta_models = [
        LogisticRegression(random_state=42),
        RandomForestClassifier(n_estimators=20, random_state=42)
    ]
    
    # 创建多层Stacking
    multi_stacking = MultiLevelStackingEnsemble(
        base_models=base_models,
        meta_models=meta_models,
        cv=3,
        verbose=True
    )
    
    # 训练模型
    print("\\n训练多层Stacking模型...")
    multi_stacking.fit(X_train, y_train)
    
    # 预测
    result = multi_stacking.predict(X_test)
    accuracy = accuracy_score(y_test, result.predictions)
    
    print(f"\\n多层Stacking准确率: {accuracy:.4f}")
    
    # 比较单层Stacking
    single_stacking = StackingEnsemble(
        base_models=base_models,
        meta_model=LogisticRegression(random_state=42),
        cv=3
    )
    single_stacking.fit(X_train, y_train)
    single_result = single_stacking.predict(X_test)
    single_accuracy = accuracy_score(y_test, single_result.predictions)
    
    print(f"单层Stacking准确率: {single_accuracy:.4f}")
    print(f"多层提升: {accuracy - single_accuracy:.4f}")
    
    return multi_stacking, result


def example_3_blending_ensemble():
    """示例3: Blending集成"""
    print("\\n" + "="*60)
    print("示例3: Blending集成 - 乳腺癌数据集")
    print("="*60)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 定义基础模型
    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        SVC(probability=True, random_state=42),
        LogisticRegression(random_state=42)
    ]
    
    model_names = ['RandomForest', 'GradientBoosting', 'SVM', 'LogisticRegression']
    
    # 创建Blending集成
    blending = BlendingEnsemble(
        base_models=base_models,
        meta_model=LogisticRegression(random_state=42),
        holdout_size=0.2,
        use_probas=True,
        model_names=model_names,
        verbose=True
    )
    
    # 训练模型
    print("\\n训练Blending模型...")
    blending.fit(X_train, y_train)
    
    # 预测
    result = blending.predict(X_test)
    accuracy = accuracy_score(y_test, result.predictions)
    
    print(f"\\nBlending集成准确率: {accuracy:.4f}")
    
    # 比较Stacking和Blending
    stacking = StackingEnsemble(
        base_models=base_models,
        meta_model=LogisticRegression(random_state=42),
        cv=5,
        use_probas=True
    )
    stacking.fit(X_train, y_train)
    stacking_result = stacking.predict(X_test)
    stacking_accuracy = accuracy_score(y_test, stacking_result.predictions)
    
    print(f"Stacking集成准确率: {stacking_accuracy:.4f}")
    print(f"方法比较 - Blending vs Stacking: {accuracy:.4f} vs {stacking_accuracy:.4f}")
    
    return blending, result


def example_4_adaptive_blending():
    """示例4: 自适应Blending"""
    print("\\n" + "="*60)
    print("示例4: 自适应Blending")
    print("="*60)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 定义基础模型
    base_models = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        GradientBoostingClassifier(n_estimators=50, random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    # 创建自适应Blending
    adaptive_blending = AdaptiveBlendingEnsemble(
        base_models=base_models,
        meta_model=LogisticRegression(random_state=42),
        adaptation_rate=0.1,
        holdout_size=0.2,
        verbose=True
    )
    
    # 训练模型
    print("\\n训练自适应Blending模型...")
    adaptive_blending.fit(X_train, y_train)
    
    # 预测（提供真实标签以进行自适应）
    result = adaptive_blending.predict(X_test, y_true=y_test)
    accuracy = accuracy_score(y_test, result.predictions)
    
    print(f"\\n自适应Blending准确率: {accuracy:.4f}")
    
    # 查看权重变化
    if hasattr(adaptive_blending, 'get_current_weights'):
        weights = adaptive_blending.get_current_weights()
        if weights is not None:
            print("\\n当前模型权重:")
            for i, weight in enumerate(weights):
                print(f"  模型 {i+1}: {weight:.4f}")
    
    return adaptive_blending, result


def example_5_model_calibration():
    """示例5: 模型校准"""
    print("\\n" + "="*60)
    print("示例5: 模型校准")
    print("="*60)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 定义基础模型
    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    # 创建基础集成
    base_ensemble = StackingEnsemble(
        base_models=base_models,
        meta_model=LogisticRegression(random_state=42),
        cv=3
    )
    
    # 创建校准器
    platt_calibrator = ModelCalibrator(method='platt')
    isotonic_calibrator = ModelCalibrator(method='isotonic')
    
    # 创建校准后的集成
    platt_ensemble = CalibratedEnsemble(
        base_ensemble=base_ensemble,
        calibrator=platt_calibrator,
        cv=3
    )
    
    isotonic_ensemble = CalibratedEnsemble(
        base_ensemble=base_ensemble,
        calibrator=isotonic_calibrator,
        cv=3
    )
    
    # 训练模型
    print("\\n训练校准模型...")
    base_ensemble.fit(X_train, y_train)
    platt_ensemble.fit(X_train, y_train)
    isotonic_ensemble.fit(X_train, y_train)
    
    # 预测
    base_result = base_ensemble.predict(X_test)
    platt_result = platt_ensemble.predict(X_test)
    isotonic_result = isotonic_ensemble.predict(X_test)
    
    # 计算准确率
    base_accuracy = accuracy_score(y_test, base_result.predictions)
    platt_accuracy = accuracy_score(y_test, platt_result.predictions)
    isotonic_accuracy = accuracy_score(y_test, isotonic_result.predictions)
    
    print(f"\\n基础集成准确率: {base_accuracy:.4f}")
    print(f"Platt校准准确率: {platt_accuracy:.4f}")
    print(f"Isotonic校准准确率: {isotonic_accuracy:.4f}")
    
    # 计算校准指标（可靠性图等）
    def reliability_diagram(y_true, y_prob, n_bins=10):
        """计算可靠性图数据"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
        
        return np.array(accuracies), np.array(confidences)
    
    if (base_result.prediction_probabilities is not None and 
        platt_result.prediction_probabilities is not None):
        
        # 获取正类概率
        base_probs = base_result.prediction_probabilities[:, 1]
        platt_probs = platt_result.prediction_probabilities[:, 1]
        
        base_acc, base_conf = reliability_diagram(y_test, base_probs)
        platt_acc, platt_conf = reliability_diagram(y_test, platt_probs)
        
        print("\\n校准质量评估完成（可靠性图数据已计算）")
    
    return platt_ensemble, isotonic_ensemble


def example_6_regression_ensemble():
    """示例6: 回归集成"""
    print("\\n" + "="*60)
    print("示例6: 回归集成 - 波士顿房价数据集")
    print("="*60)
    
    # 创建回归数据（波士顿数据集已弃用，使用模拟数据）
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=500, n_features=13, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # 定义基础回归模型
    base_models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        Ridge(alpha=1.0),
        SVR(kernel='rbf')
    ]
    
    model_names = ['RandomForest', 'Ridge', 'SVR']
    
    # Stacking回归
    stacking_reg = StackingEnsemble(
        base_models=base_models,
        meta_model=LinearRegression(),
        cv=5,
        model_names=model_names,
        verbose=True
    )
    
    # Blending回归
    blending_reg = BlendingEnsemble(
        base_models=base_models,
        meta_model=LinearRegression(),
        holdout_size=0.2,
        model_names=model_names,
        verbose=True
    )
    
    # 训练模型
    print("\\n训练回归集成模型...")
    stacking_reg.fit(X_train, y_train)
    blending_reg.fit(X_train, y_train)
    
    # 预测
    stacking_result = stacking_reg.predict(X_test)
    blending_result = blending_reg.predict(X_test)
    
    # 计算MSE
    stacking_mse = mean_squared_error(y_test, stacking_result.predictions)
    blending_mse = mean_squared_error(y_test, blending_result.predictions)
    
    print(f"\\nStacking回归MSE: {stacking_mse:.4f}")
    print(f"Blending回归MSE: {blending_mse:.4f}")
    
    # 比较各基础模型
    individual_preds = stacking_reg.get_individual_predictions(X_test)
    print("\\n各基础模型MSE:")
    for pred, name in zip(individual_preds, model_names):
        mse = mean_squared_error(y_test, pred)
        print(f"  {name}: {mse:.4f}")
    
    return stacking_reg, blending_reg


def example_7_performance_comparison():
    """示例7: 性能比较"""
    print("\\n" + "="*60)
    print("示例7: 不同集成方法性能比较")
    print("="*60)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 定义基础模型
    base_models = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        GradientBoostingClassifier(n_estimators=50, random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    # 定义不同的集成方法
    methods = {
        'Stacking': StackingEnsemble(
            base_models=base_models,
            meta_model=LogisticRegression(random_state=42),
            cv=3
        ),
        'Blending': BlendingEnsemble(
            base_models=base_models,
            meta_model=LogisticRegression(random_state=42),
            holdout_size=0.2
        )
    }
    
    # 训练和评估所有方法
    results = {}
    print("\\n训练和评估不同集成方法...")
    
    for name, method in methods.items():
        print(f"\\n训练 {name}...")
        method.fit(X_train, y_train)
        result = method.predict(X_test)
        accuracy = accuracy_score(y_test, result.predictions)
        results[name] = accuracy
        print(f"{name} 准确率: {accuracy:.4f}")
    
    # 比较基础模型
    print("\\n基础模型性能:")
    for i, model in enumerate(base_models):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"  模型 {i+1}: {acc:.4f}")
    
    # 显示最佳方法
    best_method = max(results, key=results.get)
    print(f"\\n最佳集成方法: {best_method} (准确率: {results[best_method]:.4f})")
    
    return results


def main():
    """主函数 - 运行所有示例"""
    print("Stacking和Blending集成方法使用示例")
    print("本示例展示如何使用我们实现的集成方法进行机器学习任务")
    print("="*80)
    
    try:
        # 运行所有示例
        example_1_basic_stacking()
        example_2_multi_level_stacking()
        example_3_blending_ensemble()
        example_4_adaptive_blending()
        example_5_model_calibration()
        example_6_regression_ensemble()
        example_7_performance_comparison()
        
        print("\\n" + "="*80)
        print("所有示例运行完成!")
        print("="*80)
        
    except Exception as e:
        print(f"\\n示例运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()