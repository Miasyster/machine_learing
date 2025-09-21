"""
集成方法快速入门示例

这是一个简化的示例，展示如何快速使用Stacking和Blending方法
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 导入我们的集成方法
from src.ensemble.stacking import StackingEnsemble
from src.ensemble.blending import BlendingEnsemble
from src.ensemble.calibration import CalibratedEnsemble, ModelCalibrator


def quick_start_example():
    """快速入门示例"""
    print("集成方法快速入门示例")
    print("="*50)
    
    # 1. 准备数据
    print("1. 加载数据...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"   训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 2. 定义基础模型
    print("\\n2. 定义基础模型...")
    base_models = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        LogisticRegression(random_state=42, max_iter=1000),
        SVC(probability=True, random_state=42)
    ]
    print(f"   基础模型数量: {len(base_models)}")
    
    # 3. Stacking集成
    print("\\n3. 使用Stacking集成...")
    stacking = StackingEnsemble(
        base_models=base_models,
        meta_model=LogisticRegression(random_state=42),
        cv=3,
        verbose=False  # 关闭详细输出以保持简洁
    )
    
    stacking.fit(X_train, y_train)
    stacking_pred = stacking.predict(X_test)
    stacking_accuracy = accuracy_score(y_test, stacking_pred.predictions)
    print(f"   Stacking准确率: {stacking_accuracy:.4f}")
    
    # 4. Blending集成
    print("\\n4. 使用Blending集成...")
    blending = BlendingEnsemble(
        base_models=base_models,
        meta_model=LogisticRegression(random_state=42),
        holdout_size=0.2,
        verbose=False
    )
    
    blending.fit(X_train, y_train)
    blending_pred = blending.predict(X_test)
    blending_accuracy = accuracy_score(y_test, blending_pred.predictions)
    print(f"   Blending准确率: {blending_accuracy:.4f}")
    
    # 5. 模型校准
    print("\\n5. 使用模型校准...")
    calibrated_ensemble = CalibratedEnsemble(
        models=base_models,  # 使用基础模型列表
        calibration_method='platt',
        cv=3
    )
    
    calibrated_ensemble.fit(X_train, y_train)
    calibrated_pred = calibrated_ensemble.predict(X_test)
    calibrated_accuracy = accuracy_score(y_test, calibrated_pred.predictions)
    print(f"   校准后准确率: {calibrated_accuracy:.4f}")
    
    # 6. 比较基础模型
    print("\\n6. 基础模型性能对比...")
    for i, model in enumerate(base_models):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        model_name = model.__class__.__name__
        print(f"   {model_name}: {acc:.4f}")
    
    # 7. 结果总结
    print("\\n7. 结果总结:")
    print(f"   Stacking:     {stacking_accuracy:.4f}")
    print(f"   Blending:     {blending_accuracy:.4f}")
    print(f"   校准后:       {calibrated_accuracy:.4f}")
    
    best_accuracy = max(stacking_accuracy, blending_accuracy, calibrated_accuracy)
    if best_accuracy == stacking_accuracy:
        best_method = "Stacking"
    elif best_accuracy == blending_accuracy:
        best_method = "Blending"
    else:
        best_method = "校准后集成"
    
    print(f"\\n   最佳方法: {best_method} (准确率: {best_accuracy:.4f})")
    
    return {
        'stacking': stacking_accuracy,
        'blending': blending_accuracy,
        'calibrated': calibrated_accuracy,
        'best_method': best_method
    }


def advanced_example():
    """进阶示例 - 更多参数配置"""
    print("\\n" + "="*50)
    print("进阶示例 - 参数调优")
    print("="*50)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 更多基础模型
    base_models = [
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        LogisticRegression(C=0.1, random_state=42, max_iter=1000),
        SVC(C=1.0, gamma='scale', probability=True, random_state=42)
    ]
    
    model_names = ['RF', 'LR', 'SVM']
    
    # 配置更多参数的Stacking
    advanced_stacking = StackingEnsemble(
        base_models=base_models,
        meta_model=LogisticRegression(random_state=42),
        cv=5,  # 更多折数
        use_probas=True,  # 使用概率
        use_features_in_secondary=False,  # 不使用原始特征
        model_names=model_names,
        verbose=True
    )
    
    print("\\n训练进阶Stacking模型...")
    advanced_stacking.fit(X_train, y_train)
    
    # 获取详细预测结果
    result = advanced_stacking.predict(X_test)
    accuracy = accuracy_score(y_test, result.predictions)
    
    print(f"\\n进阶Stacking准确率: {accuracy:.4f}")
    
    # 获取各模型的预测和概率
    individual_preds = advanced_stacking.get_individual_predictions(X_test)
    individual_probs = advanced_stacking.get_individual_probabilities(X_test)
    
    print("\\n各基础模型详细性能:")
    for i, (pred, prob, name) in enumerate(zip(individual_preds, individual_probs, model_names)):
        acc = accuracy_score(y_test, pred)
        print(f"  {name}: 准确率={acc:.4f}")
        if prob is not None:
            # 计算平均置信度
            avg_confidence = np.mean(np.max(prob, axis=1))
            print(f"       平均置信度={avg_confidence:.4f}")
    
    # 集成模型的置信度
    if result.prediction_probabilities is not None:
        ensemble_confidence = np.mean(np.max(result.prediction_probabilities, axis=1))
        print(f"\\n集成模型平均置信度: {ensemble_confidence:.4f}")
    
    return advanced_stacking


if __name__ == "__main__":
    print("开始运行集成方法快速入门示例...")
    
    try:
        # 运行快速入门示例
        results = quick_start_example()
        
        # 运行进阶示例
        advanced_model = advanced_example()
        
        print("\\n" + "="*50)
        print("示例运行完成!")
        print("="*50)
        print("\\n要了解更多功能，请查看 ensemble_examples.py")
        
    except Exception as e:
        print(f"\\n示例运行出错: {str(e)}")
        import traceback
        traceback.print_exc()