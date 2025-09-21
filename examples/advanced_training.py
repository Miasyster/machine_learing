"""
高级训练示例

演示超参数优化、交叉验证、模型选择和集成学习等高级功能。
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# 导入我们的框架
from src.data import DataLoader, DataPreprocessor, FeatureEngineer
from src.training import (
    ModelTrainer, ModelValidator, HyperparameterOptimizer,
    CrossValidator, ModelSelector, EnsembleTrainer
)
from src.deployment import ModelSerializer, ModelVersionManager
from src.monitoring import create_log_manager, PerformanceMonitor


def hyperparameter_optimization_example():
    """超参数优化示例"""
    print("=== 超参数优化示例 ===")
    
    # 生成数据
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建超参数优化器
    optimizer = HyperparameterOptimizer(
        model_type='random_forest',
        optimization_method='bayesian'
    )
    
    # 定义搜索空间
    search_space = {
        'n_estimators': (50, 200),
        'max_depth': (5, 20),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 5),
        'max_features': ['sqrt', 'log2', None]
    }
    
    print("1. 开始超参数优化...")
    best_params, best_score, optimization_history = optimizer.optimize(
        X_train, y_train,
        search_space=search_space,
        cv_folds=5,
        n_trials=50,
        scoring='f1_macro'
    )
    
    print(f"最佳参数: {best_params}")
    print(f"最佳得分: {best_score:.4f}")
    
    # 使用最佳参数训练模型
    trainer = ModelTrainer(
        model_type='random_forest',
        hyperparameters=best_params
    )
    
    best_model = trainer.train(X_train, y_train)
    
    # 评估模型
    validator = ModelValidator()
    test_metrics = validator.evaluate(best_model, X_test, y_test)
    
    print(f"测试集性能:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return best_model, optimization_history


def cross_validation_example():
    """交叉验证示例"""
    print("\n=== 交叉验证示例 ===")
    
    # 生成回归数据
    X, y = make_regression(
        n_samples=1000,
        n_features=15,
        noise=0.1,
        random_state=42
    )
    
    # 创建交叉验证器
    cv = CrossValidator(
        cv_method='kfold',
        n_folds=5,
        shuffle=True,
        random_state=42
    )
    
    # 定义多个模型进行比较
    models_config = {
        'linear_regression': {
            'model_type': 'linear_regression',
            'hyperparameters': {}
        },
        'random_forest': {
            'model_type': 'random_forest',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        'gradient_boosting': {
            'model_type': 'gradient_boosting',
            'hyperparameters': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
        }
    }
    
    print("1. 进行交叉验证比较...")
    cv_results = {}
    
    for model_name, config in models_config.items():
        print(f"  评估 {model_name}...")
        
        trainer = ModelTrainer(
            model_type=config['model_type'],
            hyperparameters=config['hyperparameters']
        )
        
        scores = cv.cross_validate(
            trainer, X, y,
            scoring=['mse', 'r2', 'mae']
        )
        
        cv_results[model_name] = scores
        
        print(f"    MSE: {scores['mse']['mean']:.4f} (+/- {scores['mse']['std']:.4f})")
        print(f"    R2:  {scores['r2']['mean']:.4f} (+/- {scores['r2']['std']:.4f})")
        print(f"    MAE: {scores['mae']['mean']:.4f} (+/- {scores['mae']['std']:.4f})")
    
    # 选择最佳模型
    best_model_name = min(cv_results.keys(), 
                         key=lambda x: cv_results[x]['mse']['mean'])
    
    print(f"\n最佳模型: {best_model_name}")
    
    return cv_results


def model_selection_example():
    """模型选择示例"""
    print("\n=== 模型选择示例 ===")
    
    # 生成分类数据
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建模型选择器
    selector = ModelSelector(
        selection_method='tournament',
        cv_folds=5,
        scoring='f1'
    )
    
    # 定义候选模型
    candidate_models = [
        {
            'name': 'logistic_regression',
            'model_type': 'logistic_regression',
            'hyperparameters': {'C': 1.0, 'random_state': 42}
        },
        {
            'name': 'random_forest',
            'model_type': 'random_forest',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        {
            'name': 'svm',
            'model_type': 'svm',
            'hyperparameters': {
                'C': 1.0,
                'kernel': 'rbf',
                'random_state': 42
            }
        },
        {
            'name': 'gradient_boosting',
            'model_type': 'gradient_boosting',
            'hyperparameters': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
        }
    ]
    
    print("1. 进行模型选择...")
    best_model, selection_results = selector.select_best_model(
        candidate_models, X_train, y_train
    )
    
    print(f"选择的最佳模型: {selection_results['best_model_name']}")
    print(f"最佳得分: {selection_results['best_score']:.4f}")
    
    print("\n所有模型得分:")
    for result in selection_results['all_scores']:
        print(f"  {result['name']}: {result['score']:.4f}")
    
    # 在测试集上评估最佳模型
    validator = ModelValidator()
    test_metrics = validator.evaluate(best_model, X_test, y_test)
    
    print(f"\n测试集性能:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return best_model, selection_results


def ensemble_learning_example():
    """集成学习示例"""
    print("\n=== 集成学习示例 ===")
    
    # 生成数据
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建集成训练器
    ensemble_trainer = EnsembleTrainer(
        ensemble_method='voting',
        voting_type='soft'
    )
    
    # 定义基学习器
    base_learners = [
        {
            'name': 'rf',
            'model_type': 'random_forest',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        {
            'name': 'gb',
            'model_type': 'gradient_boosting',
            'hyperparameters': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
        },
        {
            'name': 'svm',
            'model_type': 'svm',
            'hyperparameters': {
                'C': 1.0,
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42
            }
        }
    ]
    
    print("1. 训练集成模型...")
    ensemble_model = ensemble_trainer.train_ensemble(
        base_learners, X_train, y_train
    )
    
    # 评估集成模型
    validator = ModelValidator()
    ensemble_metrics = validator.evaluate(ensemble_model, X_test, y_test)
    
    print(f"集成模型性能:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 比较单个模型和集成模型
    print("\n单个模型 vs 集成模型比较:")
    
    for learner in base_learners:
        trainer = ModelTrainer(
            model_type=learner['model_type'],
            hyperparameters=learner['hyperparameters']
        )
        
        single_model = trainer.train(X_train, y_train)
        single_metrics = validator.evaluate(single_model, X_test, y_test)
        
        print(f"  {learner['name']} F1: {single_metrics['f1']:.4f}")
    
    print(f"  集成模型 F1: {ensemble_metrics['f1']:.4f}")
    
    return ensemble_model


def model_versioning_example():
    """模型版本管理示例"""
    print("\n=== 模型版本管理示例 ===")
    
    # 创建版本管理器
    version_manager = ModelVersionManager(
        models_dir="models/versioned",
        use_git=False  # 简化示例，不使用Git
    )
    
    # 生成数据并训练多个版本的模型
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练第一个版本
    print("1. 训练模型版本 v1.0.0...")
    trainer_v1 = ModelTrainer(
        model_type='random_forest',
        hyperparameters={'n_estimators': 50, 'random_state': 42}
    )
    
    model_v1 = trainer_v1.train(X_train, y_train)
    
    validator = ModelValidator()
    metrics_v1 = validator.evaluate(model_v1, X_test, y_test)
    
    # 保存第一个版本
    version_info_v1 = version_manager.create_version(
        model=model_v1,
        version="1.0.0",
        metadata={
            'model_type': 'random_forest',
            'hyperparameters': {'n_estimators': 50},
            'metrics': metrics_v1,
            'description': '初始版本，使用50棵树'
        }
    )
    
    print(f"  保存版本: {version_info_v1.version}")
    print(f"  F1得分: {metrics_v1['f1']:.4f}")
    
    # 训练第二个版本（改进的超参数）
    print("\n2. 训练模型版本 v1.1.0...")
    trainer_v2 = ModelTrainer(
        model_type='random_forest',
        hyperparameters={'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    )
    
    model_v2 = trainer_v2.train(X_train, y_train)
    metrics_v2 = validator.evaluate(model_v2, X_test, y_test)
    
    # 保存第二个版本
    version_info_v2 = version_manager.create_version(
        model=model_v2,
        version="1.1.0",
        metadata={
            'model_type': 'random_forest',
            'hyperparameters': {'n_estimators': 100, 'max_depth': 10},
            'metrics': metrics_v2,
            'description': '改进版本，增加树的数量和深度限制'
        }
    )
    
    print(f"  保存版本: {version_info_v2.version}")
    print(f"  F1得分: {metrics_v2['f1']:.4f}")
    
    # 添加标签
    if metrics_v2['f1'] > metrics_v1['f1']:
        version_manager.add_tag("1.1.0", "best_performance")
        print("  添加标签: best_performance")
    
    version_manager.add_tag("1.1.0", "production")
    print("  添加标签: production")
    
    # 列出所有版本
    print("\n3. 所有模型版本:")
    versions = version_manager.list_versions()
    for version in versions:
        print(f"  版本 {version.version}: {version.description}")
        if version.tags:
            print(f"    标签: {', '.join(version.tags)}")
    
    # 比较版本
    print("\n4. 版本比较:")
    comparison = version_manager.compare_versions("1.0.0", "1.1.0")
    print(f"  版本差异: {comparison}")
    
    # 加载特定版本
    print("\n5. 加载生产版本...")
    production_model = version_manager.load_model_by_tag("production")
    print("  生产模型加载成功")
    
    return version_manager


if __name__ == "__main__":
    # 设置日志
    log_manager = create_log_manager(
        log_level='INFO',
        log_file='logs/advanced_training.log'
    )
    
    logger = log_manager.get_logger('advanced_training')
    logger.info("开始高级训练示例")
    
    # 运行所有示例
    try:
        best_model, opt_history = hyperparameter_optimization_example()
        cv_results = cross_validation_example()
        selected_model, selection_results = model_selection_example()
        ensemble_model = ensemble_learning_example()
        version_manager = model_versioning_example()
        
        print("\n所有高级训练示例运行完成！")
        logger.info("所有高级训练示例运行完成")
        
    except Exception as e:
        logger.error(f"示例运行出错: {str(e)}")
        raise