"""
基础使用示例

演示如何使用机器学习框架进行基本的数据处理、模型训练和预测。
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 导入我们的框架
from src.data import DataLoader, DataPreprocessor, FeatureEngineer
from src.training import ModelTrainer, ModelValidator
from src.deployment import ModelSerializer, ModelInferenceEngine
from src.monitoring import PerformanceMonitor, create_log_manager


def basic_classification_example():
    """基础分类任务示例"""
    print("=== 基础分类任务示例 ===")
    
    # 1. 生成示例数据
    print("1. 生成示例数据...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # 2. 数据加载和预处理
    print("2. 数据加载和预处理...")
    data_loader = DataLoader()
    data = data_loader.load_from_dataframe(df)
    
    preprocessor = DataPreprocessor()
    preprocessor.add_step('normalize', method='standard')
    preprocessor.add_step('handle_missing', method='mean')
    
    processed_data = preprocessor.fit_transform(data)
    
    # 3. 特征工程
    print("3. 特征工程...")
    feature_engineer = FeatureEngineer()
    feature_engineer.add_transformer('polynomial', degree=2, include_bias=False)
    feature_engineer.add_selector('variance_threshold', threshold=0.01)
    
    engineered_data = feature_engineer.fit_transform(processed_data)
    
    # 4. 数据分割
    print("4. 数据分割...")
    X_train, X_test, y_train, y_test = train_test_split(
        engineered_data.drop('target', axis=1),
        engineered_data['target'],
        test_size=0.2,
        random_state=42,
        stratify=engineered_data['target']
    )
    
    # 5. 模型训练
    print("5. 模型训练...")
    trainer = ModelTrainer(
        model_type='random_forest',
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    )
    
    model = trainer.train(X_train, y_train)
    
    # 6. 模型验证
    print("6. 模型验证...")
    validator = ModelValidator()
    metrics = validator.evaluate(model, X_test, y_test)
    
    print(f"模型性能指标:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # 7. 模型序列化
    print("7. 模型序列化...")
    serializer = ModelSerializer()
    model_path = "models/basic_classifier.pkl"
    serializer.save_model(
        model=model,
        file_path=model_path,
        metadata={
            'model_type': 'random_forest',
            'features': list(X_train.columns),
            'metrics': metrics,
            'preprocessing_steps': preprocessor.get_steps(),
            'feature_engineering_steps': feature_engineer.get_transformers()
        }
    )
    
    # 8. 模型推理
    print("8. 模型推理...")
    inference_engine = ModelInferenceEngine(model_path)
    
    # 单次预测
    sample_data = X_test.iloc[:1]
    prediction = inference_engine.predict(sample_data)
    print(f"单次预测结果: {prediction}")
    
    # 批量预测
    batch_predictions = inference_engine.batch_predict(X_test.iloc[:10])
    print(f"批量预测结果: {batch_predictions}")
    
    # 9. 性能监控
    print("9. 性能监控...")
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # 模拟一些预测请求
    for i in range(100):
        sample = X_test.iloc[i:i+1]
        pred = inference_engine.predict(sample)
        monitor.record_prediction(
            input_size=len(sample),
            prediction_time=0.01,  # 模拟预测时间
            memory_usage=50  # 模拟内存使用
        )
    
    performance_stats = monitor.get_statistics()
    print(f"性能统计:")
    for stat_name, value in performance_stats.items():
        print(f"  {stat_name}: {value}")
    
    print("基础分类任务示例完成！")


def data_processing_example():
    """数据处理示例"""
    print("\n=== 数据处理示例 ===")
    
    # 创建包含缺失值和异常值的示例数据
    np.random.seed(42)
    data = {
        'numeric_feature_1': np.random.normal(0, 1, 1000),
        'numeric_feature_2': np.random.exponential(2, 1000),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.randint(0, 2, 1000)
    }
    
    # 添加缺失值
    missing_indices = np.random.choice(1000, 100, replace=False)
    data['numeric_feature_1'][missing_indices[:50]] = np.nan
    data['categorical_feature'][missing_indices[50:]] = None
    
    # 添加异常值
    outlier_indices = np.random.choice(1000, 20, replace=False)
    data['numeric_feature_1'][outlier_indices] = np.random.normal(0, 1, 20) * 10
    
    df = pd.DataFrame(data)
    
    print("1. 原始数据统计:")
    print(f"  数据形状: {df.shape}")
    print(f"  缺失值数量: {df.isnull().sum().sum()}")
    print(f"  数值特征统计:")
    print(df.describe())
    
    # 数据加载
    data_loader = DataLoader()
    loaded_data = data_loader.load_from_dataframe(df)
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    
    # 添加预处理步骤
    preprocessor.add_step('handle_missing', method='mean', columns=['numeric_feature_1'])
    preprocessor.add_step('handle_missing', method='mode', columns=['categorical_feature'])
    preprocessor.add_step('remove_outliers', method='iqr', columns=['numeric_feature_1'])
    preprocessor.add_step('normalize', method='standard', columns=['numeric_feature_1', 'numeric_feature_2'])
    preprocessor.add_step('encode_categorical', method='onehot', columns=['categorical_feature'])
    
    # 应用预处理
    processed_data = preprocessor.fit_transform(loaded_data)
    
    print("\n2. 预处理后数据统计:")
    print(f"  数据形状: {processed_data.shape}")
    print(f"  缺失值数量: {processed_data.isnull().sum().sum()}")
    print(f"  数值特征统计:")
    print(processed_data.select_dtypes(include=[np.number]).describe())
    
    # 特征工程
    feature_engineer = FeatureEngineer()
    feature_engineer.add_transformer('polynomial', degree=2, include_bias=False)
    feature_engineer.add_transformer('interaction', columns=['numeric_feature_1', 'numeric_feature_2'])
    feature_engineer.add_selector('variance_threshold', threshold=0.01)
    feature_engineer.add_selector('correlation_threshold', threshold=0.95)
    
    engineered_data = feature_engineer.fit_transform(processed_data)
    
    print("\n3. 特征工程后数据统计:")
    print(f"  数据形状: {engineered_data.shape}")
    print(f"  特征数量变化: {processed_data.shape[1]} -> {engineered_data.shape[1]}")
    
    print("数据处理示例完成！")


def monitoring_example():
    """监控系统示例"""
    print("\n=== 监控系统示例 ===")
    
    # 设置日志管理器
    log_manager = create_log_manager(
        log_level='INFO',
        log_file='logs/ml_framework.log'
    )
    
    logger = log_manager.get_logger('example')
    logger.info("开始监控示例")
    
    # 创建性能监控器
    performance_monitor = PerformanceMonitor()
    performance_monitor.start_monitoring()
    
    # 模拟模型训练和预测过程
    logger.info("模拟模型训练过程")
    
    # 记录训练指标
    for epoch in range(10):
        # 模拟训练指标
        train_loss = 1.0 - epoch * 0.1 + np.random.normal(0, 0.05)
        val_loss = 1.2 - epoch * 0.08 + np.random.normal(0, 0.08)
        accuracy = 0.5 + epoch * 0.04 + np.random.normal(0, 0.02)
        
        performance_monitor.record_training_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            accuracy=accuracy
        )
        
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, accuracy={accuracy:.4f}")
    
    # 模拟推理过程
    logger.info("模拟推理过程")
    
    for i in range(100):
        # 模拟推理指标
        inference_time = np.random.exponential(0.05)  # 推理时间
        memory_usage = np.random.normal(100, 20)  # 内存使用
        cpu_usage = np.random.normal(50, 15)  # CPU使用率
        
        performance_monitor.record_inference_metrics(
            inference_time=inference_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
    
    # 获取监控统计
    stats = performance_monitor.get_statistics()
    
    print("\n监控统计结果:")
    print(f"  平均推理时间: {stats.get('avg_inference_time', 0):.4f}s")
    print(f"  最大内存使用: {stats.get('max_memory_usage', 0):.2f}MB")
    print(f"  平均CPU使用率: {stats.get('avg_cpu_usage', 0):.2f}%")
    print(f"  总预测次数: {stats.get('total_predictions', 0)}")
    
    logger.info("监控示例完成")
    print("监控系统示例完成！")


if __name__ == "__main__":
    # 运行所有示例
    basic_classification_example()
    data_processing_example()
    monitoring_example()
    
    print("\n所有示例运行完成！")