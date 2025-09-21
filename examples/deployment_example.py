"""
模型部署示例

演示模型序列化、版本管理、推理服务和部署管理等功能。
"""

import numpy as np
import pandas as pd
import asyncio
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 导入我们的框架
from src.data import DataLoader, DataPreprocessor
from src.training import ModelTrainer, ModelValidator
from src.deployment import (
    ModelSerializer, ModelLoader, ModelVersionManager,
    ModelInferenceEngine, StreamingInferenceEngine, ModelEnsemble,
    LocalDeployment, DeploymentManager
)
from src.monitoring import create_log_manager, PerformanceMonitor


def model_serialization_example():
    """模型序列化示例"""
    print("=== 模型序列化示例 ===")
    
    # 生成和训练模型
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    trainer = ModelTrainer(
        model_type='random_forest',
        hyperparameters={'n_estimators': 100, 'random_state': 42}
    )
    
    model = trainer.train(X_train, y_train)
    
    validator = ModelValidator()
    metrics = validator.evaluate(model, X_test, y_test)
    
    # 创建序列化器
    serializer = ModelSerializer()
    
    # 测试不同的序列化格式
    formats = ['pickle', 'joblib']
    
    for fmt in formats:
        print(f"\n1. 测试 {fmt} 格式序列化...")
        
        model_path = f"models/test_model.{fmt}"
        
        # 序列化模型
        serializer.save_model(
            model=model,
            file_path=model_path,
            format=fmt,
            compress=True,
            metadata={
                'model_type': 'random_forest',
                'features': [f'feature_{i}' for i in range(X_train.shape[1])],
                'metrics': metrics,
                'training_data_shape': X_train.shape,
                'serialization_format': fmt
            }
        )
        
        print(f"  模型已保存到: {model_path}")
        
        # 加载模型
        loader = ModelLoader()
        loaded_model, loaded_metadata = loader.load_model(model_path)
        
        print(f"  模型加载成功")
        print(f"  元数据: {loaded_metadata}")
        
        # 验证加载的模型
        test_predictions = loaded_model.predict(X_test)
        original_predictions = model.predict(X_test)
        
        # 检查预测结果是否一致
        predictions_match = np.array_equal(test_predictions, original_predictions)
        print(f"  预测结果一致性: {predictions_match}")
        
        # 检查模型兼容性
        is_compatible = serializer.validate_model_compatibility(
            model_path, X_test
        )
        print(f"  模型兼容性: {is_compatible}")


def inference_engine_example():
    """推理引擎示例"""
    print("\n=== 推理引擎示例 ===")
    
    # 准备模型和数据
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练并保存模型
    trainer = ModelTrainer(
        model_type='random_forest',
        hyperparameters={'n_estimators': 100, 'random_state': 42}
    )
    
    model = trainer.train(X_train, y_train)
    
    serializer = ModelSerializer()
    model_path = "models/inference_model.pkl"
    serializer.save_model(
        model=model,
        file_path=model_path,
        metadata={
            'model_type': 'random_forest',
            'features': [f'feature_{i}' for i in range(X_train.shape[1])]
        }
    )
    
    # 创建推理引擎
    inference_engine = ModelInferenceEngine(
        model_path=model_path,
        enable_caching=True,
        cache_size=100
    )
    
    print("1. 单次预测测试...")
    sample_data = pd.DataFrame(X_test[:1], columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    
    start_time = time.time()
    prediction = inference_engine.predict(sample_data)
    prediction_time = time.time() - start_time
    
    print(f"  预测结果: {prediction}")
    print(f"  预测时间: {prediction_time:.4f}s")
    
    # 测试缓存效果
    start_time = time.time()
    cached_prediction = inference_engine.predict(sample_data)
    cached_time = time.time() - start_time
    
    print(f"  缓存预测时间: {cached_time:.4f}s")
    print(f"  加速比: {prediction_time / cached_time:.2f}x")
    
    print("\n2. 批量预测测试...")
    batch_data = pd.DataFrame(X_test[:10], columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    
    start_time = time.time()
    batch_predictions = inference_engine.batch_predict(batch_data)
    batch_time = time.time() - start_time
    
    print(f"  批量预测结果: {batch_predictions}")
    print(f"  批量预测时间: {batch_time:.4f}s")
    print(f"  平均单次预测时间: {batch_time / len(batch_data):.4f}s")
    
    print("\n3. 异步预测测试...")
    
    async def async_prediction_test():
        tasks = []
        for i in range(5):
            sample = pd.DataFrame(X_test[i:i+1], columns=[f'feature_{i}' for i in range(X_test.shape[1])])
            task = inference_engine.predict_async(sample)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        async_time = time.time() - start_time
        
        print(f"  异步预测结果: {results}")
        print(f"  异步预测总时间: {async_time:.4f}s")
        
        return results
    
    # 运行异步预测
    async_results = asyncio.run(async_prediction_test())
    
    # 获取推理统计
    stats = inference_engine.get_statistics()
    print(f"\n推理引擎统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def streaming_inference_example():
    """流式推理示例"""
    print("\n=== 流式推理示例 ===")
    
    # 准备模型
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    trainer = ModelTrainer(
        model_type='logistic_regression',
        hyperparameters={'random_state': 42}
    )
    
    model = trainer.train(X_train, y_train)
    
    serializer = ModelSerializer()
    model_path = "models/streaming_model.pkl"
    serializer.save_model(model=model, file_path=model_path)
    
    # 创建流式推理引擎
    streaming_engine = StreamingInferenceEngine(
        model_path=model_path,
        batch_size=5,
        max_latency=0.1
    )
    
    print("1. 启动流式推理引擎...")
    streaming_engine.start()
    
    # 模拟数据流
    print("2. 模拟数据流处理...")
    
    def data_generator():
        """模拟实时数据流"""
        for i in range(20):
            sample = X_test[i % len(X_test)]
            yield pd.DataFrame([sample], columns=[f'feature_{j}' for j in range(len(sample))])
            time.sleep(0.05)  # 模拟数据到达间隔
    
    results = []
    start_time = time.time()
    
    for data in data_generator():
        result = streaming_engine.process_stream(data)
        if result is not None:
            results.extend(result)
    
    # 处理剩余的批次
    final_results = streaming_engine.flush()
    if final_results:
        results.extend(final_results)
    
    total_time = time.time() - start_time
    
    print(f"  处理了 {len(results)} 个预测")
    print(f"  总处理时间: {total_time:.4f}s")
    print(f"  平均延迟: {total_time / len(results):.4f}s")
    
    # 停止流式引擎
    streaming_engine.stop()
    print("3. 流式推理引擎已停止")


def model_ensemble_example():
    """模型集成示例"""
    print("\n=== 模型集成示例 ===")
    
    # 准备数据
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练多个不同的模型
    models_config = [
        {
            'name': 'rf_model',
            'type': 'random_forest',
            'params': {'n_estimators': 100, 'random_state': 42}
        },
        {
            'name': 'gb_model',
            'type': 'gradient_boosting',
            'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
        },
        {
            'name': 'lr_model',
            'type': 'logistic_regression',
            'params': {'random_state': 42}
        }
    ]
    
    model_paths = []
    serializer = ModelSerializer()
    
    print("1. 训练基础模型...")
    for config in models_config:
        trainer = ModelTrainer(
            model_type=config['type'],
            hyperparameters=config['params']
        )
        
        model = trainer.train(X_train, y_train)
        model_path = f"models/{config['name']}.pkl"
        
        serializer.save_model(
            model=model,
            file_path=model_path,
            metadata={'model_name': config['name']}
        )
        
        model_paths.append(model_path)
        print(f"  {config['name']} 训练完成")
    
    # 创建模型集成
    print("\n2. 创建模型集成...")
    ensemble = ModelEnsemble(
        model_paths=model_paths,
        ensemble_method='voting',
        voting_type='soft'
    )
    
    # 测试集成预测
    test_data = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    
    print("3. 集成预测测试...")
    
    # 单次预测
    sample = test_data.iloc[:1]
    ensemble_pred = ensemble.predict(sample)
    print(f"  集成预测结果: {ensemble_pred}")
    
    # 批量预测
    batch_preds = ensemble.batch_predict(test_data[:10])
    print(f"  批量预测结果: {batch_preds}")
    
    # 获取预测概率
    probabilities = ensemble.predict_proba(test_data[:5])
    print(f"  预测概率: {probabilities}")
    
    # 比较单个模型和集成模型的性能
    print("\n4. 性能比较...")
    validator = ModelValidator()
    
    # 评估单个模型
    for i, model_path in enumerate(model_paths):
        loader = ModelLoader()
        model, _ = loader.load_model(model_path)
        
        single_preds = model.predict(X_test)
        single_metrics = validator.calculate_metrics(y_test, single_preds)
        
        print(f"  {models_config[i]['name']} F1: {single_metrics['f1']:.4f}")
    
    # 评估集成模型
    ensemble_preds = ensemble.batch_predict(test_data)
    ensemble_metrics = validator.calculate_metrics(y_test, ensemble_preds)
    print(f"  集成模型 F1: {ensemble_metrics['f1']:.4f}")


def deployment_management_example():
    """部署管理示例"""
    print("\n=== 部署管理示例 ===")
    
    # 准备模型
    X, y = make_classification(
        n_samples=800,
        n_features=8,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    trainer = ModelTrainer(
        model_type='random_forest',
        hyperparameters={'n_estimators': 50, 'random_state': 42}
    )
    
    model = trainer.train(X_train, y_train)
    
    serializer = ModelSerializer()
    model_path = "models/deployment_model.pkl"
    serializer.save_model(model=model, file_path=model_path)
    
    # 创建部署管理器
    deployment_manager = DeploymentManager()
    
    print("1. 创建本地部署...")
    
    # 部署配置
    deployment_config = {
        'model_path': model_path,
        'deployment_type': 'local',
        'host': 'localhost',
        'port': 8080,
        'workers': 2,
        'timeout': 30
    }
    
    # 创建部署
    deployment = deployment_manager.create_deployment(
        deployment_id="test_deployment",
        config=deployment_config
    )
    
    print(f"  部署ID: {deployment.deployment_id}")
    print(f"  部署状态: {deployment.status}")
    
    print("\n2. 启动部署...")
    deployment_manager.start_deployment("test_deployment")
    
    # 等待部署启动
    time.sleep(2)
    
    # 检查部署状态
    status = deployment_manager.get_deployment_status("test_deployment")
    print(f"  部署状态: {status}")
    
    print("\n3. 健康检查...")
    health_status = deployment_manager.health_check("test_deployment")
    print(f"  健康状态: {health_status}")
    
    print("\n4. 测试部署预测...")
    
    # 模拟预测请求
    test_sample = pd.DataFrame(X_test[:1], columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    
    try:
        prediction = deployment_manager.predict("test_deployment", test_sample)
        print(f"  预测结果: {prediction}")
    except Exception as e:
        print(f"  预测失败: {str(e)}")
    
    print("\n5. 获取部署指标...")
    metrics = deployment_manager.get_deployment_metrics("test_deployment")
    print(f"  部署指标: {metrics}")
    
    print("\n6. 停止部署...")
    deployment_manager.stop_deployment("test_deployment")
    
    # 检查最终状态
    final_status = deployment_manager.get_deployment_status("test_deployment")
    print(f"  最终状态: {final_status}")


if __name__ == "__main__":
    # 设置日志
    log_manager = create_log_manager(
        log_level='INFO',
        log_file='logs/deployment_example.log'
    )
    
    logger = log_manager.get_logger('deployment_example')
    logger.info("开始部署示例")
    
    try:
        model_serialization_example()
        inference_engine_example()
        streaming_inference_example()
        model_ensemble_example()
        deployment_management_example()
        
        print("\n所有部署示例运行完成！")
        logger.info("所有部署示例运行完成")
        
    except Exception as e:
        logger.error(f"部署示例运行出错: {str(e)}")
        raise