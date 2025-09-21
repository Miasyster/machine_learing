"""
推理引擎测试模块

测试InferenceEngine类的各种推理功能
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from unittest.mock import patch, MagicMock
import tempfile
import os
import time
import threading
import asyncio

from src.deployment.inference import InferenceEngine, InferenceError


class TestInferenceEngine(unittest.TestCase):
    """InferenceEngine类的测试用例"""
    
    def setUp(self):
        """测试前的设置"""
        # 创建测试数据和模型
        self.X, self.y = make_classification(
            n_samples=100, 
            n_features=10, 
            n_classes=2, 
            random_state=42
        )
        
        # 训练一个简单的模型
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X, self.y)
        
        # 创建推理引擎
        self.engine = InferenceEngine(self.model)
        
        # 测试数据
        self.test_sample = self.X[0:1]  # 单个样本
        self.test_batch = self.X[0:5]   # 批量样本
        
        # 临时目录
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_single_prediction(self):
        """测试单个样本预测"""
        prediction = self.engine.predict(self.test_sample)
        
        # 验证结果
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], [0, 1])  # 二分类结果
    
    def test_batch_prediction(self):
        """测试批量预测"""
        predictions = self.engine.predict(self.test_batch)
        
        # 验证结果
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_predict_proba(self):
        """测试概率预测"""
        probabilities = self.engine.predict_proba(self.test_batch)
        
        # 验证结果
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, (5, 2))  # 5个样本，2个类别
        
        # 概率值应该在[0,1]范围内，每行和为1
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)
    
    def test_predict_with_preprocessing(self):
        """测试带预处理的预测"""
        def preprocess_func(data):
            # 简单的预处理：标准化
            return (data - data.mean()) / data.std()
        
        engine_with_preprocess = InferenceEngine(
            self.model, 
            preprocessing_func=preprocess_func
        )
        
        prediction = engine_with_preprocess.predict(self.test_sample)
        
        # 验证结果
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(len(prediction), 1)
    
    def test_predict_with_postprocessing(self):
        """测试带后处理的预测"""
        def postprocess_func(predictions):
            # 简单的后处理：转换为字符串标签
            return ['positive' if pred == 1 else 'negative' for pred in predictions]
        
        engine_with_postprocess = InferenceEngine(
            self.model, 
            postprocessing_func=postprocess_func
        )
        
        prediction = engine_with_postprocess.predict(self.test_sample)
        
        # 验证结果
        self.assertIsInstance(prediction, list)
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], ['positive', 'negative'])
    
    def test_predict_with_caching(self):
        """测试缓存预测"""
        engine_with_cache = InferenceEngine(
            self.model, 
            enable_cache=True,
            cache_size=100
        )
        
        # 第一次预测
        start_time = time.time()
        prediction1 = engine_with_cache.predict(self.test_sample)
        first_time = time.time() - start_time
        
        # 第二次预测（应该使用缓存）
        start_time = time.time()
        prediction2 = engine_with_cache.predict(self.test_sample)
        second_time = time.time() - start_time
        
        # 验证结果
        np.testing.assert_array_equal(prediction1, prediction2)
        # 注意：由于测试环境的差异，时间比较可能不稳定
        # self.assertLess(second_time, first_time)  # 缓存应该更快
    
    def test_predict_invalid_input(self):
        """测试无效输入"""
        # 错误的特征数量
        invalid_input = np.array([[1, 2, 3]])  # 只有3个特征，应该是10个
        
        with self.assertRaises(ValueError):
            self.engine.predict(invalid_input)
    
    def test_predict_empty_input(self):
        """测试空输入"""
        empty_input = np.array([]).reshape(0, 10)
        
        prediction = self.engine.predict(empty_input)
        
        # 验证结果
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(len(prediction), 0)
    
    def test_batch_predict_with_batch_size(self):
        """测试指定批次大小的批量预测"""
        large_batch = self.X[0:20]  # 20个样本
        
        predictions = self.engine.batch_predict(
            large_batch, 
            batch_size=5
        )
        
        # 验证结果
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), 20)
    
    def test_async_predict(self):
        """测试异步预测"""
        async def run_async_test():
            prediction = await self.engine.predict_async(self.test_sample)
            return prediction
        
        # 运行异步测试
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            prediction = loop.run_until_complete(run_async_test())
            
            # 验证结果
            self.assertIsInstance(prediction, np.ndarray)
            self.assertEqual(len(prediction), 1)
        finally:
            loop.close()
    
    def test_concurrent_predictions(self):
        """测试并发预测"""
        def predict_worker(results, index):
            prediction = self.engine.predict(self.test_sample)
            results[index] = prediction
        
        # 创建多个线程进行并发预测
        num_threads = 5
        threads = []
        results = {}
        
        for i in range(num_threads):
            thread = threading.Thread(
                target=predict_worker, 
                args=(results, i)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(len(results), num_threads)
        for i in range(num_threads):
            self.assertIsInstance(results[i], np.ndarray)
            self.assertEqual(len(results[i]), 1)
    
    def test_model_loading_from_file(self):
        """测试从文件加载模型"""
        # 保存模型到文件
        model_path = os.path.join(self.temp_dir, 'test_model.pkl')
        import joblib
        joblib.dump(self.model, model_path)
        
        # 从文件创建推理引擎
        engine_from_file = InferenceEngine.from_file(model_path)
        
        # 测试预测
        prediction = engine_from_file.predict(self.test_sample)
        
        # 验证结果
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(len(prediction), 1)
    
    def test_model_loading_from_nonexistent_file(self):
        """测试从不存在的文件加载模型"""
        with self.assertRaises(FileNotFoundError):
            InferenceEngine.from_file('nonexistent_model.pkl')
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        info = self.engine.get_model_info()
        
        # 验证结果
        self.assertIsInstance(info, dict)
        self.assertIn('model_type', info)
        self.assertIn('input_shape', info)
        self.assertIn('output_shape', info)
        
        # 检查具体信息
        self.assertEqual(info['model_type'], 'RandomForestClassifier')
        self.assertEqual(info['input_shape'], (None, 10))  # 10个特征
    
    def test_warm_up(self):
        """测试模型预热"""
        # 预热模型
        self.engine.warm_up(num_samples=5)
        
        # 预热后预测应该正常工作
        prediction = self.engine.predict(self.test_sample)
        self.assertIsInstance(prediction, np.ndarray)
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        engine_with_monitoring = InferenceEngine(
            self.model, 
            enable_monitoring=True
        )
        
        # 执行一些预测
        for _ in range(10):
            engine_with_monitoring.predict(self.test_sample)
        
        # 获取性能统计
        stats = engine_with_monitoring.get_performance_stats()
        
        # 验证结果
        self.assertIsInstance(stats, dict)
        self.assertIn('total_predictions', stats)
        self.assertIn('average_latency', stats)
        self.assertIn('throughput', stats)
        
        # 检查统计值
        self.assertEqual(stats['total_predictions'], 10)
        self.assertGreater(stats['average_latency'], 0)
        self.assertGreater(stats['throughput'], 0)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 创建一个会抛出异常的模型
        class ErrorModel:
            def predict(self, X):
                raise RuntimeError("Model prediction failed")
        
        error_engine = InferenceEngine(ErrorModel())
        
        with self.assertRaises(InferenceError):
            error_engine.predict(self.test_sample)
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = {
            'enable_cache': True,
            'cache_size': 200,
            'enable_monitoring': True,
            'batch_size': 32
        }
        
        engine_with_config = InferenceEngine(self.model, config=config)
        
        # 验证配置
        self.assertTrue(engine_with_config.config['enable_cache'])
        self.assertEqual(engine_with_config.config['cache_size'], 200)
        self.assertTrue(engine_with_config.config['enable_monitoring'])
        self.assertEqual(engine_with_config.config['batch_size'], 32)


class TestInferenceEngineIntegration(unittest.TestCase):
    """InferenceEngine集成测试"""
    
    def setUp(self):
        """测试前的设置"""
        # 创建更复杂的测试场景
        self.X, self.y = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_classes=3, 
            random_state=42
        )
        
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X, self.y)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_inference_pipeline(self):
        """测试端到端推理管道"""
        # 定义预处理函数
        def preprocess(data):
            return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        
        # 定义后处理函数
        def postprocess(predictions):
            class_names = ['Class_A', 'Class_B', 'Class_C']
            return [class_names[pred] for pred in predictions]
        
        # 创建完整的推理引擎
        engine = InferenceEngine(
            self.model,
            preprocessing_func=preprocess,
            postprocessing_func=postprocess,
            enable_cache=True,
            enable_monitoring=True
        )
        
        # 执行推理
        test_data = self.X[0:10]
        predictions = engine.predict(test_data)
        
        # 验证结果
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(pred in ['Class_A', 'Class_B', 'Class_C'] for pred in predictions))
        
        # 检查性能统计
        stats = engine.get_performance_stats()
        self.assertEqual(stats['total_predictions'], 10)
    
    def test_high_throughput_inference(self):
        """测试高吞吐量推理"""
        engine = InferenceEngine(
            self.model,
            enable_cache=True,
            config={'batch_size': 50}
        )
        
        # 大批量数据
        large_batch = self.X[0:500]
        
        # 测量推理时间
        start_time = time.time()
        predictions = engine.batch_predict(large_batch, batch_size=50)
        end_time = time.time()
        
        # 验证结果
        self.assertEqual(len(predictions), 500)
        
        # 计算吞吐量
        throughput = len(predictions) / (end_time - start_time)
        self.assertGreater(throughput, 0)
        
        print(f"Throughput: {throughput:.2f} predictions/second")
    
    def test_model_serialization_and_inference(self):
        """测试模型序列化和推理"""
        # 保存模型
        model_path = os.path.join(self.temp_dir, 'serialized_model.pkl')
        import joblib
        joblib.dump(self.model, model_path)
        
        # 从文件加载并创建推理引擎
        loaded_engine = InferenceEngine.from_file(model_path)
        
        # 比较原始模型和加载模型的预测结果
        test_data = self.X[0:20]
        
        original_predictions = self.model.predict(test_data)
        loaded_predictions = loaded_engine.predict(test_data)
        
        # 验证结果一致性
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_concurrent_inference_stress(self):
        """测试并发推理压力测试"""
        engine = InferenceEngine(
            self.model,
            enable_cache=True,
            enable_monitoring=True
        )
        
        def stress_worker(results, worker_id, num_predictions):
            worker_results = []
            for i in range(num_predictions):
                sample_idx = (worker_id * num_predictions + i) % len(self.X)
                prediction = engine.predict(self.X[sample_idx:sample_idx+1])
                worker_results.append(prediction[0])
            results[worker_id] = worker_results
        
        # 启动多个工作线程
        num_workers = 10
        predictions_per_worker = 20
        threads = []
        results = {}
        
        start_time = time.time()
        
        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=stress_worker,
                args=(results, worker_id, predictions_per_worker)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # 验证结果
        self.assertEqual(len(results), num_workers)
        total_predictions = sum(len(worker_results) for worker_results in results.values())
        self.assertEqual(total_predictions, num_workers * predictions_per_worker)
        
        # 检查性能统计
        stats = engine.get_performance_stats()
        self.assertEqual(stats['total_predictions'], total_predictions)
        
        print(f"Concurrent inference completed: {total_predictions} predictions in {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    unittest.main()