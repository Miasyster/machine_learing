"""
模型训练器测试模块

测试ModelTrainer类的各种训练功能
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from unittest.mock import patch, MagicMock
import tempfile
import os
import joblib

from src.training.trainers import ModelTrainer, TrainingError


class TestModelTrainer(unittest.TestCase):
    """ModelTrainer类的测试用例"""
    
    def setUp(self):
        """测试前的设置"""
        self.trainer = ModelTrainer()
        
        # 创建分类测试数据
        self.X_class, self.y_class = make_classification(
            n_samples=100, 
            n_features=10, 
            n_classes=2, 
            random_state=42
        )
        self.X_train_class, self.X_test_class, self.y_train_class, self.y_test_class = train_test_split(
            self.X_class, self.y_class, test_size=0.2, random_state=42
        )
        
        # 创建回归测试数据
        self.X_reg, self.y_reg = make_regression(
            n_samples=100, 
            n_features=10, 
            noise=0.1, 
            random_state=42
        )
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(
            self.X_reg, self.y_reg, test_size=0.2, random_state=42
        )
        
        # 临时目录
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_classification_model(self):
        """测试分类模型训练"""
        model = RandomForestClassifier(random_state=42)
        
        trained_model = self.trainer.train(
            model, 
            self.X_train_class, 
            self.y_train_class
        )
        
        # 验证结果
        self.assertIsNotNone(trained_model)
        self.assertTrue(hasattr(trained_model, 'predict'))
        
        # 测试预测
        predictions = trained_model.predict(self.X_test_class)
        self.assertEqual(len(predictions), len(self.y_test_class))
        
        # 计算准确率
        accuracy = accuracy_score(self.y_test_class, predictions)
        self.assertGreater(accuracy, 0.5)  # 应该比随机猜测好
    
    def test_train_regression_model(self):
        """测试回归模型训练"""
        model = RandomForestRegressor(random_state=42)
        
        trained_model = self.trainer.train(
            model, 
            self.X_train_reg, 
            self.y_train_reg
        )
        
        # 验证结果
        self.assertIsNotNone(trained_model)
        self.assertTrue(hasattr(trained_model, 'predict'))
        
        # 测试预测
        predictions = trained_model.predict(self.X_test_reg)
        self.assertEqual(len(predictions), len(self.y_test_reg))
        
        # 计算MSE
        mse = mean_squared_error(self.y_test_reg, predictions)
        self.assertIsInstance(mse, float)
        self.assertGreater(mse, 0)
    
    def test_train_with_validation_data(self):
        """测试使用验证数据训练"""
        model = RandomForestClassifier(random_state=42)
        
        trained_model = self.trainer.train(
            model, 
            self.X_train_class, 
            self.y_train_class,
            validation_data=(self.X_test_class, self.y_test_class)
        )
        
        # 验证结果
        self.assertIsNotNone(trained_model)
        self.assertTrue(hasattr(trained_model, 'predict'))
    
    def test_train_with_early_stopping(self):
        """测试早停训练"""
        # 使用支持早停的模型（这里模拟）
        model = RandomForestClassifier(random_state=42)
        
        trained_model = self.trainer.train(
            model, 
            self.X_train_class, 
            self.y_train_class,
            validation_data=(self.X_test_class, self.y_test_class),
            early_stopping=True,
            patience=5
        )
        
        # 验证结果
        self.assertIsNotNone(trained_model)
    
    def test_train_with_callbacks(self):
        """测试使用回调函数训练"""
        callback_called = []
        
        def test_callback(epoch, logs):
            callback_called.append(epoch)
        
        model = RandomForestClassifier(random_state=42)
        
        trained_model = self.trainer.train(
            model, 
            self.X_train_class, 
            self.y_train_class,
            callbacks=[test_callback]
        )
        
        # 验证结果
        self.assertIsNotNone(trained_model)
        # 注意：对于sklearn模型，回调可能不会被调用
    
    def test_evaluate_model(self):
        """测试模型评估"""
        model = RandomForestClassifier(random_state=42)
        trained_model = model.fit(self.X_train_class, self.y_train_class)
        
        metrics = self.trainer.evaluate(
            trained_model, 
            self.X_test_class, 
            self.y_test_class,
            task_type='classification'
        )
        
        # 验证结果
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # 检查指标值的合理性
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_evaluate_regression_model(self):
        """测试回归模型评估"""
        model = RandomForestRegressor(random_state=42)
        trained_model = model.fit(self.X_train_reg, self.y_train_reg)
        
        metrics = self.trainer.evaluate(
            trained_model, 
            self.X_test_reg, 
            self.y_test_reg,
            task_type='regression'
        )
        
        # 验证结果
        self.assertIsInstance(metrics, dict)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # 检查指标值的合理性
        self.assertGreaterEqual(metrics['mse'], 0)
        self.assertGreaterEqual(metrics['rmse'], 0)
        self.assertGreaterEqual(metrics['mae'], 0)
    
    def test_cross_validation(self):
        """测试交叉验证"""
        model = RandomForestClassifier(random_state=42)
        
        cv_scores = self.trainer.cross_validate(
            model, 
            self.X_class, 
            self.y_class,
            cv=3,
            scoring='accuracy'
        )
        
        # 验证结果
        self.assertIsInstance(cv_scores, dict)
        self.assertIn('test_scores', cv_scores)
        self.assertIn('mean_score', cv_scores)
        self.assertIn('std_score', cv_scores)
        
        # 检查分数数量
        self.assertEqual(len(cv_scores['test_scores']), 3)
        
        # 检查分数合理性
        self.assertGreaterEqual(cv_scores['mean_score'], 0)
        self.assertLessEqual(cv_scores['mean_score'], 1)
    
    def test_hyperparameter_tuning_grid_search(self):
        """测试网格搜索超参数调优"""
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        best_model, best_params = self.trainer.hyperparameter_tuning(
            model, 
            self.X_train_class, 
            self.y_train_class,
            param_grid=param_grid,
            method='grid_search',
            cv=3,
            scoring='accuracy'
        )
        
        # 验证结果
        self.assertIsNotNone(best_model)
        self.assertIsInstance(best_params, dict)
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)
        
        # 验证最佳参数在搜索范围内
        self.assertIn(best_params['n_estimators'], [10, 20])
        self.assertIn(best_params['max_depth'], [3, 5])
    
    def test_hyperparameter_tuning_random_search(self):
        """测试随机搜索超参数调优"""
        model = RandomForestClassifier(random_state=42)
        param_distributions = {
            'n_estimators': [10, 20, 30],
            'max_depth': [3, 5, 7]
        }
        
        best_model, best_params = self.trainer.hyperparameter_tuning(
            model, 
            self.X_train_class, 
            self.y_train_class,
            param_grid=param_distributions,
            method='random_search',
            n_iter=5,
            cv=3,
            scoring='accuracy'
        )
        
        # 验证结果
        self.assertIsNotNone(best_model)
        self.assertIsInstance(best_params, dict)
    
    def test_hyperparameter_tuning_invalid_method(self):
        """测试无效的超参数调优方法"""
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [10, 20]}
        
        with self.assertRaises(TrainingError):
            self.trainer.hyperparameter_tuning(
                model, 
                self.X_train_class, 
                self.y_train_class,
                param_grid=param_grid,
                method='invalid_method'
            )
    
    def test_save_and_load_model(self):
        """测试模型保存和加载"""
        model = RandomForestClassifier(random_state=42)
        trained_model = model.fit(self.X_train_class, self.y_train_class)
        
        # 保存模型
        model_path = os.path.join(self.temp_dir, 'test_model.pkl')
        self.trainer.save_model(trained_model, model_path)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(model_path))
        
        # 加载模型
        loaded_model = self.trainer.load_model(model_path)
        
        # 验证加载的模型
        self.assertIsNotNone(loaded_model)
        
        # 测试预测一致性
        original_predictions = trained_model.predict(self.X_test_class)
        loaded_predictions = loaded_model.predict(self.X_test_class)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_load_nonexistent_model(self):
        """测试加载不存在的模型"""
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_model('nonexistent_model.pkl')
    
    def test_get_feature_importance(self):
        """测试获取特征重要性"""
        model = RandomForestClassifier(random_state=42)
        trained_model = model.fit(self.X_train_class, self.y_train_class)
        
        importance = self.trainer.get_feature_importance(trained_model)
        
        # 验证结果
        self.assertIsInstance(importance, np.ndarray)
        self.assertEqual(len(importance), self.X_train_class.shape[1])
        
        # 重要性值应该在合理范围内
        self.assertTrue(np.all(importance >= 0))
        self.assertAlmostEqual(np.sum(importance), 1.0, places=5)
    
    def test_get_feature_importance_unsupported_model(self):
        """测试不支持特征重要性的模型"""
        # 使用不支持feature_importances_的模型
        model = LogisticRegression()
        trained_model = model.fit(self.X_train_class, self.y_train_class)
        
        # 应该返回None或抛出异常
        try:
            importance = self.trainer.get_feature_importance(trained_model)
            # 如果返回了结果，应该是合理的
            if importance is not None:
                self.assertIsInstance(importance, np.ndarray)
        except AttributeError:
            # 如果抛出异常也是可以接受的
            pass
    
    def test_learning_curve(self):
        """测试学习曲线"""
        model = RandomForestClassifier(random_state=42)
        
        train_sizes, train_scores, val_scores = self.trainer.learning_curve(
            model, 
            self.X_class, 
            self.y_class,
            cv=3,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        
        # 验证结果
        self.assertEqual(len(train_sizes), 5)
        self.assertEqual(train_scores.shape[0], 5)
        self.assertEqual(val_scores.shape[0], 5)
        self.assertEqual(train_scores.shape[1], 3)  # cv=3
        self.assertEqual(val_scores.shape[1], 3)
    
    def test_validation_curve(self):
        """测试验证曲线"""
        model = RandomForestClassifier(random_state=42)
        param_name = 'n_estimators'
        param_range = [10, 20, 30]
        
        train_scores, val_scores = self.trainer.validation_curve(
            model, 
            self.X_class, 
            self.y_class,
            param_name=param_name,
            param_range=param_range,
            cv=3
        )
        
        # 验证结果
        self.assertEqual(train_scores.shape[0], 3)  # 3个参数值
        self.assertEqual(val_scores.shape[0], 3)
        self.assertEqual(train_scores.shape[1], 3)  # cv=3
        self.assertEqual(val_scores.shape[1], 3)
    
    def test_ensemble_training(self):
        """测试集成模型训练"""
        models = [
            ('rf', RandomForestClassifier(random_state=42)),
            ('lr', LogisticRegression(random_state=42))
        ]
        
        ensemble_model = self.trainer.train_ensemble(
            models, 
            self.X_train_class, 
            self.y_train_class,
            method='voting'
        )
        
        # 验证结果
        self.assertIsNotNone(ensemble_model)
        self.assertTrue(hasattr(ensemble_model, 'predict'))
        
        # 测试预测
        predictions = ensemble_model.predict(self.X_test_class)
        self.assertEqual(len(predictions), len(self.y_test_class))
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = {
            'random_state': 42,
            'n_jobs': -1,
            'verbose': True
        }
        
        trainer_with_config = ModelTrainer(config)
        self.assertEqual(trainer_with_config.config['random_state'], 42)
        self.assertEqual(trainer_with_config.config['n_jobs'], -1)
        self.assertTrue(trainer_with_config.config['verbose'])


class TestModelTrainerIntegration(unittest.TestCase):
    """ModelTrainer集成测试"""
    
    def setUp(self):
        """测试前的设置"""
        self.trainer = ModelTrainer()
        
        # 创建更复杂的测试数据
        self.X, self.y = make_classification(
            n_samples=500, 
            n_features=20, 
            n_informative=10,
            n_redundant=5,
            n_classes=3, 
            random_state=42
        )
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_training_pipeline(self):
        """测试完整的训练管道"""
        # 1. 超参数调优
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
        
        best_model, best_params = self.trainer.hyperparameter_tuning(
            model, 
            self.X_train, 
            self.y_train,
            param_grid=param_grid,
            method='grid_search',
            cv=3
        )
        
        # 2. 模型评估
        metrics = self.trainer.evaluate(
            best_model, 
            self.X_test, 
            self.y_test,
            task_type='classification'
        )
        
        # 3. 保存模型
        model_path = os.path.join(self.temp_dir, 'best_model.pkl')
        self.trainer.save_model(best_model, model_path)
        
        # 4. 加载模型并验证
        loaded_model = self.trainer.load_model(model_path)
        loaded_predictions = loaded_model.predict(self.X_test)
        original_predictions = best_model.predict(self.X_test)
        
        # 验证结果
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertTrue(os.path.exists(model_path))
        np.testing.assert_array_equal(loaded_predictions, original_predictions)
    
    def test_model_comparison(self):
        """测试多个模型比较"""
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            # 训练模型
            trained_model = self.trainer.train(model, self.X_train, self.y_train)
            
            # 评估模型
            metrics = self.trainer.evaluate(
                trained_model, 
                self.X_test, 
                self.y_test,
                task_type='classification'
            )
            
            results[name] = metrics
        
        # 验证结果
        self.assertEqual(len(results), 2)
        for name, metrics in results.items():
            self.assertIn('accuracy', metrics)
            self.assertGreaterEqual(metrics['accuracy'], 0)
            self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_cross_validation_with_multiple_metrics(self):
        """测试多指标交叉验证"""
        model = RandomForestClassifier(random_state=42)
        
        # 测试多个评分指标
        scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        for metric in scoring_metrics:
            cv_scores = self.trainer.cross_validate(
                model, 
                self.X, 
                self.y,
                cv=3,
                scoring=metric
            )
            
            # 验证结果
            self.assertIsInstance(cv_scores, dict)
            self.assertIn('test_scores', cv_scores)
            self.assertEqual(len(cv_scores['test_scores']), 3)


if __name__ == '__main__':
    unittest.main()