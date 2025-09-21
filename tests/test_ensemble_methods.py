"""
集成方法测试脚本

测试stacking、blending和模型校准功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import matplotlib.pyplot as plt
import logging

# 导入我们的集成方法
from src.ensemble.stacking import StackingEnsemble, MultiLevelStackingEnsemble, DynamicStackingEnsemble
from src.ensemble.blending import BlendingEnsemble, DynamicBlendingEnsemble, AdaptiveBlendingEnsemble
from src.ensemble.calibration import ModelCalibrator, CalibratedEnsemble, TemperatureScaling

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnsembleMethodTester:
    """集成方法测试类"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        
    def generate_classification_data(self, n_samples=1000, n_features=20, n_classes=3):
        """生成分类数据"""
        logger.info(f"生成分类数据: {n_samples}样本, {n_features}特征, {n_classes}类别")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            random_state=self.random_state
        )
        
        return train_test_split(X, y, test_size=0.3, random_state=self.random_state)
    
    def generate_regression_data(self, n_samples=1000, n_features=20):
        """生成回归数据"""
        logger.info(f"生成回归数据: {n_samples}样本, {n_features}特征")
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            noise=0.1,
            random_state=self.random_state
        )
        
        return train_test_split(X, y, test_size=0.3, random_state=self.random_state)
    
    def create_base_classifiers(self):
        """创建基础分类器"""
        return [
            RandomForestClassifier(n_estimators=50, random_state=self.random_state),
            LogisticRegression(random_state=self.random_state, max_iter=1000),
            SVC(probability=True, random_state=self.random_state)
        ]
    
    def create_base_regressors(self):
        """创建基础回归器"""
        return [
            RandomForestRegressor(n_estimators=50, random_state=self.random_state),
            LinearRegression(),
            SVR()
        ]
    
    def test_stacking_classification(self):
        """测试Stacking分类"""
        logger.info("=== 测试Stacking分类 ===")
        
        X_train, X_test, y_train, y_test = self.generate_classification_data()
        base_models = self.create_base_classifiers()
        
        # 基础Stacking
        stacking = StackingEnsemble(
            base_models=base_models,
            meta_model=LogisticRegression(random_state=self.random_state),
            cv=3,
            verbose=True
        )
        
        stacking.fit(X_train, y_train)
        result = stacking.predict(X_test)
        
        accuracy = accuracy_score(y_test, result.predictions)
        logger.info(f"Stacking分类准确率: {accuracy:.4f}")
        
        self.results['stacking_classification'] = {
            'accuracy': accuracy,
            'predictions': result.predictions,
            'probabilities': result.prediction_probabilities
        }
        
        # 多层Stacking
        try:
            multi_stacking = MultiLevelStackingEnsemble(
                base_models=base_models,
                meta_models=[
                    LogisticRegression(random_state=self.random_state),
                    RandomForestClassifier(n_estimators=10, random_state=self.random_state)
                ],
                cv=3,
                verbose=True
            )
            
            multi_stacking.fit(X_train, y_train)
            multi_result = multi_stacking.predict(X_test)
            
            multi_accuracy = accuracy_score(y_test, multi_result.predictions)
            logger.info(f"多层Stacking分类准确率: {multi_accuracy:.4f}")
            
            self.results['multi_stacking_classification'] = {
                'accuracy': multi_accuracy,
                'predictions': multi_result.predictions
            }
        except Exception as e:
            logger.error(f"多层Stacking测试失败: {str(e)}")
    
    def test_blending_classification(self):
        """测试Blending分类"""
        logger.info("=== 测试Blending分类 ===")
        
        X_train, X_test, y_train, y_test = self.generate_classification_data()
        base_models = self.create_base_classifiers()
        
        # 基础Blending
        blending = BlendingEnsemble(
            base_models=base_models,
            meta_model=LogisticRegression(random_state=self.random_state),
            holdout_size=0.2,
            verbose=True
        )
        
        blending.fit(X_train, y_train)
        result = blending.predict(X_test)
        
        accuracy = accuracy_score(y_test, result.predictions)
        logger.info(f"Blending分类准确率: {accuracy:.4f}")
        
        self.results['blending_classification'] = {
            'accuracy': accuracy,
            'predictions': result.predictions,
            'probabilities': result.prediction_probabilities
        }
        
        # 自适应Blending
        try:
            adaptive_blending = AdaptiveBlendingEnsemble(
                base_models=base_models,
                meta_model=LogisticRegression(random_state=self.random_state),
                adaptation_rate=0.1,
                verbose=True
            )
            
            adaptive_blending.fit(X_train, y_train)
            adaptive_result = adaptive_blending.predict(X_test, y_true=y_test)
            
            adaptive_accuracy = accuracy_score(y_test, adaptive_result.predictions)
            logger.info(f"自适应Blending分类准确率: {adaptive_accuracy:.4f}")
            
            self.results['adaptive_blending_classification'] = {
                'accuracy': adaptive_accuracy,
                'predictions': adaptive_result.predictions
            }
        except Exception as e:
            logger.error(f"自适应Blending测试失败: {str(e)}")
    
    def test_model_calibration(self):
        """测试模型校准"""
        logger.info("=== 测试模型校准 ===")
        
        X_train, X_test, y_train, y_test = self.generate_classification_data()
        base_models = self.create_base_classifiers()
        
        # 创建未校准的集成模型
        ensemble = StackingEnsemble(
            base_models=base_models,
            meta_model=LogisticRegression(random_state=self.random_state),
            cv=3
        )
        
        # 测试模型校准器
        calibrator = ModelCalibrator(method='platt')
        calibrated_ensemble = CalibratedEnsemble(
            base_ensemble=ensemble,
            calibrator=calibrator,
            cv=3
        )
        
        calibrated_ensemble.fit(X_train, y_train)
        result = calibrated_ensemble.predict(X_test)
        
        accuracy = accuracy_score(y_test, result.predictions)
        logger.info(f"校准后集成模型准确率: {accuracy:.4f}")
        
        self.results['calibrated_ensemble'] = {
            'accuracy': accuracy,
            'predictions': result.predictions,
            'probabilities': result.prediction_probabilities
        }
        
        # 测试温度缩放
        try:
            temp_scaling = TemperatureScaling()
            
            # 先训练基础模型获取logits
            ensemble.fit(X_train, y_train)
            base_result = ensemble.predict(X_train)
            
            if base_result.prediction_probabilities is not None:
                # 将概率转换为logits进行温度缩放
                logits = np.log(base_result.prediction_probabilities + 1e-8)
                temp_scaling.fit(logits, y_train)
                
                test_result = ensemble.predict(X_test)
                if test_result.prediction_probabilities is not None:
                    test_logits = np.log(test_result.prediction_probabilities + 1e-8)
                    calibrated_probs = temp_scaling.predict_proba(test_logits)
                    
                    logger.info("温度缩放校准完成")
                    self.results['temperature_scaling'] = {
                        'calibrated_probabilities': calibrated_probs
                    }
        except Exception as e:
            logger.error(f"温度缩放测试失败: {str(e)}")
    
    def test_regression_methods(self):
        """测试回归方法"""
        logger.info("=== 测试回归方法 ===")
        
        X_train, X_test, y_train, y_test = self.generate_regression_data()
        base_models = self.create_base_regressors()
        
        # Stacking回归
        stacking_reg = StackingEnsemble(
            base_models=base_models,
            meta_model=LinearRegression(),
            cv=3,
            verbose=True
        )
        
        stacking_reg.fit(X_train, y_train)
        stacking_result = stacking_reg.predict(X_test)
        
        stacking_mse = mean_squared_error(y_test, stacking_result.predictions)
        logger.info(f"Stacking回归MSE: {stacking_mse:.4f}")
        
        # Blending回归
        blending_reg = BlendingEnsemble(
            base_models=base_models,
            meta_model=LinearRegression(),
            holdout_size=0.2,
            verbose=True
        )
        
        blending_reg.fit(X_train, y_train)
        blending_result = blending_reg.predict(X_test)
        
        blending_mse = mean_squared_error(y_test, blending_result.predictions)
        logger.info(f"Blending回归MSE: {blending_mse:.4f}")
        
        self.results['regression'] = {
            'stacking_mse': stacking_mse,
            'blending_mse': blending_mse
        }
    
    def test_error_handling(self):
        """测试错误处理"""
        logger.info("=== 测试错误处理 ===")
        
        X_train, X_test, y_train, y_test = self.generate_classification_data()
        
        # 创建一个会失败的模型
        class FailingModel:
            def fit(self, X, y):
                raise ValueError("Intentional failure")
            
            def predict(self, X):
                raise ValueError("Intentional failure")
        
        base_models = self.create_base_classifiers()
        base_models.append(FailingModel())  # 添加会失败的模型
        
        try:
            stacking = StackingEnsemble(
                base_models=base_models,
                meta_model=LogisticRegression(random_state=self.random_state),
                cv=3,
                verbose=True
            )
            
            stacking.fit(X_train, y_train)
            result = stacking.predict(X_test)
            
            accuracy = accuracy_score(y_test, result.predictions)
            logger.info(f"带失败模型的Stacking准确率: {accuracy:.4f}")
            
            self.results['error_handling'] = {
                'accuracy': accuracy,
                'handled_errors': True
            }
            
        except Exception as e:
            logger.error(f"错误处理测试失败: {str(e)}")
            self.results['error_handling'] = {
                'handled_errors': False,
                'error': str(e)
            }
    
    def generate_report(self):
        """生成测试报告"""
        logger.info("=== 生成测试报告 ===")
        
        print("\n" + "="*60)
        print("集成方法测试报告")
        print("="*60)
        
        for test_name, results in self.results.items():
            print(f"\n{test_name.upper()}:")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, bool):
                    print(f"  {key}: {value}")
                elif isinstance(value, str):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: <数组数据>")
        
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行所有集成方法测试...")
        
        try:
            self.test_stacking_classification()
        except Exception as e:
            logger.error(f"Stacking分类测试失败: {str(e)}")
        
        try:
            self.test_blending_classification()
        except Exception as e:
            logger.error(f"Blending分类测试失败: {str(e)}")
        
        try:
            self.test_model_calibration()
        except Exception as e:
            logger.error(f"模型校准测试失败: {str(e)}")
        
        try:
            self.test_regression_methods()
        except Exception as e:
            logger.error(f"回归方法测试失败: {str(e)}")
        
        try:
            self.test_error_handling()
        except Exception as e:
            logger.error(f"错误处理测试失败: {str(e)}")
        
        self.generate_report()


def main():
    """主函数"""
    print("集成方法测试脚本")
    print("测试stacking、blending和模型校准功能")
    print("-" * 50)
    
    # 创建测试器并运行所有测试
    tester = EnsembleMethodTester(random_state=42)
    tester.run_all_tests()


if __name__ == "__main__":
    main()