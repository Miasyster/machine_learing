"""
数据预处理器测试模块

测试DataPreprocessor类的各种数据预处理功能
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from unittest.mock import patch, MagicMock

from src.data.preprocessors import DataPreprocessor, PreprocessingError


class TestDataPreprocessor(unittest.TestCase):
    """DataPreprocessor类的测试用例"""
    
    def setUp(self):
        """测试前的设置"""
        self.preprocessor = DataPreprocessor()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'text': ['hello world', 'machine learning', 'data science', 'python', 'ai'],
            'missing': [1.0, np.nan, 3.0, np.nan, 5.0],
            'target': [0, 1, 0, 1, 1]
        })
        
        # 包含异常值的数据
        self.outlier_data = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]  # 100是异常值
        })
    
    def test_handle_missing_values_drop(self):
        """测试删除缺失值"""
        result = self.preprocessor.handle_missing_values(
            self.test_data, 
            strategy='drop'
        )
        
        # 验证结果
        self.assertEqual(len(result), 3)  # 应该剩下3行没有缺失值的数据
        self.assertFalse(result.isnull().any().any())
    
    def test_handle_missing_values_fill_mean(self):
        """测试用均值填充缺失值"""
        result = self.preprocessor.handle_missing_values(
            self.test_data, 
            strategy='fill',
            fill_value='mean'
        )
        
        # 验证结果
        self.assertEqual(len(result), 5)  # 行数不变
        self.assertFalse(result['missing'].isnull().any())
        # 缺失值应该被均值(3.0)填充
        expected_mean = self.test_data['missing'].mean()
        filled_values = result.loc[result.index[1], 'missing']
        self.assertEqual(filled_values, expected_mean)
    
    def test_handle_missing_values_fill_median(self):
        """测试用中位数填充缺失值"""
        result = self.preprocessor.handle_missing_values(
            self.test_data, 
            strategy='fill',
            fill_value='median'
        )
        
        # 验证结果
        self.assertEqual(len(result), 5)
        self.assertFalse(result['missing'].isnull().any())
    
    def test_handle_missing_values_fill_mode(self):
        """测试用众数填充缺失值"""
        result = self.preprocessor.handle_missing_values(
            self.test_data, 
            strategy='fill',
            fill_value='mode'
        )
        
        # 验证结果
        self.assertEqual(len(result), 5)
        self.assertFalse(result['missing'].isnull().any())
    
    def test_handle_missing_values_fill_constant(self):
        """测试用常数填充缺失值"""
        result = self.preprocessor.handle_missing_values(
            self.test_data, 
            strategy='fill',
            fill_value=999
        )
        
        # 验证结果
        self.assertEqual(len(result), 5)
        self.assertFalse(result['missing'].isnull().any())
        # 检查缺失值是否被999填充
        filled_indices = self.test_data['missing'].isnull()
        self.assertTrue((result.loc[filled_indices, 'missing'] == 999).all())
    
    def test_handle_missing_values_invalid_strategy(self):
        """测试无效的缺失值处理策略"""
        with self.assertRaises(PreprocessingError):
            self.preprocessor.handle_missing_values(
                self.test_data, 
                strategy='invalid_strategy'
            )
    
    def test_scale_features_standard(self):
        """测试标准化缩放"""
        numeric_data = self.test_data[['numeric1', 'numeric2']]
        result = self.preprocessor.scale_features(
            numeric_data, 
            method='standard'
        )
        
        # 验证结果
        self.assertEqual(result.shape, numeric_data.shape)
        # 标准化后均值应该接近0，标准差接近1
        self.assertAlmostEqual(result['numeric1'].mean(), 0, places=10)
        self.assertAlmostEqual(result['numeric1'].std(), 1, places=10)
    
    def test_scale_features_minmax(self):
        """测试最小-最大缩放"""
        numeric_data = self.test_data[['numeric1', 'numeric2']]
        result = self.preprocessor.scale_features(
            numeric_data, 
            method='minmax'
        )
        
        # 验证结果
        self.assertEqual(result.shape, numeric_data.shape)
        # 最小-最大缩放后值应该在[0,1]范围内
        self.assertTrue((result >= 0).all().all())
        self.assertTrue((result <= 1).all().all())
    
    def test_scale_features_robust(self):
        """测试鲁棒缩放"""
        numeric_data = self.test_data[['numeric1', 'numeric2']]
        result = self.preprocessor.scale_features(
            numeric_data, 
            method='robust'
        )
        
        # 验证结果
        self.assertEqual(result.shape, numeric_data.shape)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_scale_features_invalid_method(self):
        """测试无效的缩放方法"""
        numeric_data = self.test_data[['numeric1', 'numeric2']]
        with self.assertRaises(PreprocessingError):
            self.preprocessor.scale_features(
                numeric_data, 
                method='invalid_method'
            )
    
    def test_encode_categorical_onehot(self):
        """测试独热编码"""
        categorical_data = self.test_data[['categorical']]
        result = self.preprocessor.encode_categorical(
            categorical_data, 
            method='onehot'
        )
        
        # 验证结果
        self.assertEqual(len(result), len(categorical_data))
        # 独热编码后应该有3列（A, B, C）
        self.assertEqual(result.shape[1], 3)
        # 每行的和应该为1
        self.assertTrue((result.sum(axis=1) == 1).all())
    
    def test_encode_categorical_label(self):
        """测试标签编码"""
        categorical_data = self.test_data[['categorical']]
        result = self.preprocessor.encode_categorical(
            categorical_data, 
            method='label'
        )
        
        # 验证结果
        self.assertEqual(result.shape, categorical_data.shape)
        # 标签编码后应该是整数
        self.assertTrue(result['categorical'].dtype in ['int64', 'int32'])
    
    def test_encode_categorical_target(self):
        """测试目标编码"""
        categorical_data = self.test_data[['categorical']]
        target = self.test_data['target']
        result = self.preprocessor.encode_categorical(
            categorical_data, 
            method='target',
            target=target
        )
        
        # 验证结果
        self.assertEqual(result.shape, categorical_data.shape)
        self.assertTrue(result['categorical'].dtype in ['float64', 'float32'])
    
    def test_encode_categorical_invalid_method(self):
        """测试无效的编码方法"""
        categorical_data = self.test_data[['categorical']]
        with self.assertRaises(PreprocessingError):
            self.preprocessor.encode_categorical(
                categorical_data, 
                method='invalid_method'
            )
    
    def test_detect_outliers_iqr(self):
        """测试IQR方法检测异常值"""
        outliers = self.preprocessor.detect_outliers(
            self.outlier_data, 
            method='iqr'
        )
        
        # 验证结果
        self.assertIsInstance(outliers, pd.Series)
        self.assertEqual(len(outliers), len(self.outlier_data))
        # 应该检测到异常值（100）
        self.assertTrue(outliers.any())
    
    def test_detect_outliers_zscore(self):
        """测试Z-score方法检测异常值"""
        outliers = self.preprocessor.detect_outliers(
            self.outlier_data, 
            method='zscore',
            threshold=2
        )
        
        # 验证结果
        self.assertIsInstance(outliers, pd.Series)
        self.assertEqual(len(outliers), len(self.outlier_data))
        # 应该检测到异常值
        self.assertTrue(outliers.any())
    
    def test_detect_outliers_isolation_forest(self):
        """测试孤立森林方法检测异常值"""
        outliers = self.preprocessor.detect_outliers(
            self.outlier_data, 
            method='isolation_forest'
        )
        
        # 验证结果
        self.assertIsInstance(outliers, pd.Series)
        self.assertEqual(len(outliers), len(self.outlier_data))
    
    def test_detect_outliers_invalid_method(self):
        """测试无效的异常值检测方法"""
        with self.assertRaises(PreprocessingError):
            self.preprocessor.detect_outliers(
                self.outlier_data, 
                method='invalid_method'
            )
    
    def test_remove_outliers(self):
        """测试移除异常值"""
        # 先检测异常值
        outliers = self.preprocessor.detect_outliers(
            self.outlier_data, 
            method='iqr'
        )
        
        # 移除异常值
        result = self.preprocessor.remove_outliers(
            self.outlier_data, 
            outliers
        )
        
        # 验证结果
        self.assertLess(len(result), len(self.outlier_data))
        # 异常值100应该被移除
        self.assertNotIn(100, result['values'].values)
    
    def test_feature_selection_correlation(self):
        """测试基于相关性的特征选择"""
        # 创建高相关性的特征
        corr_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],  # 与feature1高度相关
            'feature3': [5, 3, 1, 4, 2],   # 与其他特征低相关
            'target': [0, 1, 0, 1, 1]
        })
        
        result = self.preprocessor.feature_selection(
            corr_data.drop('target', axis=1), 
            corr_data['target'],
            method='correlation',
            k=2
        )
        
        # 验证结果
        self.assertEqual(result.shape[1], 2)
    
    def test_feature_selection_mutual_info(self):
        """测试基于互信息的特征选择"""
        features = self.test_data[['numeric1', 'numeric2', 'missing']].fillna(0)
        target = self.test_data['target']
        
        result = self.preprocessor.feature_selection(
            features, 
            target,
            method='mutual_info',
            k=2
        )
        
        # 验证结果
        self.assertEqual(result.shape[1], 2)
    
    def test_feature_selection_chi2(self):
        """测试基于卡方检验的特征选择"""
        # 创建非负特征数据
        features = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'feature3': [1, 1, 2, 2, 3]
        })
        target = pd.Series([0, 1, 0, 1, 1])
        
        result = self.preprocessor.feature_selection(
            features, 
            target,
            method='chi2',
            k=2
        )
        
        # 验证结果
        self.assertEqual(result.shape[1], 2)
    
    def test_feature_selection_invalid_method(self):
        """测试无效的特征选择方法"""
        features = self.test_data[['numeric1', 'numeric2']]
        target = self.test_data['target']
        
        with self.assertRaises(PreprocessingError):
            self.preprocessor.feature_selection(
                features, 
                target,
                method='invalid_method'
            )
    
    def test_create_polynomial_features(self):
        """测试创建多项式特征"""
        features = self.test_data[['numeric1', 'numeric2']]
        result = self.preprocessor.create_polynomial_features(
            features, 
            degree=2
        )
        
        # 验证结果
        self.assertGreater(result.shape[1], features.shape[1])
        # 2次多项式应该包含原特征、交互项和平方项
        expected_features = 2 + 1 + 2  # 原特征 + 交互项 + 平方项
        self.assertEqual(result.shape[1], expected_features)
    
    def test_create_interaction_features(self):
        """测试创建交互特征"""
        features = self.test_data[['numeric1', 'numeric2']]
        result = self.preprocessor.create_interaction_features(features)
        
        # 验证结果
        self.assertGreater(result.shape[1], features.shape[1])
        # 应该包含原特征和交互项
        self.assertEqual(result.shape[1], 3)  # 2个原特征 + 1个交互项
    
    def test_pipeline_execution(self):
        """测试预处理管道执行"""
        # 定义预处理步骤
        steps = [
            ('missing', {'strategy': 'fill', 'fill_value': 'mean'}),
            ('scale', {'method': 'standard'}),
            ('outliers', {'method': 'iqr'})
        ]
        
        # 执行管道
        result = self.preprocessor.create_pipeline(steps)
        
        # 验证结果
        self.assertIsNotNone(result)
    
    def test_transform_text_features(self):
        """测试文本特征转换"""
        text_data = self.test_data[['text']]
        result = self.preprocessor.transform_text_features(
            text_data, 
            method='tfidf',
            max_features=100
        )
        
        # 验证结果
        self.assertEqual(len(result), len(text_data))
        self.assertGreater(result.shape[1], 0)
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = {
            'default_scaler': 'standard',
            'missing_strategy': 'mean',
            'outlier_method': 'iqr'
        }
        
        preprocessor_with_config = DataPreprocessor(config)
        self.assertEqual(preprocessor_with_config.config['default_scaler'], 'standard')
        self.assertEqual(preprocessor_with_config.config['missing_strategy'], 'mean')
        self.assertEqual(preprocessor_with_config.config['outlier_method'], 'iqr')


class TestDataPreprocessorIntegration(unittest.TestCase):
    """DataPreprocessor集成测试"""
    
    def setUp(self):
        """测试前的设置"""
        self.preprocessor = DataPreprocessor()
        
        # 创建复杂的测试数据
        np.random.seed(42)
        self.complex_data = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, 100),
            'numeric2': np.random.exponential(2, 100),
            'categorical1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical2': np.random.choice(['X', 'Y'], 100),
            'text': ['text_' + str(i) for i in range(100)],
            'target': np.random.choice([0, 1], 100)
        })
        
        # 添加一些缺失值
        missing_indices = np.random.choice(100, 10, replace=False)
        self.complex_data.loc[missing_indices, 'numeric1'] = np.nan
    
    def test_full_preprocessing_pipeline(self):
        """测试完整的预处理管道"""
        # 分离特征和目标
        features = self.complex_data.drop('target', axis=1)
        target = self.complex_data['target']
        
        # 处理缺失值
        features_no_missing = self.preprocessor.handle_missing_values(
            features, 
            strategy='fill', 
            fill_value='mean'
        )
        
        # 编码分类特征
        categorical_cols = ['categorical1', 'categorical2']
        categorical_encoded = self.preprocessor.encode_categorical(
            features_no_missing[categorical_cols], 
            method='onehot'
        )
        
        # 缩放数值特征
        numeric_cols = ['numeric1', 'numeric2']
        numeric_scaled = self.preprocessor.scale_features(
            features_no_missing[numeric_cols], 
            method='standard'
        )
        
        # 验证结果
        self.assertFalse(features_no_missing.isnull().any().any())
        self.assertEqual(len(categorical_encoded), len(features))
        self.assertEqual(len(numeric_scaled), len(features))
        
        # 检查缩放后的数值特征
        self.assertAlmostEqual(numeric_scaled['numeric1'].mean(), 0, places=1)
        self.assertAlmostEqual(numeric_scaled['numeric1'].std(), 1, places=1)
    
    def test_preprocessing_with_outliers(self):
        """测试包含异常值的预处理"""
        # 添加异常值
        outlier_data = self.complex_data.copy()
        outlier_data.loc[0, 'numeric1'] = 1000  # 极端异常值
        
        # 检测异常值
        outliers = self.preprocessor.detect_outliers(
            outlier_data[['numeric1', 'numeric2']], 
            method='iqr'
        )
        
        # 移除异常值
        clean_data = self.preprocessor.remove_outliers(outlier_data, outliers)
        
        # 验证结果
        self.assertLess(len(clean_data), len(outlier_data))
        self.assertNotIn(1000, clean_data['numeric1'].values)
    
    def test_feature_engineering_pipeline(self):
        """测试特征工程管道"""
        # 选择数值特征
        numeric_features = self.complex_data[['numeric1', 'numeric2']].fillna(0)
        
        # 创建多项式特征
        poly_features = self.preprocessor.create_polynomial_features(
            numeric_features, 
            degree=2
        )
        
        # 创建交互特征
        interaction_features = self.preprocessor.create_interaction_features(
            numeric_features
        )
        
        # 验证结果
        self.assertGreater(poly_features.shape[1], numeric_features.shape[1])
        self.assertGreater(interaction_features.shape[1], numeric_features.shape[1])


if __name__ == '__main__':
    unittest.main()