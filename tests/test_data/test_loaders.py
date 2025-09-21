"""
数据加载器测试模块

测试DataLoader类的各种数据加载功能
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from src.data.loaders import DataLoader, DataError


class TestDataLoader(unittest.TestCase):
    """DataLoader类的测试用例"""
    
    def setUp(self):
        """测试前的设置"""
        self.loader = DataLoader()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000]
        })
    
    def tearDown(self):
        """测试后的清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_csv_file(self):
        """测试CSV文件加载"""
        # 创建临时CSV文件
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        self.test_data.to_csv(csv_path, index=False)
        
        # 加载数据
        loaded_data = self.loader.load_from_file(csv_path)
        
        # 验证结果
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 5)
        self.assertListEqual(list(loaded_data.columns), ['id', 'name', 'age', 'salary'])
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    def test_load_json_file(self):
        """测试JSON文件加载"""
        # 创建临时JSON文件
        json_path = os.path.join(self.temp_dir, 'test.json')
        self.test_data.to_json(json_path, orient='records')
        
        # 加载数据
        loaded_data = self.loader.load_from_file(json_path)
        
        # 验证结果
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 5)
        self.assertListEqual(list(loaded_data.columns), ['id', 'name', 'age', 'salary'])
    
    def test_load_excel_file(self):
        """测试Excel文件加载"""
        # 创建临时Excel文件
        excel_path = os.path.join(self.temp_dir, 'test.xlsx')
        self.test_data.to_excel(excel_path, index=False)
        
        # 加载数据
        loaded_data = self.loader.load_from_file(excel_path)
        
        # 验证结果
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 5)
        self.assertListEqual(list(loaded_data.columns), ['id', 'name', 'age', 'salary'])
    
    def test_load_parquet_file(self):
        """测试Parquet文件加载"""
        # 创建临时Parquet文件
        parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.test_data.to_parquet(parquet_path, index=False)
        
        # 加载数据
        loaded_data = self.loader.load_from_file(parquet_path)
        
        # 验证结果
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 5)
        self.assertListEqual(list(loaded_data.columns), ['id', 'name', 'age', 'salary'])
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_from_file('nonexistent.csv')
    
    def test_load_unsupported_format(self):
        """测试加载不支持的文件格式"""
        # 创建不支持的文件格式
        txt_path = os.path.join(self.temp_dir, 'test.txt')
        with open(txt_path, 'w') as f:
            f.write('some text')
        
        with self.assertRaises(DataError):
            self.loader.load_from_file(txt_path)
    
    def test_load_csv_with_custom_params(self):
        """测试使用自定义参数加载CSV"""
        # 创建带分隔符的CSV文件
        csv_path = os.path.join(self.temp_dir, 'test_semicolon.csv')
        self.test_data.to_csv(csv_path, index=False, sep=';')
        
        # 使用自定义分隔符加载
        loaded_data = self.loader.load_from_file(csv_path, sep=';')
        
        # 验证结果
        self.assertEqual(len(loaded_data), 5)
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    @patch('pandas.read_sql')
    def test_load_from_database(self, mock_read_sql):
        """测试数据库加载"""
        # 模拟数据库查询结果
        mock_read_sql.return_value = self.test_data
        
        # 执行数据库加载
        connection_string = "sqlite:///test.db"
        query = "SELECT * FROM users"
        loaded_data = self.loader.load_from_database(connection_string, query)
        
        # 验证结果
        mock_read_sql.assert_called_once_with(query, connection_string)
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    @patch('requests.get')
    def test_load_from_api_success(self, mock_get):
        """测试API加载成功"""
        # 模拟API响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.test_data.to_dict('records')
        mock_get.return_value = mock_response
        
        # 执行API加载
        url = "https://api.example.com/data"
        loaded_data = self.loader.load_from_api(url)
        
        # 验证结果
        mock_get.assert_called_once_with(url, headers=None, params=None)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 5)
    
    @patch('requests.get')
    def test_load_from_api_failure(self, mock_get):
        """测试API加载失败"""
        # 模拟API错误响应
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response
        
        # 执行API加载并验证异常
        url = "https://api.example.com/data"
        with self.assertRaises(Exception):
            self.loader.load_from_api(url)
    
    @patch('requests.get')
    def test_load_from_api_with_headers_and_params(self, mock_get):
        """测试带请求头和参数的API加载"""
        # 模拟API响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.test_data.to_dict('records')
        mock_get.return_value = mock_response
        
        # 执行API加载
        url = "https://api.example.com/data"
        headers = {"Authorization": "Bearer token"}
        params = {"limit": 100}
        loaded_data = self.loader.load_from_api(url, headers=headers, params=params)
        
        # 验证结果
        mock_get.assert_called_once_with(url, headers=headers, params=params)
        self.assertIsInstance(loaded_data, pd.DataFrame)
    
    def test_load_empty_csv(self):
        """测试加载空CSV文件"""
        # 创建空CSV文件
        empty_csv_path = os.path.join(self.temp_dir, 'empty.csv')
        pd.DataFrame().to_csv(empty_csv_path, index=False)
        
        # 加载数据
        loaded_data = self.loader.load_from_file(empty_csv_path)
        
        # 验证结果
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 0)
    
    def test_load_csv_with_missing_values(self):
        """测试加载包含缺失值的CSV"""
        # 创建包含缺失值的数据
        data_with_na = self.test_data.copy()
        data_with_na.loc[1, 'age'] = np.nan
        data_with_na.loc[3, 'salary'] = np.nan
        
        # 保存到CSV
        csv_path = os.path.join(self.temp_dir, 'test_na.csv')
        data_with_na.to_csv(csv_path, index=False)
        
        # 加载数据
        loaded_data = self.loader.load_from_file(csv_path)
        
        # 验证结果
        self.assertEqual(len(loaded_data), 5)
        self.assertTrue(pd.isna(loaded_data.loc[1, 'age']))
        self.assertTrue(pd.isna(loaded_data.loc[3, 'salary']))
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = {
            'default_format': 'csv',
            'encoding': 'utf-8',
            'cache_enabled': True
        }
        
        loader_with_config = DataLoader(config)
        self.assertEqual(loader_with_config.config['default_format'], 'csv')
        self.assertEqual(loader_with_config.config['encoding'], 'utf-8')
        self.assertTrue(loader_with_config.config['cache_enabled'])


class TestDataLoaderIntegration(unittest.TestCase):
    """DataLoader集成测试"""
    
    def setUp(self):
        """测试前的设置"""
        self.loader = DataLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_multiple_files(self):
        """测试加载多个文件"""
        # 创建多个测试文件
        data1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        data2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        
        csv1_path = os.path.join(self.temp_dir, 'data1.csv')
        csv2_path = os.path.join(self.temp_dir, 'data2.csv')
        
        data1.to_csv(csv1_path, index=False)
        data2.to_csv(csv2_path, index=False)
        
        # 加载多个文件
        loaded_data1 = self.loader.load_from_file(csv1_path)
        loaded_data2 = self.loader.load_from_file(csv2_path)
        
        # 验证结果
        pd.testing.assert_frame_equal(loaded_data1, data1)
        pd.testing.assert_frame_equal(loaded_data2, data2)
    
    def test_load_large_file_chunks(self):
        """测试分块加载大文件"""
        # 创建较大的测试数据
        large_data = pd.DataFrame({
            'id': range(1000),
            'value': np.random.randn(1000)
        })
        
        csv_path = os.path.join(self.temp_dir, 'large_data.csv')
        large_data.to_csv(csv_path, index=False)
        
        # 分块加载
        chunks = []
        for chunk in pd.read_csv(csv_path, chunksize=100):
            chunks.append(chunk)
        
        # 验证结果
        self.assertEqual(len(chunks), 10)  # 1000 / 100 = 10 chunks
        combined_data = pd.concat(chunks, ignore_index=True)
        self.assertEqual(len(combined_data), 1000)


if __name__ == '__main__':
    unittest.main()