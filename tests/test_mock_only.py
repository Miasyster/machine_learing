"""
仅测试模拟模块的测试文件
用于验证模拟模块是否正常工作
"""

import pytest


class TestMockModules:
    """测试模拟模块"""
    
    def test_mock_torch_import(self):
        """测试模拟torch导入"""
        import torch
        assert torch is not None
        
        # 测试基本功能
        tensor = torch.tensor([1, 2, 3])
        assert tensor is not None
        
        # 测试神经网络
        linear = torch.nn.Linear(10, 5)
        assert linear is not None
        
        # 测试优化器
        optimizer = torch.optim.Adam([])
        assert optimizer is not None
        
        # 测试分布
        dist = torch.distributions.Categorical(logits=[0.1, 0.2, 0.3])
        assert dist is not None
    
    def test_mock_tensorflow_import(self):
        """测试模拟tensorflow导入"""
        import tensorflow as tf
        assert tf is not None
        
        # 测试基本功能
        tensor = tf.constant([1, 2, 3])
        assert tensor is not None
        
        # 测试模型
        model = tf.keras.Sequential()
        assert model is not None
        
        # 测试层
        dense = tf.keras.layers.Dense(10)
        assert dense is not None
    
    def test_cuda_availability(self):
        """测试CUDA可用性检查"""
        import torch
        # 模拟环境中CUDA不可用
        assert not torch.cuda.is_available()
        assert torch.cuda.device_count() == 0


if __name__ == "__main__":
    pytest.main([__file__])