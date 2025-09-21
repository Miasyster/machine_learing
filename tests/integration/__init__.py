"""
集成测试模块

测试各模块之间的集成和端到端功能
"""

# 集成测试配置
INTEGRATION_CONFIG = {
    'random_seed': 42,
    'test_data_size': 500,
    'large_data_size': 5000,
    'timeout': 120,
    'tolerance': 1e-4
}