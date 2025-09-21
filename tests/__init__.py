"""
测试模块

提供全面的模型测试套件
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 测试配置
TEST_CONFIG = {
    'random_seed': 42,
    'test_data_size': 1000,
    'tolerance': 1e-6,
    'timeout': 300,  # 5分钟超时
    'parallel_jobs': 2
}

__version__ = '1.0.0'