"""
测试配置文件
包含测试相关的配置和工具函数
"""

import sys
import os
from pathlib import Path

# 获取项目根目录
def get_project_root():
    """获取项目根目录路径"""
    current_file = Path(__file__).resolve()
    # 从 src/tests/test_config.py 向上两级到项目根目录
    return current_file.parent.parent.parent

# 添加项目路径到sys.path
def setup_test_environment():
    """设置测试环境，添加必要的路径"""
    project_root = get_project_root()
    
    # 添加项目根目录到Python路径
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 确保数据目录存在
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 确保各个子目录存在
    (data_dir / "raw" / "klines").mkdir(parents=True, exist_ok=True)
    (data_dir / "raw" / "tickers").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed" / "features").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed" / "signals").mkdir(parents=True, exist_ok=True)
    (data_dir / "external").mkdir(parents=True, exist_ok=True)
    
    return project_root

# 测试用的默认配置
TEST_CONFIG = {
    'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    'intervals': ['1h'],
    'limit': 50,  # 测试时只获取少量数据
    'delay': 0.2  # API调用延迟
}