"""
简单测试用例，用于验证测试环境配置
"""

import pytest
import sys
import os


class TestBasic:
    """基础测试类"""
    
    def test_python_version(self):
        """测试Python版本"""
        assert sys.version_info >= (3, 8)
    
    def test_imports(self):
        """测试基础库导入"""
        import numpy as np
        import pandas as pd
        import sklearn
        assert np.__version__
        assert pd.__version__
        assert sklearn.__version__
    
    def test_project_structure(self):
        """测试项目结构"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 检查主要目录存在
        assert os.path.exists(os.path.join(project_root, "src"))
        assert os.path.exists(os.path.join(project_root, "tests"))
        assert os.path.exists(os.path.join(project_root, "requirements.txt"))
    
    def test_math_operations(self):
        """测试基础数学运算"""
        assert 2 + 2 == 4
        assert 3 * 3 == 9
        assert 10 / 2 == 5.0
    
    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8)
    ])
    def test_parametrized(self, input_val, expected):
        """参数化测试"""
        assert input_val * 2 == expected


class TestConfiguration:
    """配置测试类"""
    
    def test_pytest_markers(self):
        """测试pytest标记"""
        # 这个测试验证pytest标记配置正确
        pass
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """慢速测试标记"""
        import time
        time.sleep(0.1)  # 模拟慢速操作
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """集成测试标记"""
        assert True
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """单元测试标记"""
        assert True


if __name__ == "__main__":
    pytest.main([__file__])