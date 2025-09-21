"""
Pytest配置文件

全局测试配置和fixtures
"""

import pytest
import numpy as np
import warnings
import os
import sys
from pathlib import Path
import tempfile
import shutil
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import platform

# 在导入其他模块之前设置模拟
def setup_mocks():
    """设置模拟模块"""
    # 模拟torch
    if 'torch' not in sys.modules:
        from tests.mocks.mock_torch import torch
        sys.modules['torch'] = torch
        sys.modules['torch.nn'] = torch.nn
        sys.modules['torch.optim'] = torch.optim
        sys.modules['torch.utils'] = torch.utils
        sys.modules['torch.utils.data'] = torch.utils.data
        sys.modules['torch.nn.functional'] = torch.nn.functional
        sys.modules['torch.distributions'] = torch.distributions
    
    # 模拟tensorflow
    if 'tensorflow' not in sys.modules:
        from tests.mocks.mock_tensorflow import tensorflow
        sys.modules['tensorflow'] = tensorflow
        sys.modules['tf'] = tensorflow

# 在pytest收集阶段之前设置模拟
setup_mocks()

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 忽略警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def pytest_configure(config):
    """Pytest配置"""
    # 设置随机种子
    np.random.seed(42)
    
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    # 为慢速测试添加标记
    slow_marker = pytest.mark.slow
    for item in items:
        if "slow" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(slow_marker)


@pytest.fixture(scope="session")
def test_config():
    """测试配置fixture"""
    return {
        'random_seed': 42,
        'test_data_size': 100,
        'tolerance': 1e-6,
        'timeout': 30,
        'temp_dir': 'temp_test_data'
    }


@pytest.fixture(scope="session")
def sample_data():
    """样本数据fixture"""
    np.random.seed(42)
    
    # 回归数据
    X_reg = np.random.randn(100, 5)
    y_reg = X_reg.sum(axis=1) + np.random.randn(100) * 0.1
    
    # 分类数据
    X_cls = np.random.randn(100, 5)
    y_cls = (X_cls.sum(axis=1) > 0).astype(int)
    
    # 时间序列数据
    time_steps = 100
    ts_data = np.cumsum(np.random.randn(time_steps)) + np.sin(np.arange(time_steps) * 0.1)
    
    return {
        'regression': (X_reg, y_reg),
        'classification': (X_cls, y_cls),
        'time_series': ts_data
    }


@pytest.fixture
def temp_directory(tmp_path):
    """临时目录fixture"""
    return tmp_path


@pytest.fixture(autouse=True)
def reset_random_state():
    """自动重置随机状态"""
    np.random.seed(42)
    yield
    np.random.seed(42)


@pytest.fixture
def mock_model():
    """模拟模型fixture"""
    from tests.test_utils import MockModel
    return MockModel()


@pytest.fixture
def data_generator():
    """数据生成器fixture"""
    from tests.test_utils import TestDataGenerator
    return TestDataGenerator()


@pytest.fixture
def model_helper():
    """模型测试辅助工具fixture"""
    from tests.test_utils import ModelTestHelper
    return ModelTestHelper


@pytest.fixture(scope="session")
def performance_config():
    """性能测试配置fixture"""
    return {
        'small_dataset_size': 100,
        'medium_dataset_size': 1000,
        'large_dataset_size': 10000,
        'timeout_seconds': 60,
        'memory_limit_mb': 500,
        'cpu_cores': os.cpu_count() or 1
    }


# 跳过条件
skip_if_no_sklearn = pytest.mark.skipif(
    not pytest.importorskip("sklearn", minversion="0.24"),
    reason="Scikit-learn not available"
)

try:
    import torch
    skip_if_no_torch = pytest.mark.skipif(False, reason="PyTorch available")
except ImportError:
    skip_if_no_torch = pytest.mark.skipif(True, reason="PyTorch not available")

try:
    import tensorflow
    skip_if_no_tensorflow = pytest.mark.skipif(False, reason="TensorFlow available")
except ImportError:
    skip_if_no_tensorflow = pytest.mark.skipif(True, reason="TensorFlow not available")

try:
    import torch
    skip_if_no_gpu = pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )
except ImportError:
    skip_if_no_gpu = pytest.mark.skipif(True, reason="PyTorch not available for GPU check")


def pytest_runtest_setup(item):
    """测试运行前设置"""
    # 检查GPU标记
    if item.get_closest_marker("gpu"):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available for GPU test")
    
    # 检查慢速测试标记
    if item.get_closest_marker("slow"):
        if item.config.getoption("-m") == "not slow":
            pytest.skip("Skipping slow test")


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-gpu", action="store_true", default=False,
        help="run GPU tests"
    )
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="run integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture(scope="function")
def cleanup_files():
    """清理文件fixture"""
    created_files = []
    
    def _add_file(filepath):
        created_files.append(filepath)
    
    yield _add_file
    
    # 清理创建的文件
    for filepath in created_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


@pytest.fixture(scope="function")
def capture_warnings():
    """捕获警告fixture"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


# HTML报告配置 (需要pytest-html插件)
# def pytest_html_report_title(report):
#     """自定义HTML报告标题"""
#     report.title = "机器学习项目测试报告"


# def pytest_html_results_summary(prefix, summary, postfix):
#     """自定义HTML报告摘要"""
#     prefix.extend([
#         "<h2>测试环境信息</h2>",
#         f"<p>Python版本: {sys.version}</p>",
#         f"<p>操作系统: {platform.system()} {platform.release()}</p>",
#         f"<p>测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
#     ])


# 性能测试相关
@pytest.fixture
def benchmark_config():
    """基准测试配置"""
    return {
        'warmup_rounds': 3,
        'test_rounds': 10,
        'timeout': 60
    }


@pytest.fixture
def memory_profiler():
    """内存分析器fixture"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        def get_memory_usage():
            return process.memory_info().rss / 1024 / 1024  # MB
        
        return get_memory_usage
    except ImportError:
        pytest.skip("psutil not available for memory profiling")


# 数据验证相关
@pytest.fixture
def data_validator():
    """数据验证器fixture"""
    def validate_data(X, y=None):
        """验证数据格式"""
        assert isinstance(X, np.ndarray), "X must be numpy array"
        assert X.ndim >= 2, "X must be at least 2D"
        assert not np.any(np.isnan(X)), "X contains NaN values"
        assert not np.any(np.isinf(X)), "X contains infinite values"
        
        if y is not None:
            assert isinstance(y, np.ndarray), "y must be numpy array"
            assert len(X) == len(y), "X and y must have same length"
            assert not np.any(np.isnan(y)), "y contains NaN values"
            assert not np.any(np.isinf(y)), "y contains infinite values"
    
    return validate_data


# 模型验证相关
@pytest.fixture
def model_validator():
    """模型验证器fixture"""
    def validate_model(model, X, y=None):
        """验证模型接口"""
        # 检查必要方法
        assert hasattr(model, 'fit'), "Model must have fit method"
        assert hasattr(model, 'predict'), "Model must have predict method"
        
        # 检查fit方法
        if y is not None:
            model.fit(X, y)
        else:
            model.fit(X)
        
        # 检查predict方法
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray), "Predictions must be numpy array"
        assert len(predictions) == len(X), "Predictions length must match input length"
        
        return predictions
    
    return validate_model