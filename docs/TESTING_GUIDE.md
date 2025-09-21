# 测试指南

本文档提供了机器学习项目的完整测试指南，包括单元测试、集成测试、性能测试和CI/CD配置。

## 目录

- [测试环境设置](#测试环境设置)
- [运行测试](#运行测试)
- [测试类型](#测试类型)
- [测试配置](#测试配置)
- [覆盖率报告](#覆盖率报告)
- [CI/CD集成](#cicd集成)
- [最佳实践](#最佳实践)

## 测试环境设置

### 1. 安装测试依赖

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 或者使用Makefile
make install-dev
```

### 2. 配置测试环境

测试配置文件位于 `pytest.ini`，包含以下配置：

- 测试发现规则
- 覆盖率配置
- 测试标记
- 输出格式

## 运行测试

### 基础测试命令

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_simple.py

# 运行特定测试类
pytest tests/test_simple.py::TestBasic

# 运行特定测试方法
pytest tests/test_simple.py::TestBasic::test_python_version
```

### 使用Makefile

```bash
# 运行所有测试
make test

# 运行单元测试
make test-unit

# 运行集成测试
make test-integration

# 运行性能测试
make test-performance

# 运行覆盖率测试
make test-coverage
```

### 测试标记

项目使用pytest标记来分类测试：

```bash
# 运行单元测试
pytest -m unit

# 运行集成测试
pytest -m integration

# 运行慢速测试
pytest -m slow

# 跳过慢速测试
pytest -m "not slow"

# 运行需要GPU的测试
pytest -m gpu

# 运行需要网络的测试
pytest -m network
```

## 测试类型

### 1. 单元测试

位于 `tests/unit/` 目录，测试单个函数或类的功能。

```python
# 示例：tests/unit/test_example.py
import pytest
from src.module import function

class TestFunction:
    def test_basic_functionality(self):
        result = function(input_data)
        assert result == expected_output
    
    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6)
    ])
    def test_parametrized(self, input_val, expected):
        assert function(input_val) == expected
```

### 2. 集成测试

位于 `tests/integration/` 目录，测试多个组件的交互。

```python
# 示例：tests/integration/test_pipeline.py
import pytest

@pytest.mark.integration
class TestDataPipeline:
    def test_end_to_end_pipeline(self):
        # 测试完整的数据处理流程
        pass
```

### 3. 性能测试

位于 `tests/performance/` 目录，测试性能和基准。

```python
# 示例：tests/performance/test_benchmarks.py
import pytest
import time

@pytest.mark.slow
class TestPerformance:
    def test_processing_speed(self):
        start_time = time.time()
        # 执行操作
        end_time = time.time()
        assert (end_time - start_time) < 1.0  # 应在1秒内完成
```

## 测试配置

### pytest.ini 配置

```ini
[tool:pytest]
# 测试发现
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# 输出配置
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

# 标记定义
markers =
    slow: 慢速测试
    integration: 集成测试
    unit: 单元测试
    gpu: 需要GPU的测试
    network: 需要网络的测试
```

### conftest.py 配置

`tests/conftest.py` 包含：

- 测试夹具（fixtures）
- 模拟对象设置
- 测试环境配置
- 依赖跳过标记

## 覆盖率报告

### 生成覆盖率报告

```bash
# 生成HTML覆盖率报告
pytest --cov=src --cov-report=html

# 生成终端覆盖率报告
pytest --cov=src --cov-report=term-missing

# 生成XML覆盖率报告（用于CI）
pytest --cov=src --cov-report=xml
```

### 查看覆盖率报告

- HTML报告：打开 `htmlcov/index.html`
- 终端报告：直接在命令行查看
- XML报告：`coverage.xml` 文件

### 覆盖率要求

项目要求最低覆盖率为80%。可以在 `pytest.ini` 中配置：

```ini
addopts = --cov-fail-under=80
```

## CI/CD集成

### GitHub Actions

项目使用GitHub Actions进行持续集成，配置文件：`.github/workflows/ci.yml`

#### 测试矩阵

- Python版本：3.8, 3.9, 3.10, 3.11
- 操作系统：Ubuntu Latest

#### 测试流程

1. **代码检查**：flake8, mypy
2. **单元测试**：pytest with coverage
3. **安全检查**：bandit, safety
4. **构建检查**：package building
5. **部署**：PyPI发布（仅main分支）

### 本地CI测试

使用tox进行本地多环境测试：

```bash
# 运行所有环境测试
tox

# 运行特定环境
tox -e py310

# 运行代码检查
tox -e flake8

# 运行类型检查
tox -e mypy

# 运行覆盖率测试
tox -e coverage
```

## 最佳实践

### 1. 测试命名

- 测试文件：`test_*.py` 或 `*_test.py`
- 测试类：`Test*`
- 测试方法：`test_*`

### 2. 测试结构

```python
# 推荐的测试结构
class TestClassName:
    def setup_method(self):
        """每个测试方法前执行"""
        pass
    
    def teardown_method(self):
        """每个测试方法后执行"""
        pass
    
    def test_method_name(self):
        # Arrange（准备）
        input_data = "test_input"
        expected = "expected_output"
        
        # Act（执行）
        result = function_under_test(input_data)
        
        # Assert（断言）
        assert result == expected
```

### 3. 使用夹具

```python
@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

### 4. 参数化测试

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
def test_uppercase(input, expected):
    assert input.upper() == expected
```

### 5. 模拟和补丁

```python
from unittest.mock import patch, MagicMock

@patch('module.external_service')
def test_with_mock(mock_service):
    mock_service.return_value = "mocked_response"
    result = function_that_uses_service()
    assert result == "expected_result"
```

### 6. 异常测试

```python
def test_exception_handling():
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_should_raise(invalid_input)
```

### 7. 临时文件和目录

```python
def test_file_operations(tmp_path):
    # tmp_path是pytest提供的临时目录夹具
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    assert test_file.read_text() == "test content"
```

## 故障排除

### 常见问题

1. **导入错误**：确保项目根目录在Python路径中
2. **依赖缺失**：安装所有测试依赖
3. **权限问题**：确保测试文件有执行权限
4. **环境变量**：设置必要的环境变量

### 调试测试

```bash
# 详细输出
pytest -v -s

# 进入调试器
pytest --pdb

# 在第一个失败时停止
pytest -x

# 显示最慢的10个测试
pytest --durations=10
```

## 相关文档

- [部署指南](DEPLOYMENT.md)
- [API参考](API_REFERENCE.md)
- [最佳实践](BEST_PRACTICES.md)
- [快速开始](QUICK_START.md)