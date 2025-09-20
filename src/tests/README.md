# 测试文件说明

本目录包含项目的所有测试文件，已重新组织并确保能够正常运行。

## 文件结构

```
src/tests/
├── __init__.py                    # 测试包初始化文件
├── test_config.py                 # 测试配置和环境设置
├── test_data_pipeline.py          # 原始数据管道测试（功能测试风格）
├── test_data_pipeline_pytest.py   # pytest风格的数据管道测试
└── README.md                      # 本说明文件
```

## 运行测试

### 方法1: 使用测试脚本（推荐）
```bash
# 运行所有测试（详细输出）
python run_tests.py

# 快速测试模式
python run_tests.py --quick
python run_tests.py -q

# 只运行数据管道测试
python run_tests.py --data-pipeline
python run_tests.py -d

# 查看帮助
python run_tests.py --help
```

### 方法2: 使用pytest
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest src/tests/test_data_pipeline_pytest.py

# 详细输出
pytest -v

# 安静模式
pytest -q

# 显示测试覆盖率
pytest --cov=src
```

### 方法3: 直接运行单个测试文件
```bash
python src/tests/test_data_pipeline.py
```

## 测试内容

### 数据管道测试
- **单个交易对数据流程**：测试获取和存储单个交易对的K线数据
- **多个交易对数据流程**：测试批量获取和存储多个交易对的数据
- **数据管理功能**：测试数据摘要、文件管理等功能
- **最新价格获取**：测试实时价格和24小时统计获取
- **数据质量检查**：验证数据的完整性和合理性

## 测试配置

- 测试使用真实的币安API（无需API密钥）
- 测试数据量较小（每个交易对50条记录）
- 包含API调用延迟以避免限流
- 自动创建必要的数据目录结构

## 注意事项

1. 测试需要网络连接以访问币安API
2. 首次运行可能需要安装依赖：`pip install pytest pytest-cov`
3. 测试会在 `data/` 目录下创建实际的数据文件
4. 所有测试都应该通过，如有失败请检查网络连接和API可用性