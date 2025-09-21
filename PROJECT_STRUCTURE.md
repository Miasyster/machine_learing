# 项目目录结构说明

本文档描述了整理后的项目目录结构和各目录的用途。

## 根目录文件
- `README.md` - 项目主要说明文档
- `DEPLOYMENT.md` - 部署指南
- `AIRFLOW_USAGE.md` - Airflow使用说明
- `AIRFLOW_WINDOWS_GUIDE.md` - Windows下Airflow配置指南
- `Makefile` - 构建和部署脚本
- `requirements.txt` / `requirements-dev.txt` - Python依赖
- `.env.example` - 环境变量模板
- `.gitignore` - Git忽略文件配置

## 主要目录结构

### `/src` - 源代码
项目的核心源代码，按功能模块组织：
- `ensemble/` - 集成学习模块
- `training/` - 训练模块
- `evaluation/` - 评估模块
- `monitoring/` - 监控模块
- `deployment/` - 部署模块
- `explainability/` - 模型解释性模块
- `backtest/` - 回测引擎模块
- `etl/` - 数据处理模块
- `features/` - 特征工程模块
- `optimization/` - 优化模块
- `augmentation/` - 数据增强模块
- `utils/` - 工具函数

### `/tests` - 测试代码
单元测试、集成测试和性能测试：
- `unit/` - 单元测试
- `integration/` - 集成测试
- `performance/` - 性能测试
- `mocks/` - 测试模拟对象
- `test_data/` - 测试数据

### `/examples` - 示例代码
按功能分类的示例代码：
- `training/` - 训练相关示例
- `optimization/` - 优化相关示例
- `data_processing/` - 数据处理示例
- `airflow/` - Airflow集成示例

### `/scripts` - 脚本文件
部署、启动和维护脚本：
- `deploy.ps1` - 部署脚本
- `start_airflow_local.ps1` - 本地Airflow启动脚本
- `run_tests.py` - 测试运行脚本

### `/tools` - 工具和实用程序
开发和调试工具：
- `test_chart_display.py` - 图表显示测试
- `test_optimizers.py` - 优化器测试
- `verify_fixes.py` - 修复验证工具
- `view_chart.py` - 图表查看工具

### `/configs` - 配置文件
项目配置和设置：
- `strategy_config.yaml` - 策略配置
- `pytest.ini` - pytest配置
- `tox.ini` - tox配置
- `.pre-commit-config.yaml` - pre-commit配置

### `/docs` - 文档
项目文档和报告：
- `API_REFERENCE.md` - API参考
- `BEST_PRACTICES.md` - 最佳实践
- `QUICK_START.md` - 快速开始指南
- `TESTING_GUIDE.md` - 测试指南
- `data_augmentation_report.md` - 数据增强报告

### `/dags` - Airflow DAGs
Airflow工作流定义：
- `config/` - DAG配置文件
- `operators/` - 自定义操作符
- `utils/` - DAG工具函数

### 其他目录
- `/airflow_env` - Airflow虚拟环境
- `/airflow_home` - Airflow主目录
- `/docker-compose` - Docker配置
- `/infra` - 基础设施配置
- `/init-scripts` - 初始化脚本
- `/notebooks` - Jupyter笔记本
- `/experiments` - 实验代码
- `/plugins` - 插件
- `/script` - 额外脚本

## 文件组织原则

1. **按功能分类** - 相关功能的文件放在同一目录
2. **清晰的层次结构** - 避免过深的嵌套
3. **一致的命名** - 使用清晰、一致的命名约定
4. **分离关注点** - 源码、测试、文档、配置分别组织
5. **易于导航** - 目录结构直观，便于查找文件

## 使用建议

- 新增功能时，请按照现有的目录结构组织代码
- 添加示例时，请放入相应的examples子目录
- 配置文件统一放在configs目录
- 工具脚本放在scripts或tools目录
- 保持README文件的更新