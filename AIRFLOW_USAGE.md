# Airflow 使用指南

## 🚨 当前状态

由于网络连接和依赖版本冲突问题，Docker Compose 和本地安装都遇到了一些技术障碍。但是我已经为您创建了完整的Airflow DAGs和配置文件。

## 📁 已创建的文件结构

```
machine_learning/
├── dags/                           # Airflow DAGs目录
│   ├── data_ingestion_dag.py       # 数据获取DAG
│   ├── data_cleaning_dag.py        # 数据清洗DAG  
│   ├── feature_engineering_dag.py  # 特征工程DAG
│   ├── monitoring_dag.py           # 数据质量监控DAG
│   ├── lineage_dag.py             # 数据血缘追踪DAG
│   ├── operators/                  # 自定义操作符
│   │   ├── data_ingestion_operator.py
│   │   ├── data_cleaning_operator.py
│   │   ├── feature_engineering_operator.py
│   │   ├── monitoring_operator.py
│   │   └── lineage_operator.py
│   ├── utils/                      # 工具函数
│   │   └── task_helpers.py
│   └── config/                     # 配置文件
│       ├── dag_config.yaml
│       ├── data_sources.yaml
│       ├── quality_thresholds.yaml
│       └── governance_config.yaml
├── docker-compose.yml             # Docker部署配置
├── docker-compose-simple.yml      # 简化版Docker配置
├── Dockerfile                     # 自定义镜像
├── .env.example                   # 环境变量模板
├── deploy.ps1                     # 部署脚本
└── start_airflow_local.ps1        # 本地启动脚本
```

## 🔗 DAGs与您的代码集成

### 1. 数据获取DAG (`data_ingestion_dag.py`)
- **使用模块**: `src.etl.binance_data_fetcher.BinanceDataFetcher`
- **功能**: 自动从币安API获取市场数据
- **调度**: 每小时执行一次
- **输出**: 存储到数据库的原始市场数据

### 2. 数据清洗DAG (`data_cleaning_dag.py`)
- **使用模块**: `src.etl.data_cleaner.DataCleaner`
- **功能**: 清洗和预处理原始数据
- **依赖**: 数据获取DAG完成后触发
- **输出**: 清洗后的标准化数据

### 3. 特征工程DAG (`feature_engineering_dag.py`)
- **使用模块**: `src.features.feature_engineering.FeatureEngineer`
- **功能**: 计算技术指标和特征
- **依赖**: 数据清洗DAG完成后触发
- **输出**: 可用于模型训练的特征数据

### 4. 数据质量监控DAG (`monitoring_dag.py`)
- **功能**: 监控数据质量、完整性和异常
- **调度**: 每天执行一次
- **告警**: 发现问题时发送通知

### 5. 数据血缘追踪DAG (`lineage_dag.py`)
- **功能**: 追踪数据流转和依赖关系
- **输出**: 数据血缘图和元数据

## 🛠️ 如何使用

### 方法1: Docker部署（推荐生产环境）

1. **准备环境**:
   ```powershell
   # 复制环境变量文件
   Copy-Item .env.example .env
   
   # 编辑.env文件，设置您的配置
   notepad .env
   ```

2. **启动服务**:
   ```powershell
   # 使用简化版配置
   docker-compose -f docker-compose-simple.yml up -d
   
   # 或使用完整配置
   docker-compose up -d
   ```

3. **访问Web界面**:
   - URL: http://localhost:8080
   - 用户名: admin
   - 密码: admin

### 方法2: 本地安装（推荐开发环境）

1. **安装Airflow**:
   ```powershell
   # 设置环境变量
   $env:AIRFLOW_HOME = "$PWD\airflow_home"
   $env:AIRFLOW__CORE__DAGS_FOLDER = "$PWD\dags"
   $env:AIRFLOW__CORE__LOAD_EXAMPLES = "False"
   
   # 安装Airflow
   pip install apache-airflow==2.7.0
   
   # 初始化数据库
   airflow db init
   
   # 创建管理员用户
   airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
   ```

2. **启动服务**:
   ```powershell
   # 启动调度器（新终端）
   airflow scheduler
   
   # 启动Web服务器（新终端）
   airflow webserver --port 8080
   ```

### 方法3: 使用提供的脚本

```powershell
# 完整安装和初始化
.\start_airflow_local.ps1 setup

# 启动服务
.\start_airflow_local.ps1 start

# 查看状态
.\start_airflow_local.ps1 status

# 停止服务
.\start_airflow_local.ps1 stop
```

## 📊 DAG工作流程

```
数据获取DAG (每小时)
    ↓
数据清洗DAG (触发式)
    ↓
特征工程DAG (触发式)
    ↓
[您的模型训练代码]

数据质量监控DAG (每天) → 告警通知
数据血缘追踪DAG (每天) → 元数据更新
```

## 🔧 自定义配置

### 1. 修改调度频率
编辑DAG文件中的 `schedule_interval` 参数：
```python
# 每小时执行
schedule_interval='@hourly'

# 每天执行
schedule_interval='@daily'

# 自定义cron表达式
schedule_interval='0 */6 * * *'  # 每6小时
```

### 2. 添加新的数据源
1. 在 `dags/config/data_sources.yaml` 中添加配置
2. 在 `operators/` 目录下创建新的操作符
3. 在相应的DAG中引用新操作符

### 3. 自定义监控规则
编辑 `dags/config/quality_thresholds.yaml` 文件：
```yaml
data_quality:
  completeness_threshold: 0.95
  freshness_hours: 2
  volume_change_threshold: 0.3
```

## 🚀 下一步

1. **解决依赖问题**: 如果遇到安装问题，可以使用虚拟环境或Docker
2. **测试DAGs**: 在Airflow Web界面中手动触发DAGs进行测试
3. **集成模型训练**: 在特征工程DAG后添加模型训练任务
4. **设置告警**: 配置邮件或Slack通知
5. **监控优化**: 根据实际运行情况调整资源配置

## 📞 故障排除

### 常见问题:

1. **端口冲突**: 修改docker-compose.yml中的端口映射
2. **权限问题**: 确保Docker有足够的权限访问项目目录
3. **内存不足**: 增加Docker的内存限制
4. **网络问题**: 检查防火墙和代理设置

### 日志查看:
```powershell
# Docker日志
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler

# 本地日志
# 查看 $AIRFLOW_HOME/logs/ 目录
```

## 💡 提示

- 所有DAGs都已配置为使用您现有的 `src/` 目录下的代码
- 数据流程完全自动化，无需手动干预
- Web界面提供了丰富的监控和调试功能
- 支持任务重试、依赖管理和并行执行