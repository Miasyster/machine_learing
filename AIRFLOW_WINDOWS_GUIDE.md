# Airflow Windows 使用指南

## 🚨 当前状态

由于Airflow在Windows系统上存在兼容性问题（缺少`pwd`模块），我们提供以下几种解决方案：

## 🔧 解决方案

### 方案1: 使用WSL2 (推荐)

1. **安装WSL2**:
   ```powershell
   wsl --install
   ```

2. **在WSL2中安装Airflow**:
   ```bash
   # 在WSL2 Ubuntu中执行
   pip install apache-airflow==2.7.0
   export AIRFLOW_HOME=~/airflow
   airflow db init
   airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
   airflow webserver --port 8080 &
   airflow scheduler &
   ```

3. **访问Web界面**: http://localhost:8080

### 方案2: 使用Docker Desktop (当前可用)

1. **确保Docker Desktop运行正常**
2. **使用我们的docker-compose配置**:
   ```powershell
   docker-compose -f docker-compose-simple.yml up -d
   ```
3. **等待镜像下载完成** (可能需要10-15分钟)
4. **访问Web界面**: http://localhost:8080

### 方案3: 使用虚拟机

1. 安装VirtualBox或VMware
2. 创建Ubuntu虚拟机
3. 在虚拟机中安装Airflow

## 📋 DAGs 已经准备就绪

您的项目中已经包含完整的Airflow DAGs:

### 🔄 数据流水线DAGs

1. **数据获取DAG** (`dags/data_ingestion_dag.py`)
   - 使用您的 `src.etl.binance_data_fetcher.BinanceDataFetcher`
   - 自动获取币安数据
   - 调度: 每小时执行

2. **数据清洗DAG** (`dags/data_cleaning_dag.py`)
   - 使用您的 `src.etl.data_cleaner.DataCleaner`
   - 数据质量检查和清洗
   - 依赖: 数据获取完成后执行

3. **特征工程DAG** (`dags/feature_engineering_dag.py`)
   - 使用您的 `src.features.feature_engineering.FeatureEngineer`
   - 计算技术指标
   - 依赖: 数据清洗完成后执行

4. **监控DAG** (`dags/monitoring_dag.py`)
   - 数据质量监控
   - 系统健康检查
   - 异常检测和告警

5. **血缘追踪DAG** (`dags/lineage_dag.py`)
   - 数据血缘记录
   - 元数据管理

## 🎯 Airflow Web界面使用指南

### 登录信息
- **URL**: http://localhost:8080
- **用户名**: admin
- **密码**: admin

### 主要功能

#### 1. DAGs 视图
- **查看所有DAGs**: 主页显示所有数据流水线
- **启用/禁用DAGs**: 点击开关按钮
- **手动触发**: 点击"Trigger DAG"按钮
- **查看DAG详情**: 点击DAG名称

#### 2. 任务监控
- **Graph View**: 查看任务依赖关系图
- **Tree View**: 查看历史执行记录
- **Gantt Chart**: 查看任务执行时间线
- **Task Duration**: 分析任务执行时长

#### 3. 日志查看
- **点击任务节点** → **View Log** 查看详细日志
- **实时监控**: 任务执行状态实时更新

#### 4. 配置管理
- **Admin** → **Connections**: 配置数据库连接
- **Admin** → **Variables**: 设置全局变量
- **Admin** → **Configuration**: 查看系统配置

## ⚙️ 需要手动配置的项目

### 1. 数据库连接
```python
# 在Airflow Web界面中配置
# Admin → Connections → Create
Connection Id: postgres_default
Connection Type: Postgres
Host: localhost
Schema: your_database
Login: your_username
Password: your_password
Port: 5432
```

### 2. API密钥配置
```python
# Admin → Variables → Create
Key: BINANCE_API_KEY
Value: your_binance_api_key

Key: BINANCE_SECRET_KEY
Value: your_binance_secret_key
```

### 3. 邮件告警配置
```python
# 在airflow.cfg中配置SMTP
[smtp]
smtp_host = smtp.gmail.com
smtp_starttls = True
smtp_ssl = False
smtp_user = your_email@gmail.com
smtp_password = your_app_password
smtp_port = 587
smtp_mail_from = your_email@gmail.com
```

## 🔍 验证DAGs

### 检查DAG语法
```powershell
# 在项目根目录执行
python -m py_compile dags/data_ingestion_dag.py
python -m py_compile dags/data_cleaning_dag.py
python -m py_compile dags/feature_engineering_dag.py
```

### 测试任务执行
```bash
# 在Airflow环境中测试
airflow tasks test data_ingestion_dag fetch_binance_data 2024-01-01
```

## 🚀 生产环境部署建议

1. **使用PostgreSQL**: 替换SQLite数据库
2. **配置Redis**: 用于Celery Executor
3. **设置负载均衡**: 多个Worker节点
4. **监控告警**: 集成Prometheus + Grafana
5. **日志管理**: 配置远程日志存储

## 📞 故障排除

### 常见问题

1. **DAG不显示**:
   - 检查DAG文件语法错误
   - 确认DAGS_FOLDER路径正确
   - 查看Airflow日志

2. **任务执行失败**:
   - 查看任务日志
   - 检查依赖包是否安装
   - 验证数据库连接

3. **Web界面无法访问**:
   - 确认端口8080未被占用
   - 检查防火墙设置
   - 查看webserver日志

## 💡 下一步操作

1. **选择合适的部署方案** (WSL2推荐)
2. **启动Airflow服务**
3. **配置数据库连接和API密钥**
4. **测试DAG执行**
5. **设置监控告警**
6. **根据需要调整调度频率**

您的ETL代码已经完美集成到Airflow中，只需要解决Windows兼容性问题即可开始使用！