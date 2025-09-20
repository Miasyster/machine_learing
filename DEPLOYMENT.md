# Airflow ML项目部署指南

本文档详细说明如何部署和运行基于Apache Airflow的机器学习项目。

## 目录

- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [详细部署步骤](#详细部署步骤)
- [配置说明](#配置说明)
- [服务管理](#服务管理)
- [监控和维护](#监控和维护)
- [故障排除](#故障排除)
- [生产环境部署](#生产环境部署)

## 系统要求

### 硬件要求
- **内存**: 最少8GB，推荐16GB或更多
- **CPU**: 最少4核，推荐8核或更多
- **磁盘**: 最少50GB可用空间，推荐100GB或更多
- **网络**: 稳定的互联网连接（用于下载数据）

### 软件要求
- **操作系统**: Windows 10/11, macOS, 或 Linux
- **Docker**: 版本20.10或更高
- **Docker Compose**: 版本2.0或更高
- **PowerShell**: Windows环境下需要PowerShell 5.1或更高

### 验证环境
```powershell
# 检查Docker版本
docker --version
docker-compose --version

# 检查可用资源
docker system info
```

## 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd machine_learning
```

### 2. 初始化项目
```powershell
# Windows
.\deploy.ps1 init

# Linux/macOS
./deploy.sh init
```

### 3. 启动服务
```powershell
# Windows
.\deploy.ps1 start

# Linux/macOS
./deploy.sh start
```

### 4. 访问界面
- **Airflow Web界面**: http://localhost:8080
  - 用户名: `admin`
  - 密码: `admin123`
- **Flower监控**: http://localhost:5555 (可选)
- **Jupyter Notebook**: http://localhost:8888 (可选)
- **MLflow**: http://localhost:5000 (可选)

## 详细部署步骤

### 步骤1: 环境准备

1. **安装Docker Desktop**
   - 下载并安装Docker Desktop
   - 确保Docker服务正在运行
   - 分配足够的资源（内存≥8GB，CPU≥4核）

2. **配置环境变量**
   ```powershell
   # 复制环境变量模板
   Copy-Item .env.example .env
   
   # 编辑.env文件，设置必要的配置
   notepad .env
   ```

3. **重要环境变量配置**
   ```bash
   # 数据库密码（必须修改）
   POSTGRES_PASSWORD=your_secure_password
   
   # API密钥（如果使用Binance API）
   BINANCE_API_KEY=your_api_key
   BINANCE_SECRET_KEY=your_secret_key
   
   # 邮件配置（用于告警）
   SMTP_HOST=smtp.gmail.com
   SMTP_USER=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   
   # Slack通知（可选）
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   ```

### 步骤2: 项目初始化

```powershell
# 运行初始化脚本
.\deploy.ps1 init
```

初始化脚本会执行以下操作：
- 创建必要的目录结构
- 设置文件权限
- 构建Docker镜像
- 初始化数据库
- 创建默认用户和连接

### 步骤3: 启动服务

```powershell
# 启动所有服务
.\deploy.ps1 start

# 或者分步启动
.\deploy.ps1 start -Service postgres    # 先启动数据库
.\deploy.ps1 start -Service redis       # 启动Redis
.\deploy.ps1 start -Service webserver   # 启动Web服务器
.\deploy.ps1 start -Service scheduler   # 启动调度器
```

### 步骤4: 验证部署

1. **检查服务状态**
   ```powershell
   .\deploy.ps1 status
   ```

2. **查看日志**
   ```powershell
   # 查看所有服务日志
   .\deploy.ps1 logs
   
   # 查看特定服务日志
   .\deploy.ps1 logs -Service webserver
   ```

3. **访问Web界面**
   - 打开浏览器访问 http://localhost:8080
   - 使用默认凭据登录
   - 检查DAGs是否正常加载

## 配置说明

### 核心配置文件

1. **docker-compose.yml**: Docker服务编排配置
2. **.env**: 环境变量配置
3. **airflow.cfg**: Airflow核心配置（自动生成）
4. **dags/config/**: DAG特定配置文件

### 重要配置项

#### Airflow配置
```yaml
# 执行器类型
AIRFLOW__CORE__EXECUTOR: CeleryExecutor

# 数据库连接
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://...

# 并发设置
AIRFLOW__CORE__PARALLELISM: 32
AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG: 16
```

#### 数据库配置
```yaml
# PostgreSQL配置
POSTGRES_HOST: postgres
POSTGRES_PORT: 5432
POSTGRES_DB: airflow
POSTGRES_USER: airflow
POSTGRES_PASSWORD: your_password
```

#### 安全配置
```yaml
# Web服务器密钥
AIRFLOW__WEBSERVER__SECRET_KEY: your-secret-key

# Fernet加密密钥
AIRFLOW__CORE__FERNET_KEY: your-fernet-key
```

## 服务管理

### 常用命令

```powershell
# 启动服务
.\deploy.ps1 start

# 停止服务
.\deploy.ps1 stop

# 重启服务
.\deploy.ps1 restart

# 查看状态
.\deploy.ps1 status

# 查看日志
.\deploy.ps1 logs

# 重新构建镜像
.\deploy.ps1 build

# 清理所有资源
.\deploy.ps1 clean -Force
```

### 服务组件

| 服务 | 端口 | 描述 |
|------|------|------|
| airflow-webserver | 8080 | Web界面 |
| airflow-scheduler | - | 任务调度器 |
| airflow-worker | - | 任务执行器 |
| postgres | 5432 | Airflow数据库 |
| redis | 6379 | 消息队列 |
| ml-postgres | 5433 | ML数据库 |
| flower | 5555 | Celery监控 |
| jupyter | 8888 | 开发环境 |
| mlflow | 5000 | 实验跟踪 |

### 扩展服务

```powershell
# 启动可选服务
docker-compose --profile flower up -d     # Celery监控
docker-compose --profile jupyter up -d    # Jupyter Notebook
docker-compose --profile mlflow up -d     # MLflow跟踪
```

## 监控和维护

### 健康检查

```powershell
# 检查所有服务健康状态
docker-compose ps

# 检查特定服务
docker-compose exec airflow-webserver python /opt/airflow/healthcheck.py webserver
```

### 日志管理

```powershell
# 查看实时日志
.\deploy.ps1 logs -Service scheduler

# 查看历史日志
docker-compose logs --since 1h scheduler

# 清理日志
docker-compose exec airflow-scheduler airflow tasks clear <dag_id>
```

### 数据备份

```powershell
# 备份Airflow数据库
docker-compose exec postgres pg_dump -U airflow airflow > backup_airflow.sql

# 备份ML数据库
docker-compose exec ml-postgres pg_dump -U ml_user ml_data > backup_ml.sql

# 备份数据文件
docker run --rm -v machine_learning_data:/data -v ${PWD}:/backup alpine tar czf /backup/data_backup.tar.gz /data
```

### 性能监控

1. **系统资源监控**
   ```powershell
   # 查看容器资源使用
   docker stats
   
   # 查看磁盘使用
   docker system df
   ```

2. **Airflow监控**
   - Web界面的Admin > Metrics菜单
   - Flower界面监控Celery任务
   - 数据库查询监控表

3. **自定义监控**
   - 配置Prometheus + Grafana
   - 设置告警规则
   - 集成外部监控系统

## 故障排除

### 常见问题

#### 1. 服务启动失败
```powershell
# 检查日志
.\deploy.ps1 logs

# 检查端口占用
netstat -an | findstr :8080

# 重新构建镜像
.\deploy.ps1 build -Force
```

#### 2. 数据库连接失败
```powershell
# 检查数据库状态
docker-compose exec postgres pg_isready -U airflow

# 重置数据库
docker-compose down -v
.\deploy.ps1 init
```

#### 3. DAG加载失败
```powershell
# 检查DAG语法
docker-compose exec airflow-scheduler airflow dags list

# 查看DAG错误
docker-compose exec airflow-scheduler airflow dags show <dag_id>
```

#### 4. 任务执行失败
```powershell
# 查看任务日志
# 在Web界面中点击任务 > View Log

# 手动运行任务
docker-compose exec airflow-scheduler airflow tasks run <dag_id> <task_id> <execution_date>
```

### 调试模式

```powershell
# 启动调试容器
docker-compose run --rm airflow-cli bash

# 在容器内调试
airflow dags list
airflow tasks list <dag_id>
python -c "import sys; print(sys.path)"
```

### 重置环境

```powershell
# 完全重置（谨慎使用）
.\deploy.ps1 clean -Force
.\deploy.ps1 init
```

## 生产环境部署

### 安全加固

1. **更改默认密码**
   ```bash
   # 在.env文件中设置强密码
   _AIRFLOW_WWW_USER_PASSWORD=your_strong_password
   POSTGRES_PASSWORD=your_strong_db_password
   ```

2. **启用HTTPS**
   ```yaml
   # 在docker-compose.yml中配置SSL
   environment:
     AIRFLOW__WEBSERVER__WEB_SERVER_SSL_CERT: /path/to/cert.pem
     AIRFLOW__WEBSERVER__WEB_SERVER_SSL_KEY: /path/to/key.pem
   ```

3. **网络安全**
   ```yaml
   # 限制端口暴露
   ports:
     - "127.0.0.1:8080:8080"  # 只绑定本地接口
   ```

### 性能优化

1. **资源配置**
   ```yaml
   # 增加worker数量
   deploy:
     replicas: 3
     resources:
       limits:
         memory: 2G
         cpus: '1.0'
   ```

2. **数据库优化**
   ```sql
   -- PostgreSQL配置优化
   shared_buffers = 256MB
   effective_cache_size = 1GB
   work_mem = 4MB
   ```

3. **缓存配置**
   ```yaml
   # Redis配置优化
   REDIS_MAXMEMORY: 512mb
   REDIS_MAXMEMORY_POLICY: allkeys-lru
   ```

### 高可用部署

1. **多节点部署**
   - 使用Docker Swarm或Kubernetes
   - 配置负载均衡器
   - 设置数据库集群

2. **备份策略**
   - 自动化数据库备份
   - 代码版本控制
   - 配置文件备份

3. **监控告警**
   - 集成Prometheus/Grafana
   - 设置关键指标告警
   - 配置日志聚合

### 部署检查清单

- [ ] 环境变量已正确配置
- [ ] 默认密码已更改
- [ ] 网络安全已配置
- [ ] 资源限制已设置
- [ ] 备份策略已实施
- [ ] 监控告警已配置
- [ ] 日志轮转已设置
- [ ] 健康检查已验证
- [ ] 性能测试已完成
- [ ] 文档已更新

## 支持和维护

### 获取帮助

1. **查看文档**
   - [Apache Airflow官方文档](https://airflow.apache.org/docs/)
   - [Docker Compose文档](https://docs.docker.com/compose/)

2. **社区支持**
   - Airflow Slack频道
   - Stack Overflow
   - GitHub Issues

3. **日志分析**
   ```powershell
   # 收集诊断信息
   .\deploy.ps1 status > diagnosis.txt
   .\deploy.ps1 logs >> diagnosis.txt
   ```

### 版本升级

```powershell
# 备份当前环境
.\deploy.ps1 stop
docker-compose exec postgres pg_dump -U airflow airflow > backup_before_upgrade.sql

# 更新镜像版本
# 编辑docker-compose.yml中的镜像版本

# 重新构建和启动
.\deploy.ps1 build -Force
.\deploy.ps1 start
```

### 维护计划

- **每日**: 检查服务状态和日志
- **每周**: 清理旧日志和临时文件
- **每月**: 数据库维护和性能检查
- **每季度**: 安全更新和版本升级

---

如有问题，请查看日志文件或联系系统管理员。