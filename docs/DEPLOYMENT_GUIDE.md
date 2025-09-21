# 部署指南

本文档详细介绍了机器学习项目的部署流程、配置和最佳实践。

## 目录

- [概述](#概述)
- [环境准备](#环境准备)
- [本地部署](#本地部署)
- [容器化部署](#容器化部署)
- [云平台部署](#云平台部署)
- [生产环境配置](#生产环境配置)
- [监控和维护](#监控和维护)
- [故障排除](#故障排除)

## 概述

本项目支持多种部署方式：

1. **本地开发部署**：用于开发和测试
2. **容器化部署**：使用Docker进行标准化部署
3. **云平台部署**：支持AWS、Azure、GCP等云平台
4. **生产环境部署**：高可用、可扩展的生产部署

## 环境准备

### 系统要求

#### 最低配置
- **CPU**：2核心
- **内存**：4GB RAM
- **存储**：20GB可用空间
- **操作系统**：Ubuntu 20.04+, CentOS 8+, Windows 10+, macOS 10.15+

#### 推荐配置
- **CPU**：4核心以上
- **内存**：8GB RAM以上
- **存储**：50GB SSD
- **GPU**：NVIDIA GPU（用于深度学习训练）

### 软件依赖

#### Python环境
```bash
# Python 3.8-3.11
python --version

# 虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

#### 系统依赖
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-dev build-essential libssl-dev libffi-dev

# CentOS/RHEL
sudo yum install -y python3-devel gcc openssl-devel libffi-devel

# macOS
brew install python@3.9
```

## 本地部署

### 1. 克隆项目

```bash
git clone https://github.com/your-username/machine_learning.git
cd machine_learning
```

### 2. 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 配置文件

创建配置文件 `config/local.yaml`：

```yaml
# 本地开发配置
app:
  name: "ML Project"
  version: "1.0.0"
  debug: true
  host: "localhost"
  port: 8000

database:
  url: "sqlite:///data/local.db"
  echo: true

logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/app.log"

model:
  cache_dir: "models/cache"
  max_memory: "2GB"
  device: "cpu"  # 或 "cuda" 如果有GPU
```

### 4. 数据库初始化

```bash
# 创建数据库表
python -m src.database.init_db

# 加载示例数据（可选）
python -m src.database.load_sample_data
```

### 5. 启动服务

```bash
# 开发模式
python -m src.main --config config/local.yaml

# 或使用Makefile
make run-dev
```

### 6. 验证部署

```bash
# 健康检查
curl http://localhost:8000/health

# API测试
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [1, 2, 3, 4, 5]}'
```

## 容器化部署

### 1. Docker配置

#### Dockerfile

```dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "-m", "src.main", "--config", "config/production.yaml"]
```

#### .dockerignore

```
.git
.gitignore
README.md
Dockerfile
.dockerignore
.pytest_cache
.coverage
htmlcov/
.tox/
venv/
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
pip-log.txt
pip-delete-this-directory.txt
.DS_Store
.vscode/
.idea/
*.log
```

### 2. Docker Compose

#### docker-compose.yml

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - DATABASE_URL=postgresql://user:password@db:5432/mldb
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=mldb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://app/health;
            access_log off;
        }
    }
}
```

### 3. 构建和运行

```bash
# 构建镜像
docker build -t ml-project:latest .

# 运行容器
docker run -d \
  --name ml-app \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  ml-project:latest

# 使用Docker Compose
docker-compose up -d

# 查看日志
docker-compose logs -f app

# 停止服务
docker-compose down
```

## 云平台部署

### AWS部署

#### 1. ECS部署

```yaml
# ecs-task-definition.json
{
  "family": "ml-project",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "ml-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/ml-project:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:ml-db-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ml-project",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 2. Lambda部署

```python
# lambda_handler.py
import json
from src.api.lambda_adapter import LambdaAdapter

adapter = LambdaAdapter()

def lambda_handler(event, context):
    """AWS Lambda处理函数"""
    try:
        response = adapter.handle(event, context)
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

#### 3. CloudFormation模板

```yaml
# cloudformation.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'ML Project Infrastructure'

Parameters:
  ImageUri:
    Type: String
    Description: ECR image URI

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true

  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: ml-project-cluster

  ECSService:
    Type: AWS::ECS::Service
    Properties:
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref SecurityGroup
          Subnets:
            - !Ref PrivateSubnet1
            - !Ref PrivateSubnet2

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: ml-project
      Cpu: 512
      Memory: 1024
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
      ContainerDefinitions:
        - Name: ml-app
          Image: !Ref ImageUri
          PortMappings:
            - ContainerPort: 8000
```

### Azure部署

#### 1. Container Instances

```yaml
# azure-container-instance.yaml
apiVersion: 2019-12-01
location: eastus
name: ml-project-aci
properties:
  containers:
  - name: ml-app
    properties:
      image: yourregistry.azurecr.io/ml-project:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
      ports:
      - port: 8000
        protocol: TCP
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
  imageRegistryCredentials:
  - server: yourregistry.azurecr.io
    username: username
    password: password
```

#### 2. App Service

```yaml
# azure-app-service.yaml
apiVersion: 2020-12-01
type: Microsoft.Web/sites
name: ml-project-app
location: East US
properties:
  serverFarmId: /subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.Web/serverfarms/{app-service-plan}
  siteConfig:
    linuxFxVersion: DOCKER|yourregistry.azurecr.io/ml-project:latest
    appSettings:
    - name: WEBSITES_ENABLE_APP_SERVICE_STORAGE
      value: false
    - name: ENV
      value: production
```

### Google Cloud部署

#### 1. Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ml-project
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      containers:
      - image: gcr.io/project-id/ml-project:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENV
          value: production
        resources:
          limits:
            cpu: 1000m
            memory: 2Gi
```

#### 2. GKE部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-project
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-project
  template:
    metadata:
      labels:
        app: ml-project
    spec:
      containers:
      - name: ml-app
        image: gcr.io/project-id/ml-project:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENV
          value: production
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
---
apiVersion: v1
kind: Service
metadata:
  name: ml-project-service
spec:
  selector:
    app: ml-project
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 生产环境配置

### 1. 环境变量

```bash
# 生产环境变量
export ENV=production
export DEBUG=false
export DATABASE_URL=postgresql://user:password@db:5432/mldb
export REDIS_URL=redis://redis:6379/0
export SECRET_KEY=your-secret-key
export API_KEY=your-api-key
export LOG_LEVEL=INFO
export MAX_WORKERS=4
export TIMEOUT=30
```

### 2. 配置文件

#### config/production.yaml

```yaml
app:
  name: "ML Project"
  version: "1.0.0"
  debug: false
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30

database:
  url: "${DATABASE_URL}"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
  echo: false

cache:
  url: "${REDIS_URL}"
  timeout: 300
  max_connections: 20

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - type: "file"
      filename: "/app/logs/app.log"
      max_bytes: 10485760  # 10MB
      backup_count: 5
    - type: "console"

security:
  secret_key: "${SECRET_KEY}"
  api_key: "${API_KEY}"
  cors_origins:
    - "https://yourdomain.com"
    - "https://api.yourdomain.com"

model:
  cache_dir: "/app/models/cache"
  max_memory: "4GB"
  device: "cuda"
  batch_size: 32
  timeout: 60

monitoring:
  metrics_enabled: true
  health_check_interval: 30
  prometheus_port: 9090
```

### 3. 安全配置

#### SSL/TLS配置

```nginx
# nginx-ssl.conf
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

#### 防火墙配置

```bash
# UFW配置
sudo ufw enable
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8000/tcp  # 只允许内部访问
```

### 4. 性能优化

#### Gunicorn配置

```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
preload_app = True
```

#### 数据库优化

```sql
-- PostgreSQL优化
-- 连接池配置
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';

-- 索引优化
CREATE INDEX CONCURRENTLY idx_model_predictions_created_at 
ON model_predictions(created_at);

CREATE INDEX CONCURRENTLY idx_users_email 
ON users(email);
```

## 监控和维护

### 1. 健康检查

```python
# src/api/health.py
from fastapi import APIRouter, Depends
from src.database import get_db
from src.cache import get_cache

router = APIRouter()

@router.get("/health")
async def health_check(db=Depends(get_db), cache=Depends(get_cache)):
    """系统健康检查"""
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # 数据库检查
    try:
        await db.execute("SELECT 1")
        checks["checks"]["database"] = "healthy"
    except Exception as e:
        checks["checks"]["database"] = f"unhealthy: {str(e)}"
        checks["status"] = "unhealthy"
    
    # 缓存检查
    try:
        await cache.ping()
        checks["checks"]["cache"] = "healthy"
    except Exception as e:
        checks["checks"]["cache"] = f"unhealthy: {str(e)}"
        checks["status"] = "unhealthy"
    
    # 模型检查
    try:
        from src.models import get_model
        model = get_model()
        checks["checks"]["model"] = "healthy"
    except Exception as e:
        checks["checks"]["model"] = f"unhealthy: {str(e)}"
        checks["status"] = "unhealthy"
    
    return checks
```

### 2. 日志配置

```python
# src/logging_config.py
import logging
import logging.handlers
from pathlib import Path

def setup_logging(config):
    """配置日志系统"""
    log_level = getattr(logging, config.get('level', 'INFO'))
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 配置根日志器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[]
    )
    
    logger = logging.getLogger()
    
    # 文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    return logger
```

### 3. 指标监控

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# 定义指标
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total model predictions')
MODEL_LATENCY = Histogram('model_prediction_duration_seconds', 'Model prediction duration')

def start_metrics_server(port=9090):
    """启动指标服务器"""
    start_http_server(port)

def record_request(method, endpoint, status_code, duration):
    """记录请求指标"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
    REQUEST_DURATION.observe(duration)

def record_prediction(duration):
    """记录预测指标"""
    MODEL_PREDICTIONS.inc()
    MODEL_LATENCY.observe(duration)
```

### 4. 备份策略

```bash
#!/bin/bash
# backup.sh

# 数据库备份
pg_dump $DATABASE_URL > backups/db_$(date +%Y%m%d_%H%M%S).sql

# 模型备份
tar -czf backups/models_$(date +%Y%m%d_%H%M%S).tar.gz models/

# 配置备份
tar -czf backups/config_$(date +%Y%m%d_%H%M%S).tar.gz config/

# 清理旧备份（保留30天）
find backups/ -name "*.sql" -mtime +30 -delete
find backups/ -name "*.tar.gz" -mtime +30 -delete
```

## 故障排除

### 常见问题

#### 1. 容器启动失败

```bash
# 检查容器日志
docker logs ml-app

# 检查容器状态
docker inspect ml-app

# 进入容器调试
docker exec -it ml-app /bin/bash
```

#### 2. 数据库连接问题

```bash
# 检查数据库连接
psql $DATABASE_URL -c "SELECT 1"

# 检查网络连接
telnet db-host 5432

# 检查防火墙
sudo ufw status
```

#### 3. 性能问题

```bash
# 检查系统资源
top
htop
free -h
df -h

# 检查网络
netstat -tulpn
ss -tulpn

# 检查应用指标
curl http://localhost:9090/metrics
```

#### 4. SSL证书问题

```bash
# 检查证书有效期
openssl x509 -in cert.pem -text -noout

# 测试SSL连接
openssl s_client -connect yourdomain.com:443

# 更新证书（Let's Encrypt）
certbot renew
```

### 调试工具

#### 1. 应用调试

```python
# 启用调试模式
import pdb; pdb.set_trace()

# 性能分析
import cProfile
cProfile.run('your_function()')

# 内存分析
import tracemalloc
tracemalloc.start()
```

#### 2. 网络调试

```bash
# 网络连接测试
curl -v http://localhost:8000/health
wget --spider http://localhost:8000/health

# DNS解析测试
nslookup yourdomain.com
dig yourdomain.com

# 端口扫描
nmap -p 8000 localhost
```

#### 3. 数据库调试

```sql
-- 查看活动连接
SELECT * FROM pg_stat_activity;

-- 查看慢查询
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- 查看锁等待
SELECT * FROM pg_locks WHERE NOT granted;
```

## 相关文档

- [CI/CD指南](CI_CD_GUIDE.md)
- [测试指南](TESTING_GUIDE.md)
- [API参考](API_REFERENCE.md)
- [最佳实践](BEST_PRACTICES.md)