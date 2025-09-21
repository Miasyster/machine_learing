# 基于官方Airflow镜像构建自定义镜像
FROM apache/airflow:2.7.0-python3.10

# 切换到root用户安装系统依赖
USER root

# 安装系统依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
        wget \
        vim \
        htop \
        postgresql-client \
        redis-tools \
        libpq-dev \
        libffi-dev \
        libssl-dev \
        python3-dev \
        pkg-config \
        libhdf5-dev \
        libnetcdf-dev \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建必要的目录
RUN mkdir -p /opt/airflow/ml_project \
    && mkdir -p /opt/airflow/data \
    && mkdir -p /opt/airflow/models \
    && mkdir -p /opt/airflow/logs/ml \
    && mkdir -p /opt/airflow/plugins/ml

# 切换回airflow用户
USER airflow

# 设置工作目录
WORKDIR /opt/airflow

# 复制requirements文件
COPY requirements.txt /opt/airflow/requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY --chown=airflow:root ./src /opt/airflow/ml_project/src
COPY --chown=airflow:root ./configs /opt/airflow/ml_project/configs
COPY --chown=airflow:root ./dags /opt/airflow/dags
COPY --chown=airflow:root ./plugins /opt/airflow/plugins

# 设置Python路径
ENV PYTHONPATH="/opt/airflow/ml_project/src:${PYTHONPATH}"
ENV PYTHONPATH="/opt/airflow/plugins:${PYTHONPATH}"

# 设置ML项目环境变量
ENV ML_PROJECT_ROOT="/opt/airflow/ml_project"
ENV ML_DATA_PATH="/opt/airflow/data"
ENV ML_MODELS_PATH="/opt/airflow/models"
ENV ML_LOGS_PATH="/opt/airflow/logs/ml"

# 创建启动脚本
COPY --chown=airflow:root <<EOF /opt/airflow/entrypoint-custom.sh
#!/bin/bash
set -e

# 初始化ML项目目录结构
mkdir -p \${ML_DATA_PATH}/{raw,processed,features,models}
mkdir -p \${ML_MODELS_PATH}/{trained,staging,production}
mkdir -p \${ML_LOGS_PATH}/{training,inference,monitoring}

# 设置权限
chmod -R 755 \${ML_DATA_PATH}
chmod -R 755 \${ML_MODELS_PATH}
chmod -R 755 \${ML_LOGS_PATH}

# 初始化Airflow连接（如果需要）
if [ "\$1" = "webserver" ] || [ "\$1" = "scheduler" ]; then
    echo "Initializing Airflow connections..."
    
    # 等待数据库就绪
    airflow db check || sleep 10
    
    # 创建ML相关的连接
    airflow connections add 'ml_postgres' \
        --conn-type 'postgres' \
        --conn-host 'ml-postgres' \
        --conn-port 5432 \
        --conn-login '\${POSTGRES_USER:-ml_user}' \
        --conn-password '\${POSTGRES_PASSWORD:-ml_password}' \
        --conn-schema 'ml_data' || true
    
    airflow connections add 'ml_redis' \
        --conn-type 'redis' \
        --conn-host 'ml-redis' \
        --conn-port 6379 \
        --conn-extra '{"db": 1}' || true
    
    # 创建变量
    airflow variables set ML_PROJECT_ROOT "\${ML_PROJECT_ROOT}"
    airflow variables set ML_DATA_PATH "\${ML_DATA_PATH}"
    airflow variables set ML_MODELS_PATH "\${ML_MODELS_PATH}"
    airflow variables set ML_LOGS_PATH "\${ML_LOGS_PATH}"
    
    echo "Airflow connections and variables initialized."
fi

# 执行原始入口点
exec /entrypoint "\$@"
EOF

# 设置启动脚本权限
RUN chmod +x /opt/airflow/entrypoint-custom.sh

# 健康检查脚本
COPY --chown=airflow:root <<EOF /opt/airflow/healthcheck.py
#!/usr/bin/env python3
"""
Airflow健康检查脚本
"""
import sys
import requests
import psycopg2
import redis
import os
from datetime import datetime

def check_airflow_webserver():
    """检查Airflow Web服务器"""
    try:
        response = requests.get('http://localhost:8080/health', timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Webserver health check failed: {e}")
        return False

def check_database():
    """检查数据库连接"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=os.getenv('POSTGRES_PORT', 5432),
            user='airflow',
            password='airflow',
            database='airflow'
        )
        conn.close()
        return True
    except Exception as e:
        print(f"Database health check failed: {e}")
        return False

def check_redis():
    """检查Redis连接"""
    try:
        r = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=os.getenv('REDIS_PORT', 6379),
            db=0
        )
        r.ping()
        return True
    except Exception as e:
        print(f"Redis health check failed: {e}")
        return False

def check_ml_database():
    """检查ML数据库连接"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'ml-postgres'),
            port=5432,
            user=os.getenv('POSTGRES_USER', 'ml_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'ml_password'),
            database='ml_data'
        )
        conn.close()
        return True
    except Exception as e:
        print(f"ML Database health check failed: {e}")
        return False

def main():
    """主健康检查函数"""
    checks = {
        'database': check_database(),
        'redis': check_redis(),
        'ml_database': check_ml_database(),
    }
    
    # 如果是webserver，检查web服务
    if 'webserver' in sys.argv:
        checks['webserver'] = check_airflow_webserver()
    
    # 打印检查结果
    print(f"Health check results at {datetime.now()}:")
    for service, status in checks.items():
        status_str = "✓ HEALTHY" if status else "✗ UNHEALTHY"
        print(f"  {service}: {status_str}")
    
    # 如果所有检查都通过，返回0，否则返回1
    if all(checks.values()):
        print("All health checks passed!")
        sys.exit(0)
    else:
        print("Some health checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# 设置健康检查脚本权限
RUN chmod +x /opt/airflow/healthcheck.py

# 使用自定义入口点
ENTRYPOINT ["/opt/airflow/entrypoint-custom.sh"]

# 默认命令
CMD ["webserver"]