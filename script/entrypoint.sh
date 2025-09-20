#!/usr/bin/env bash
set -e

# 安装额外的Python包
if [ -f "/opt/airflow/requirements.txt" ]; then
    pip install --user -r /opt/airflow/requirements.txt
fi

# 等待数据库准备就绪
function wait_for_port() {
    local name="$1" host="$2" port="$3"
    local j=0
    while ! nc -z "$host" "$port" >/dev/null 2>&1 < /dev/null; do
        j=$((j+1))
        if [ $j -ge 200 ]; then
            echo >&2 "$(date) - $host:$port still not reachable, giving up"
            exit 1
        fi
        echo "$(date) - waiting for $name... $j/200"
        sleep 1
    done
}

wait_for_port "Postgres" "postgres" "5432"

# 初始化数据库（仅在webserver启动时执行一次）
if [ "$1" = "webserver" ]; then
    echo "Initializing database..."
    airflow db init
    
    # 创建管理员用户
    echo "Creating admin user..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin || true
fi

# 启动Airflow服务
exec airflow "$@"