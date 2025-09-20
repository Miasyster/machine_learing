#!/usr/bin/env python3
"""
Airflow连接设置脚本
用于创建和配置Airflow中的连接和变量
"""

import os
import json
from airflow.models import Connection, Variable
from airflow.utils.db import create_session
from airflow import settings


def create_binance_connection():
    """创建Binance API连接"""
    conn_id = 'binance_api'
    
    # 检查连接是否已存在
    with create_session() as session:
        existing_conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()
        if existing_conn:
            print(f"连接 {conn_id} 已存在，跳过创建")
            return
    
    # 创建新连接
    new_conn = Connection(
        conn_id=conn_id,
        conn_type='http',
        host='https://api.binance.com',
        extra=json.dumps({
            'api_key': os.getenv('BINANCE_API_KEY', ''),
            'api_secret': os.getenv('BINANCE_API_SECRET', ''),
            'rate_limit': 1200,
            'timeout': 30
        })
    )
    
    with create_session() as session:
        session.add(new_conn)
        session.commit()
    
    print(f"已创建Binance API连接: {conn_id}")


def create_postgres_connection():
    """创建PostgreSQL数据库连接"""
    conn_id = 'postgres_ml'
    
    # 检查连接是否已存在
    with create_session() as session:
        existing_conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()
        if existing_conn:
            print(f"连接 {conn_id} 已存在，跳过创建")
            return
    
    # 创建新连接
    new_conn = Connection(
        conn_id=conn_id,
        conn_type='postgres',
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        schema=os.getenv('POSTGRES_DB', 'ml_database'),
        login=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', '')
    )
    
    with create_session() as session:
        session.add(new_conn)
        session.commit()
    
    print(f"已创建PostgreSQL连接: {conn_id}")


def create_redis_connection():
    """创建Redis连接"""
    conn_id = 'redis_cache'
    
    # 检查连接是否已存在
    with create_session() as session:
        existing_conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()
        if existing_conn:
            print(f"连接 {conn_id} 已存在，跳过创建")
            return
    
    # 创建新连接
    new_conn = Connection(
        conn_id=conn_id,
        conn_type='redis',
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', '6379')),
        password=os.getenv('REDIS_PASSWORD', ''),
        extra=json.dumps({
            'db': int(os.getenv('REDIS_DB', '0'))
        })
    )
    
    with create_session() as session:
        session.add(new_conn)
        session.commit()
    
    print(f"已创建Redis连接: {conn_id}")


def create_file_system_connection():
    """创建文件系统连接"""
    conn_id = 'fs_ml_data'
    
    # 检查连接是否已存在
    with create_session() as session:
        existing_conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()
        if existing_conn:
            print(f"连接 {conn_id} 已存在，跳过创建")
            return
    
    # 创建新连接
    data_path = os.getenv('ML_DATA_PATH', '/opt/airflow/data')
    new_conn = Connection(
        conn_id=conn_id,
        conn_type='fs',
        extra=json.dumps({
            'path': data_path
        })
    )
    
    with create_session() as session:
        session.add(new_conn)
        session.commit()
    
    print(f"已创建文件系统连接: {conn_id}")


def create_slack_connection():
    """创建Slack通知连接"""
    conn_id = 'slack_alerts'
    
    # 检查连接是否已存在
    with create_session() as session:
        existing_conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()
        if existing_conn:
            print(f"连接 {conn_id} 已存在，跳过创建")
            return
    
    # 创建新连接
    webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')
    if webhook_url:
        new_conn = Connection(
            conn_id=conn_id,
            conn_type='http',
            host='hooks.slack.com',
            extra=json.dumps({
                'webhook_url': webhook_url,
                'channel': '#data-alerts'
            })
        )
        
        with create_session() as session:
            session.add(new_conn)
            session.commit()
        
        print(f"已创建Slack连接: {conn_id}")
    else:
        print("未设置SLACK_WEBHOOK_URL环境变量，跳过Slack连接创建")


def create_airflow_variables():
    """创建Airflow变量"""
    variables = {
        # 数据路径变量
        'ml_data_base_path': os.getenv('ML_DATA_PATH', '/opt/airflow/data'),
        'ml_raw_data_path': os.getenv('ML_RAW_DATA_PATH', '/opt/airflow/data/raw'),
        'ml_processed_data_path': os.getenv('ML_PROCESSED_DATA_PATH', '/opt/airflow/data/processed'),
        'ml_features_data_path': os.getenv('ML_FEATURES_DATA_PATH', '/opt/airflow/data/features'),
        
        # 配置文件路径
        'dag_config_path': '/opt/airflow/dags/config/dag_config.yaml',
        'data_sources_config_path': '/opt/airflow/dags/config/data_sources.yaml',
        'quality_thresholds_config_path': '/opt/airflow/dags/config/quality_thresholds.yaml',
        
        # 交易对配置
        'default_trading_pairs': json.dumps(['BTCUSDT', 'ETHUSDT', 'BNBUSDT']),
        'default_intervals': json.dumps(['1h', '4h', '1d']),
        
        # 数据质量阈值
        'data_quality_threshold': '80.0',
        'missing_data_threshold': '5.0',
        'price_anomaly_threshold': '10.0',
        
        # 通知配置
        'enable_email_alerts': 'true',
        'enable_slack_alerts': 'true',
        'alert_recipients': json.dumps(['admin@company.com']),
        
        # 环境配置
        'environment': os.getenv('AIRFLOW_ENV', 'development'),
        'debug_mode': os.getenv('DEBUG_MODE', 'false'),
        
        # 资源配置
        'max_parallel_tasks': '4',
        'task_timeout_minutes': '60',
        'retry_delay_minutes': '5'
    }
    
    for key, value in variables.items():
        try:
            # 检查变量是否已存在
            existing_var = Variable.get(key, default_var=None)
            if existing_var is not None:
                print(f"变量 {key} 已存在，跳过创建")
                continue
            
            # 创建新变量
            Variable.set(key, value)
            print(f"已创建变量: {key} = {value}")
            
        except Exception as e:
            print(f"创建变量 {key} 时出错: {str(e)}")


def setup_email_configuration():
    """设置邮件配置变量"""
    email_config = {
        'smtp_host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
        'smtp_port': os.getenv('SMTP_PORT', '587'),
        'smtp_user': os.getenv('SMTP_USER', ''),
        'smtp_password': os.getenv('SMTP_PASSWORD', ''),
        'smtp_starttls': 'true',
        'smtp_ssl': 'false'
    }
    
    for key, value in email_config.items():
        var_key = f'email_{key}'
        try:
            existing_var = Variable.get(var_key, default_var=None)
            if existing_var is not None:
                print(f"邮件配置变量 {var_key} 已存在，跳过创建")
                continue
            
            Variable.set(var_key, value)
            print(f"已创建邮件配置变量: {var_key}")
            
        except Exception as e:
            print(f"创建邮件配置变量 {var_key} 时出错: {str(e)}")


def main():
    """主函数：设置所有连接和变量"""
    print("开始设置Airflow连接和变量...")
    
    # 创建连接
    print("\n=== 创建连接 ===")
    create_binance_connection()
    create_postgres_connection()
    create_redis_connection()
    create_file_system_connection()
    create_slack_connection()
    
    # 创建变量
    print("\n=== 创建变量 ===")
    create_airflow_variables()
    
    # 设置邮件配置
    print("\n=== 设置邮件配置 ===")
    setup_email_configuration()
    
    print("\n✅ Airflow连接和变量设置完成！")
    print("\n📝 请确保以下环境变量已正确设置：")
    print("   - BINANCE_API_KEY")
    print("   - BINANCE_API_SECRET")
    print("   - POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD")
    print("   - REDIS_HOST, REDIS_PASSWORD (可选)")
    print("   - SLACK_WEBHOOK_URL (可选)")
    print("   - SMTP_HOST, SMTP_USER, SMTP_PASSWORD (可选)")


if __name__ == '__main__':
    main()