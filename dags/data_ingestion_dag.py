"""
数据获取DAG
从Binance API获取加密货币K线数据
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from operators.data_ingestion_operator import BinanceDataIngestionOperator, MultiSymbolDataIngestionOperator
from utils.task_helpers import (
    create_default_args, get_trading_pairs, get_default_intervals,
    get_data_path, get_task_timeout, get_retry_config
)


# DAG配置
DAG_ID = 'data_ingestion_dag'
DESCRIPTION = '从Binance API获取加密货币K线数据'

# 获取配置
default_args = create_default_args()
trading_pairs = get_trading_pairs()
intervals = get_default_intervals()
raw_data_path = get_data_path('raw')

# 创建DAG
dag = DAG(
    dag_id=DAG_ID,
    description=DESCRIPTION,
    default_args=default_args,
    schedule_interval='0 * * * *',  # 每小时执行
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['data', 'ingestion', 'binance', 'crypto']
)


def check_api_connection(**context):
    """检查Binance API连接"""
    import requests
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # 测试Binance API连接
        response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
        response.raise_for_status()
        
        logger.info("Binance API连接正常")
        return True
        
    except Exception as e:
        logger.error(f"Binance API连接失败: {str(e)}")
        raise


def validate_raw_data(**context):
    """验证原始数据"""
    import logging
    
    logger = logging.getLogger(__name__)
    
    # 从XCom获取数据获取结果
    task_instance = context['task_instance']
    
    # 获取所有数据获取任务的结果
    results = {}
    for symbol in trading_pairs:
        for interval in intervals:
            task_id = f"fetch_data.{symbol.lower()}_{interval}_data"
            try:
                result = task_instance.xcom_pull(task_ids=task_id, key='ingestion_result')
                if result:
                    results[f"{symbol}_{interval}"] = result
            except Exception as e:
                logger.warning(f"无法获取 {task_id} 的结果: {str(e)}")
    
    # 验证结果
    total_tasks = len(trading_pairs) * len(intervals)
    successful_tasks = len(results)
    success_rate = successful_tasks / total_tasks * 100
    
    logger.info(f"数据获取成功率: {success_rate:.1f}% ({successful_tasks}/{total_tasks})")
    
    # 检查成功率阈值
    if success_rate < 80:
        logger.warning(f"数据获取成功率过低: {success_rate:.1f}%")
    
    # 计算总数据量
    total_data_points = sum(result.get('data_points', 0) for result in results.values())
    total_file_size = sum(result.get('file_size_mb', 0) for result in results.values())
    
    summary = {
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'success_rate': success_rate,
        'total_data_points': total_data_points,
        'total_file_size_mb': round(total_file_size, 2),
        'results': results
    }
    
    logger.info(f"数据获取汇总: {total_data_points} 条数据, {total_file_size:.2f} MB")
    
    # 推送汇总结果到XCom
    task_instance.xcom_push(key='validation_summary', value=summary)
    
    return summary


def send_completion_notification(**context):
    """发送完成通知"""
    import logging
    
    logger = logging.getLogger(__name__)
    
    # 获取验证结果
    task_instance = context['task_instance']
    summary = task_instance.xcom_pull(task_ids='validate_raw_data', key='validation_summary')
    
    if summary:
        success_rate = summary.get('success_rate', 0)
        total_data_points = summary.get('total_data_points', 0)
        
        if success_rate >= 90:
            status = "✅ 成功"
        elif success_rate >= 70:
            status = "⚠️ 部分成功"
        else:
            status = "❌ 失败"
        
        message = f"""
数据获取任务完成 {status}

📊 执行摘要:
- 成功率: {success_rate:.1f}%
- 数据量: {total_data_points:,} 条
- 文件大小: {summary.get('total_file_size_mb', 0):.2f} MB
- 执行时间: {context['execution_date']}

🔗 详细信息请查看Airflow Web界面
        """
        
        logger.info(message)
        
        # TODO: 实现实际的通知发送（邮件、Slack等）
        # 这里可以集成邮件或Slack通知
        
    return "通知发送完成"


# 开始任务
start_task = EmptyOperator(
    task_id='start',
    dag=dag
)

# API连接检查
api_check_task = PythonOperator(
    task_id='check_api_connection',
    python_callable=check_api_connection,
    dag=dag
)

# 数据获取任务组
with TaskGroup('fetch_data', dag=dag) as fetch_data_group:
    
    for symbol in trading_pairs:
        for interval in intervals:
            # 为每个交易对和时间间隔创建获取任务
            fetch_task = BinanceDataIngestionOperator(
                task_id=f'{symbol.lower()}_{interval}_data',
                symbol=symbol,
                interval=interval,
                limit=1000,
                connection_id='binance_api',
                storage_path=raw_data_path,
                file_format='csv',
                validate_data=True,
                **get_retry_config('api'),
                dag=dag
            )

# 原始数据验证
validate_task = PythonOperator(
    task_id='validate_raw_data',
    python_callable=validate_raw_data,
    dag=dag
)

# 数据目录检查（确保数据已保存）
check_data_dir = BashOperator(
    task_id='check_data_directory',
    bash_command=f'ls -la {raw_data_path} && echo "数据目录检查完成"',
    dag=dag
)

# 完成通知
notification_task = PythonOperator(
    task_id='send_completion_notification',
    python_callable=send_completion_notification,
    dag=dag
)

# 结束任务
end_task = EmptyOperator(
    task_id='end',
    dag=dag
)

# 定义任务依赖关系
start_task >> api_check_task >> fetch_data_group >> validate_task >> check_data_dir >> notification_task >> end_task


# 添加SLA监控
def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """SLA超时回调函数"""
    import logging
    
    logger = logging.getLogger(__name__)
    logger.error(f"SLA超时: DAG={dag.dag_id}, 任务={[task.task_id for task in task_list]}")
    
    # TODO: 发送SLA超时告警


# 设置SLA
dag.sla_miss_callback = sla_miss_callback

# 添加文档
dag.doc_md = """
# 数据获取DAG

## 功能描述
从Binance API获取加密货币K线数据，支持多个交易对和时间间隔。

## 执行流程
1. **API连接检查**: 验证Binance API可用性
2. **数据获取**: 并行获取多个交易对的K线数据
3. **数据验证**: 验证获取的数据质量和完整性
4. **目录检查**: 确认数据文件已正确保存
5. **完成通知**: 发送执行结果通知

## 配置说明
- **调度频率**: 每小时执行一次
- **交易对**: 从配置文件或Airflow变量获取
- **时间间隔**: 支持1h、4h、1d等多种间隔
- **数据格式**: CSV格式存储
- **存储路径**: 按交易对和日期分层存储

## 监控指标
- 数据获取成功率
- 数据量统计
- 文件大小统计
- 任务执行时间

## 告警机制
- API连接失败告警
- 数据获取成功率低于阈值告警
- SLA超时告警
"""