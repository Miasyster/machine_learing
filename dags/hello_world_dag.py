"""
简单的Hello World DAG - 用于测试Airflow基本功能
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# 默认参数
default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 创建DAG
dag = DAG(
    'hello_world_dag',
    default_args=default_args,
    description='简单的Hello World测试DAG',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['test', 'hello_world']
)

def print_hello():
    """打印Hello World消息"""
    print("Hello World from Airflow!")
    print("这是一个简单的测试任务")
    return "Hello World task completed"

def print_date():
    """打印当前日期时间"""
    from datetime import datetime
    current_time = datetime.now()
    print(f"当前时间: {current_time}")
    print(f"DAG运行成功!")
    return f"Date task completed at {current_time}"

# 任务定义
start_task = EmptyOperator(
    task_id='start',
    dag=dag
)

hello_task = PythonOperator(
    task_id='print_hello',
    python_callable=print_hello,
    dag=dag
)

date_task = PythonOperator(
    task_id='print_date',
    python_callable=print_date,
    dag=dag
)

bash_task = BashOperator(
    task_id='bash_command',
    bash_command='echo "这是一个Bash命令任务" && date',
    dag=dag
)

end_task = EmptyOperator(
    task_id='end',
    dag=dag
)

# 定义任务依赖关系
start_task >> [hello_task, date_task] >> bash_task >> end_task