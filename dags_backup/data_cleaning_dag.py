"""
数据清洗和完整性检查DAG
集成DataCleaner和DataIntegrityChecker模块
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.email import EmailOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.utils.dates import days_ago

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from operators.data_cleaning_operator import (
    DataCleaningOperator,
    DataIntegrityCheckOperator,
    DataQualityReportOperator
)
from utils.task_helpers import (
    load_dag_config,
    get_dag_default_args,
    validate_symbols_and_intervals,
    send_notification
)

# 加载配置
config = load_dag_config()
dag_config = config.get('data_cleaning_dag', {})

# DAG配置
DAG_ID = 'data_cleaning_dag'
DESCRIPTION = """
数据清洗和完整性检查DAG

主要功能：
1. 对原始数据进行清洗处理
2. 执行数据完整性检查
3. 生成数据质量报告
4. 发送质量告警通知

依赖关系：
- 依赖于data_ingestion_dag的输出数据
- 为feature_engineering_dag提供清洗后的数据
"""

# 默认参数
default_args = get_dag_default_args()
default_args.update({
    'depends_on_past': True,  # 依赖前一次执行成功
    'wait_for_downstream': True,  # 等待下游任务完成
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': dag_config.get('retries', 2),
    'retry_delay': timedelta(minutes=dag_config.get('retry_delay_minutes', 5)),
    'sla': timedelta(hours=dag_config.get('sla_hours', 2))
})

# 交易对和时间间隔配置
SYMBOLS = Variable.get('ml_symbols', 'BTCUSDT,ETHUSDT,ADAUSDT').split(',')
INTERVALS = Variable.get('ml_intervals', '1h,4h,1d').split(',')

# 数据质量阈值
QUALITY_THRESHOLD = float(Variable.get('ml_quality_threshold', '80.0'))

# 创建DAG
dag = DAG(
    dag_id=DAG_ID,
    description=DESCRIPTION,
    default_args=default_args,
    schedule_interval=dag_config.get('schedule_interval', '0 */6 * * *'),  # 每6小时执行一次
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['data-processing', 'cleaning', 'quality-check'],
    doc_md=DESCRIPTION
)


def check_upstream_data(**context):
    """
    检查上游数据获取任务的状态
    """
    from airflow.models import DagRun
    from airflow.utils.state import State
    
    execution_date = context['execution_date']
    
    # 检查data_ingestion_dag是否成功完成
    upstream_dag_id = 'data_ingestion_dag'
    upstream_dag_runs = DagRun.find(
        dag_id=upstream_dag_id,
        execution_date=execution_date,
        state=State.SUCCESS
    )
    
    if not upstream_dag_runs:
        raise Exception(f"上游DAG {upstream_dag_id} 在 {execution_date} 未成功完成")
    
    print(f"上游DAG {upstream_dag_id} 检查通过")
    return True


def validate_data_paths(**context):
    """
    验证数据路径和文件存在性
    """
    import os
    from airflow.models import Variable
    
    raw_data_path = Variable.get('ml_raw_data_path', '/opt/airflow/data/raw')
    execution_date = context['execution_date']
    
    missing_files = []
    
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            date_str = execution_date.strftime('%Y-%m-%d')
            file_path = os.path.join(
                raw_data_path,
                symbol.lower(),
                interval,
                execution_date.strftime('%Y'),
                execution_date.strftime('%m'),
                f"{symbol.lower()}_{interval}_{date_str}.csv"
            )
            
            if not os.path.exists(file_path):
                missing_files.append(file_path)
    
    if missing_files:
        raise Exception(f"以下数据文件不存在: {missing_files}")
    
    print(f"数据路径验证通过，共检查 {len(SYMBOLS) * len(INTERVALS)} 个文件")
    return True


# 开始任务
start_task = DummyOperator(
    task_id='start',
    dag=dag
)

# 检查上游数据
check_upstream_task = PythonOperator(
    task_id='check_upstream_data',
    python_callable=check_upstream_data,
    dag=dag
)

# 验证数据路径
validate_paths_task = PythonOperator(
    task_id='validate_data_paths',
    python_callable=validate_data_paths,
    dag=dag
)

# 数据清洗任务组
with TaskGroup('clean_data', dag=dag) as clean_data_group:
    
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            task_id = f'{symbol.lower()}_{interval}_cleaning'
            
            # 数据清洗任务
            cleaning_task = DataCleaningOperator(
                task_id=task_id,
                symbol=symbol,
                interval=interval,
                cleaning_config={
                    'remove_duplicates': True,
                    'handle_missing_values': True,
                    'validate_price_logic': True,
                    'normalize_timestamps': True,
                    'remove_outliers': True,
                    'outlier_method': 'iqr',
                    'outlier_threshold': 3.0
                },
                file_format='csv',
                validate_output=True,
                dag=dag
            )

# 数据完整性检查任务组
with TaskGroup('integrity_check', dag=dag) as integrity_check_group:
    
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            task_id = f'{symbol.lower()}_{interval}_integrity'
            
            # 完整性检查任务
            integrity_task = DataIntegrityCheckOperator(
                task_id=task_id,
                symbol=symbol,
                interval=interval,
                check_config={
                    'fail_on_quality_threshold': False,  # 不因质量问题中断流程
                    'detailed_report': True
                },
                quality_threshold=QUALITY_THRESHOLD,
                file_format='csv',
                dag=dag
            )

# 数据质量报告任务
quality_report_task = DataQualityReportOperator(
    task_id='generate_quality_report',
    symbols=SYMBOLS,
    intervals=INTERVALS,
    report_format='json',
    dag=dag
)


def check_quality_alerts(**context):
    """
    检查数据质量告警
    """
    task_instance = context['task_instance']
    
    # 获取质量报告
    quality_report = task_instance.xcom_pull(
        task_ids='generate_quality_report',
        key='quality_report'
    )
    
    if not quality_report:
        print("未找到质量报告")
        return
    
    # 检查质量问题
    alerts = []
    
    # 检查平均质量分数
    avg_score = quality_report['summary']['average_quality_score']
    if avg_score < QUALITY_THRESHOLD:
        alerts.append(f"平均数据质量分数 {avg_score:.2f} 低于阈值 {QUALITY_THRESHOLD}")
    
    # 检查质量分布
    quality_dist = quality_report['summary']['quality_distribution']
    poor_count = quality_dist.get('poor', 0)
    total_datasets = quality_report['summary']['total_datasets']
    
    if poor_count > 0:
        poor_percentage = (poor_count / total_datasets) * 100
        alerts.append(f"{poor_count} 个数据集质量较差 ({poor_percentage:.1f}%)")
    
    # 检查清洗移除率
    for key, result in quality_report.get('cleaning_results', {}).items():
        removal_rate = result.get('cleaning_report', {}).get('removal_percentage', 0)
        if removal_rate > 15:  # 超过15%的数据被移除
            alerts.append(f"{key} 数据移除率过高: {removal_rate:.1f}%")
    
    # 如果有告警，推送到XCom
    if alerts:
        context['task_instance'].xcom_push(key='quality_alerts', value=alerts)
        print(f"发现 {len(alerts)} 个质量告警")
        for alert in alerts:
            print(f"- {alert}")
    else:
        print("数据质量检查通过，无告警")
    
    return alerts


# 质量告警检查任务
quality_alert_task = PythonOperator(
    task_id='check_quality_alerts',
    python_callable=check_quality_alerts,
    dag=dag
)


def send_quality_notification(**context):
    """
    发送质量通知
    """
    task_instance = context['task_instance']
    
    # 获取质量报告和告警
    quality_report = task_instance.xcom_pull(
        task_ids='generate_quality_report',
        key='quality_report'
    )
    
    alerts = task_instance.xcom_pull(
        task_ids='check_quality_alerts',
        key='quality_alerts'
    )
    
    # 构造通知消息
    execution_date = context['execution_date']
    avg_score = quality_report['summary']['average_quality_score']
    
    if alerts:
        # 有告警的通知
        subject = f"数据质量告警 - {execution_date.strftime('%Y-%m-%d')}"
        message = f"""
数据清洗和质量检查完成，发现以下问题：

平均质量分数: {avg_score:.2f}
处理数据集: {quality_report['summary']['cleaned_datasets']}/{quality_report['summary']['total_datasets']}

告警信息:
"""
        for alert in alerts:
            message += f"• {alert}\n"
        
        message += f"\n建议措施:\n"
        for rec in quality_report.get('recommendations', []):
            message += f"• {rec}\n"
        
        # 发送告警通知
        send_notification(
            subject=subject,
            message=message,
            notification_type='warning',
            context=context
        )
    else:
        # 正常完成的通知
        subject = f"数据质量检查通过 - {execution_date.strftime('%Y-%m-%d')}"
        message = f"""
数据清洗和质量检查成功完成：

平均质量分数: {avg_score:.2f}
处理数据集: {quality_report['summary']['cleaned_datasets']}/{quality_report['summary']['total_datasets']}
质量分布: 优秀 {quality_report['summary']['quality_distribution']['excellent']}，良好 {quality_report['summary']['quality_distribution']['good']}，可接受 {quality_report['summary']['quality_distribution']['acceptable']}，较差 {quality_report['summary']['quality_distribution']['poor']}

所有数据已准备就绪，可进行特征工程处理。
"""
        
        # 发送成功通知
        send_notification(
            subject=subject,
            message=message,
            notification_type='success',
            context=context
        )


# 通知任务
notification_task = PythonOperator(
    task_id='send_quality_notification',
    python_callable=send_quality_notification,
    dag=dag
)

# 清理临时文件任务
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command="""
    # 清理超过7天的临时文件
    find {{ var.value.ml_temp_path | default('/tmp/ml_processing') }} -name "*.tmp" -mtime +7 -delete || true
    
    # 清理处理日志
    find {{ var.value.ml_log_path | default('/opt/airflow/logs') }}/data_cleaning -name "*.log" -mtime +30 -delete || true
    
    echo "临时文件清理完成"
    """,
    dag=dag
)

# 结束任务
end_task = DummyOperator(
    task_id='end',
    dag=dag
)

# 设置任务依赖关系
start_task >> check_upstream_task >> validate_paths_task

# 清洗任务依赖验证任务
validate_paths_task >> clean_data_group

# 完整性检查依赖清洗任务
clean_data_group >> integrity_check_group

# 质量报告依赖完整性检查
integrity_check_group >> quality_report_task

# 告警检查依赖质量报告
quality_report_task >> quality_alert_task

# 通知依赖告警检查
quality_alert_task >> notification_task

# 清理和结束
notification_task >> cleanup_task >> end_task

# 设置清洗任务组内的依赖关系
for symbol in SYMBOLS:
    for interval in INTERVALS:
        cleaning_task_id = f'clean_data.{symbol.lower()}_{interval}_cleaning'
        integrity_task_id = f'integrity_check.{symbol.lower()}_{interval}_integrity'
        
        # 确保对应的完整性检查依赖于对应的清洗任务
        dag.get_task(cleaning_task_id) >> dag.get_task(integrity_task_id)

# DAG文档
dag.doc_md = """
# 数据清洗和完整性检查DAG

## 概述
此DAG负责对原始交易数据进行清洗处理和完整性检查，确保数据质量满足后续分析需求。

## 主要功能

### 1. 数据清洗 (clean_data)
- 移除重复数据
- 处理缺失值
- 验证价格逻辑
- 标准化时间戳
- 移除异常值

### 2. 完整性检查 (integrity_check)
- 数据完整性验证
- 质量评分计算
- 异常检测
- 一致性检查

### 3. 质量报告
- 生成详细的质量报告
- 统计清洗效果
- 提供改进建议

### 4. 告警机制
- 质量阈值监控
- 自动告警通知
- 问题追踪

## 配置参数

### 环境变量
- `ml_symbols`: 交易对列表 (默认: BTCUSDT,ETHUSDT,ADAUSDT)
- `ml_intervals`: 时间间隔列表 (默认: 1h,4h,1d)
- `ml_quality_threshold`: 质量阈值 (默认: 80.0)
- `ml_raw_data_path`: 原始数据路径
- `ml_processed_data_path`: 处理后数据路径

### DAG配置
- 调度间隔: 每6小时执行一次
- 重试次数: 2次
- SLA: 2小时
- 最大并行运行数: 1

## 依赖关系
- **上游**: data_ingestion_dag (数据获取)
- **下游**: feature_engineering_dag (特征工程)

## 监控指标
- 数据清洗成功率
- 平均质量分数
- 数据移除率
- 处理时间

## 故障排除

### 常见问题
1. **上游数据不存在**: 检查data_ingestion_dag执行状态
2. **质量分数过低**: 检查原始数据源质量
3. **清洗失败**: 查看具体错误日志，可能需要调整清洗参数

### 恢复步骤
1. 检查上游DAG状态
2. 验证数据文件存在性
3. 检查配置参数
4. 重新运行失败任务
"""