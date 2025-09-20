"""
数据质量监控和告警DAG

提供全面的数据质量监控、告警和健康检查功能
监控整个数据处理流水线的质量指标和性能表现
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

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

# 导入自定义操作符和工具函数
from operators.monitoring_operator import DataQualityMonitorOperator, HealthCheckOperator
from utils.task_helpers import load_config, get_dag_default_args, validate_data_path

# 加载配置
config = load_config()

# DAG默认参数
default_args = get_dag_default_args()
default_args.update({
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=1)
})

# DAG定义
dag = DAG(
    'monitoring_dag',
    default_args=default_args,
    description='数据质量监控和告警DAG - 监控整个数据处理流水线的质量指标和性能表现',
    schedule_interval='0 */6 * * *',  # 每6小时执行一次
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['monitoring', 'quality', 'alerts', 'health-check'],
    doc_md="""
    # 数据质量监控和告警DAG
    
    ## 功能概述
    
    本DAG提供全面的数据质量监控、告警和健康检查功能，包括：
    
    ### 监控功能
    - **流水线状态监控**: 监控所有相关DAG的执行状态和成功率
    - **数据质量监控**: 收集和分析数据完整性、准确性、一致性指标
    - **性能监控**: 监控处理时间、资源使用情况和吞吐量
    - **系统健康检查**: 检查数据库、存储、API连接等系统组件
    
    ### 告警功能
    - **智能告警**: 基于阈值和趋势的异常检测
    - **多渠道通知**: 支持邮件、Slack、日志等多种告警渠道
    - **告警分级**: 根据严重程度进行告警分级
    - **告警抑制**: 避免重复告警和告警风暴
    
    ### 报告功能
    - **质量报告**: 生成详细的数据质量监控报告
    - **趋势分析**: 分析质量指标的历史趋势
    - **建议措施**: 基于监控结果提供改进建议
    
    ## 执行流程
    
    1. **系统健康检查**: 检查系统各组件的健康状态
    2. **数据质量监控**: 收集和分析质量指标
    3. **异常检测**: 检测质量异常和性能问题
    4. **告警处理**: 发送告警通知和生成报告
    5. **清理维护**: 清理过期的监控数据和报告
    
    ## 配置说明
    
    ### Airflow变量配置
    - `ml_monitoring_reports_path`: 监控报告存储路径
    - `alert_emails`: 告警邮件地址列表
    - `slack_webhook_url`: Slack告警Webhook URL
    - `quality_alert_thresholds`: 质量告警阈值配置
    
    ### 连接配置
    - `postgres_ml`: PostgreSQL数据库连接
    - `binance_api`: Binance API连接
    - `slack_webhook`: Slack Webhook连接
    
    ## 监控指标
    
    ### 流水线指标
    - DAG成功率
    - 任务失败率
    - 执行时间
    - 资源使用情况
    
    ### 数据质量指标
    - 数据完整性
    - 数据准确性
    - 数据一致性
    - 特征质量
    
    ### 性能指标
    - 处理时间
    - 吞吐量
    - 延迟
    - 错误率
    
    ## 告警规则
    
    ### 高优先级告警
    - 整体质量分数 < 70%
    - DAG成功率 < 80%
    - 系统组件不可用
    - 数据丢失或严重质量问题
    
    ### 中优先级告警
    - 整体质量分数 < 85%
    - 处理时间超过阈值
    - 数据质量轻微下降
    
    ### 低优先级告警
    - 性能轻微下降
    - 非关键组件问题
    
    ## 维护说明
    
    - 监控报告保留30天
    - 告警历史保留90天
    - 定期检查和更新告警阈值
    - 监控系统自身的性能和可用性
    """
)


def check_monitoring_prerequisites(**context) -> bool:
    """
    检查监控前置条件
    
    Args:
        context: Airflow任务上下文
        
    Returns:
        bool: 检查结果
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # 检查必要的路径
        reports_path = Variable.get('ml_monitoring_reports_path', '/opt/airflow/data/monitoring/reports')
        if not validate_data_path(reports_path, create_if_missing=True):
            raise Exception(f"监控报告路径不可用: {reports_path}")
        
        # 检查配置文件
        config_files = [
            'config/data_sources.yaml',
            'config/quality_thresholds.yaml'
        ]
        
        for config_file in config_files:
            config_path = os.path.join(os.path.dirname(__file__), config_file)
            if not os.path.exists(config_path):
                raise Exception(f"配置文件不存在: {config_file}")
        
        logger.info("监控前置条件检查通过")
        return True
        
    except Exception as e:
        logger.error(f"监控前置条件检查失败: {str(e)}")
        raise


def cleanup_old_monitoring_data(**context) -> None:
    """
    清理过期的监控数据
    
    Args:
        context: Airflow任务上下文
    """
    import logging
    import glob
    from datetime import datetime, timedelta
    
    logger = logging.getLogger(__name__)
    
    try:
        # 获取清理配置
        retention_days = int(Variable.get('monitoring_data_retention_days', '30'))
        reports_path = Variable.get('ml_monitoring_reports_path', '/opt/airflow/data/monitoring/reports')
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # 清理过期报告
        deleted_count = 0
        for root, dirs, files in os.walk(reports_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff_date:
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"删除文件失败 {file_path}: {str(e)}")
        
        # 清理空目录
        for root, dirs, files in os.walk(reports_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # 空目录
                        os.rmdir(dir_path)
                except Exception as e:
                    logger.warning(f"删除空目录失败 {dir_path}: {str(e)}")
        
        logger.info(f"清理完成，删除了 {deleted_count} 个过期文件")
        
        # 将结果推送到XCom
        context['task_instance'].xcom_push(
            key='cleanup_result',
            value={
                'deleted_files': deleted_count,
                'retention_days': retention_days,
                'cleanup_timestamp': datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"清理监控数据失败: {str(e)}")
        raise


def generate_monitoring_summary(**context) -> Dict[str, Any]:
    """
    生成监控摘要
    
    Args:
        context: Airflow任务上下文
        
    Returns:
        Dict: 监控摘要
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        task_instance = context['task_instance']
        
        # 获取各个监控任务的结果
        health_check_result = task_instance.xcom_pull(
            task_ids='system_health_check',
            key='health_check_result'
        )
        
        quality_monitoring_result = task_instance.xcom_pull(
            task_ids='data_quality_monitoring',
            key='monitoring_result'
        )
        
        cleanup_result = task_instance.xcom_pull(
            task_ids='cleanup_old_data',
            key='cleanup_result'
        )
        
        # 生成摘要
        summary = {
            'monitoring_timestamp': datetime.now().isoformat(),
            'execution_date': context['execution_date'].isoformat(),
            'system_health': health_check_result.get('overall_health', 'unknown') if health_check_result else 'unknown',
            'quality_score': quality_monitoring_result.get('quality_assessment', {}).get('overall_score', 0) if quality_monitoring_result else 0,
            'alert_count': len(quality_monitoring_result.get('alerts', [])) if quality_monitoring_result else 0,
            'high_priority_alerts': len([
                a for a in quality_monitoring_result.get('alerts', []) 
                if a.get('severity') == 'high'
            ]) if quality_monitoring_result else 0,
            'cleanup_files_deleted': cleanup_result.get('deleted_files', 0) if cleanup_result else 0,
            'components_checked': len(health_check_result.get('component_status', {})) if health_check_result else 0,
            'healthy_components': len([
                status for status in health_check_result.get('component_status', {}).values()
                if status.get('status') == 'healthy'
            ]) if health_check_result else 0
        }
        
        # 计算整体状态
        if summary['system_health'] == 'healthy' and summary['quality_score'] >= 85 and summary['high_priority_alerts'] == 0:
            summary['overall_status'] = 'excellent'
        elif summary['system_health'] in ['healthy', 'degraded'] and summary['quality_score'] >= 70:
            summary['overall_status'] = 'good'
        elif summary['high_priority_alerts'] == 0:
            summary['overall_status'] = 'acceptable'
        else:
            summary['overall_status'] = 'needs_attention'
        
        logger.info(f"监控摘要生成完成: {summary['overall_status']}")
        
        # 将摘要推送到XCom
        context['task_instance'].xcom_push(key='monitoring_summary', value=summary)
        
        return summary
        
    except Exception as e:
        logger.error(f"生成监控摘要失败: {str(e)}")
        raise


# 任务定义

# 开始任务
start_monitoring = DummyOperator(
    task_id='start_monitoring',
    dag=dag
)

# 检查监控前置条件
check_prerequisites = PythonOperator(
    task_id='check_monitoring_prerequisites',
    python_callable=check_monitoring_prerequisites,
    dag=dag,
    doc_md="""
    检查监控系统的前置条件：
    - 验证监控报告存储路径
    - 检查配置文件完整性
    - 确认必要的Airflow变量和连接
    """
)

# 系统健康检查任务组
with TaskGroup('health_checks', dag=dag) as health_checks_group:
    
    # 系统健康检查
    system_health_check = HealthCheckOperator(
        task_id='system_health_check',
        check_components=['database', 'storage', 'api_connections', 'airflow_services'],
        dag=dag,
        doc_md="检查系统各组件的健康状态，包括数据库、存储、API连接和Airflow服务"
    )
    
    # 数据路径检查
    check_data_paths = BashOperator(
        task_id='check_data_paths',
        bash_command="""
        echo "检查数据路径可访问性..."
        
        # 检查主要数据目录
        for path in "{{ var.value.ml_raw_data_path }}" "{{ var.value.ml_processed_data_path }}" "{{ var.value.ml_features_data_path }}"; do
            if [ -d "$path" ]; then
                echo "✓ 路径可访问: $path"
                # 检查磁盘空间
                df -h "$path"
            else
                echo "✗ 路径不存在: $path"
                exit 1
            fi
        done
        
        echo "数据路径检查完成"
        """,
        dag=dag,
        doc_md="检查所有数据存储路径的可访问性和磁盘空间"
    )

# 数据质量监控任务组
with TaskGroup('quality_monitoring', dag=dag) as quality_monitoring_group:
    
    # 数据质量监控
    data_quality_monitoring = DataQualityMonitorOperator(
        task_id='data_quality_monitoring',
        monitoring_config={
            'check_pipeline_status': True,
            'check_data_quality': True,
            'check_performance': True,
            'generate_alerts': True
        },
        quality_thresholds={
            'data_completeness': 95.0,
            'data_accuracy': 90.0,
            'data_consistency': 85.0,
            'pipeline_success_rate': 95.0,
            'processing_time_threshold': 3600
        },
        alert_channels=['email', 'log', 'slack'],
        report_format='json',
        dag=dag,
        doc_md="执行全面的数据质量监控，包括流水线状态、数据质量指标和性能监控"
    )
    
    # 生成HTML报告
    generate_html_report = BashOperator(
        task_id='generate_html_report',
        bash_command="""
        echo "生成HTML监控报告..."
        
        # 获取JSON报告路径
        REPORTS_PATH="{{ var.value.ml_monitoring_reports_path }}"
        EXECUTION_DATE="{{ ds }}"
        
        # 查找最新的JSON报告
        JSON_REPORT=$(find "$REPORTS_PATH" -name "*quality_monitoring_${EXECUTION_DATE//-/_}*.json" | head -1)
        
        if [ -f "$JSON_REPORT" ]; then
            echo "找到JSON报告: $JSON_REPORT"
            
            # 生成HTML报告路径
            HTML_REPORT="${JSON_REPORT%.json}.html"
            
            # 这里可以添加JSON到HTML的转换逻辑
            echo "HTML报告将生成到: $HTML_REPORT"
        else
            echo "未找到JSON报告文件"
        fi
        """,
        dag=dag,
        doc_md="将JSON格式的监控报告转换为HTML格式，便于查看和分享"
    )

# 告警处理任务组
with TaskGroup('alert_processing', dag=dag) as alert_processing_group:
    
    # 检查告警状态
    check_alert_status = PythonOperator(
        task_id='check_alert_status',
        python_callable=lambda **context: context['task_instance'].xcom_pull(
            task_ids='quality_monitoring.data_quality_monitoring',
            key='monitoring_result'
        ).get('alerts', []),
        dag=dag,
        doc_md="检查数据质量监控产生的告警状态"
    )
    
    # 发送摘要邮件
    send_summary_email = EmailOperator(
        task_id='send_summary_email',
        to="{{ var.value.alert_emails }}".split(','),
        subject="数据质量监控摘要 - {{ ds }}",
        html_content="""
        <h2>数据质量监控摘要</h2>
        <p><strong>执行日期:</strong> {{ ds }}</p>
        <p><strong>监控时间:</strong> {{ ts }}</p>
        
        <h3>监控结果</h3>
        <ul>
            <li>系统健康状态: {{ ti.xcom_pull(task_ids='health_checks.system_health_check', key='health_check_result').get('overall_health', 'unknown') }}</li>
            <li>数据质量分数: {{ ti.xcom_pull(task_ids='quality_monitoring.data_quality_monitoring', key='monitoring_result').get('quality_assessment', {}).get('overall_score', 0) }}</li>
            <li>告警数量: {{ ti.xcom_pull(task_ids='quality_monitoring.data_quality_monitoring', key='monitoring_result').get('alerts', [])|length }}</li>
        </ul>
        
        <p>详细报告请查看监控系统或联系数据团队。</p>
        """,
        dag=dag,
        trigger_rule='none_failed',
        doc_md="发送监控摘要邮件给相关人员"
    )

# 维护任务组
with TaskGroup('maintenance', dag=dag) as maintenance_group:
    
    # 清理过期数据
    cleanup_old_data = PythonOperator(
        task_id='cleanup_old_data',
        python_callable=cleanup_old_monitoring_data,
        dag=dag,
        doc_md="清理过期的监控报告和数据文件"
    )
    
    # 更新监控统计
    update_monitoring_stats = BashOperator(
        task_id='update_monitoring_stats',
        bash_command="""
        echo "更新监控统计信息..."
        
        # 统计监控报告数量
        REPORTS_PATH="{{ var.value.ml_monitoring_reports_path }}"
        REPORT_COUNT=$(find "$REPORTS_PATH" -name "*.json" | wc -l)
        
        echo "当前监控报告数量: $REPORT_COUNT"
        
        # 可以在这里添加更多统计逻辑
        # 例如：计算平均质量分数、告警趋势等
        
        echo "监控统计更新完成"
        """,
        dag=dag,
        doc_md="更新监控系统的统计信息和元数据"
    )

# 生成监控摘要
generate_summary = PythonOperator(
    task_id='generate_monitoring_summary',
    python_callable=generate_monitoring_summary,
    dag=dag,
    doc_md="生成本次监控执行的综合摘要报告"
)

# 完成任务
complete_monitoring = DummyOperator(
    task_id='complete_monitoring',
    dag=dag,
    trigger_rule='none_failed'
)

# 任务依赖关系
start_monitoring >> check_prerequisites

check_prerequisites >> health_checks_group
check_prerequisites >> quality_monitoring_group

health_checks_group >> alert_processing_group
quality_monitoring_group >> alert_processing_group

alert_processing_group >> maintenance_group
maintenance_group >> generate_summary

generate_summary >> complete_monitoring

# 设置任务组内部依赖
# 健康检查组
health_checks_group.system_health_check >> health_checks_group.check_data_paths

# 质量监控组
quality_monitoring_group.data_quality_monitoring >> quality_monitoring_group.generate_html_report

# 告警处理组
alert_processing_group.check_alert_status >> alert_processing_group.send_summary_email

# 维护组
maintenance_group.cleanup_old_data >> maintenance_group.update_monitoring_stats