"""
特征工程DAG
集成FeatureEngineer模块，对清洗后的数据进行特征计算
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.sensors.external_task import ExternalTaskSensor

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from operators.feature_engineering_operator import (
    FeatureEngineeringOperator,
    MultiSymbolFeatureEngineeringOperator
)
from utils.task_helpers import (
    load_dag_config,
    get_dag_default_args,
    validate_symbols_and_intervals,
    send_notification
)

# 加载配置
config = load_dag_config()
dag_config = config.get('feature_engineering_dag', {})

# DAG配置
DAG_ID = 'feature_engineering_dag'
DESCRIPTION = """
特征工程DAG

主要功能：
1. 对清洗后的数据计算技术指标特征
2. 生成移动平均、波动率、成交量等特征
3. 验证特征质量和完整性
4. 保存特征数据和元数据
5. 生成特征工程报告

依赖关系：
- 依赖于data_cleaning_dag的输出数据
- 为机器学习模型训练提供特征数据
"""

# 默认参数
default_args = get_dag_default_args()
default_args.update({
    'depends_on_past': True,
    'wait_for_downstream': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': dag_config.get('retries', 2),
    'retry_delay': timedelta(minutes=dag_config.get('retry_delay_minutes', 5)),
    'sla': timedelta(hours=dag_config.get('sla_hours', 3))
})

# 交易对和时间间隔配置
SYMBOLS = Variable.get('ml_symbols', 'BTCUSDT,ETHUSDT,ADAUSDT').split(',')
INTERVALS = Variable.get('ml_intervals', '1h,4h,1d').split(',')

# 特征工程配置
FEATURE_CONFIG = {
    'ma_periods': [5, 10, 20, 50, 100, 200],  # 移动平均周期
    'volatility_periods': [10, 20, 30],       # 波动率计算周期
    'volume_periods': [10, 20],               # 成交量指标周期
    'price_change_periods': [1, 5, 10],       # 价格变化周期
    'rsi_period': 14,                         # RSI周期
    'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},  # MACD参数
    'bollinger_period': 20,                   # 布林带周期
    'bollinger_std': 2                        # 布林带标准差倍数
}

# 指标列表（可选择性计算）
INDICATORS = Variable.get('ml_indicators', '').split(',') if Variable.get('ml_indicators', '') else None

# 创建DAG
dag = DAG(
    dag_id=DAG_ID,
    description=DESCRIPTION,
    default_args=default_args,
    schedule_interval=dag_config.get('schedule_interval', '0 */8 * * *'),  # 每8小时执行一次
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['feature-engineering', 'ml-pipeline', 'technical-indicators'],
    doc_md=DESCRIPTION
)


def check_upstream_cleaning_dag(**context):
    """
    检查上游数据清洗DAG的状态
    """
    from airflow.models import DagRun
    from airflow.utils.state import State
    
    execution_date = context['execution_date']
    
    # 检查data_cleaning_dag是否成功完成
    upstream_dag_id = 'data_cleaning_dag'
    upstream_dag_runs = DagRun.find(
        dag_id=upstream_dag_id,
        execution_date=execution_date,
        state=State.SUCCESS
    )
    
    if not upstream_dag_runs:
        raise Exception(f"上游DAG {upstream_dag_id} 在 {execution_date} 未成功完成")
    
    print(f"上游DAG {upstream_dag_id} 检查通过")
    return True


def validate_cleaned_data_paths(**context):
    """
    验证清洗后数据路径和文件存在性
    """
    import os
    from airflow.models import Variable
    
    processed_data_path = Variable.get('ml_processed_data_path', '/opt/airflow/data/processed')
    execution_date = context['execution_date']
    
    missing_files = []
    
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            date_str = execution_date.strftime('%Y-%m-%d')
            file_path = os.path.join(
                processed_data_path,
                symbol.lower(),
                interval,
                execution_date.strftime('%Y'),
                execution_date.strftime('%m'),
                f"{symbol.lower()}_{interval}_{date_str}_cleaned.csv"
            )
            
            if not os.path.exists(file_path):
                missing_files.append(file_path)
    
    if missing_files:
        raise Exception(f"以下清洗后数据文件不存在: {missing_files}")
    
    print(f"清洗后数据路径验证通过，共检查 {len(SYMBOLS) * len(INTERVALS)} 个文件")
    return True


def prepare_feature_directories(**context):
    """
    准备特征数据目录
    """
    import os
    from airflow.models import Variable
    
    features_data_path = Variable.get('ml_features_data_path', '/opt/airflow/data/features')
    execution_date = context['execution_date']
    
    created_dirs = []
    
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            dir_path = os.path.join(
                features_data_path,
                symbol.lower(),
                interval,
                execution_date.strftime('%Y'),
                execution_date.strftime('%m')
            )
            
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                created_dirs.append(dir_path)
    
    print(f"特征数据目录准备完成，创建了 {len(created_dirs)} 个目录")
    return created_dirs


# 开始任务
start_task = DummyOperator(
    task_id='start',
    dag=dag
)

# 等待上游数据清洗DAG完成
wait_for_cleaning = ExternalTaskSensor(
    task_id='wait_for_data_cleaning',
    external_dag_id='data_cleaning_dag',
    external_task_id='end',
    timeout=3600,  # 1小时超时
    poke_interval=300,  # 每5分钟检查一次
    mode='poke',
    dag=dag
)

# 检查上游清洗DAG
check_upstream_task = PythonOperator(
    task_id='check_upstream_cleaning_dag',
    python_callable=check_upstream_cleaning_dag,
    dag=dag
)

# 验证清洗后数据路径
validate_paths_task = PythonOperator(
    task_id='validate_cleaned_data_paths',
    python_callable=validate_cleaned_data_paths,
    dag=dag
)

# 准备特征目录
prepare_dirs_task = PythonOperator(
    task_id='prepare_feature_directories',
    python_callable=prepare_feature_directories,
    dag=dag
)

# 特征工程任务组
with TaskGroup('compute_features', dag=dag) as compute_features_group:
    
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            task_id = f'{symbol.lower()}_{interval}_features'
            
            # 特征工程任务
            feature_task = FeatureEngineeringOperator(
                task_id=task_id,
                symbol=symbol,
                interval=interval,
                feature_config=FEATURE_CONFIG,
                indicators=INDICATORS,
                file_format='csv',
                validate_features=True,
                save_metadata=True,
                dag=dag
            )


def validate_all_features(**context):
    """
    验证所有特征数据
    """
    import os
    import pandas as pd
    from airflow.models import Variable
    
    features_data_path = Variable.get('ml_features_data_path', '/opt/airflow/data/features')
    execution_date = context['execution_date']
    
    validation_results = {}
    
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            date_str = execution_date.strftime('%Y-%m-%d')
            file_path = os.path.join(
                features_data_path,
                symbol.lower(),
                interval,
                execution_date.strftime('%Y'),
                execution_date.strftime('%m'),
                f"{symbol.lower()}_{interval}_{date_str}_features.csv"
            )
            
            key = f"{symbol}_{interval}"
            
            try:
                if not os.path.exists(file_path):
                    validation_results[key] = {'status': 'missing', 'error': 'File not found'}
                    continue
                
                # 加载并验证特征数据
                data = pd.read_csv(file_path)
                
                # 基本验证
                if data.empty:
                    validation_results[key] = {'status': 'error', 'error': 'Empty data'}
                    continue
                
                # 检查必需的原始列
                required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    validation_results[key] = {
                        'status': 'error', 
                        'error': f'Missing required columns: {missing_columns}'
                    }
                    continue
                
                # 检查特征列
                feature_columns = [col for col in data.columns if col not in required_columns]
                if not feature_columns:
                    validation_results[key] = {'status': 'warning', 'error': 'No feature columns found'}
                    continue
                
                # 检查数据质量
                null_count = data.isnull().sum().sum()
                inf_count = data.select_dtypes(include=['number']).apply(lambda x: x.isin([float('inf'), float('-inf')])).sum().sum()
                
                validation_results[key] = {
                    'status': 'success',
                    'records': len(data),
                    'feature_count': len(feature_columns),
                    'null_values': null_count,
                    'inf_values': inf_count,
                    'file_size_mb': round(os.path.getsize(file_path) / 1024 / 1024, 2)
                }
                
            except Exception as e:
                validation_results[key] = {'status': 'error', 'error': str(e)}
    
    # 统计验证结果
    success_count = sum(1 for r in validation_results.values() if r['status'] == 'success')
    warning_count = sum(1 for r in validation_results.values() if r['status'] == 'warning')
    error_count = sum(1 for r in validation_results.values() if r['status'] == 'error')
    
    print(f"特征验证完成: 成功 {success_count}, 警告 {warning_count}, 错误 {error_count}")
    
    # 将结果推送到XCom
    context['task_instance'].xcom_push(key='validation_results', value=validation_results)
    
    return validation_results


# 特征验证任务
validate_features_task = PythonOperator(
    task_id='validate_all_features',
    python_callable=validate_all_features,
    dag=dag
)


def generate_feature_summary_report(**context):
    """
    生成特征工程汇总报告
    """
    task_instance = context['task_instance']
    
    # 收集所有特征工程结果
    feature_results = {}
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            task_id = f'compute_features.{symbol.lower()}_{interval}_features'
            try:
                result = task_instance.xcom_pull(task_ids=task_id, key='feature_result')
                if result:
                    feature_results[f"{symbol}_{interval}"] = result
            except Exception as e:
                print(f"无法获取特征结果 {task_id}: {str(e)}")
    
    # 获取验证结果
    validation_results = task_instance.xcom_pull(
        task_ids='validate_all_features',
        key='validation_results'
    )
    
    # 生成汇总报告
    execution_date = context['execution_date']
    
    summary_report = {
        'execution_date': execution_date.isoformat(),
        'generation_timestamp': datetime.now().isoformat(),
        'summary': {
            'total_datasets': len(SYMBOLS) * len(INTERVALS),
            'processed_datasets': len(feature_results),
            'validated_datasets': len(validation_results) if validation_results else 0,
            'success_rate': len(feature_results) / (len(SYMBOLS) * len(INTERVALS)) * 100,
            'total_features_generated': 0,
            'total_records_processed': 0,
            'average_features_per_dataset': 0
        },
        'feature_results': feature_results,
        'validation_results': validation_results,
        'feature_config': FEATURE_CONFIG,
        'indicators_used': INDICATORS,
        'symbols': SYMBOLS,
        'intervals': INTERVALS
    }
    
    # 计算统计信息
    if feature_results:
        feature_counts = [r.get('feature_count', 0) for r in feature_results.values()]
        record_counts = [r.get('feature_records', 0) for r in feature_results.values()]
        
        summary_report['summary']['total_features_generated'] = sum(feature_counts)
        summary_report['summary']['total_records_processed'] = sum(record_counts)
        summary_report['summary']['average_features_per_dataset'] = sum(feature_counts) / len(feature_counts)
    
    # 保存报告
    features_data_path = Variable.get('ml_features_data_path', '/opt/airflow/data/features')
    report_file = os.path.join(
        features_data_path,
        'reports',
        f'feature_summary_{execution_date.strftime("%Y-%m-%d")}.json'
    )
    
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    import json
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    print(f"特征工程汇总报告已保存到: {report_file}")
    
    # 将结果推送到XCom
    context['task_instance'].xcom_push(key='summary_report', value=summary_report)
    
    return summary_report


# 汇总报告任务
summary_report_task = PythonOperator(
    task_id='generate_feature_summary_report',
    python_callable=generate_feature_summary_report,
    dag=dag
)


def send_feature_notification(**context):
    """
    发送特征工程完成通知
    """
    task_instance = context['task_instance']
    
    # 获取汇总报告
    summary_report = task_instance.xcom_pull(
        task_ids='generate_feature_summary_report',
        key='summary_report'
    )
    
    if not summary_report:
        print("未找到汇总报告")
        return
    
    # 构造通知消息
    execution_date = context['execution_date']
    summary = summary_report['summary']
    
    subject = f"特征工程完成 - {execution_date.strftime('%Y-%m-%d')}"
    
    message = f"""
特征工程处理完成：

处理概况:
• 数据集: {summary['processed_datasets']}/{summary['total_datasets']} (成功率: {summary['success_rate']:.1f}%)
• 总特征数: {summary['total_features_generated']}
• 总记录数: {summary['total_records_processed']}
• 平均特征数/数据集: {summary['average_features_per_dataset']:.1f}

处理的交易对: {', '.join(SYMBOLS)}
时间间隔: {', '.join(INTERVALS)}

特征配置:
• 移动平均周期: {FEATURE_CONFIG['ma_periods']}
• 波动率周期: {FEATURE_CONFIG['volatility_periods']}
• 成交量周期: {FEATURE_CONFIG['volume_periods']}
• RSI周期: {FEATURE_CONFIG['rsi_period']}

数据已准备就绪，可进行机器学习模型训练。
"""
    
    # 检查是否有失败的任务
    failed_count = summary['total_datasets'] - summary['processed_datasets']
    if failed_count > 0:
        message += f"\n注意: {failed_count} 个数据集处理失败，请检查日志。"
        notification_type = 'warning'
    else:
        notification_type = 'success'
    
    # 发送通知
    send_notification(
        subject=subject,
        message=message,
        notification_type=notification_type,
        context=context
    )


# 通知任务
notification_task = PythonOperator(
    task_id='send_feature_notification',
    python_callable=send_feature_notification,
    dag=dag
)

# 清理任务
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command="""
    # 清理超过7天的临时特征文件
    find {{ var.value.ml_temp_path | default('/tmp/ml_processing') }} -name "*_features_temp*" -mtime +7 -delete || true
    
    # 清理特征处理日志
    find {{ var.value.ml_log_path | default('/opt/airflow/logs') }}/feature_engineering -name "*.log" -mtime +30 -delete || true
    
    # 压缩旧的特征报告
    find {{ var.value.ml_features_data_path | default('/opt/airflow/data/features') }}/reports -name "*.json" -mtime +30 -exec gzip {} \; || true
    
    echo "特征工程临时文件清理完成"
    """,
    dag=dag
)

# 结束任务
end_task = DummyOperator(
    task_id='end',
    dag=dag
)

# 设置任务依赖关系
start_task >> wait_for_cleaning >> check_upstream_task >> validate_paths_task >> prepare_dirs_task

# 特征计算依赖目录准备
prepare_dirs_task >> compute_features_group

# 验证依赖特征计算
compute_features_group >> validate_features_task

# 报告依赖验证
validate_features_task >> summary_report_task

# 通知依赖报告
summary_report_task >> notification_task

# 清理和结束
notification_task >> cleanup_task >> end_task

# DAG文档
dag.doc_md = """
# 特征工程DAG

## 概述
此DAG负责对清洗后的交易数据进行特征工程处理，计算各种技术指标和衍生特征，为机器学习模型提供输入数据。

## 主要功能

### 1. 技术指标计算
- **移动平均**: SMA, EMA (多个周期)
- **波动率指标**: 标准差, ATR, 波动率
- **成交量指标**: 成交量移动平均, 成交量比率
- **价格指标**: 价格变化, 收益率, 价格比率
- **动量指标**: RSI, MACD, 动量
- **趋势指标**: 布林带, 趋势方向

### 2. 特征验证
- 数据完整性检查
- 特征质量验证
- 异常值检测
- 统计信息计算

### 3. 元数据管理
- 特征描述信息
- 计算参数记录
- 数据血缘追踪
- 版本控制

### 4. 质量监控
- 特征生成成功率
- 数据质量评分
- 异常告警
- 性能监控

## 配置参数

### 环境变量
- `ml_symbols`: 交易对列表
- `ml_intervals`: 时间间隔列表
- `ml_indicators`: 指定计算的指标列表
- `ml_features_data_path`: 特征数据存储路径

### 特征配置
```yaml
feature_config:
  ma_periods: [5, 10, 20, 50, 100, 200]
  volatility_periods: [10, 20, 30]
  volume_periods: [10, 20]
  price_change_periods: [1, 5, 10]
  rsi_period: 14
  macd_params: {fast: 12, slow: 26, signal: 9}
  bollinger_period: 20
  bollinger_std: 2
```

## 依赖关系
- **上游**: data_cleaning_dag (数据清洗)
- **下游**: ML模型训练DAG

## 输出数据

### 特征数据文件
- 路径: `{features_data_path}/{symbol}/{interval}/{year}/{month}/{symbol}_{interval}_{date}_features.csv`
- 格式: CSV/Parquet
- 内容: 原始数据 + 计算的特征

### 元数据文件
- 路径: `{features_data_path}/{symbol}/{interval}/{year}/{month}/{symbol}_{interval}_{date}_features_metadata.json`
- 内容: 特征描述、统计信息、计算参数

### 汇总报告
- 路径: `{features_data_path}/reports/feature_summary_{date}.json`
- 内容: 处理概况、质量统计、配置信息

## 监控指标
- 特征生成成功率
- 平均特征数量
- 处理时间
- 数据质量分数
- 存储空间使用

## 故障排除

### 常见问题
1. **上游数据不存在**: 检查data_cleaning_dag执行状态
2. **特征计算失败**: 检查数据格式和计算参数
3. **内存不足**: 调整批处理大小或增加资源
4. **存储空间不足**: 清理旧数据或扩展存储

### 恢复步骤
1. 检查上游DAG状态
2. 验证清洗后数据文件
3. 检查特征配置参数
4. 重新运行失败任务
5. 监控资源使用情况

## 性能优化
- 并行处理多个交易对
- 批量计算相似指标
- 内存优化和垃圾回收
- 增量特征更新
"""