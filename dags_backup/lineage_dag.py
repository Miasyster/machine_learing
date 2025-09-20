"""
数据血缘追踪和版本管理DAG

该DAG负责：
1. 追踪数据处理流程中的血缘关系
2. 管理数据版本控制
3. 生成血缘关系报告
4. 监控数据治理指标
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor
from airflow.operators.email import EmailOperator

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义操作符和工具
from operators.lineage_operator import (
    DataLineageOperator, 
    DataVersionOperator, 
    LineageReportOperator
)
from utils.task_helpers import (
    load_config,
    get_dag_config,
    validate_data_paths,
    send_notification
)

# DAG配置
DAG_ID = 'lineage_tracking_dag'
DAG_CONFIG = get_dag_config(DAG_ID)

# 默认参数
default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=2),
    'email': Variable.get('ml_notification_email', 'admin@company.com').split(',')
}

# 创建DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='数据血缘追踪和版本管理流水线',
    schedule_interval=timedelta(hours=6),  # 每6小时运行一次
    max_active_runs=1,
    catchup=False,
    tags=['ml', 'lineage', 'governance', 'versioning'],
    doc_md="""
    # 数据血缘追踪和版本管理DAG
    
    ## 功能概述
    
    该DAG实现了完整的数据血缘追踪和版本管理功能：
    
    ### 主要功能
    1. **数据血缘追踪**: 记录数据处理过程中的血缘关系
    2. **版本管理**: 管理数据的版本控制和历史记录
    3. **报告生成**: 生成血缘关系和版本管理报告
    4. **治理监控**: 监控数据治理指标和合规性
    
    ### 执行流程
    1. 初始化血缘数据库
    2. 收集现有数据资产信息
    3. 追踪数据血缘关系
    4. 管理数据版本
    5. 生成治理报告
    6. 发送通知和告警
    
    ### 监控指标
    - 数据资产数量
    - 血缘关系完整性
    - 版本管理覆盖率
    - 数据治理合规性
    
    ### 告警条件
    - 血缘关系缺失
    - 版本管理异常
    - 数据治理违规
    - 系统性能问题
    """
)


def check_lineage_prerequisites(**context) -> bool:
    """
    检查血缘追踪前置条件
    
    Returns:
        bool: 检查结果
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # 检查配置
        config = load_config()
        
        # 检查必要的路径
        required_paths = [
            config.get('data_raw_path'),
            config.get('data_processed_path'),
            config.get('data_features_path')
        ]
        
        missing_paths = []
        for path in required_paths:
            if path and not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            logger.warning(f"以下路径不存在: {missing_paths}")
        
        # 检查血缘数据库路径
        lineage_db_path = Variable.get('ml_lineage_db_path', '/opt/airflow/data/lineage/lineage.db')
        lineage_db_dir = os.path.dirname(lineage_db_path)
        
        if not os.path.exists(lineage_db_dir):
            os.makedirs(lineage_db_dir, exist_ok=True)
            logger.info(f"创建血缘数据库目录: {lineage_db_dir}")
        
        logger.info("血缘追踪前置条件检查完成")
        return True
        
    except Exception as e:
        logger.error(f"血缘追踪前置条件检查失败: {str(e)}")
        raise


def discover_data_assets(**context) -> Dict[str, List[Dict[str, Any]]]:
    """
    发现和注册数据资产
    
    Returns:
        Dict: 发现的数据资产信息
    """
    import logging
    import glob
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    try:
        config = load_config()
        
        # 定义数据路径和类型映射
        data_paths = {
            'raw_data': config.get('data_raw_path', '/opt/airflow/data/raw'),
            'processed_data': config.get('data_processed_path', '/opt/airflow/data/processed'),
            'features': config.get('data_features_path', '/opt/airflow/data/features'),
            'models': config.get('data_models_path', '/opt/airflow/data/models')
        }
        
        discovered_assets = {}
        
        for asset_type, base_path in data_paths.items():
            if not os.path.exists(base_path):
                logger.warning(f"数据路径不存在: {base_path}")
                discovered_assets[asset_type] = []
                continue
            
            assets = []
            
            # 搜索CSV文件
            csv_files = glob.glob(os.path.join(base_path, '**', '*.csv'), recursive=True)
            for csv_file in csv_files:
                rel_path = os.path.relpath(csv_file, base_path)
                asset_id = f"{asset_type}_{rel_path.replace(os.sep, '_').replace('.csv', '')}"
                
                assets.append({
                    'asset_id': asset_id,
                    'asset_name': os.path.basename(csv_file),
                    'asset_type': asset_type,
                    'asset_path': csv_file,
                    'file_format': 'csv',
                    'relative_path': rel_path
                })
            
            # 搜索Parquet文件
            parquet_files = glob.glob(os.path.join(base_path, '**', '*.parquet'), recursive=True)
            for parquet_file in parquet_files:
                rel_path = os.path.relpath(parquet_file, base_path)
                asset_id = f"{asset_type}_{rel_path.replace(os.sep, '_').replace('.parquet', '')}"
                
                assets.append({
                    'asset_id': asset_id,
                    'asset_name': os.path.basename(parquet_file),
                    'asset_type': asset_type,
                    'asset_path': parquet_file,
                    'file_format': 'parquet',
                    'relative_path': rel_path
                })
            
            # 搜索模型文件
            if asset_type == 'models':
                model_files = glob.glob(os.path.join(base_path, '**', '*.pkl'), recursive=True)
                for model_file in model_files:
                    rel_path = os.path.relpath(model_file, base_path)
                    asset_id = f"{asset_type}_{rel_path.replace(os.sep, '_').replace('.pkl', '')}"
                    
                    assets.append({
                        'asset_id': asset_id,
                        'asset_name': os.path.basename(model_file),
                        'asset_type': asset_type,
                        'asset_path': model_file,
                        'file_format': 'pickle',
                        'relative_path': rel_path
                    })
            
            discovered_assets[asset_type] = assets
            logger.info(f"发现 {len(assets)} 个 {asset_type} 资产")
        
        # 将结果推送到XCom
        total_assets = sum(len(assets) for assets in discovered_assets.values())
        logger.info(f"总共发现 {total_assets} 个数据资产")
        
        context['task_instance'].xcom_push(key='discovered_assets', value=discovered_assets)
        
        return discovered_assets
        
    except Exception as e:
        logger.error(f"发现数据资产失败: {str(e)}")
        raise


def analyze_lineage_relationships(**context) -> Dict[str, Any]:
    """
    分析数据血缘关系
    
    Returns:
        Dict: 血缘关系分析结果
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # 获取发现的资产
        discovered_assets = context['task_instance'].xcom_pull(
            task_ids='discover_data_assets',
            key='discovered_assets'
        )
        
        if not discovered_assets:
            logger.warning("未找到已发现的数据资产")
            return {}
        
        # 分析血缘关系
        lineage_relationships = []
        
        # 原始数据 -> 处理数据的关系
        raw_assets = discovered_assets.get('raw_data', [])
        processed_assets = discovered_assets.get('processed_data', [])
        
        for raw_asset in raw_assets:
            for processed_asset in processed_assets:
                # 基于文件名相似性推断关系
                if raw_asset['asset_name'].split('.')[0] in processed_asset['asset_name']:
                    lineage_relationships.append({
                        'source_asset': raw_asset,
                        'target_asset': processed_asset,
                        'relationship_type': 'transform',
                        'confidence': 0.8
                    })
        
        # 处理数据 -> 特征数据的关系
        features_assets = discovered_assets.get('features', [])
        
        for processed_asset in processed_assets:
            for feature_asset in features_assets:
                if processed_asset['asset_name'].split('.')[0] in feature_asset['asset_name']:
                    lineage_relationships.append({
                        'source_asset': processed_asset,
                        'target_asset': feature_asset,
                        'relationship_type': 'feature_engineering',
                        'confidence': 0.7
                    })
        
        # 特征数据 -> 模型的关系
        model_assets = discovered_assets.get('models', [])
        
        for feature_asset in features_assets:
            for model_asset in model_assets:
                if feature_asset['asset_name'].split('.')[0] in model_asset['asset_name']:
                    lineage_relationships.append({
                        'source_asset': feature_asset,
                        'target_asset': model_asset,
                        'relationship_type': 'training',
                        'confidence': 0.9
                    })
        
        result = {
            'total_relationships': len(lineage_relationships),
            'relationships': lineage_relationships,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"分析完成，发现 {len(lineage_relationships)} 个血缘关系")
        
        # 将结果推送到XCom
        context['task_instance'].xcom_push(key='lineage_analysis', value=result)
        
        return result
        
    except Exception as e:
        logger.error(f"分析血缘关系失败: {str(e)}")
        raise


def validate_governance_compliance(**context) -> Dict[str, Any]:
    """
    验证数据治理合规性
    
    Returns:
        Dict: 合规性验证结果
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # 获取血缘分析结果
        lineage_analysis = context['task_instance'].xcom_pull(
            task_ids='analyze_lineage_relationships',
            key='lineage_analysis'
        )
        
        # 获取发现的资产
        discovered_assets = context['task_instance'].xcom_pull(
            task_ids='discover_data_assets',
            key='discovered_assets'
        )
        
        compliance_issues = []
        compliance_score = 100.0
        
        # 检查血缘关系完整性
        total_assets = sum(len(assets) for assets in discovered_assets.values())
        total_relationships = lineage_analysis.get('total_relationships', 0)
        
        if total_assets > 0:
            relationship_coverage = total_relationships / total_assets
            if relationship_coverage < 0.5:
                compliance_issues.append({
                    'type': 'lineage_coverage',
                    'severity': 'medium',
                    'message': f'血缘关系覆盖率过低: {relationship_coverage:.2%}',
                    'recommendation': '增加血缘关系追踪覆盖范围'
                })
                compliance_score -= 20
        
        # 检查版本管理覆盖率
        versioned_assets = 0
        for asset_type, assets in discovered_assets.items():
            for asset in assets:
                # 检查是否有版本管理
                asset_path = asset.get('asset_path', '')
                if 'v' in os.path.basename(asset_path) or 'version' in asset_path.lower():
                    versioned_assets += 1
        
        if total_assets > 0:
            version_coverage = versioned_assets / total_assets
            if version_coverage < 0.3:
                compliance_issues.append({
                    'type': 'version_coverage',
                    'severity': 'high',
                    'message': f'版本管理覆盖率过低: {version_coverage:.2%}',
                    'recommendation': '为更多数据资产实施版本管理'
                })
                compliance_score -= 30
        
        # 检查数据质量文档
        quality_documented = 0
        for asset_type, assets in discovered_assets.items():
            for asset in assets:
                # 检查是否有质量文档
                asset_dir = os.path.dirname(asset.get('asset_path', ''))
                quality_files = [
                    os.path.join(asset_dir, 'quality_report.json'),
                    os.path.join(asset_dir, 'data_profile.json'),
                    os.path.join(asset_dir, 'README.md')
                ]
                
                if any(os.path.exists(f) for f in quality_files):
                    quality_documented += 1
        
        if total_assets > 0:
            quality_coverage = quality_documented / total_assets
            if quality_coverage < 0.4:
                compliance_issues.append({
                    'type': 'quality_documentation',
                    'severity': 'medium',
                    'message': f'数据质量文档覆盖率过低: {quality_coverage:.2%}',
                    'recommendation': '为数据资产添加质量文档'
                })
                compliance_score -= 15
        
        # 计算最终合规性等级
        if compliance_score >= 90:
            compliance_level = 'excellent'
        elif compliance_score >= 75:
            compliance_level = 'good'
        elif compliance_score >= 60:
            compliance_level = 'fair'
        else:
            compliance_level = 'poor'
        
        result = {
            'compliance_score': compliance_score,
            'compliance_level': compliance_level,
            'total_issues': len(compliance_issues),
            'issues': compliance_issues,
            'metrics': {
                'total_assets': total_assets,
                'total_relationships': total_relationships,
                'versioned_assets': versioned_assets,
                'quality_documented': quality_documented,
                'relationship_coverage': total_relationships / total_assets if total_assets > 0 else 0,
                'version_coverage': versioned_assets / total_assets if total_assets > 0 else 0,
                'quality_coverage': quality_documented / total_assets if total_assets > 0 else 0
            },
            'validation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"治理合规性验证完成，合规分数: {compliance_score:.1f}")
        
        # 将结果推送到XCom
        context['task_instance'].xcom_push(key='compliance_result', value=result)
        
        return result
        
    except Exception as e:
        logger.error(f"验证治理合规性失败: {str(e)}")
        raise


def cleanup_lineage_data(**context) -> None:
    """
    清理过期的血缘数据
    """
    import logging
    import sqlite3
    logger = logging.getLogger(__name__)
    
    try:
        lineage_db_path = Variable.get('ml_lineage_db_path', '/opt/airflow/data/lineage/lineage.db')
        
        if not os.path.exists(lineage_db_path):
            logger.info("血缘数据库不存在，跳过清理")
            return
        
        # 清理30天前的质量历史记录
        cutoff_date = datetime.now() - timedelta(days=30)
        
        with sqlite3.connect(lineage_db_path) as conn:
            cursor = conn.cursor()
            
            # 清理过期的质量历史
            cursor.execute("""
                DELETE FROM quality_history 
                WHERE checked_at < ?
            """, (cutoff_date,))
            
            deleted_quality = cursor.rowcount
            
            # 清理过期的血缘关系（保留最近的关系）
            cursor.execute("""
                DELETE FROM lineage_relationships 
                WHERE created_at < ? 
                AND id NOT IN (
                    SELECT MAX(id) 
                    FROM lineage_relationships 
                    GROUP BY source_asset_id, target_asset_id, relationship_type
                )
            """, (cutoff_date,))
            
            deleted_lineage = cursor.rowcount
            
            # 清理孤立的数据版本（保留最新的5个版本）
            cursor.execute("""
                DELETE FROM data_versions 
                WHERE id NOT IN (
                    SELECT id FROM (
                        SELECT id, ROW_NUMBER() OVER (
                            PARTITION BY asset_id ORDER BY created_at DESC
                        ) as rn
                        FROM data_versions
                    ) ranked
                    WHERE rn <= 5
                )
            """)
            
            deleted_versions = cursor.rowcount
            
            conn.commit()
            
            logger.info(f"清理完成 - 质量记录: {deleted_quality}, 血缘关系: {deleted_lineage}, 版本: {deleted_versions}")
        
        # 压缩数据库
        with sqlite3.connect(lineage_db_path) as conn:
            conn.execute("VACUUM")
            logger.info("数据库压缩完成")
        
    except Exception as e:
        logger.error(f"清理血缘数据失败: {str(e)}")
        raise


# 任务定义

# 1. 前置条件检查
check_prerequisites = PythonOperator(
    task_id='check_lineage_prerequisites',
    python_callable=check_lineage_prerequisites,
    dag=dag,
    doc_md="检查血缘追踪前置条件，包括路径、权限和配置"
)

# 2. 数据资产发现
discover_assets = PythonOperator(
    task_id='discover_data_assets',
    python_callable=discover_data_assets,
    dag=dag,
    doc_md="自动发现和注册数据资产"
)

# 3. 血缘关系分析
analyze_lineage = PythonOperator(
    task_id='analyze_lineage_relationships',
    python_callable=analyze_lineage_relationships,
    dag=dag,
    doc_md="分析数据处理流程中的血缘关系"
)

# 4. 血缘追踪任务组
with TaskGroup('lineage_tracking', dag=dag) as lineage_tracking_group:
    
    # 记录原始数据血缘
    track_raw_data_lineage = DataLineageOperator(
        task_id='track_raw_data_lineage',
        source_assets="{{ task_instance.xcom_pull(task_ids='discover_data_assets', key='discovered_assets')['raw_data'] or [] }}",
        target_assets="{{ task_instance.xcom_pull(task_ids='discover_data_assets', key='discovered_assets')['processed_data'] or [] }}",
        relationship_type='data_ingestion',
        transformation_info={
            'process': 'data_ingestion',
            'description': '原始数据摄取和初步处理'
        },
        dag=dag
    )
    
    # 记录特征工程血缘
    track_feature_lineage = DataLineageOperator(
        task_id='track_feature_lineage',
        source_assets="{{ task_instance.xcom_pull(task_ids='discover_data_assets', key='discovered_assets')['processed_data'] or [] }}",
        target_assets="{{ task_instance.xcom_pull(task_ids='discover_data_assets', key='discovered_assets')['features'] or [] }}",
        relationship_type='feature_engineering',
        transformation_info={
            'process': 'feature_engineering',
            'description': '特征工程和数据转换'
        },
        dag=dag
    )
    
    # 记录模型训练血缘
    track_model_lineage = DataLineageOperator(
        task_id='track_model_lineage',
        source_assets="{{ task_instance.xcom_pull(task_ids='discover_data_assets', key='discovered_assets')['features'] or [] }}",
        target_assets="{{ task_instance.xcom_pull(task_ids='discover_data_assets', key='discovered_assets')['models'] or [] }}",
        relationship_type='model_training',
        transformation_info={
            'process': 'model_training',
            'description': '机器学习模型训练'
        },
        dag=dag
    )

# 5. 版本管理任务组
with TaskGroup('version_management', dag=dag) as version_management_group:
    
    # 为处理数据创建版本
    version_processed_data = DataVersionOperator(
        task_id='version_processed_data',
        asset_id='processed_data_{{ ds }}',
        data_path="{{ var.value.ml_data_processed_path }}/{{ ds }}/",
        auto_version=True,
        version_metadata={
            'data_type': 'processed',
            'processing_date': '{{ ds }}',
            'dag_run_id': '{{ dag_run.run_id }}'
        },
        dag=dag
    )
    
    # 为特征数据创建版本
    version_features_data = DataVersionOperator(
        task_id='version_features_data',
        asset_id='features_data_{{ ds }}',
        data_path="{{ var.value.ml_data_features_path }}/{{ ds }}/",
        auto_version=True,
        version_metadata={
            'data_type': 'features',
            'processing_date': '{{ ds }}',
            'dag_run_id': '{{ dag_run.run_id }}'
        },
        dag=dag
    )
    
    # 为模型创建版本
    version_models = DataVersionOperator(
        task_id='version_models',
        asset_id='models_{{ ds }}',
        data_path="{{ var.value.ml_data_models_path }}/{{ ds }}/",
        auto_version=True,
        version_metadata={
            'data_type': 'models',
            'processing_date': '{{ ds }}',
            'dag_run_id': '{{ dag_run.run_id }}'
        },
        dag=dag
    )

# 6. 治理合规性验证
validate_compliance = PythonOperator(
    task_id='validate_governance_compliance',
    python_callable=validate_governance_compliance,
    dag=dag,
    doc_md="验证数据治理合规性和最佳实践"
)

# 7. 报告生成任务组
with TaskGroup('report_generation', dag=dag) as report_generation_group:
    
    # 生成血缘关系报告
    generate_lineage_report = LineageReportOperator(
        task_id='generate_lineage_report',
        report_type='full',
        report_format='json',
        report_config={
            'include_statistics': True,
            'include_graphs': True,
            'max_depth': 10
        },
        dag=dag
    )
    
    # 生成摘要报告
    generate_summary_report = LineageReportOperator(
        task_id='generate_summary_report',
        report_type='summary',
        report_format='html',
        report_config={
            'include_charts': True,
            'include_metrics': True
        },
        dag=dag
    )
    
    # 生成影响分析报告
    generate_impact_report = LineageReportOperator(
        task_id='generate_impact_report',
        report_type='impact_analysis',
        report_format='json',
        report_config={
            'analyze_dependencies': True,
            'include_risk_assessment': True
        },
        dag=dag
    )

# 8. 通知任务
send_lineage_notification = PythonOperator(
    task_id='send_lineage_notification',
    python_callable=send_notification,
    op_kwargs={
        'notification_type': 'lineage_tracking',
        'message_template': 'lineage_completion',
        'include_attachments': True
    },
    dag=dag,
    doc_md="发送血缘追踪完成通知"
)

# 9. 清理任务
cleanup_data = PythonOperator(
    task_id='cleanup_lineage_data',
    python_callable=cleanup_lineage_data,
    dag=dag,
    doc_md="清理过期的血缘数据和版本信息"
)

# 10. 完成标记
lineage_complete = DummyOperator(
    task_id='lineage_tracking_complete',
    dag=dag,
    doc_md="血缘追踪和版本管理流程完成标记"
)

# 定义任务依赖关系
check_prerequisites >> discover_assets >> analyze_lineage

# 血缘追踪依赖分析结果
analyze_lineage >> lineage_tracking_group

# 版本管理可以并行进行
analyze_lineage >> version_management_group

# 合规性验证依赖血缘追踪和版本管理
[lineage_tracking_group, version_management_group] >> validate_compliance

# 报告生成依赖合规性验证
validate_compliance >> report_generation_group

# 通知依赖报告生成
report_generation_group >> send_lineage_notification

# 清理任务在通知后执行
send_lineage_notification >> cleanup_data

# 最终完成标记
cleanup_data >> lineage_complete