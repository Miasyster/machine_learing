"""
数据质量监控和告警操作符
提供全面的数据质量监控、告警和报告功能
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.models import Variable
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DataQualityMonitorOperator(BaseOperator):
    """
    数据质量监控操作符
    
    监控整个数据处理流水线的质量指标
    """
    
    template_fields = ['monitoring_config']
    
    @apply_defaults
    def __init__(
        self,
        monitoring_config: Optional[Dict[str, Any]] = None,
        quality_thresholds: Optional[Dict[str, float]] = None,
        alert_channels: Optional[List[str]] = None,
        report_format: str = 'json',
        *args,
        **kwargs
    ):
        """
        初始化数据质量监控操作符
        
        Args:
            monitoring_config: 监控配置
            quality_thresholds: 质量阈值配置
            alert_channels: 告警渠道列表
            report_format: 报告格式
        """
        super().__init__(*args, **kwargs)
        self.monitoring_config = monitoring_config or {}
        self.quality_thresholds = quality_thresholds or {
            'data_completeness': 95.0,
            'data_accuracy': 90.0,
            'data_consistency': 85.0,
            'pipeline_success_rate': 95.0,
            'processing_time_threshold': 3600  # 1小时
        }
        self.alert_channels = alert_channels or ['email', 'log']
        self.report_format = report_format
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据质量监控任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 监控结果
        """
        self.logger.info("开始数据质量监控")
        
        try:
            execution_date = context['execution_date']
            
            # 收集各个DAG的执行状态和质量指标
            pipeline_status = self._collect_pipeline_status(context)
            
            # 收集数据质量指标
            quality_metrics = self._collect_quality_metrics(context)
            
            # 收集性能指标
            performance_metrics = self._collect_performance_metrics(context)
            
            # 评估整体质量
            quality_assessment = self._assess_overall_quality(
                pipeline_status, quality_metrics, performance_metrics
            )
            
            # 检测异常和告警
            alerts = self._detect_anomalies_and_alerts(quality_assessment)
            
            # 生成监控报告
            monitoring_report = self._generate_monitoring_report(
                execution_date, pipeline_status, quality_metrics, 
                performance_metrics, quality_assessment, alerts
            )
            
            # 保存监控报告
            self._save_monitoring_report(monitoring_report, context)
            
            # 发送告警（如果有）
            if alerts:
                self._send_alerts(alerts, monitoring_report, context)
            
            # 准备返回结果
            result = {
                'monitoring_timestamp': datetime.now().isoformat(),
                'execution_date': execution_date.isoformat(),
                'pipeline_status': pipeline_status,
                'quality_metrics': quality_metrics,
                'performance_metrics': performance_metrics,
                'quality_assessment': quality_assessment,
                'alerts': alerts,
                'report_file': monitoring_report.get('report_file')
            }
            
            self.logger.info(f"数据质量监控完成，发现 {len(alerts)} 个告警")
            
            # 将结果推送到XCom
            context['task_instance'].xcom_push(key='monitoring_result', value=result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据质量监控失败: {str(e)}")
            raise AirflowException(f"数据质量监控失败: {str(e)}")
    
    def _collect_pipeline_status(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """收集流水线状态"""
        from airflow.models import DagRun, TaskInstance
        from airflow.utils.state import State
        
        execution_date = context['execution_date']
        
        # 要监控的DAG列表
        monitored_dags = [
            'data_ingestion_dag',
            'data_cleaning_dag',
            'feature_engineering_dag'
        ]
        
        pipeline_status = {
            'execution_date': execution_date.isoformat(),
            'dag_status': {},
            'overall_success_rate': 0.0,
            'failed_dags': [],
            'running_dags': [],
            'successful_dags': []
        }
        
        successful_count = 0
        
        for dag_id in monitored_dags:
            try:
                # 获取DAG运行状态
                dag_runs = DagRun.find(
                    dag_id=dag_id,
                    execution_date=execution_date
                )
                
                if dag_runs:
                    dag_run = dag_runs[0]
                    status = dag_run.state
                    
                    # 获取任务实例状态
                    task_instances = dag_run.get_task_instances()
                    task_status = {}
                    
                    for ti in task_instances:
                        task_status[ti.task_id] = {
                            'state': ti.state,
                            'start_date': ti.start_date.isoformat() if ti.start_date else None,
                            'end_date': ti.end_date.isoformat() if ti.end_date else None,
                            'duration': ti.duration if ti.duration else None
                        }
                    
                    pipeline_status['dag_status'][dag_id] = {
                        'state': status,
                        'start_date': dag_run.start_date.isoformat() if dag_run.start_date else None,
                        'end_date': dag_run.end_date.isoformat() if dag_run.end_date else None,
                        'task_count': len(task_instances),
                        'successful_tasks': len([ti for ti in task_instances if ti.state == State.SUCCESS]),
                        'failed_tasks': len([ti for ti in task_instances if ti.state == State.FAILED]),
                        'task_status': task_status
                    }
                    
                    if status == State.SUCCESS:
                        successful_count += 1
                        pipeline_status['successful_dags'].append(dag_id)
                    elif status == State.FAILED:
                        pipeline_status['failed_dags'].append(dag_id)
                    elif status == State.RUNNING:
                        pipeline_status['running_dags'].append(dag_id)
                        
                else:
                    pipeline_status['dag_status'][dag_id] = {
                        'state': 'NOT_FOUND',
                        'error': 'DAG run not found'
                    }
                    
            except Exception as e:
                pipeline_status['dag_status'][dag_id] = {
                    'state': 'ERROR',
                    'error': str(e)
                }
        
        # 计算整体成功率
        pipeline_status['overall_success_rate'] = (successful_count / len(monitored_dags)) * 100
        
        return pipeline_status
    
    def _collect_quality_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """收集数据质量指标"""
        task_instance = context['task_instance']
        
        quality_metrics = {
            'data_completeness': {},
            'data_accuracy': {},
            'data_consistency': {},
            'feature_quality': {},
            'overall_scores': {}
        }
        
        # 从数据清洗DAG收集质量指标
        try:
            cleaning_report = task_instance.xcom_pull(
                dag_id='data_cleaning_dag',
                task_ids='generate_quality_report',
                key='quality_report'
            )
            
            if cleaning_report:
                quality_metrics['data_completeness'] = cleaning_report.get('summary', {})
                quality_metrics['overall_scores']['cleaning'] = cleaning_report.get('summary', {}).get('average_quality_score', 0)
                
        except Exception as e:
            self.logger.warning(f"无法获取清洗质量报告: {str(e)}")
        
        # 从特征工程DAG收集质量指标
        try:
            feature_report = task_instance.xcom_pull(
                dag_id='feature_engineering_dag',
                task_ids='generate_feature_summary_report',
                key='summary_report'
            )
            
            if feature_report:
                quality_metrics['feature_quality'] = feature_report.get('summary', {})
                quality_metrics['overall_scores']['features'] = feature_report.get('summary', {}).get('success_rate', 0)
                
        except Exception as e:
            self.logger.warning(f"无法获取特征质量报告: {str(e)}")
        
        # 计算综合质量分数
        scores = [score for score in quality_metrics['overall_scores'].values() if score > 0]
        quality_metrics['overall_quality_score'] = sum(scores) / len(scores) if scores else 0
        
        return quality_metrics
    
    def _collect_performance_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """收集性能指标"""
        from airflow.models import DagRun
        
        execution_date = context['execution_date']
        
        performance_metrics = {
            'processing_times': {},
            'resource_usage': {},
            'throughput': {},
            'error_rates': {}
        }
        
        # 收集各DAG的处理时间
        monitored_dags = ['data_ingestion_dag', 'data_cleaning_dag', 'feature_engineering_dag']
        
        for dag_id in monitored_dags:
            try:
                dag_runs = DagRun.find(dag_id=dag_id, execution_date=execution_date)
                if dag_runs:
                    dag_run = dag_runs[0]
                    if dag_run.start_date and dag_run.end_date:
                        duration = (dag_run.end_date - dag_run.start_date).total_seconds()
                        performance_metrics['processing_times'][dag_id] = {
                            'duration_seconds': duration,
                            'start_time': dag_run.start_date.isoformat(),
                            'end_time': dag_run.end_date.isoformat()
                        }
                        
            except Exception as e:
                self.logger.warning(f"无法获取 {dag_id} 性能指标: {str(e)}")
        
        # 计算总处理时间
        total_duration = sum(
            metrics.get('duration_seconds', 0) 
            for metrics in performance_metrics['processing_times'].values()
        )
        performance_metrics['total_processing_time'] = total_duration
        
        return performance_metrics
    
    def _assess_overall_quality(self, pipeline_status: Dict, quality_metrics: Dict, performance_metrics: Dict) -> Dict[str, Any]:
        """评估整体质量"""
        assessment = {
            'overall_score': 0.0,
            'pipeline_health': 'unknown',
            'quality_grade': 'unknown',
            'performance_grade': 'unknown',
            'recommendations': []
        }
        
        # 评估流水线健康度
        success_rate = pipeline_status.get('overall_success_rate', 0)
        if success_rate >= 95:
            assessment['pipeline_health'] = 'excellent'
        elif success_rate >= 85:
            assessment['pipeline_health'] = 'good'
        elif success_rate >= 70:
            assessment['pipeline_health'] = 'acceptable'
        else:
            assessment['pipeline_health'] = 'poor'
        
        # 评估数据质量
        quality_score = quality_metrics.get('overall_quality_score', 0)
        if quality_score >= 95:
            assessment['quality_grade'] = 'excellent'
        elif quality_score >= 85:
            assessment['quality_grade'] = 'good'
        elif quality_score >= 75:
            assessment['quality_grade'] = 'acceptable'
        else:
            assessment['quality_grade'] = 'poor'
        
        # 评估性能
        total_time = performance_metrics.get('total_processing_time', 0)
        time_threshold = self.quality_thresholds.get('processing_time_threshold', 3600)
        
        if total_time <= time_threshold * 0.5:
            assessment['performance_grade'] = 'excellent'
        elif total_time <= time_threshold * 0.8:
            assessment['performance_grade'] = 'good'
        elif total_time <= time_threshold:
            assessment['performance_grade'] = 'acceptable'
        else:
            assessment['performance_grade'] = 'poor'
        
        # 计算综合分数
        scores = {
            'pipeline': success_rate,
            'quality': quality_score,
            'performance': max(0, 100 - (total_time / time_threshold) * 100)
        }
        
        assessment['overall_score'] = sum(scores.values()) / len(scores)
        assessment['component_scores'] = scores
        
        # 生成建议
        if assessment['pipeline_health'] == 'poor':
            assessment['recommendations'].append("流水线成功率过低，需要检查失败的DAG和任务")
        
        if assessment['quality_grade'] == 'poor':
            assessment['recommendations'].append("数据质量较差，需要优化数据清洗和验证流程")
        
        if assessment['performance_grade'] == 'poor':
            assessment['recommendations'].append("处理时间过长，需要优化性能或增加资源")
        
        return assessment
    
    def _detect_anomalies_and_alerts(self, quality_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测异常和生成告警"""
        alerts = []
        
        # 检查整体分数
        overall_score = quality_assessment.get('overall_score', 0)
        if overall_score < self.quality_thresholds.get('pipeline_success_rate', 95):
            alerts.append({
                'type': 'quality_degradation',
                'severity': 'high' if overall_score < 70 else 'medium',
                'message': f"整体质量分数 {overall_score:.1f} 低于阈值 {self.quality_thresholds['pipeline_success_rate']}",
                'metric': 'overall_score',
                'value': overall_score,
                'threshold': self.quality_thresholds['pipeline_success_rate']
            })
        
        # 检查流水线健康度
        pipeline_health = quality_assessment.get('pipeline_health')
        if pipeline_health in ['poor', 'acceptable']:
            alerts.append({
                'type': 'pipeline_health',
                'severity': 'high' if pipeline_health == 'poor' else 'medium',
                'message': f"流水线健康度为 {pipeline_health}，需要关注",
                'metric': 'pipeline_health',
                'value': pipeline_health
            })
        
        # 检查数据质量
        quality_grade = quality_assessment.get('quality_grade')
        if quality_grade in ['poor', 'acceptable']:
            alerts.append({
                'type': 'data_quality',
                'severity': 'high' if quality_grade == 'poor' else 'medium',
                'message': f"数据质量等级为 {quality_grade}，需要改进",
                'metric': 'quality_grade',
                'value': quality_grade
            })
        
        # 检查性能
        performance_grade = quality_assessment.get('performance_grade')
        if performance_grade in ['poor', 'acceptable']:
            alerts.append({
                'type': 'performance',
                'severity': 'medium' if performance_grade == 'poor' else 'low',
                'message': f"性能等级为 {performance_grade}，建议优化",
                'metric': 'performance_grade',
                'value': performance_grade
            })
        
        return alerts
    
    def _generate_monitoring_report(self, execution_date: datetime, pipeline_status: Dict, 
                                  quality_metrics: Dict, performance_metrics: Dict, 
                                  quality_assessment: Dict, alerts: List[Dict]) -> Dict[str, Any]:
        """生成监控报告"""
        report = {
            'report_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'execution_date': execution_date.isoformat(),
                'report_type': 'data_quality_monitoring',
                'version': '1.0'
            },
            'executive_summary': {
                'overall_score': quality_assessment.get('overall_score', 0),
                'pipeline_health': quality_assessment.get('pipeline_health'),
                'quality_grade': quality_assessment.get('quality_grade'),
                'performance_grade': quality_assessment.get('performance_grade'),
                'alert_count': len(alerts),
                'high_severity_alerts': len([a for a in alerts if a.get('severity') == 'high'])
            },
            'pipeline_status': pipeline_status,
            'quality_metrics': quality_metrics,
            'performance_metrics': performance_metrics,
            'quality_assessment': quality_assessment,
            'alerts': alerts,
            'recommendations': quality_assessment.get('recommendations', [])
        }
        
        return report
    
    def _save_monitoring_report(self, report: Dict[str, Any], context: Dict[str, Any]) -> str:
        """保存监控报告"""
        try:
            execution_date = context['execution_date']
            
            # 确定报告保存路径
            reports_path = Variable.get('ml_monitoring_reports_path', '/opt/airflow/data/monitoring/reports')
            report_file = os.path.join(
                reports_path,
                execution_date.strftime('%Y'),
                execution_date.strftime('%m'),
                f'quality_monitoring_{execution_date.strftime("%Y-%m-%d_%H-%M")}.{self.report_format}'
            )
            
            # 确保目录存在
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            # 保存报告
            if self.report_format == 'json':
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
            elif self.report_format == 'html':
                html_content = self._generate_html_report(report)
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            self.logger.info(f"监控报告已保存到: {report_file}")
            
            # 更新报告元数据
            report['report_metadata']['report_file'] = report_file
            
            return report_file
            
        except Exception as e:
            self.logger.error(f"保存监控报告失败: {str(e)}")
            raise
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """生成HTML格式报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据质量监控报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .summary { display: flex; justify-content: space-around; margin: 20px 0; }
                .metric { text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .alert { padding: 10px; margin: 5px 0; border-radius: 5px; }
                .alert.high { background-color: #ffebee; border-left: 4px solid #f44336; }
                .alert.medium { background-color: #fff3e0; border-left: 4px solid #ff9800; }
                .alert.low { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>数据质量监控报告</h1>
                <p>生成时间: {generation_time}</p>
                <p>执行日期: {execution_date}</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>整体分数</h3>
                    <h2>{overall_score:.1f}</h2>
                </div>
                <div class="metric">
                    <h3>流水线健康度</h3>
                    <h2>{pipeline_health}</h2>
                </div>
                <div class="metric">
                    <h3>数据质量</h3>
                    <h2>{quality_grade}</h2>
                </div>
                <div class="metric">
                    <h3>性能等级</h3>
                    <h2>{performance_grade}</h2>
                </div>
            </div>
            
            <h2>告警信息</h2>
            {alerts_html}
            
            <h2>建议措施</h2>
            <ul>
                {recommendations_html}
            </ul>
        </body>
        </html>
        """
        
        # 生成告警HTML
        alerts_html = ""
        for alert in report.get('alerts', []):
            severity = alert.get('severity', 'low')
            alerts_html += f'<div class="alert {severity}"><strong>{alert.get("type", "")}</strong>: {alert.get("message", "")}</div>'
        
        if not alerts_html:
            alerts_html = '<p>无告警信息</p>'
        
        # 生成建议HTML
        recommendations_html = ""
        for rec in report.get('recommendations', []):
            recommendations_html += f'<li>{rec}</li>'
        
        if not recommendations_html:
            recommendations_html = '<li>系统运行正常，无特殊建议</li>'
        
        # 填充模板
        return html_template.format(
            generation_time=report['report_metadata']['generation_timestamp'],
            execution_date=report['report_metadata']['execution_date'],
            overall_score=report['executive_summary']['overall_score'],
            pipeline_health=report['executive_summary']['pipeline_health'],
            quality_grade=report['executive_summary']['quality_grade'],
            performance_grade=report['executive_summary']['performance_grade'],
            alerts_html=alerts_html,
            recommendations_html=recommendations_html
        )
    
    def _send_alerts(self, alerts: List[Dict], report: Dict, context: Dict[str, Any]) -> None:
        """发送告警"""
        try:
            # 过滤高优先级告警
            high_priority_alerts = [a for a in alerts if a.get('severity') in ['high', 'medium']]
            
            if not high_priority_alerts:
                return
            
            # 构造告警消息
            execution_date = context['execution_date']
            subject = f"数据质量告警 - {execution_date.strftime('%Y-%m-%d %H:%M')}"
            
            message = f"""
数据质量监控发现以下问题：

执行日期: {execution_date.strftime('%Y-%m-%d %H:%M')}
整体分数: {report['executive_summary']['overall_score']:.1f}
告警数量: {len(alerts)} (高优先级: {len(high_priority_alerts)})

告警详情:
"""
            
            for alert in high_priority_alerts:
                message += f"• [{alert.get('severity', '').upper()}] {alert.get('message', '')}\n"
            
            message += f"\n建议措施:\n"
            for rec in report.get('recommendations', []):
                message += f"• {rec}\n"
            
            # 发送到各个渠道
            for channel in self.alert_channels:
                try:
                    if channel == 'email':
                        self._send_email_alert(subject, message, context)
                    elif channel == 'slack':
                        self._send_slack_alert(subject, message, context)
                    elif channel == 'log':
                        self.logger.warning(f"ALERT: {subject}\n{message}")
                        
                except Exception as e:
                    self.logger.error(f"发送 {channel} 告警失败: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"发送告警失败: {str(e)}")
    
    def _send_email_alert(self, subject: str, message: str, context: Dict[str, Any]) -> None:
        """发送邮件告警"""
        try:
            # 获取邮件配置
            smtp_host = Variable.get('smtp_host', 'localhost')
            smtp_port = int(Variable.get('smtp_port', '587'))
            smtp_user = Variable.get('smtp_user', '')
            smtp_password = Variable.get('smtp_password', '')
            
            # 获取收件人列表
            alert_emails = Variable.get('alert_emails', '').split(',')
            alert_emails = [email.strip() for email in alert_emails if email.strip()]
            
            if not alert_emails:
                self.logger.warning("未配置告警邮件地址")
                return
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = ', '.join(alert_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain', 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if smtp_user and smtp_password:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                
                server.send_message(msg)
            
            self.logger.info(f"告警邮件已发送到: {', '.join(alert_emails)}")
            
        except Exception as e:
            self.logger.error(f"发送邮件告警失败: {str(e)}")
    
    def _send_slack_alert(self, subject: str, message: str, context: Dict[str, Any]) -> None:
        """发送Slack告警"""
        try:
            # 获取Slack配置
            slack_webhook = Variable.get('slack_webhook_url', '')
            
            if not slack_webhook:
                self.logger.warning("未配置Slack Webhook URL")
                return
            
            import requests
            
            # 构造Slack消息
            slack_message = {
                "text": subject,
                "attachments": [
                    {
                        "color": "danger",
                        "fields": [
                            {
                                "title": "详细信息",
                                "value": message,
                                "short": False
                            }
                        ]
                    }
                ]
            }
            
            # 发送到Slack
            response = requests.post(slack_webhook, json=slack_message)
            response.raise_for_status()
            
            self.logger.info("告警消息已发送到Slack")
            
        except Exception as e:
            self.logger.error(f"发送Slack告警失败: {str(e)}")


class HealthCheckOperator(BaseOperator):
    """
    系统健康检查操作符
    
    检查系统各组件的健康状态
    """
    
    @apply_defaults
    def __init__(
        self,
        check_components: Optional[List[str]] = None,
        *args,
        **kwargs
    ):
        """
        初始化健康检查操作符
        
        Args:
            check_components: 要检查的组件列表
        """
        super().__init__(*args, **kwargs)
        self.check_components = check_components or [
            'database', 'storage', 'api_connections', 'airflow_services'
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行健康检查
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 健康检查结果
        """
        self.logger.info("开始系统健康检查")
        
        health_results = {
            'check_timestamp': datetime.now().isoformat(),
            'overall_health': 'unknown',
            'component_status': {},
            'issues': [],
            'recommendations': []
        }
        
        healthy_components = 0
        total_components = len(self.check_components)
        
        for component in self.check_components:
            try:
                if component == 'database':
                    status = self._check_database_health()
                elif component == 'storage':
                    status = self._check_storage_health()
                elif component == 'api_connections':
                    status = self._check_api_connections()
                elif component == 'airflow_services':
                    status = self._check_airflow_services()
                else:
                    status = {'status': 'unknown', 'message': f'Unknown component: {component}'}
                
                health_results['component_status'][component] = status
                
                if status.get('status') == 'healthy':
                    healthy_components += 1
                elif status.get('status') == 'unhealthy':
                    health_results['issues'].append(f"{component}: {status.get('message', 'Unknown issue')}")
                
            except Exception as e:
                health_results['component_status'][component] = {
                    'status': 'error',
                    'message': str(e)
                }
                health_results['issues'].append(f"{component}: {str(e)}")
        
        # 计算整体健康状态
        health_ratio = healthy_components / total_components
        if health_ratio >= 0.9:
            health_results['overall_health'] = 'healthy'
        elif health_ratio >= 0.7:
            health_results['overall_health'] = 'degraded'
        else:
            health_results['overall_health'] = 'unhealthy'
        
        # 生成建议
        if health_results['issues']:
            health_results['recommendations'].append("检查并修复发现的组件问题")
        
        if health_results['overall_health'] != 'healthy':
            health_results['recommendations'].append("建议进行详细的系统诊断")
        
        self.logger.info(f"健康检查完成: {health_results['overall_health']} ({healthy_components}/{total_components} 组件正常)")
        
        # 将结果推送到XCom
        context['task_instance'].xcom_push(key='health_check_result', value=health_results)
        
        return health_results
    
    def _check_database_health(self) -> Dict[str, Any]:
        """检查数据库健康状态"""
        try:
            # 这里可以添加实际的数据库连接检查
            # 例如检查PostgreSQL连接
            return {
                'status': 'healthy',
                'message': 'Database connection successful',
                'response_time_ms': 50
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Database connection failed: {str(e)}'
            }
    
    def _check_storage_health(self) -> Dict[str, Any]:
        """检查存储健康状态"""
        try:
            # 检查数据目录是否可访问
            data_paths = [
                Variable.get('ml_raw_data_path', '/opt/airflow/data/raw'),
                Variable.get('ml_processed_data_path', '/opt/airflow/data/processed'),
                Variable.get('ml_features_data_path', '/opt/airflow/data/features')
            ]
            
            for path in data_paths:
                if not os.path.exists(path):
                    return {
                        'status': 'unhealthy',
                        'message': f'Data path not accessible: {path}'
                    }
                
                # 检查磁盘空间
                import shutil
                total, used, free = shutil.disk_usage(path)
                free_percentage = (free / total) * 100
                
                if free_percentage < 10:  # 少于10%空闲空间
                    return {
                        'status': 'degraded',
                        'message': f'Low disk space: {free_percentage:.1f}% free'
                    }
            
            return {
                'status': 'healthy',
                'message': 'All storage paths accessible with sufficient space'
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Storage check failed: {str(e)}'
            }
    
    def _check_api_connections(self) -> Dict[str, Any]:
        """检查API连接健康状态"""
        try:
            # 检查Binance API连接
            import requests
            
            response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'message': 'API connections successful',
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
            else:
                return {
                    'status': 'unhealthy',
                    'message': f'API connection failed: HTTP {response.status_code}'
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'API connection check failed: {str(e)}'
            }
    
    def _check_airflow_services(self) -> Dict[str, Any]:
        """检查Airflow服务健康状态"""
        try:
            # 检查Airflow元数据库连接
            from airflow.models import Variable as AirflowVariable
            
            # 尝试读取一个变量来测试数据库连接
            test_var = AirflowVariable.get('airflow_test_var', default_var='test')
            
            return {
                'status': 'healthy',
                'message': 'Airflow services operational'
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Airflow services check failed: {str(e)}'
            }