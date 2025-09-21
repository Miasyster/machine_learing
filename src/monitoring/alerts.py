"""
异常告警模块

提供告警管理、告警处理器和告警通知功能
"""

import smtplib
import json
import requests
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from collections import defaultdict, deque
import logging

from .base import (
    BaseAlertHandler, AlertRule, AlertEvent, AlertSeverity, 
    MonitoringConfig, MetricData
)

logger = logging.getLogger(__name__)


class EmailAlertHandler(BaseAlertHandler):
    """
    邮件告警处理器
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化邮件告警处理器
        
        Args:
            config: 邮件配置
                - smtp_server: SMTP服务器
                - smtp_port: SMTP端口
                - username: 用户名
                - password: 密码
                - from_email: 发送邮箱
                - to_emails: 接收邮箱列表
                - use_tls: 是否使用TLS
        """
        super().__init__(config)
        self.smtp_server = config['smtp_server']
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config['username']
        self.password = config['password']
        self.from_email = config['from_email']
        self.to_emails = config['to_emails']
        self.use_tls = config.get('use_tls', True)
    
    def send_alert(self, alert: AlertEvent) -> bool:
        """
        发送邮件告警
        
        Args:
            alert: 告警事件
            
        Returns:
            是否发送成功
        """
        try:
            # 创建邮件内容
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] ML Alert: {alert.rule_name}"
            
            # 邮件正文
            body = self._create_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for rule: {alert.rule_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_email_body(self, alert: AlertEvent) -> str:
        """
        创建邮件正文
        
        Args:
            alert: 告警事件
            
        Returns:
            HTML格式的邮件正文
        """
        severity_colors = {
            AlertSeverity.LOW: '#28a745',
            AlertSeverity.MEDIUM: '#ffc107',
            AlertSeverity.HIGH: '#fd7e14',
            AlertSeverity.CRITICAL: '#dc3545'
        }
        
        color = severity_colors.get(alert.severity, '#6c757d')
        
        return f"""
        <html>
        <body>
            <h2 style="color: {color};">ML System Alert</h2>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <td><strong>Rule Name</strong></td>
                    <td>{alert.rule_name}</td>
                </tr>
                <tr>
                    <td><strong>Metric</strong></td>
                    <td>{alert.metric_name}</td>
                </tr>
                <tr>
                    <td><strong>Severity</strong></td>
                    <td style="color: {color}; font-weight: bold;">{alert.severity.value.upper()}</td>
                </tr>
                <tr>
                    <td><strong>Current Value</strong></td>
                    <td>{alert.current_value}</td>
                </tr>
                <tr>
                    <td><strong>Threshold</strong></td>
                    <td>{alert.threshold}</td>
                </tr>
                <tr>
                    <td><strong>Message</strong></td>
                    <td>{alert.message}</td>
                </tr>
                <tr>
                    <td><strong>Timestamp</strong></td>
                    <td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
                <tr>
                    <td><strong>Tags</strong></td>
                    <td>{', '.join(f'{k}={v}' for k, v in alert.tags.items())}</td>
                </tr>
            </table>
            <p><em>This is an automated alert from the ML monitoring system.</em></p>
        </body>
        </html>
        """
    
    def test_connection(self) -> bool:
        """
        测试邮件连接
        
        Returns:
            连接是否正常
        """
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
            return True
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False


class SlackAlertHandler(BaseAlertHandler):
    """
    Slack告警处理器
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Slack告警处理器
        
        Args:
            config: Slack配置
                - webhook_url: Webhook URL
                - channel: 频道名称
                - username: 机器人用户名
        """
        super().__init__(config)
        self.webhook_url = config['webhook_url']
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'ML Monitor')
    
    def send_alert(self, alert: AlertEvent) -> bool:
        """
        发送Slack告警
        
        Args:
            alert: 告警事件
            
        Returns:
            是否发送成功
        """
        try:
            # 创建Slack消息
            payload = self._create_slack_payload(alert)
            
            # 发送消息
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent for rule: {alert.rule_name}")
                return True
            else:
                logger.error(f"Slack alert failed with status: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _create_slack_payload(self, alert: AlertEvent) -> Dict[str, Any]:
        """
        创建Slack消息载荷
        
        Args:
            alert: 告警事件
            
        Returns:
            Slack消息载荷
        """
        severity_colors = {
            AlertSeverity.LOW: 'good',
            AlertSeverity.MEDIUM: 'warning',
            AlertSeverity.HIGH: 'danger',
            AlertSeverity.CRITICAL: 'danger'
        }
        
        color = severity_colors.get(alert.severity, 'good')
        
        return {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "color": color,
                    "title": f"ML Alert: {alert.rule_name}",
                    "fields": [
                        {
                            "title": "Metric",
                            "value": alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": str(alert.current_value),
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": str(alert.threshold),
                            "short": True
                        },
                        {
                            "title": "Message",
                            "value": alert.message,
                            "short": False
                        },
                        {
                            "title": "Tags",
                            "value": ', '.join(f'{k}={v}' for k, v in alert.tags.items()),
                            "short": False
                        }
                    ],
                    "footer": "ML Monitoring System",
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }
    
    def test_connection(self) -> bool:
        """
        测试Slack连接
        
        Returns:
            连接是否正常
        """
        try:
            test_payload = {
                "channel": self.channel,
                "username": self.username,
                "text": "Test connection from ML monitoring system"
            }
            
            response = requests.post(
                self.webhook_url,
                json=test_payload,
                timeout=10
            )
            
            return response.status_code == 200
        
        except Exception as e:
            logger.error(f"Slack connection test failed: {e}")
            return False


class WebhookAlertHandler(BaseAlertHandler):
    """
    Webhook告警处理器
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Webhook告警处理器
        
        Args:
            config: Webhook配置
                - url: Webhook URL
                - headers: 请求头
                - timeout: 超时时间
        """
        super().__init__(config)
        self.url = config['url']
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 10)
    
    def send_alert(self, alert: AlertEvent) -> bool:
        """
        发送Webhook告警
        
        Args:
            alert: 告警事件
            
        Returns:
            是否发送成功
        """
        try:
            # 创建请求载荷
            payload = alert.to_dict()
            
            # 发送请求
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Webhook alert sent for rule: {alert.rule_name}")
                return True
            else:
                logger.error(f"Webhook alert failed with status: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        测试Webhook连接
        
        Returns:
            连接是否正常
        """
        try:
            test_payload = {
                "test": True,
                "message": "Test connection from ML monitoring system",
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(
                self.url,
                json=test_payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            return response.status_code in [200, 201, 202]
        
        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False


class AlertManager:
    """
    告警管理器
    """
    
    def __init__(self, config: MonitoringConfig):
        """
        初始化告警管理器
        
        Args:
            config: 监控配置
        """
        self.config = config
        self.rules: Dict[str, AlertRule] = {}
        self.handlers: Dict[str, BaseAlertHandler] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        
        # 指标缓存
        self.metrics_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        添加告警规则
        
        Args:
            rule: 告警规则
        """
        with self._lock:
            self.rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> None:
        """
        移除告警规则
        
        Args:
            rule_name: 规则名称
        """
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
    
    def add_handler(self, name: str, handler: BaseAlertHandler) -> None:
        """
        添加告警处理器
        
        Args:
            name: 处理器名称
            handler: 告警处理器
        """
        self.handlers[name] = handler
        logger.info(f"Added alert handler: {name}")
    
    def remove_handler(self, name: str) -> None:
        """
        移除告警处理器
        
        Args:
            name: 处理器名称
        """
        if name in self.handlers:
            del self.handlers[name]
            logger.info(f"Removed alert handler: {name}")
    
    def process_metrics(self, metrics: List[MetricData]) -> None:
        """
        处理指标数据并检查告警
        
        Args:
            metrics: 指标数据列表
        """
        with self._lock:
            # 更新指标缓存
            for metric in metrics:
                self.metrics_cache[metric.name].append(metric.value)
            
            # 检查告警规则
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                self._check_rule(rule)
    
    def _check_rule(self, rule: AlertRule) -> None:
        """
        检查单个告警规则
        
        Args:
            rule: 告警规则
        """
        # 检查冷却时间
        if self._is_in_cooldown(rule.name):
            return
        
        # 获取指标数据
        if rule.metric_name not in self.metrics_cache:
            return
        
        metric_values = list(self.metrics_cache[rule.metric_name])
        
        # 评估规则
        if rule.evaluate(metric_values):
            self._trigger_alert(rule, metric_values[-1] if metric_values else 0)
        else:
            # 检查是否需要解决现有告警
            self._resolve_alert(rule.name)
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """
        检查是否在冷却时间内
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否在冷却时间内
        """
        if rule_name not in self.cooldown_tracker:
            return False
        
        last_alert_time = self.cooldown_tracker[rule_name]
        cooldown_duration = self.rules[rule_name].cooldown
        
        return (datetime.now() - last_alert_time).total_seconds() < cooldown_duration
    
    def _trigger_alert(self, rule: AlertRule, current_value: float) -> None:
        """
        触发告警
        
        Args:
            rule: 告警规则
            current_value: 当前值
        """
        # 创建告警事件
        alert = AlertEvent(
            rule_name=rule.name,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            message=f"Metric {rule.metric_name} {rule.condition} {rule.threshold} (current: {current_value})",
            tags=rule.tags
        )
        
        # 记录告警
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        self.cooldown_tracker[rule.name] = datetime.now()
        
        # 发送告警
        self._send_alert(alert)
        
        logger.warning(f"Alert triggered: {rule.name} - {alert.message}")
    
    def _resolve_alert(self, rule_name: str) -> None:
        """
        解决告警
        
        Args:
            rule_name: 规则名称
        """
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolve()
            del self.active_alerts[rule_name]
            
            logger.info(f"Alert resolved: {rule_name}")
    
    def _send_alert(self, alert: AlertEvent) -> None:
        """
        发送告警到所有处理器
        
        Args:
            alert: 告警事件
        """
        for name, handler in self.handlers.items():
            if not handler.is_enabled():
                continue
            
            try:
                success = handler.send_alert(alert)
                if success:
                    logger.info(f"Alert sent via {name}")
                else:
                    logger.error(f"Failed to send alert via {name}")
            
            except Exception as e:
                logger.error(f"Error sending alert via {name}: {e}")
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """
        获取活跃告警
        
        Returns:
            活跃告警列表
        """
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, 
                         limit: Optional[int] = None,
                         severity: Optional[AlertSeverity] = None) -> List[AlertEvent]:
        """
        获取告警历史
        
        Args:
            limit: 限制数量
            severity: 严重程度过滤
            
        Returns:
            告警历史列表
        """
        alerts = list(self.alert_history)
        
        # 按严重程度过滤
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        # 按时间倒序排序
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        # 限制数量
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while not self._stop_event.is_set():
            try:
                # 清理过期的指标缓存
                self._cleanup_metrics_cache()
                
                # 清理过期的冷却时间
                self._cleanup_cooldown_tracker()
                
                time.sleep(60)  # 每分钟清理一次
            
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                time.sleep(5)
    
    def _cleanup_metrics_cache(self) -> None:
        """清理指标缓存"""
        # 这里可以实现更复杂的清理逻辑
        pass
    
    def _cleanup_cooldown_tracker(self) -> None:
        """清理冷却时间跟踪器"""
        now = datetime.now()
        expired_rules = []
        
        for rule_name, last_alert_time in self.cooldown_tracker.items():
            if rule_name in self.rules:
                cooldown_duration = self.rules[rule_name].cooldown
                if (now - last_alert_time).total_seconds() > cooldown_duration:
                    expired_rules.append(rule_name)
        
        for rule_name in expired_rules:
            del self.cooldown_tracker[rule_name]


def create_alert_manager(config: MonitoringConfig) -> AlertManager:
    """
    创建告警管理器
    
    Args:
        config: 监控配置
        
    Returns:
        告警管理器实例
    """
    return AlertManager(config)