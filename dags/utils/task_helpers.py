"""
DAG任务辅助函数
提供通用的任务处理功能
"""

import os
import yaml
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from airflow.models import Variable
from airflow.exceptions import AirflowException


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict: 配置内容
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败 {config_path}: {str(e)}")
        raise AirflowException(f"加载配置文件失败: {str(e)}")


def get_dag_config() -> Dict[str, Any]:
    """
    获取DAG配置
    
    Returns:
        Dict: DAG配置
    """
    try:
        config_path = Variable.get('dag_config_path', '/opt/airflow/dags/config/dag_config.yaml')
        return load_config(config_path)
    except Exception:
        # 返回默认配置
        return {
            'default_args': {
                'owner': 'ml_team',
                'depends_on_past': False,
                'start_date': '2024-01-01',
                'email_on_failure': True,
                'email_on_retry': False,
                'retries': 2,
                'retry_delay_minutes': 5
            }
        }


def get_data_sources_config() -> Dict[str, Any]:
    """
    获取数据源配置
    
    Returns:
        Dict: 数据源配置
    """
    try:
        config_path = Variable.get('data_sources_config_path', '/opt/airflow/dags/config/data_sources.yaml')
        return load_config(config_path)
    except Exception:
        # 返回默认配置
        return {
            'trading_pairs': {
                'major_pairs': [
                    {'symbol': 'BTCUSDT', 'intervals': ['1h', '4h', '1d']},
                    {'symbol': 'ETHUSDT', 'intervals': ['1h', '4h', '1d']}
                ]
            }
        }


def get_quality_thresholds_config() -> Dict[str, Any]:
    """
    获取数据质量阈值配置
    
    Returns:
        Dict: 质量阈值配置
    """
    try:
        config_path = Variable.get('quality_thresholds_config_path', '/opt/airflow/dags/config/quality_thresholds.yaml')
        return load_config(config_path)
    except Exception:
        # 返回默认配置
        return {
            'data_integrity': {
                'missing_values': {'max_missing_percentage': 5.0},
                'duplicates': {'max_duplicate_percentage': 2.0}
            }
        }


def get_trading_pairs() -> List[str]:
    """
    获取交易对列表
    
    Returns:
        List: 交易对符号列表
    """
    try:
        # 从Airflow变量获取
        pairs_json = Variable.get('default_trading_pairs', '["BTCUSDT", "ETHUSDT", "BNBUSDT"]')
        return json.loads(pairs_json)
    except Exception:
        # 从配置文件获取
        config = get_data_sources_config()
        pairs = []
        for pair_group in config.get('trading_pairs', {}).values():
            if isinstance(pair_group, list):
                pairs.extend([pair['symbol'] for pair in pair_group if 'symbol' in pair])
        return pairs if pairs else ['BTCUSDT', 'ETHUSDT']


def get_default_intervals() -> List[str]:
    """
    获取默认时间间隔列表
    
    Returns:
        List: 时间间隔列表
    """
    try:
        intervals_json = Variable.get('default_intervals', '["1h", "4h", "1d"]')
        return json.loads(intervals_json)
    except Exception:
        return ['1h', '4h', '1d']


def get_data_path(path_type: str) -> str:
    """
    获取数据路径
    
    Args:
        path_type: 路径类型 ('raw', 'processed', 'features', 'base')
        
    Returns:
        str: 数据路径
    """
    path_mapping = {
        'base': 'ml_data_base_path',
        'raw': 'ml_raw_data_path',
        'processed': 'ml_processed_data_path',
        'features': 'ml_features_data_path'
    }
    
    default_paths = {
        'base': '/opt/airflow/data',
        'raw': '/opt/airflow/data/raw',
        'processed': '/opt/airflow/data/processed',
        'features': '/opt/airflow/data/features'
    }
    
    variable_name = path_mapping.get(path_type)
    if not variable_name:
        raise AirflowException(f"未知的路径类型: {path_type}")
    
    try:
        return Variable.get(variable_name, default_paths[path_type])
    except Exception:
        return default_paths[path_type]


def create_default_args() -> Dict[str, Any]:
    """
    创建默认DAG参数
    
    Returns:
        Dict: 默认参数
    """
    config = get_dag_config()
    default_args = config.get('default_args', {})
    
    # 处理日期字符串
    if 'start_date' in default_args and isinstance(default_args['start_date'], str):
        default_args['start_date'] = datetime.strptime(default_args['start_date'], '%Y-%m-%d')
    
    # 处理重试延迟
    if 'retry_delay_minutes' in default_args:
        default_args['retry_delay'] = timedelta(minutes=default_args.pop('retry_delay_minutes'))
    
    return default_args


def get_task_timeout(task_type: str) -> Optional[timedelta]:
    """
    获取任务超时时间
    
    Args:
        task_type: 任务类型
        
    Returns:
        timedelta: 超时时间
    """
    config = get_dag_config()
    timeout_config = config.get('timeout_config', {})
    
    timeout_mapping = {
        'data_fetch': 'data_fetch_timeout_minutes',
        'data_processing': 'data_processing_timeout_minutes',
        'feature_calculation': 'feature_calculation_timeout_minutes',
        'quality_check': 'quality_check_timeout_minutes'
    }
    
    timeout_key = timeout_mapping.get(task_type)
    if timeout_key and timeout_key in timeout_config:
        return timedelta(minutes=timeout_config[timeout_key])
    
    return None


def get_retry_config(task_type: str) -> Dict[str, Any]:
    """
    获取重试配置
    
    Args:
        task_type: 任务类型
        
    Returns:
        Dict: 重试配置
    """
    config = get_dag_config()
    retry_config = config.get('retry_config', {})
    
    task_mapping = {
        'api': 'api_tasks',
        'processing': 'processing_tasks',
        'quality': 'quality_checks'
    }
    
    config_key = task_mapping.get(task_type, 'processing_tasks')
    task_config = retry_config.get(config_key, {})
    
    result = {
        'retries': task_config.get('retries', 2),
        'retry_delay': timedelta(minutes=task_config.get('retry_delay_minutes', 5))
    }
    
    if task_config.get('retry_exponential_backoff', False):
        result['retry_exponential_backoff'] = True
    
    return result


def format_file_path(base_path: str, symbol: str, interval: str, date: str, file_format: str = 'csv') -> str:
    """
    格式化文件路径
    
    Args:
        base_path: 基础路径
        symbol: 交易对符号
        interval: 时间间隔
        date: 日期字符串
        file_format: 文件格式
        
    Returns:
        str: 格式化的文件路径
    """
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')
    day = date_obj.strftime('%d')
    
    # 创建分层目录结构
    file_path = os.path.join(
        base_path,
        symbol.lower(),
        interval,
        year,
        month,
        f"{symbol.lower()}_{interval}_{date}.{file_format}"
    )
    
    return file_path


def ensure_directory_exists(file_path: str) -> None:
    """
    确保目录存在
    
    Args:
        file_path: 文件路径
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def validate_symbol(symbol: str) -> bool:
    """
    验证交易对符号格式
    
    Args:
        symbol: 交易对符号
        
    Returns:
        bool: 是否有效
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # 基本格式检查
    if len(symbol) < 6 or len(symbol) > 12:
        return False
    
    # 检查是否包含USDT
    if not symbol.endswith('USDT'):
        return False
    
    return True


def validate_interval(interval: str) -> bool:
    """
    验证时间间隔格式
    
    Args:
        interval: 时间间隔
        
    Returns:
        bool: 是否有效
    """
    valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    return interval in valid_intervals


def calculate_data_quality_score(metrics: Dict[str, float]) -> float:
    """
    计算数据质量评分
    
    Args:
        metrics: 质量指标字典
        
    Returns:
        float: 质量评分 (0-100)
    """
    config = get_quality_thresholds_config()
    weights = config.get('quality_scoring', {}).get('weights', {})
    
    # 默认权重
    default_weights = {
        'data_completeness': 0.25,
        'price_logic': 0.25,
        'volume_logic': 0.15,
        'time_series_continuity': 0.20,
        'feature_quality': 0.15
    }
    
    total_score = 0.0
    total_weight = 0.0
    
    for metric, value in metrics.items():
        weight = weights.get(metric, default_weights.get(metric, 0.1))
        total_score += value * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return min(100.0, max(0.0, total_score / total_weight))


def should_send_alert(alert_type: str, current_value: float) -> bool:
    """
    判断是否应该发送告警
    
    Args:
        alert_type: 告警类型
        current_value: 当前值
        
    Returns:
        bool: 是否发送告警
    """
    config = get_quality_thresholds_config()
    alerting_config = config.get('alerting', {})
    
    # 获取告警阈值
    alert_levels = alerting_config.get('alert_levels', {})
    
    # 检查严重告警
    critical_thresholds = alert_levels.get('critical', {})
    if alert_type in critical_thresholds:
        if current_value < critical_thresholds[alert_type]:
            return True
    
    # 检查警告告警
    warning_thresholds = alert_levels.get('warning', {})
    if alert_type in warning_thresholds:
        if current_value < warning_thresholds[alert_type]:
            return True
    
    return False


def get_environment() -> str:
    """
    获取当前环境
    
    Returns:
        str: 环境名称 ('development', 'staging', 'production')
    """
    try:
        return Variable.get('environment', 'development')
    except Exception:
        return 'development'


def is_debug_mode() -> bool:
    """
    检查是否为调试模式
    
    Returns:
        bool: 是否为调试模式
    """
    try:
        debug_mode = Variable.get('debug_mode', 'false')
        return debug_mode.lower() in ['true', '1', 'yes']
    except Exception:
        return False