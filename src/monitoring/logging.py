"""
日志记录模块

提供结构化日志记录、日志聚合、日志分析和日志管理功能。
"""

import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue
import gzip
import shutil

from .base import MonitoringError


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """日志条目数据类"""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: str
    process_id: int
    extra_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        return data
    
    def to_json(self) -> str:
        """转换为JSON格式"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 设置文件处理器
        self._setup_file_handler()
        
        # 设置控制台处理器
        self._setup_console_handler()
        
        # 额外数据
        self.extra_data = {}
    
    def _setup_file_handler(self):
        """设置文件处理器"""
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 自定义格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def _setup_console_handler(self):
        """设置控制台处理器"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def set_extra_data(self, **kwargs):
        """设置额外数据"""
        self.extra_data.update(kwargs)
    
    def clear_extra_data(self):
        """清除额外数据"""
        self.extra_data.clear()
    
    def _create_log_entry(self, level: LogLevel, message: str, 
                         extra: Optional[Dict[str, Any]] = None) -> LogEntry:
        """创建日志条目"""
        import inspect
        import threading
        import os
        
        # 获取调用信息
        frame = inspect.currentframe().f_back.f_back
        
        # 合并额外数据
        combined_extra = self.extra_data.copy()
        if extra:
            combined_extra.update(extra)
        
        return LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            logger_name=self.name,
            module=frame.f_globals.get('__name__', 'unknown'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=threading.current_thread().name,
            process_id=os.getpid(),
            extra_data=combined_extra
        )
    
    def debug(self, message: str, **extra):
        """记录DEBUG级别日志"""
        entry = self._create_log_entry(LogLevel.DEBUG, message, extra)
        self.logger.debug(message, extra=extra)
        return entry
    
    def info(self, message: str, **extra):
        """记录INFO级别日志"""
        entry = self._create_log_entry(LogLevel.INFO, message, extra)
        self.logger.info(message, extra=extra)
        return entry
    
    def warning(self, message: str, **extra):
        """记录WARNING级别日志"""
        entry = self._create_log_entry(LogLevel.WARNING, message, extra)
        self.logger.warning(message, extra=extra)
        return entry
    
    def error(self, message: str, **extra):
        """记录ERROR级别日志"""
        entry = self._create_log_entry(LogLevel.ERROR, message, extra)
        self.logger.error(message, extra=extra)
        return entry
    
    def critical(self, message: str, **extra):
        """记录CRITICAL级别日志"""
        entry = self._create_log_entry(LogLevel.CRITICAL, message, extra)
        self.logger.critical(message, extra=extra)
        return entry


class LogAggregator:
    """日志聚合器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        self.log_queue = Queue()
        self.is_running = False
        self.worker_thread = None
        
        # 聚合配置
        self.batch_size = 100
        self.flush_interval = 60  # 秒
        
    def start(self):
        """启动日志聚合"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.start()
    
    def stop(self):
        """停止日志聚合"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def add_log_entry(self, entry: LogEntry):
        """添加日志条目"""
        self.log_queue.put(entry)
    
    def _worker(self):
        """工作线程"""
        batch = []
        last_flush = time.time()
        
        while self.is_running:
            try:
                # 获取日志条目
                if not self.log_queue.empty():
                    entry = self.log_queue.get(timeout=1)
                    batch.append(entry)
                
                # 检查是否需要刷新
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    current_time - last_flush >= self.flush_interval
                )
                
                if should_flush and batch:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush = current_time
                    
            except Exception as e:
                print(f"日志聚合错误: {e}")
        
        # 处理剩余日志
        if batch:
            self._flush_batch(batch)
    
    def _flush_batch(self, batch: List[LogEntry]):
        """刷新批次日志"""
        if not batch:
            return
        
        # 按日期分组
        date_groups = {}
        for entry in batch:
            date_key = entry.timestamp.strftime('%Y-%m-%d')
            if date_key not in date_groups:
                date_groups[date_key] = []
            date_groups[date_key].append(entry)
        
        # 写入文件
        for date_key, entries in date_groups.items():
            log_file = self.output_dir / f"aggregated_{date_key}.jsonl"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                for entry in entries:
                    f.write(entry.to_json() + '\n')


class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
    
    def analyze_logs(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """分析日志"""
        log_files = self._get_log_files(start_date, end_date)
        
        analysis = {
            'total_entries': 0,
            'level_counts': {level.value: 0 for level in LogLevel},
            'error_patterns': {},
            'top_modules': {},
            'hourly_distribution': {},
            'performance_metrics': {}
        }
        
        for log_file in log_files:
            self._analyze_file(log_file, analysis)
        
        return analysis
    
    def _get_log_files(self, start_date: datetime, end_date: datetime) -> List[Path]:
        """获取日期范围内的日志文件"""
        log_files = []
        current_date = start_date.date()
        end_date = end_date.date()
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            log_file = self.log_dir / f"aggregated_{date_str}.jsonl"
            
            if log_file.exists():
                log_files.append(log_file)
            
            current_date += timedelta(days=1)
        
        return log_files
    
    def _analyze_file(self, log_file: Path, analysis: Dict[str, Any]):
        """分析单个日志文件"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry_data = json.loads(line)
                        self._analyze_entry(entry_data, analysis)
        except Exception as e:
            print(f"分析日志文件 {log_file} 时出错: {e}")
    
    def _analyze_entry(self, entry_data: Dict[str, Any], analysis: Dict[str, Any]):
        """分析单个日志条目"""
        analysis['total_entries'] += 1
        
        # 级别统计
        level = entry_data.get('level', 'UNKNOWN')
        if level in analysis['level_counts']:
            analysis['level_counts'][level] += 1
        
        # 模块统计
        module = entry_data.get('module', 'unknown')
        analysis['top_modules'][module] = analysis['top_modules'].get(module, 0) + 1
        
        # 错误模式分析
        if level in ['ERROR', 'CRITICAL']:
            message = entry_data.get('message', '')
            # 简单的错误模式提取
            error_key = message.split(':')[0] if ':' in message else message[:50]
            analysis['error_patterns'][error_key] = analysis['error_patterns'].get(error_key, 0) + 1
        
        # 时间分布
        timestamp_str = entry_data.get('timestamp', '')
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                hour_key = timestamp.strftime('%H:00')
                analysis['hourly_distribution'][hour_key] = analysis['hourly_distribution'].get(hour_key, 0) + 1
            except:
                pass
    
    def get_error_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """获取错误摘要"""
        analysis = self.analyze_logs(start_date, end_date)
        
        return {
            'total_errors': analysis['level_counts']['ERROR'] + analysis['level_counts']['CRITICAL'],
            'error_rate': (analysis['level_counts']['ERROR'] + analysis['level_counts']['CRITICAL']) / max(analysis['total_entries'], 1),
            'top_error_patterns': sorted(
                analysis['error_patterns'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'error_trend': self._calculate_error_trend(start_date, end_date)
        }
    
    def _calculate_error_trend(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """计算错误趋势"""
        trend = []
        current_date = start_date.date()
        end_date = end_date.date()
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            log_file = self.log_dir / f"aggregated_{date_str}.jsonl"
            
            daily_errors = 0
            daily_total = 0
            
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                entry_data = json.loads(line)
                                daily_total += 1
                                if entry_data.get('level') in ['ERROR', 'CRITICAL']:
                                    daily_errors += 1
                except:
                    pass
            
            trend.append({
                'date': date_str,
                'errors': daily_errors,
                'total': daily_total,
                'error_rate': daily_errors / max(daily_total, 1)
            })
            
            current_date += timedelta(days=1)
        
        return trend


class LogManager:
    """日志管理器"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        self.loggers = {}
        self.aggregator = LogAggregator(log_dir / "aggregated")
        self.analyzer = LogAnalyzer(log_dir / "aggregated")
        
        # 启动聚合器
        self.aggregator.start()
    
    def get_logger(self, name: str) -> StructuredLogger:
        """获取日志记录器"""
        if name not in self.loggers:
            logger = StructuredLogger(name, self.log_dir)
            self.loggers[name] = logger
            
            # 重写日志方法以支持聚合
            original_methods = {}
            for level in ['debug', 'info', 'warning', 'error', 'critical']:
                original_method = getattr(logger, level)
                original_methods[level] = original_method
                
                def create_wrapper(orig_method):
                    def wrapper(message: str, **extra):
                        entry = orig_method(message, **extra)
                        self.aggregator.add_log_entry(entry)
                        return entry
                    return wrapper
                
                setattr(logger, level, create_wrapper(original_method))
        
        return self.loggers[name]
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """清理旧日志"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in self.log_dir.rglob("*.log"):
            try:
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
            except Exception as e:
                print(f"删除日志文件 {log_file} 时出错: {e}")
    
    def compress_old_logs(self, days_to_compress: int = 7):
        """压缩旧日志"""
        cutoff_date = datetime.now() - timedelta(days=days_to_compress)
        
        for log_file in self.log_dir.rglob("*.log"):
            try:
                if (log_file.stat().st_mtime < cutoff_date.timestamp() and
                    not log_file.name.endswith('.gz')):
                    
                    compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    log_file.unlink()
                    
            except Exception as e:
                print(f"压缩日志文件 {log_file} 时出错: {e}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        stats = {
            'total_loggers': len(self.loggers),
            'log_files': 0,
            'total_size': 0,
            'compressed_files': 0
        }
        
        for log_file in self.log_dir.rglob("*"):
            if log_file.is_file():
                stats['log_files'] += 1
                stats['total_size'] += log_file.stat().st_size
                
                if log_file.name.endswith('.gz'):
                    stats['compressed_files'] += 1
        
        return stats
    
    def shutdown(self):
        """关闭日志管理器"""
        self.aggregator.stop()


def create_log_manager(log_dir: Optional[Path] = None) -> LogManager:
    """创建日志管理器"""
    if log_dir is None:
        log_dir = Path("logs")
    
    return LogManager(log_dir)


def setup_application_logging(app_name: str, log_level: str = "INFO") -> StructuredLogger:
    """设置应用程序日志"""
    log_manager = create_log_manager()
    logger = log_manager.get_logger(app_name)
    
    # 设置日志级别
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    if log_level in level_map:
        logger.logger.setLevel(level_map[log_level])
    
    return logger