"""
数据获取操作符
集成现有的BinanceDataFetcher模块到Airflow任务中
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.exceptions import AirflowException

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.etl.binance_data_fetcher import BinanceDataFetcher
from src.etl.data_storage import DataStorage


class BinanceDataIngestionOperator(BaseOperator):
    """
    Binance数据获取操作符
    
    从Binance API获取K线数据并保存到存储系统
    """
    
    template_fields = ['symbol', 'interval', 'start_date', 'end_date']
    
    @apply_defaults
    def __init__(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 1000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        connection_id: str = 'binance_api',
        storage_path: Optional[str] = None,
        file_format: str = 'csv',
        validate_data: bool = True,
        *args,
        **kwargs
    ):
        """
        初始化数据获取操作符
        
        Args:
            symbol: 交易对符号，如 'BTCUSDT'
            interval: 时间间隔，如 '1h', '4h', '1d'
            limit: 获取的K线数量限制
            start_date: 开始日期 (YYYY-MM-DD格式)
            end_date: 结束日期 (YYYY-MM-DD格式)
            connection_id: Airflow连接ID
            storage_path: 数据存储路径
            file_format: 文件格式 ('csv' 或 'parquet')
            validate_data: 是否验证数据
        """
        super().__init__(*args, **kwargs)
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.start_date = start_date
        self.end_date = end_date
        self.connection_id = connection_id
        self.storage_path = storage_path
        self.file_format = file_format
        self.validate_data = validate_data
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据获取任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 任务执行结果
        """
        self.logger.info(f"开始获取 {self.symbol} 的 {self.interval} K线数据")
        
        try:
            # 获取连接配置
            connection = self._get_connection_config()
            
            # 初始化数据获取器
            fetcher = BinanceDataFetcher(
                api_key=connection.get('api_key'),
                api_secret=connection.get('api_secret')
            )
            
            # 获取数据存储路径
            storage_path = self._get_storage_path()
            
            # 初始化数据存储器
            storage = DataStorage(base_path=storage_path)
            
            # 获取K线数据
            data = self._fetch_klines_data(fetcher, context)
            
            if data is None or data.empty:
                raise AirflowException(f"未获取到 {self.symbol} 的数据")
            
            # 验证数据
            if self.validate_data:
                self._validate_data(data)
            
            # 保存数据
            file_path = self._save_data(storage, data, context)
            
            # 准备返回结果
            result = {
                'symbol': self.symbol,
                'interval': self.interval,
                'data_points': len(data),
                'file_path': file_path,
                'start_time': data['open_time'].min().isoformat(),
                'end_time': data['open_time'].max().isoformat(),
                'file_size_mb': round(os.path.getsize(file_path) / 1024 / 1024, 2)
            }
            
            self.logger.info(f"成功获取并保存 {len(data)} 条 {self.symbol} 数据到 {file_path}")
            
            # 将结果推送到XCom
            context['task_instance'].xcom_push(key='ingestion_result', value=result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据获取失败: {str(e)}")
            raise AirflowException(f"数据获取失败: {str(e)}")
    
    def _get_connection_config(self) -> Dict[str, str]:
        """获取连接配置"""
        try:
            connection = BaseHook.get_connection(self.connection_id)
            extra = json.loads(connection.extra) if connection.extra else {}
            
            return {
                'api_key': extra.get('api_key', ''),
                'api_secret': extra.get('api_secret', ''),
                'rate_limit': extra.get('rate_limit', 1200),
                'timeout': extra.get('timeout', 30)
            }
        except Exception as e:
            self.logger.error(f"获取连接配置失败: {str(e)}")
            raise AirflowException(f"获取连接配置失败: {str(e)}")
    
    def _get_storage_path(self) -> str:
        """获取数据存储路径"""
        if self.storage_path:
            return self.storage_path
        
        try:
            # 从Airflow变量获取路径
            base_path = Variable.get('ml_raw_data_path', '/opt/airflow/data/raw')
            return base_path
        except Exception:
            # 使用默认路径
            return '/opt/airflow/data/raw'
    
    def _fetch_klines_data(self, fetcher: BinanceDataFetcher, context: Dict[str, Any]):
        """获取K线数据"""
        try:
            # 处理日期参数
            start_date = self.start_date
            end_date = self.end_date
            
            # 如果没有指定日期，使用默认逻辑
            if not start_date:
                # 获取最近的数据
                execution_date = context['execution_date']
                start_date = (execution_date - timedelta(days=1)).strftime('%Y-%m-%d')
            
            if not end_date:
                execution_date = context['execution_date']
                end_date = execution_date.strftime('%Y-%m-%d')
            
            self.logger.info(f"获取数据范围: {start_date} 到 {end_date}")
            
            # 调用数据获取器
            data = fetcher.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=self.limit,
                start_str=start_date,
                end_str=end_date
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {str(e)}")
            raise
    
    def _validate_data(self, data) -> None:
        """验证数据质量"""
        if data is None or data.empty:
            raise AirflowException("数据为空")
        
        # 检查必需的列
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise AirflowException(f"缺少必需的列: {missing_columns}")
        
        # 检查数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not data[col].dtype.kind in 'biufc':  # 数值类型
                raise AirflowException(f"列 {col} 不是数值类型")
        
        # 检查价格逻辑
        invalid_ohlc = data[
            (data['high'] < data['low']) |
            (data['open'] < data['low']) | (data['open'] > data['high']) |
            (data['close'] < data['low']) | (data['close'] > data['high'])
        ]
        
        if not invalid_ohlc.empty:
            self.logger.warning(f"发现 {len(invalid_ohlc)} 条OHLC逻辑异常的数据")
        
        # 检查成交量
        negative_volume = data[data['volume'] < 0]
        if not negative_volume.empty:
            raise AirflowException(f"发现 {len(negative_volume)} 条负成交量数据")
        
        self.logger.info(f"数据验证通过: {len(data)} 条记录")
    
    def _save_data(self, storage: DataStorage, data, context: Dict[str, Any]) -> str:
        """保存数据"""
        try:
            execution_date = context['execution_date']
            date_str = execution_date.strftime('%Y-%m-%d')
            
            # 保存数据
            file_path = storage.save_klines_data(
                data=data,
                symbol=self.symbol,
                interval=self.interval,
                date=date_str,
                file_format=self.file_format
            )
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {str(e)}")
            raise AirflowException(f"保存数据失败: {str(e)}")


class MultiSymbolDataIngestionOperator(BaseOperator):
    """
    多交易对数据获取操作符
    
    批量获取多个交易对的数据
    """
    
    template_fields = ['symbols', 'interval', 'start_date', 'end_date']
    
    @apply_defaults
    def __init__(
        self,
        symbols: List[str],
        interval: str = '1h',
        limit: int = 1000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        connection_id: str = 'binance_api',
        storage_path: Optional[str] = None,
        file_format: str = 'csv',
        validate_data: bool = True,
        parallel: bool = False,
        *args,
        **kwargs
    ):
        """
        初始化多交易对数据获取操作符
        
        Args:
            symbols: 交易对符号列表
            interval: 时间间隔
            limit: 获取的K线数量限制
            start_date: 开始日期
            end_date: 结束日期
            connection_id: Airflow连接ID
            storage_path: 数据存储路径
            file_format: 文件格式
            validate_data: 是否验证数据
            parallel: 是否并行获取（暂未实现）
        """
        super().__init__(*args, **kwargs)
        self.symbols = symbols
        self.interval = interval
        self.limit = limit
        self.start_date = start_date
        self.end_date = end_date
        self.connection_id = connection_id
        self.storage_path = storage_path
        self.file_format = file_format
        self.validate_data = validate_data
        self.parallel = parallel
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行多交易对数据获取任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 任务执行结果
        """
        self.logger.info(f"开始获取 {len(self.symbols)} 个交易对的数据")
        
        results = {}
        errors = {}
        
        for symbol in self.symbols:
            try:
                # 创建单个交易对的获取操作符
                operator = BinanceDataIngestionOperator(
                    task_id=f"fetch_{symbol.lower()}",
                    symbol=symbol,
                    interval=self.interval,
                    limit=self.limit,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    connection_id=self.connection_id,
                    storage_path=self.storage_path,
                    file_format=self.file_format,
                    validate_data=self.validate_data
                )
                
                # 执行获取任务
                result = operator.execute(context)
                results[symbol] = result
                
                self.logger.info(f"成功获取 {symbol} 数据: {result['data_points']} 条记录")
                
            except Exception as e:
                error_msg = f"获取 {symbol} 数据失败: {str(e)}"
                self.logger.error(error_msg)
                errors[symbol] = error_msg
        
        # 准备汇总结果
        summary = {
            'total_symbols': len(self.symbols),
            'successful_symbols': len(results),
            'failed_symbols': len(errors),
            'results': results,
            'errors': errors,
            'success_rate': len(results) / len(self.symbols) * 100
        }
        
        self.logger.info(f"数据获取完成: {summary['successful_symbols']}/{summary['total_symbols']} 成功")
        
        # 如果有失败的交易对，记录警告
        if errors:
            self.logger.warning(f"以下交易对获取失败: {list(errors.keys())}")
        
        # 将结果推送到XCom
        context['task_instance'].xcom_push(key='multi_ingestion_result', value=summary)
        
        return summary