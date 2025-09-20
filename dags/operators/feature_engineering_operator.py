"""
特征工程操作符
集成现有的FeatureEngineer模块到Airflow任务中
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.models import Variable
from airflow.exceptions import AirflowException

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.feature_engineering import FeatureEngineer
from src.etl.data_storage import DataStorage


class FeatureEngineeringOperator(BaseOperator):
    """
    特征工程操作符
    
    使用FeatureEngineer对清洗后的数据进行特征计算
    """
    
    template_fields = ['input_path', 'output_path', 'symbol', 'interval']
    
    @apply_defaults
    def __init__(
        self,
        symbol: str,
        interval: str,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        feature_config: Optional[Dict[str, Any]] = None,
        indicators: Optional[List[str]] = None,
        file_format: str = 'csv',
        validate_features: bool = True,
        save_metadata: bool = True,
        *args,
        **kwargs
    ):
        """
        初始化特征工程操作符
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            input_path: 输入数据路径
            output_path: 输出数据路径
            feature_config: 特征配置
            indicators: 指定要计算的指标列表
            file_format: 文件格式
            validate_features: 是否验证特征
            save_metadata: 是否保存元数据
        """
        super().__init__(*args, **kwargs)
        self.symbol = symbol
        self.interval = interval
        self.input_path = input_path
        self.output_path = output_path
        self.feature_config = feature_config or {}
        self.indicators = indicators
        self.file_format = file_format
        self.validate_features = validate_features
        self.save_metadata = save_metadata
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行特征工程任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 任务执行结果
        """
        self.logger.info(f"开始为 {self.symbol} 的 {self.interval} 数据计算特征")
        
        try:
            # 获取输入和输出路径
            input_file = self._get_input_path(context)
            output_file = self._get_output_path(context)
            
            # 加载清洗后的数据
            data = self._load_data(input_file)
            
            if data is None or data.empty:
                raise AirflowException(f"输入数据为空: {input_file}")
            
            self.logger.info(f"加载数据: {len(data)} 条记录")
            
            # 初始化特征工程器
            feature_engineer = FeatureEngineer()
            
            # 注册指定的指标（如果提供）
            if self.indicators:
                self._register_custom_indicators(feature_engineer)
            
            # 计算特征
            features_data = self._compute_features(feature_engineer, data)
            
            # 验证特征
            if self.validate_features:
                self._validate_features(features_data)
            
            # 保存特征数据
            self._save_features(features_data, output_file)
            
            # 保存元数据
            metadata = None
            if self.save_metadata:
                metadata = self._save_feature_metadata(features_data, output_file, context)
            
            # 生成特征报告
            feature_report = self._generate_feature_report(data, features_data)
            
            # 准备返回结果
            result = {
                'symbol': self.symbol,
                'interval': self.interval,
                'input_file': input_file,
                'output_file': output_file,
                'original_records': len(data),
                'feature_records': len(features_data),
                'feature_count': len(features_data.columns) - len(data.columns),
                'feature_report': feature_report,
                'metadata_file': metadata.get('metadata_file') if metadata else None,
                'file_size_mb': round(os.path.getsize(output_file) / 1024 / 1024, 2)
            }
            
            self.logger.info(f"特征工程完成: 生成 {result['feature_count']} 个特征")
            
            # 将结果推送到XCom
            context['task_instance'].xcom_push(key='feature_result', value=result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"特征工程失败: {str(e)}")
            raise AirflowException(f"特征工程失败: {str(e)}")
    
    def _get_input_path(self, context: Dict[str, Any]) -> str:
        """获取输入数据路径"""
        if self.input_path:
            return self.input_path
        
        # 从上游清洗任务获取文件路径
        task_instance = context['task_instance']
        upstream_result = task_instance.xcom_pull(
            task_ids=f'clean_data.{self.symbol.lower()}_{self.interval}_cleaning',
            key='cleaning_result',
            dag_id='data_cleaning_dag'
        )
        
        if upstream_result and 'output_file' in upstream_result:
            return upstream_result['output_file']
        
        # 构造默认路径（使用处理后的数据）
        processed_data_path = Variable.get('ml_processed_data_path', '/opt/airflow/data/processed')
        execution_date = context['execution_date']
        date_str = execution_date.strftime('%Y-%m-%d')
        
        return os.path.join(
            processed_data_path,
            self.symbol.lower(),
            self.interval,
            execution_date.strftime('%Y'),
            execution_date.strftime('%m'),
            f"{self.symbol.lower()}_{self.interval}_{date_str}_cleaned.{self.file_format}"
        )
    
    def _get_output_path(self, context: Dict[str, Any]) -> str:
        """获取输出数据路径"""
        if self.output_path:
            return self.output_path
        
        features_data_path = Variable.get('ml_features_data_path', '/opt/airflow/data/features')
        execution_date = context['execution_date']
        date_str = execution_date.strftime('%Y-%m-%d')
        
        output_file = os.path.join(
            features_data_path,
            self.symbol.lower(),
            self.interval,
            execution_date.strftime('%Y'),
            execution_date.strftime('%m'),
            f"{self.symbol.lower()}_{self.interval}_{date_str}_features.{self.file_format}"
        )
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        return output_file
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据"""
        try:
            if not os.path.exists(file_path):
                raise AirflowException(f"输入文件不存在: {file_path}")
            
            if self.file_format.lower() == 'csv':
                data = pd.read_csv(file_path)
            elif self.file_format.lower() == 'parquet':
                data = pd.read_parquet(file_path)
            else:
                raise AirflowException(f"不支持的文件格式: {self.file_format}")
            
            # 确保时间列为datetime类型
            if 'open_time' in data.columns:
                data['open_time'] = pd.to_datetime(data['open_time'])
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def _register_custom_indicators(self, feature_engineer: FeatureEngineer) -> None:
        """注册自定义指标"""
        try:
            # 如果指定了特定指标，只注册这些指标
            if self.indicators:
                # 清除默认注册的指标
                feature_engineer.indicators = {}
                
                # 重新注册指定的指标
                for indicator_name in self.indicators:
                    if hasattr(feature_engineer, f'_register_{indicator_name}'):
                        getattr(feature_engineer, f'_register_{indicator_name}')()
                    else:
                        self.logger.warning(f"未找到指标: {indicator_name}")
            
            self.logger.info(f"已注册 {len(feature_engineer.indicators)} 个技术指标")
            
        except Exception as e:
            self.logger.error(f"注册指标失败: {str(e)}")
            raise
    
    def _compute_features(self, feature_engineer: FeatureEngineer, data: pd.DataFrame) -> pd.DataFrame:
        """计算特征"""
        try:
            # 使用特征工程器计算所有特征
            features_data = feature_engineer.compute_all_features(
                data=data,
                **self.feature_config
            )
            
            return features_data
            
        except Exception as e:
            self.logger.error(f"特征计算失败: {str(e)}")
            raise
    
    def _validate_features(self, features_data: pd.DataFrame) -> None:
        """验证特征"""
        if features_data is None or features_data.empty:
            raise AirflowException("特征数据为空")
        
        # 检查是否有无穷大或NaN值
        inf_count = np.isinf(features_data.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            self.logger.warning(f"发现 {inf_count} 个无穷大值")
        
        nan_count = features_data.isnull().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"发现 {nan_count} 个NaN值")
        
        # 检查特征数量
        original_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in features_data.columns if col not in original_columns]
        
        if len(feature_columns) == 0:
            raise AirflowException("未生成任何特征")
        
        self.logger.info(f"特征验证通过: {len(feature_columns)} 个特征，{len(features_data)} 条记录")
    
    def _save_features(self, features_data: pd.DataFrame, output_file: str) -> None:
        """保存特征数据"""
        try:
            if self.file_format.lower() == 'csv':
                features_data.to_csv(output_file, index=False)
            elif self.file_format.lower() == 'parquet':
                features_data.to_parquet(output_file, index=False)
            else:
                raise AirflowException(f"不支持的输出格式: {self.file_format}")
            
            self.logger.info(f"特征数据已保存到: {output_file}")
            
        except Exception as e:
            self.logger.error(f"保存特征数据失败: {str(e)}")
            raise
    
    def _save_feature_metadata(self, features_data: pd.DataFrame, output_file: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """保存特征元数据"""
        try:
            # 生成元数据
            original_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in features_data.columns if col not in original_columns]
            
            metadata = {
                'symbol': self.symbol,
                'interval': self.interval,
                'generation_timestamp': datetime.now().isoformat(),
                'execution_date': context['execution_date'].isoformat(),
                'data_records': len(features_data),
                'original_columns': original_columns,
                'feature_columns': feature_columns,
                'feature_count': len(feature_columns),
                'feature_statistics': {},
                'data_range': {
                    'start_time': features_data['open_time'].min().isoformat() if 'open_time' in features_data.columns else None,
                    'end_time': features_data['open_time'].max().isoformat() if 'open_time' in features_data.columns else None
                },
                'feature_config': self.feature_config,
                'indicators_used': self.indicators
            }
            
            # 计算特征统计信息
            for col in feature_columns:
                if pd.api.types.is_numeric_dtype(features_data[col]):
                    metadata['feature_statistics'][col] = {
                        'mean': float(features_data[col].mean()),
                        'std': float(features_data[col].std()),
                        'min': float(features_data[col].min()),
                        'max': float(features_data[col].max()),
                        'null_count': int(features_data[col].isnull().sum()),
                        'null_percentage': float(features_data[col].isnull().sum() / len(features_data) * 100)
                    }
            
            # 保存元数据文件
            metadata_file = output_file.replace(f'.{self.file_format}', '_metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"特征元数据已保存到: {metadata_file}")
            
            return {
                'metadata_file': metadata_file,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"保存特征元数据失败: {str(e)}")
            raise
    
    def _generate_feature_report(self, original_data: pd.DataFrame, features_data: pd.DataFrame) -> Dict[str, Any]:
        """生成特征报告"""
        original_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in features_data.columns if col not in original_columns]
        
        report = {
            'original_records': len(original_data),
            'feature_records': len(features_data),
            'original_columns': len(original_columns),
            'feature_columns': len(feature_columns),
            'total_columns': len(features_data.columns),
            'feature_categories': self._categorize_features(feature_columns),
            'data_quality': {
                'null_values': features_data.isnull().sum().sum(),
                'inf_values': np.isinf(features_data.select_dtypes(include=[np.number])).sum().sum(),
                'duplicate_rows': features_data.duplicated().sum()
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _categorize_features(self, feature_columns: List[str]) -> Dict[str, List[str]]:
        """对特征进行分类"""
        categories = {
            'moving_averages': [],
            'volatility': [],
            'volume': [],
            'price_based': [],
            'momentum': [],
            'trend': [],
            'other': []
        }
        
        for col in feature_columns:
            col_lower = col.lower()
            if 'ma_' in col_lower or 'sma_' in col_lower or 'ema_' in col_lower:
                categories['moving_averages'].append(col)
            elif 'volatility' in col_lower or 'std_' in col_lower or 'atr' in col_lower:
                categories['volatility'].append(col)
            elif 'volume' in col_lower or 'vol_' in col_lower:
                categories['volume'].append(col)
            elif 'price' in col_lower or 'return' in col_lower or 'change' in col_lower:
                categories['price_based'].append(col)
            elif 'rsi' in col_lower or 'momentum' in col_lower or 'macd' in col_lower:
                categories['momentum'].append(col)
            elif 'trend' in col_lower or 'direction' in col_lower:
                categories['trend'].append(col)
            else:
                categories['other'].append(col)
        
        return categories


class MultiSymbolFeatureEngineeringOperator(BaseOperator):
    """
    多交易对特征工程操作符
    
    批量处理多个交易对的特征工程
    """
    
    template_fields = ['symbols', 'intervals']
    
    @apply_defaults
    def __init__(
        self,
        symbols: List[str],
        intervals: List[str],
        feature_config: Optional[Dict[str, Any]] = None,
        indicators: Optional[List[str]] = None,
        file_format: str = 'csv',
        parallel_processing: bool = False,
        max_workers: int = 4,
        *args,
        **kwargs
    ):
        """
        初始化多交易对特征工程操作符
        
        Args:
            symbols: 交易对符号列表
            intervals: 时间间隔列表
            feature_config: 特征配置
            indicators: 指定要计算的指标列表
            file_format: 文件格式
            parallel_processing: 是否并行处理
            max_workers: 最大工作线程数
        """
        super().__init__(*args, **kwargs)
        self.symbols = symbols
        self.intervals = intervals
        self.feature_config = feature_config or {}
        self.indicators = indicators
        self.file_format = file_format
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行多交易对特征工程任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 任务执行结果
        """
        self.logger.info(f"开始批量特征工程: {len(self.symbols)} 个交易对, {len(self.intervals)} 个时间间隔")
        
        try:
            results = {}
            
            if self.parallel_processing:
                results = self._process_parallel(context)
            else:
                results = self._process_sequential(context)
            
            # 生成汇总报告
            summary_report = self._generate_summary_report(results)
            
            # 准备返回结果
            final_result = {
                'processed_count': len(results),
                'total_expected': len(self.symbols) * len(self.intervals),
                'success_rate': len(results) / (len(self.symbols) * len(self.intervals)) * 100,
                'results': results,
                'summary_report': summary_report,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"批量特征工程完成: {len(results)}/{len(self.symbols) * len(self.intervals)} 个任务成功")
            
            # 将结果推送到XCom
            context['task_instance'].xcom_push(key='batch_feature_result', value=final_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"批量特征工程失败: {str(e)}")
            raise AirflowException(f"批量特征工程失败: {str(e)}")
    
    def _process_sequential(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """顺序处理"""
        results = {}
        
        for symbol in self.symbols:
            for interval in self.intervals:
                try:
                    # 创建单个特征工程操作符
                    operator = FeatureEngineeringOperator(
                        task_id=f'temp_{symbol.lower()}_{interval}',
                        symbol=symbol,
                        interval=interval,
                        feature_config=self.feature_config,
                        indicators=self.indicators,
                        file_format=self.file_format
                    )
                    
                    # 执行特征工程
                    result = operator.execute(context)
                    results[f"{symbol}_{interval}"] = result
                    
                    self.logger.info(f"完成 {symbol} {interval} 特征工程")
                    
                except Exception as e:
                    self.logger.error(f"处理 {symbol} {interval} 失败: {str(e)}")
                    results[f"{symbol}_{interval}"] = {'error': str(e)}
        
        return results
    
    def _process_parallel(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """并行处理"""
        import concurrent.futures
        
        results = {}
        
        def process_single(symbol: str, interval: str) -> tuple:
            try:
                operator = FeatureEngineeringOperator(
                    task_id=f'temp_{symbol.lower()}_{interval}',
                    symbol=symbol,
                    interval=interval,
                    feature_config=self.feature_config,
                    indicators=self.indicators,
                    file_format=self.file_format
                )
                
                result = operator.execute(context)
                return f"{symbol}_{interval}", result
                
            except Exception as e:
                return f"{symbol}_{interval}", {'error': str(e)}
        
        # 创建任务列表
        tasks = [(symbol, interval) for symbol in self.symbols for interval in self.intervals]
        
        # 并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(process_single, symbol, interval): (symbol, interval)
                for symbol, interval in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                symbol, interval = future_to_task[future]
                try:
                    key, result = future.result()
                    results[key] = result
                    self.logger.info(f"完成 {symbol} {interval} 特征工程")
                except Exception as e:
                    self.logger.error(f"处理 {symbol} {interval} 失败: {str(e)}")
                    results[f"{symbol}_{interval}"] = {'error': str(e)}
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成汇总报告"""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        failed_results = {k: v for k, v in results.items() if 'error' in v}
        
        summary = {
            'total_tasks': len(results),
            'successful_tasks': len(successful_results),
            'failed_tasks': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100 if results else 0,
            'total_features_generated': 0,
            'total_records_processed': 0,
            'average_feature_count': 0,
            'failed_tasks_details': list(failed_results.keys()) if failed_results else []
        }
        
        if successful_results:
            feature_counts = [r.get('feature_count', 0) for r in successful_results.values()]
            record_counts = [r.get('feature_records', 0) for r in successful_results.values()]
            
            summary['total_features_generated'] = sum(feature_counts)
            summary['total_records_processed'] = sum(record_counts)
            summary['average_feature_count'] = sum(feature_counts) / len(feature_counts)
        
        return summary