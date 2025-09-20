"""
数据清洗操作符
集成现有的DataCleaner和DataIntegrityChecker模块到Airflow任务中
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.models import Variable
from airflow.exceptions import AirflowException

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.etl.data_cleaner import DataCleaner
from src.etl.data_storage import DataStorage
from src.etl.data_integrity import DataIntegrityChecker


class DataCleaningOperator(BaseOperator):
    """
    数据清洗操作符
    
    使用DataCleaner对原始数据进行清洗处理
    """
    
    template_fields = ['input_path', 'output_path', 'symbol', 'interval']
    
    @apply_defaults
    def __init__(
        self,
        symbol: str,
        interval: str,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        cleaning_config: Optional[Dict[str, Any]] = None,
        file_format: str = 'csv',
        validate_output: bool = True,
        *args,
        **kwargs
    ):
        """
        初始化数据清洗操作符
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            input_path: 输入数据路径
            output_path: 输出数据路径
            cleaning_config: 清洗配置
            file_format: 文件格式
            validate_output: 是否验证输出数据
        """
        super().__init__(*args, **kwargs)
        self.symbol = symbol
        self.interval = interval
        self.input_path = input_path
        self.output_path = output_path
        self.cleaning_config = cleaning_config or {}
        self.file_format = file_format
        self.validate_output = validate_output
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据清洗任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 任务执行结果
        """
        self.logger.info(f"开始清洗 {self.symbol} 的 {self.interval} 数据")
        
        try:
            # 获取输入和输出路径
            input_file = self._get_input_path(context)
            output_file = self._get_output_path(context)
            
            # 加载原始数据
            data = self._load_data(input_file)
            
            if data is None or data.empty:
                raise AirflowException(f"输入数据为空: {input_file}")
            
            self.logger.info(f"加载数据: {len(data)} 条记录")
            
            # 初始化数据清洗器
            cleaner = DataCleaner()
            
            # 执行数据清洗
            cleaned_data = self._clean_data(cleaner, data)
            
            # 验证清洗后的数据
            if self.validate_output:
                self._validate_cleaned_data(cleaned_data)
            
            # 保存清洗后的数据
            self._save_cleaned_data(cleaned_data, output_file)
            
            # 生成清洗报告
            cleaning_report = self._generate_cleaning_report(data, cleaned_data)
            
            # 准备返回结果
            result = {
                'symbol': self.symbol,
                'interval': self.interval,
                'input_file': input_file,
                'output_file': output_file,
                'original_records': len(data),
                'cleaned_records': len(cleaned_data),
                'cleaning_report': cleaning_report,
                'file_size_mb': round(os.path.getsize(output_file) / 1024 / 1024, 2)
            }
            
            self.logger.info(f"数据清洗完成: {len(data)} -> {len(cleaned_data)} 条记录")
            
            # 将结果推送到XCom
            context['task_instance'].xcom_push(key='cleaning_result', value=result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据清洗失败: {str(e)}")
            raise AirflowException(f"数据清洗失败: {str(e)}")
    
    def _get_input_path(self, context: Dict[str, Any]) -> str:
        """获取输入数据路径"""
        if self.input_path:
            return self.input_path
        
        # 从上游任务获取文件路径
        task_instance = context['task_instance']
        upstream_result = task_instance.xcom_pull(
            task_ids=f'fetch_data.{self.symbol.lower()}_{self.interval}_data',
            key='ingestion_result'
        )
        
        if upstream_result and 'file_path' in upstream_result:
            return upstream_result['file_path']
        
        # 构造默认路径
        raw_data_path = Variable.get('ml_raw_data_path', '/opt/airflow/data/raw')
        execution_date = context['execution_date']
        date_str = execution_date.strftime('%Y-%m-%d')
        
        return os.path.join(
            raw_data_path,
            self.symbol.lower(),
            self.interval,
            execution_date.strftime('%Y'),
            execution_date.strftime('%m'),
            f"{self.symbol.lower()}_{self.interval}_{date_str}.{self.file_format}"
        )
    
    def _get_output_path(self, context: Dict[str, Any]) -> str:
        """获取输出数据路径"""
        if self.output_path:
            return self.output_path
        
        processed_data_path = Variable.get('ml_processed_data_path', '/opt/airflow/data/processed')
        execution_date = context['execution_date']
        date_str = execution_date.strftime('%Y-%m-%d')
        
        output_file = os.path.join(
            processed_data_path,
            self.symbol.lower(),
            self.interval,
            execution_date.strftime('%Y'),
            execution_date.strftime('%m'),
            f"{self.symbol.lower()}_{self.interval}_{date_str}_cleaned.{self.file_format}"
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
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def _clean_data(self, cleaner: DataCleaner, data: pd.DataFrame) -> pd.DataFrame:
        """执行数据清洗"""
        try:
            # 使用完整的清洗流水线
            cleaned_data = cleaner.clean_data_pipeline(
                data=data,
                **self.cleaning_config
            )
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"数据清洗处理失败: {str(e)}")
            raise
    
    def _validate_cleaned_data(self, data: pd.DataFrame) -> None:
        """验证清洗后的数据"""
        if data is None or data.empty:
            raise AirflowException("清洗后数据为空")
        
        # 检查必需的列
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise AirflowException(f"清洗后数据缺少必需的列: {missing_columns}")
        
        # 检查数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise AirflowException(f"列 {col} 不是数值类型")
        
        self.logger.info(f"清洗后数据验证通过: {len(data)} 条记录")
    
    def _save_cleaned_data(self, data: pd.DataFrame, output_file: str) -> None:
        """保存清洗后的数据"""
        try:
            if self.file_format.lower() == 'csv':
                data.to_csv(output_file, index=False)
            elif self.file_format.lower() == 'parquet':
                data.to_parquet(output_file, index=False)
            else:
                raise AirflowException(f"不支持的输出格式: {self.file_format}")
            
            self.logger.info(f"清洗后数据已保存到: {output_file}")
            
        except Exception as e:
            self.logger.error(f"保存清洗后数据失败: {str(e)}")
            raise
    
    def _generate_cleaning_report(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> Dict[str, Any]:
        """生成清洗报告"""
        report = {
            'original_records': len(original_data),
            'cleaned_records': len(cleaned_data),
            'records_removed': len(original_data) - len(cleaned_data),
            'removal_percentage': (len(original_data) - len(cleaned_data)) / len(original_data) * 100,
            'missing_values_before': original_data.isnull().sum().to_dict(),
            'missing_values_after': cleaned_data.isnull().sum().to_dict(),
            'cleaning_timestamp': datetime.now().isoformat()
        }
        
        return report


class DataIntegrityCheckOperator(BaseOperator):
    """
    数据完整性检查操作符
    
    使用DataIntegrityChecker对数据进行完整性检查
    """
    
    template_fields = ['input_path', 'symbol', 'interval']
    
    @apply_defaults
    def __init__(
        self,
        symbol: str,
        interval: str,
        input_path: Optional[str] = None,
        check_config: Optional[Dict[str, Any]] = None,
        quality_threshold: float = 80.0,
        file_format: str = 'csv',
        *args,
        **kwargs
    ):
        """
        初始化数据完整性检查操作符
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            input_path: 输入数据路径
            check_config: 检查配置
            quality_threshold: 质量阈值
            file_format: 文件格式
        """
        super().__init__(*args, **kwargs)
        self.symbol = symbol
        self.interval = interval
        self.input_path = input_path
        self.check_config = check_config or {}
        self.quality_threshold = quality_threshold
        self.file_format = file_format
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据完整性检查任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 检查结果
        """
        self.logger.info(f"开始检查 {self.symbol} 的 {self.interval} 数据完整性")
        
        try:
            # 获取输入路径
            input_file = self._get_input_path(context)
            
            # 加载数据
            data = self._load_data(input_file)
            
            if data is None or data.empty:
                raise AirflowException(f"输入数据为空: {input_file}")
            
            # 初始化完整性检查器
            checker = DataIntegrityChecker()
            
            # 执行完整性检查
            integrity_result = checker.comprehensive_integrity_check(data)
            
            # 评估检查结果
            overall_score = integrity_result.get('overall_score', 0)
            
            # 检查是否满足质量阈值
            quality_passed = overall_score >= self.quality_threshold
            
            # 准备返回结果
            result = {
                'symbol': self.symbol,
                'interval': self.interval,
                'input_file': input_file,
                'data_records': len(data),
                'integrity_result': integrity_result,
                'overall_score': overall_score,
                'quality_threshold': self.quality_threshold,
                'quality_passed': quality_passed,
                'check_timestamp': datetime.now().isoformat()
            }
            
            # 记录检查结果
            if quality_passed:
                self.logger.info(f"数据完整性检查通过: 评分 {overall_score:.2f} >= 阈值 {self.quality_threshold}")
            else:
                self.logger.warning(f"数据完整性检查未通过: 评分 {overall_score:.2f} < 阈值 {self.quality_threshold}")
            
            # 将结果推送到XCom
            context['task_instance'].xcom_push(key='integrity_result', value=result)
            
            # 如果质量不达标，可以选择是否抛出异常
            if not quality_passed and self.check_config.get('fail_on_quality_threshold', False):
                raise AirflowException(f"数据质量不达标: {overall_score:.2f} < {self.quality_threshold}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据完整性检查失败: {str(e)}")
            raise AirflowException(f"数据完整性检查失败: {str(e)}")
    
    def _get_input_path(self, context: Dict[str, Any]) -> str:
        """获取输入数据路径"""
        if self.input_path:
            return self.input_path
        
        # 从上游清洗任务获取文件路径
        task_instance = context['task_instance']
        upstream_result = task_instance.xcom_pull(
            task_ids=f'clean_data.{self.symbol.lower()}_{self.interval}_cleaning',
            key='cleaning_result'
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
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            raise


class DataQualityReportOperator(BaseOperator):
    """
    数据质量报告操作符
    
    汇总数据清洗和完整性检查结果，生成质量报告
    """
    
    @apply_defaults
    def __init__(
        self,
        symbols: List[str],
        intervals: List[str],
        report_format: str = 'json',
        output_path: Optional[str] = None,
        *args,
        **kwargs
    ):
        """
        初始化数据质量报告操作符
        
        Args:
            symbols: 交易对符号列表
            intervals: 时间间隔列表
            report_format: 报告格式 ('json', 'html', 'csv')
            output_path: 输出路径
        """
        super().__init__(*args, **kwargs)
        self.symbols = symbols
        self.intervals = intervals
        self.report_format = report_format
        self.output_path = output_path
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据质量报告生成任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 报告结果
        """
        self.logger.info("开始生成数据质量报告")
        
        try:
            # 收集所有清洗和检查结果
            cleaning_results = self._collect_cleaning_results(context)
            integrity_results = self._collect_integrity_results(context)
            
            # 生成汇总报告
            quality_report = self._generate_quality_report(cleaning_results, integrity_results)
            
            # 保存报告
            if self.output_path:
                self._save_report(quality_report, context)
            
            # 将结果推送到XCom
            context['task_instance'].xcom_push(key='quality_report', value=quality_report)
            
            self.logger.info("数据质量报告生成完成")
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"生成数据质量报告失败: {str(e)}")
            raise AirflowException(f"生成数据质量报告失败: {str(e)}")
    
    def _collect_cleaning_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """收集清洗结果"""
        task_instance = context['task_instance']
        results = {}
        
        for symbol in self.symbols:
            for interval in self.intervals:
                task_id = f'clean_data.{symbol.lower()}_{interval}_cleaning'
                try:
                    result = task_instance.xcom_pull(task_ids=task_id, key='cleaning_result')
                    if result:
                        results[f"{symbol}_{interval}"] = result
                except Exception as e:
                    self.logger.warning(f"无法获取清洗结果 {task_id}: {str(e)}")
        
        return results
    
    def _collect_integrity_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """收集完整性检查结果"""
        task_instance = context['task_instance']
        results = {}
        
        for symbol in self.symbols:
            for interval in self.intervals:
                task_id = f'integrity_check.{symbol.lower()}_{interval}_integrity'
                try:
                    result = task_instance.xcom_pull(task_ids=task_id, key='integrity_result')
                    if result:
                        results[f"{symbol}_{interval}"] = result
                except Exception as e:
                    self.logger.warning(f"无法获取完整性检查结果 {task_id}: {str(e)}")
        
        return results
    
    def _generate_quality_report(self, cleaning_results: Dict, integrity_results: Dict) -> Dict[str, Any]:
        """生成质量报告"""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_datasets': len(self.symbols) * len(self.intervals),
                'cleaned_datasets': len(cleaning_results),
                'checked_datasets': len(integrity_results),
                'average_quality_score': 0.0,
                'quality_distribution': {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}
            },
            'cleaning_results': cleaning_results,
            'integrity_results': integrity_results,
            'recommendations': []
        }
        
        # 计算平均质量分数
        if integrity_results:
            scores = [result.get('overall_score', 0) for result in integrity_results.values()]
            report['summary']['average_quality_score'] = sum(scores) / len(scores)
            
            # 质量分布统计
            for score in scores:
                if score >= 95:
                    report['summary']['quality_distribution']['excellent'] += 1
                elif score >= 85:
                    report['summary']['quality_distribution']['good'] += 1
                elif score >= 75:
                    report['summary']['quality_distribution']['acceptable'] += 1
                else:
                    report['summary']['quality_distribution']['poor'] += 1
        
        # 生成建议
        report['recommendations'] = self._generate_recommendations(cleaning_results, integrity_results)
        
        return report
    
    def _generate_recommendations(self, cleaning_results: Dict, integrity_results: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于清洗结果的建议
        high_removal_rate = []
        for key, result in cleaning_results.items():
            removal_rate = result.get('cleaning_report', {}).get('removal_percentage', 0)
            if removal_rate > 10:  # 超过10%的数据被移除
                high_removal_rate.append(key)
        
        if high_removal_rate:
            recommendations.append(f"以下数据集的数据移除率较高，建议检查数据源质量: {', '.join(high_removal_rate)}")
        
        # 基于完整性检查的建议
        low_quality_datasets = []
        for key, result in integrity_results.items():
            score = result.get('overall_score', 0)
            if score < 80:
                low_quality_datasets.append(key)
        
        if low_quality_datasets:
            recommendations.append(f"以下数据集质量较低，建议进一步清洗: {', '.join(low_quality_datasets)}")
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any], context: Dict[str, Any]) -> None:
        """保存报告"""
        try:
            execution_date = context['execution_date']
            date_str = execution_date.strftime('%Y-%m-%d')
            
            if not self.output_path:
                base_path = Variable.get('ml_processed_data_path', '/opt/airflow/data/processed')
                output_file = os.path.join(base_path, 'reports', f'quality_report_{date_str}.{self.report_format}')
            else:
                output_file = self.output_path
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存报告
            if self.report_format == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
            elif self.report_format == 'csv':
                # 将报告转换为CSV格式（简化版）
                import pandas as pd
                summary_df = pd.DataFrame([report['summary']])
                summary_df.to_csv(output_file, index=False)
            
            self.logger.info(f"质量报告已保存到: {output_file}")
            
        except Exception as e:
            self.logger.error(f"保存质量报告失败: {str(e)}")
            raise