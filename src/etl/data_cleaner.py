"""
数据清洗模块
用于处理金融时序数据的缺失值、异常值和时序对齐
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
from scipy import stats
import warnings

# 导入数据完整性检查模块
from .data_integrity import DataIntegrityChecker

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, 
                 price_columns: List[str] = None,
                 volume_columns: List[str] = None):
        """
        初始化数据清洗器
        
        Args:
            price_columns: 价格相关列名
            volume_columns: 成交量相关列名
        """
        self.price_columns = price_columns or ['open', 'high', 'low', 'close']
        self.volume_columns = volume_columns or ['volume', 'quote_asset_volume']
        
        logger.info("数据清洗器初始化完成")
    
    def detect_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测缺失值
        
        Args:
            data: 输入数据
            
        Returns:
            缺失值统计信息
        """
        logger.info("开始检测缺失值...")
        
        missing_info = {
            'total_rows': len(data),
            'missing_by_column': {},
            'missing_percentage': {},
            'rows_with_missing': 0,
            'complete_rows': 0
        }
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            missing_info['missing_by_column'][column] = missing_count
            missing_info['missing_percentage'][column] = (missing_count / len(data)) * 100
        
        # 统计有缺失值的行数
        missing_info['rows_with_missing'] = data.isnull().any(axis=1).sum()
        missing_info['complete_rows'] = len(data) - missing_info['rows_with_missing']
        
        logger.info(f"缺失值检测完成: {missing_info['rows_with_missing']} 行有缺失值")
        
        return missing_info
    
    def fill_missing_values(self, 
                           data: pd.DataFrame,
                           method: str = 'smart',
                           forward_limit: int = 3,
                           backward_limit: int = 1) -> pd.DataFrame:
        """
        填充缺失值
        
        Args:
            data: 输入数据
            method: 填充方法 ('forward', 'backward', 'interpolate', 'smart')
            forward_limit: 前向填充的最大连续缺失值数量
            backward_limit: 后向填充的最大连续缺失值数量
            
        Returns:
            填充后的数据
        """
        logger.info(f"开始使用 {method} 方法填充缺失值...")
        
        data_filled = data.copy()
        
        if method == 'forward':
            data_filled = data_filled.fillna(method='ffill', limit=forward_limit)
            
        elif method == 'backward':
            data_filled = data_filled.fillna(method='bfill', limit=backward_limit)
            
        elif method == 'interpolate':
            # 对价格数据使用线性插值
            for col in self.price_columns:
                if col in data_filled.columns:
                    data_filled[col] = data_filled[col].interpolate(method='linear')
            
            # 对成交量数据使用前向填充
            for col in self.volume_columns:
                if col in data_filled.columns:
                    data_filled[col] = data_filled[col].fillna(method='ffill')
                    
        elif method == 'smart':
            # 智能填充：结合多种方法
            data_filled = self._smart_fill(data_filled, forward_limit, backward_limit)
            
        else:
            raise ValueError(f"不支持的填充方法: {method}")
        
        # 统计填充效果
        original_missing = data.isnull().sum().sum()
        remaining_missing = data_filled.isnull().sum().sum()
        filled_count = original_missing - remaining_missing
        
        logger.info(f"填充完成: 原有 {original_missing} 个缺失值，填充了 {filled_count} 个，剩余 {remaining_missing} 个")
        
        return data_filled
    
    def _smart_fill(self, 
                   data: pd.DataFrame, 
                   forward_limit: int, 
                   backward_limit: int) -> pd.DataFrame:
        """
        智能填充策略
        
        Args:
            data: 输入数据
            forward_limit: 前向填充限制
            backward_limit: 后向填充限制
            
        Returns:
            填充后的数据
        """
        data_filled = data.copy()
        
        # 1. 对价格数据：先线性插值，再前向填充
        for col in self.price_columns:
            if col in data_filled.columns:
                # 线性插值
                data_filled[col] = data_filled[col].interpolate(method='linear')
                # 前向填充剩余缺失值
                data_filled[col] = data_filled[col].fillna(method='ffill', limit=forward_limit)
                # 后向填充开头的缺失值
                data_filled[col] = data_filled[col].fillna(method='bfill', limit=backward_limit)
        
        # 2. 对成交量数据：前向填充
        for col in self.volume_columns:
            if col in data_filled.columns:
                data_filled[col] = data_filled[col].fillna(method='ffill', limit=forward_limit)
                # 如果还有缺失值，用0填充（成交量可以为0）
                data_filled[col] = data_filled[col].fillna(0)
        
        # 3. 对其他数值列：前向填充
        numeric_cols = data_filled.select_dtypes(include=[np.number]).columns
        other_cols = [col for col in numeric_cols 
                     if col not in self.price_columns + self.volume_columns]
        
        for col in other_cols:
            data_filled[col] = data_filled[col].fillna(method='ffill', limit=forward_limit)
            data_filled[col] = data_filled[col].fillna(method='bfill', limit=backward_limit)
        
        return data_filled
    
    def detect_outliers(self, 
                       data: pd.DataFrame,
                       method: str = 'iqr',
                       columns: List[str] = None,
                       z_threshold: float = 3.0,
                       iqr_multiplier: float = 1.5) -> Dict[str, Any]:
        """
        检测异常值
        
        Args:
            data: 输入数据
            method: 检测方法 ('z_score', 'iqr', 'business_rules')
            columns: 要检测的列名，默认为价格列
            z_threshold: Z分数阈值
            iqr_multiplier: IQR倍数
            
        Returns:
            异常值检测结果
        """
        logger.info(f"开始使用 {method} 方法检测异常值...")
        
        if columns is None:
            columns = self.price_columns
        
        outlier_info = {
            'method': method,
            'outliers_by_column': {},
            'outlier_indices': set(),
            'total_outliers': 0
        }
        
        for column in columns:
            if column not in data.columns:
                continue
                
            column_data = data[column].dropna()
            
            if method == 'z_score':
                outliers = self._detect_outliers_zscore(column_data, z_threshold)
            elif method == 'iqr':
                outliers = self._detect_outliers_iqr(column_data, iqr_multiplier)
            elif method == 'business_rules':
                outliers = self._detect_outliers_business_rules(data, column)
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")
            
            outlier_info['outliers_by_column'][column] = outliers
            outlier_info['outlier_indices'].update(outliers)
        
        outlier_info['total_outliers'] = len(outlier_info['outlier_indices'])
        
        logger.info(f"异常值检测完成: 发现 {outlier_info['total_outliers']} 个异常值")
        
        return outlier_info
    
    def _detect_outliers_zscore(self, data: pd.Series, threshold: float) -> List[int]:
        """使用Z分数检测异常值"""
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > threshold].index.tolist()
        return outliers
    
    def _detect_outliers_iqr(self, data: pd.Series, multiplier: float) -> List[int]:
        """使用IQR方法检测异常值"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)].index.tolist()
        return outliers
    
    def _detect_outliers_business_rules(self, data: pd.DataFrame, column: str) -> List[int]:
        """使用业务规则检测异常值"""
        outliers = []
        
        if column in self.price_columns:
            # 价格异常规则
            # 1. 价格为负数或零
            negative_prices = data[data[column] <= 0].index.tolist()
            outliers.extend(negative_prices)
            
            # 2. 价格关系异常 (high < low, close > high, close < low等)
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # high应该是最高价
                invalid_high = data[
                    (data['high'] < data['low']) | 
                    (data['high'] < data['open']) | 
                    (data['high'] < data['close'])
                ].index.tolist()
                outliers.extend(invalid_high)
                
                # low应该是最低价
                invalid_low = data[
                    (data['low'] > data['high']) | 
                    (data['low'] > data['open']) | 
                    (data['low'] > data['close'])
                ].index.tolist()
                outliers.extend(invalid_low)
            
            # 3. 价格变动过大（超过20%）
            if len(data) > 1:
                price_change = data[column].pct_change().abs()
                large_changes = data[price_change > 0.2].index.tolist()
                outliers.extend(large_changes)
        
        elif column in self.volume_columns:
            # 成交量异常规则
            # 1. 成交量为负数
            negative_volume = data[data[column] < 0].index.tolist()
            outliers.extend(negative_volume)
            
            # 2. 成交量异常大（超过平均值的10倍）
            mean_volume = data[column].mean()
            large_volume = data[data[column] > mean_volume * 10].index.tolist()
            outliers.extend(large_volume)
        
        return list(set(outliers))  # 去重
    
    def handle_outliers(self, 
                       data: pd.DataFrame,
                       outlier_info: Dict[str, Any],
                       method: str = 'clip') -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            data: 输入数据
            outlier_info: 异常值检测结果
            method: 处理方法 ('remove', 'clip', 'interpolate')
            
        Returns:
            处理后的数据
        """
        logger.info(f"开始使用 {method} 方法处理异常值...")
        
        data_cleaned = data.copy()
        
        if method == 'remove':
            # 删除异常值行
            outlier_indices = list(outlier_info['outlier_indices'])
            data_cleaned = data_cleaned.drop(outlier_indices)
            
        elif method == 'clip':
            # 截断异常值
            for column, outliers in outlier_info['outliers_by_column'].items():
                if column in data_cleaned.columns and outliers:
                    # 计算正常值的范围
                    normal_data = data_cleaned[column].drop(outliers)
                    lower_bound = normal_data.quantile(0.05)
                    upper_bound = normal_data.quantile(0.95)
                    
                    # 截断异常值
                    data_cleaned[column] = data_cleaned[column].clip(lower_bound, upper_bound)
                    
        elif method == 'interpolate':
            # 用插值替换异常值
            for column, outliers in outlier_info['outliers_by_column'].items():
                if column in data_cleaned.columns and outliers:
                    # 将异常值设为NaN
                    data_cleaned.loc[outliers, column] = np.nan
                    # 插值填充
                    data_cleaned[column] = data_cleaned[column].interpolate(method='linear')
                    
        else:
            raise ValueError(f"不支持的异常值处理方法: {method}")
        
        logger.info(f"异常值处理完成，使用方法: {method}")
        
        return data_cleaned
    
    def align_time_series(self, 
                         data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                         freq: str = '1H',
                         method: str = 'outer',
                         fill_method: str = 'forward') -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        时序数据对齐
        
        Args:
            data: 单个DataFrame或多个DataFrame的字典
            freq: 目标频率 ('1H', '1D', '1T'等)
            method: 对齐方法 ('outer', 'inner', 'left', 'right')
            fill_method: 缺失值填充方法
            
        Returns:
            对齐后的数据
        """
        logger.info(f"开始时序数据对齐，目标频率: {freq}")
        
        if isinstance(data, pd.DataFrame):
            return self._align_single_dataframe(data, freq, fill_method)
        elif isinstance(data, dict):
            return self._align_multiple_dataframes(data, freq, method, fill_method)
        else:
            raise ValueError("数据类型必须是DataFrame或DataFrame字典")
    
    def _align_single_dataframe(self, 
                               data: pd.DataFrame, 
                               freq: str, 
                               fill_method: str) -> pd.DataFrame:
        """对齐单个DataFrame的时序"""
        # 确保索引是时间类型
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是DatetimeIndex类型")
        
        # 创建完整的时间范围
        start_time = data.index.min()
        end_time = data.index.max()
        complete_index = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # 重新索引
        aligned_data = data.reindex(complete_index)
        
        # 填充缺失值
        if fill_method == 'forward':
            aligned_data = aligned_data.fillna(method='ffill')
        elif fill_method == 'backward':
            aligned_data = aligned_data.fillna(method='bfill')
        elif fill_method == 'interpolate':
            aligned_data = aligned_data.interpolate(method='linear')
        
        logger.info(f"单个数据对齐完成: {len(data)} -> {len(aligned_data)} 行")
        
        return aligned_data
    
    def _align_multiple_dataframes(self, 
                                  data_dict: Dict[str, pd.DataFrame], 
                                  freq: str, 
                                  method: str, 
                                  fill_method: str) -> Dict[str, pd.DataFrame]:
        """对齐多个DataFrame的时序"""
        if not data_dict:
            return {}
        
        # 找到所有时间范围
        all_indices = []
        for df in data_dict.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_indices.extend(df.index.tolist())
        
        if not all_indices:
            raise ValueError("所有DataFrame的索引都必须是DatetimeIndex类型")
        
        # 创建统一的时间范围
        start_time = min(all_indices)
        end_time = max(all_indices)
        complete_index = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # 对齐所有DataFrame
        aligned_dict = {}
        for symbol, df in data_dict.items():
            aligned_df = df.reindex(complete_index)
            
            # 填充缺失值
            if fill_method == 'forward':
                aligned_df = aligned_df.fillna(method='ffill')
            elif fill_method == 'backward':
                aligned_df = aligned_df.fillna(method='bfill')
            elif fill_method == 'interpolate':
                aligned_df = aligned_df.interpolate(method='linear')
            
            aligned_dict[symbol] = aligned_df
        
        logger.info(f"多个数据对齐完成: {len(data_dict)} 个数据集")
        
        return aligned_dict
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        数据质量验证
        
        Args:
            data: 输入数据
            
        Returns:
            数据质量报告
        """
        logger.info("开始数据质量验证...")
        
        quality_report = {
            'basic_info': {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'memory_usage': data.memory_usage(deep=True).sum(),
                'date_range': {
                    'start': data.index.min() if isinstance(data.index, pd.DatetimeIndex) else None,
                    'end': data.index.max() if isinstance(data.index, pd.DatetimeIndex) else None
                }
            },
            'missing_values': self.detect_missing_values(data),
            'data_types': data.dtypes.to_dict(),
            'duplicates': data.duplicated().sum(),
            'price_validation': {},
            'volume_validation': {}
        }
        
        # 价格数据验证
        for col in self.price_columns:
            if col in data.columns:
                quality_report['price_validation'][col] = {
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'negative_values': (data[col] <= 0).sum(),
                    'infinite_values': np.isinf(data[col]).sum()
                }
        
        # 成交量数据验证
        for col in self.volume_columns:
            if col in data.columns:
                quality_report['volume_validation'][col] = {
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'mean': data[col].mean(),
                    'zero_values': (data[col] == 0).sum(),
                    'negative_values': (data[col] < 0).sum()
                }
        
        # OHLC关系验证
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            quality_report['ohlc_validation'] = {
                'high_lt_low': (data['high'] < data['low']).sum(),
                'high_lt_open': (data['high'] < data['open']).sum(),
                'high_lt_close': (data['high'] < data['close']).sum(),
                'low_gt_open': (data['low'] > data['open']).sum(),
                'low_gt_close': (data['low'] > data['close']).sum()
            }
        
        logger.info("数据质量验证完成")
        
        return quality_report
    
    def clean_data_pipeline(self, 
                           data: pd.DataFrame,
                           missing_method: str = 'smart',
                           outlier_detection: str = 'iqr',
                           outlier_handling: str = 'clip',
                           align_freq: Optional[str] = None,
                           enable_integrity_check: bool = True,
                           price_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        完整的数据清洗流水线
        
        Args:
            data: 输入数据
            missing_method: 缺失值处理方法
            outlier_detection: 异常值检测方法
            outlier_handling: 异常值处理方法
            align_freq: 时序对齐频率
            enable_integrity_check: 是否启用数据完整性检查
            price_columns: 价格列名列表，用于完整性检查
            
        Returns:
            清洗后的数据和处理报告
        """
        logger.info("开始执行完整数据清洗流水线...")
        
        # 初始化报告
        cleaning_report = {
            'original_shape': data.shape,
            'steps_performed': [],
            'quality_before': self.validate_data_quality(data),
            'quality_after': None
        }
        
        cleaned_data = data.copy()
        
        # 1. 缺失值处理
        if cleaned_data.isnull().any().any():
            logger.info("步骤1: 处理缺失值")
            cleaned_data = self.fill_missing_values(cleaned_data, method=missing_method)
            cleaning_report['steps_performed'].append('missing_values_filled')
        
        # 2. 异常值检测和处理
        logger.info("步骤2: 检测和处理异常值")
        outlier_info = self.detect_outliers(cleaned_data, method=outlier_detection)
        if outlier_info['total_outliers'] > 0:
            cleaned_data = self.handle_outliers(cleaned_data, outlier_info, method=outlier_handling)
            cleaning_report['steps_performed'].append('outliers_handled')
            cleaning_report['outliers_detected'] = outlier_info['total_outliers']
        
        # 3. 时序对齐
        if align_freq:
            logger.info("步骤3: 时序对齐")
            cleaned_data = self.align_time_series(cleaned_data, freq=align_freq)
            cleaning_report['steps_performed'].append('time_series_aligned')
        
        # 4. 数据完整性检查
        if enable_integrity_check:
            logger.info("步骤4: 数据完整性检查")
            try:
                # 初始化完整性检查器
                if price_columns is None:
                    price_columns = ['open', 'high', 'low', 'close']
                
                integrity_checker = DataIntegrityChecker(price_columns=price_columns)
                
                # 执行完整性检查
                integrity_result = integrity_checker.comprehensive_integrity_check(cleaned_data)
                
                # 将完整性检查结果添加到报告中
                cleaning_report['integrity_check'] = {
                    'overall_score': integrity_result['overall_score'],
                    'overall_status': integrity_result['overall_status'],
                    'date_continuity': integrity_result['checks']['date_continuity'],
                    'ohlc_logic': integrity_result['checks']['ohlc_logic'],
                    'volume_logic': integrity_result['checks']['volume_logic'],
                    'lookback_consistency': integrity_result['checks']['lookback_consistency']
                }
                
                cleaning_report['steps_performed'].append('integrity_check_completed')
                
                # 如果完整性检查失败，记录警告
                if integrity_result['overall_status'] == 'FAIL':
                    logger.warning(f"数据完整性检查失败，总体评分: {integrity_result['overall_score']:.2f}%")
                else:
                    logger.info(f"数据完整性检查通过，总体评分: {integrity_result['overall_score']:.2f}%")
                    
            except Exception as e:
                logger.error(f"数据完整性检查失败: {e}")
                cleaning_report['integrity_check'] = {'error': str(e)}
        
        # 5. 最终质量验证
        cleaning_report['quality_after'] = self.validate_data_quality(cleaned_data)
        cleaning_report['final_shape'] = cleaned_data.shape
        
        logger.info(f"数据清洗流水线完成: {data.shape} -> {cleaned_data.shape}")
        
        return cleaned_data, cleaning_report
    
    def generate_cleaning_summary(self, cleaning_report: Dict[str, Any]) -> str:
        """
        生成数据清洗摘要报告
        
        Args:
            cleaning_report: 清洗报告
            
        Returns:
            格式化的摘要文本
        """
        summary = []
        summary.append("=" * 50)
        summary.append("数据清洗摘要报告")
        summary.append("=" * 50)
        
        # 基本信息
        summary.append(f"原始数据形状: {cleaning_report['original_shape']}")
        summary.append(f"清洗后形状: {cleaning_report['final_shape']}")
        summary.append(f"执行的步骤: {', '.join(cleaning_report['steps_performed'])}")
        
        # 缺失值信息
        before_missing = cleaning_report['quality_before']['missing_values']['rows_with_missing']
        after_missing = cleaning_report['quality_after']['missing_values']['rows_with_missing']
        summary.append(f"缺失值行数: {before_missing} -> {after_missing}")
        
        # 异常值信息
        if 'outliers_detected' in cleaning_report:
            summary.append(f"检测到异常值: {cleaning_report['outliers_detected']} 个")
        
        # 数据质量改善
        summary.append("\n数据质量改善:")
        before_complete = cleaning_report['quality_before']['missing_values']['complete_rows']
        after_complete = cleaning_report['quality_after']['missing_values']['complete_rows']
        improvement = after_complete - before_complete
        summary.append(f"完整数据行数增加: {improvement}")
        
        # 完整性检查结果
        if 'integrity_check' in cleaning_report:
            summary.append("\n数据完整性检查:")
            integrity = cleaning_report['integrity_check']
            if 'error' in integrity:
                summary.append(f"检查状态: 错误 - {integrity['error']}")
            else:
                summary.append(f"总体评分: {integrity['overall_score']:.2f}%")
                summary.append(f"总体状态: {integrity['overall_status']}")
                summary.append(f"日期连续性: {'通过' if integrity['date_continuity']['is_continuous'] else '失败'}")
                summary.append(f"OHLC逻辑: {'通过' if integrity['ohlc_logic']['is_valid'] else '失败'}")
                summary.append(f"成交量逻辑: {'通过' if integrity['volume_logic']['is_valid'] else '失败'}")
                summary.append(f"回溯一致性: {'通过' if integrity['lookback_consistency']['is_consistent'] else '失败'}")
        
        summary.append("=" * 50)
        
        return "\n".join(summary)


# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    test_data = pd.DataFrame({
        'open': np.random.normal(50000, 1000, 100),
        'high': np.random.normal(51000, 1000, 100),
        'low': np.random.normal(49000, 1000, 100),
        'close': np.random.normal(50000, 1000, 100),
        'volume': np.random.normal(1000, 200, 100)
    }, index=dates)
    
    # 人为添加一些问题
    test_data.loc[test_data.index[10:15], 'close'] = np.nan  # 缺失值
    test_data.loc[test_data.index[20], 'high'] = 100000  # 异常值
    test_data.loc[test_data.index[30], 'volume'] = -100  # 负成交量
    
    # 创建清洗器
    cleaner = DataCleaner()
    
    # 执行完整清洗流水线
    cleaned_data, report = cleaner.clean_data_pipeline(
        test_data,
        missing_method='smart',
        outlier_detection='business_rules',
        outlier_handling='clip'
    )
    
    # 打印摘要
    print(cleaner.generate_cleaning_summary(report))