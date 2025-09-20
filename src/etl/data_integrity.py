"""
数据完整性检查模块
用于验证金融时序数据的完整性，包括日期连续性、OHLC逻辑、回溯一致性等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')


class DataIntegrityChecker:
    """数据完整性检查器"""
    
    def __init__(self, 
                 price_columns: List[str] = None,
                 volume_columns: List[str] = None,
                 datetime_column: str = None):
        """
        初始化数据完整性检查器
        
        Args:
            price_columns: 价格相关列名
            volume_columns: 成交量相关列名
            datetime_column: 时间列名（如果不是索引）
        """
        self.price_columns = price_columns or ['open', 'high', 'low', 'close']
        self.volume_columns = volume_columns or ['volume']
        self.datetime_column = datetime_column
        
        logger.info("数据完整性检查器初始化完成")
    
    def check_date_continuity(self, 
                             data: pd.DataFrame,
                             expected_freq: str = '1H',
                             tolerance: float = 0.1) -> Dict[str, Any]:
        """
        检查日期连续性
        
        Args:
            data: 输入数据（索引为时间或包含时间列）
            expected_freq: 期望的频率（如'1H', '1D', '1min'）
            tolerance: 容忍的缺失比例
            
        Returns:
            日期连续性检查结果
        """
        logger.info("开始检查日期连续性...")
        
        # 获取时间索引
        if self.datetime_column and self.datetime_column in data.columns:
            time_index = pd.to_datetime(data[self.datetime_column])
        else:
            time_index = data.index
            
        if not isinstance(time_index, pd.DatetimeIndex):
            time_index = pd.to_datetime(time_index)
        
        # 生成期望的完整时间序列
        start_time = time_index.min()
        end_time = time_index.max()
        expected_index = pd.date_range(start=start_time, end=end_time, freq=expected_freq)
        
        # 检查缺失的时间点
        missing_dates = expected_index.difference(time_index)
        duplicate_dates = time_index[time_index.duplicated()]
        
        # 计算统计信息
        total_expected = len(expected_index)
        total_actual = len(time_index)
        missing_count = len(missing_dates)
        duplicate_count = len(duplicate_dates)
        
        continuity_ratio = (total_actual - duplicate_count) / total_expected
        
        result = {
            'is_continuous': missing_count == 0 and duplicate_count == 0,
            'continuity_ratio': continuity_ratio,
            'total_expected_points': total_expected,
            'total_actual_points': total_actual,
            'missing_points': missing_count,
            'duplicate_points': duplicate_count,
            'missing_dates': missing_dates.tolist(),
            'duplicate_dates': duplicate_dates.tolist(),
            'passes_tolerance': continuity_ratio >= (1 - tolerance),
            'expected_frequency': expected_freq,
            'start_date': start_time,
            'end_date': end_time
        }
        
        logger.info(f"日期连续性检查完成: 连续性比例 {continuity_ratio:.2%}")
        
        return result
    
    def check_ohlc_logic(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检查OHLC价格逻辑
        
        Args:
            data: 包含OHLC数据的DataFrame
            
        Returns:
            OHLC逻辑检查结果
        """
        logger.info("开始检查OHLC价格逻辑...")
        
        required_columns = self.price_columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            return {
                'is_valid': False,
                'error': f"缺少必要的价格列: {missing_columns}",
                'violations': []
            }
        
        violations = []
        
        # 获取列名映射（假设标准顺序：open, high, low, close）
        if len(required_columns) >= 4:
            open_col, high_col, low_col, close_col = required_columns[:4]
        else:
            # 如果列数不足，尝试按名称匹配
            open_col = next((col for col in required_columns if 'open' in col.lower() or col.lower() == 'o'), None)
            high_col = next((col for col in required_columns if 'high' in col.lower() or col.lower() == 'h'), None)
            low_col = next((col for col in required_columns if 'low' in col.lower() or col.lower() == 'l'), None)
            close_col = next((col for col in required_columns if 'close' in col.lower() or col.lower() == 'c'), None)
        
        high_violations = []
        low_violations = []
        
        # 检查 high >= max(open, close) 和 low <= min(open, close)
        if high_col and open_col and close_col:
            high_violations = data[
                (data[high_col] < data[open_col]) | 
                (data[high_col] < data[close_col])
            ].index.tolist()
        
        if low_col and open_col and close_col:
            low_violations = data[
                (data[low_col] > data[open_col]) | 
                (data[low_col] > data[close_col])
            ].index.tolist()
        
        # 检查价格为负值
        negative_price_violations = {}
        for col in required_columns:
            negative_indices = data[data[col] < 0].index.tolist()
            if negative_indices:
                negative_price_violations[col] = negative_indices
        
        # 检查价格为零
        zero_price_violations = {}
        for col in required_columns:
            zero_indices = data[data[col] == 0].index.tolist()
            if zero_indices:
                zero_price_violations[col] = zero_indices
        
        # 检查异常的价格跳跃（超过50%的变化）
        price_jump_violations = []
        if len(data) > 1 and open_col and close_col:
            for i in range(1, len(data)):
                prev_close = data.iloc[i-1][close_col]
                curr_open = data.iloc[i][open_col]
                if prev_close > 0:
                    jump_ratio = abs(curr_open - prev_close) / prev_close
                    if jump_ratio > 0.5:  # 50%的跳跃
                        price_jump_violations.append(data.index[i])
        
        violations = {
            'high_violations': high_violations,
            'low_violations': low_violations,
            'negative_price_violations': negative_price_violations,
            'zero_price_violations': zero_price_violations,
            'price_jump_violations': price_jump_violations
        }
        
        total_violations = (len(high_violations) + len(low_violations) + 
                          sum(len(v) for v in negative_price_violations.values()) +
                          sum(len(v) for v in zero_price_violations.values()) +
                          len(price_jump_violations))
        
        result = {
            'is_valid': total_violations == 0,
            'total_violations': total_violations,
            'violation_ratio': total_violations / len(data) if len(data) > 0 else 0,
            'violations': violations,
            'summary': {
                'high_low_logic_errors': len(high_violations) + len(low_violations),
                'negative_price_errors': sum(len(v) for v in negative_price_violations.values()),
                'zero_price_errors': sum(len(v) for v in zero_price_violations.values()),
                'price_jump_errors': len(price_jump_violations)
            }
        }
        
        logger.info(f"OHLC逻辑检查完成: {total_violations} 个违规项")
        
        return result
    
    def check_volume_logic(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检查成交量逻辑
        
        Args:
            data: 包含成交量数据的DataFrame
            
        Returns:
            成交量逻辑检查结果
        """
        logger.info("开始检查成交量逻辑...")
        
        violations = {}
        
        for vol_col in self.volume_columns:
            if vol_col not in data.columns:
                continue
                
            # 检查负成交量
            negative_volume = data[data[vol_col] < 0].index.tolist()
            
            # 检查异常大的成交量（超过平均值的10倍）
            if len(data) > 10:
                mean_volume = data[vol_col].mean()
                std_volume = data[vol_col].std()
                threshold = mean_volume + 10 * std_volume
                abnormal_volume = data[data[vol_col] > threshold].index.tolist()
            else:
                abnormal_volume = []
            
            violations[vol_col] = {
                'negative_volume': negative_volume,
                'abnormal_volume': abnormal_volume
            }
        
        total_violations = sum(
            len(v['negative_volume']) + len(v['abnormal_volume']) 
            for v in violations.values()
        )
        
        result = {
            'is_valid': total_violations == 0,
            'total_violations': total_violations,
            'violations': violations
        }
        
        logger.info(f"成交量逻辑检查完成: {total_violations} 个违规项")
        
        return result
    
    def check_lookback_consistency(self, 
                                  data: pd.DataFrame,
                                  lookback_periods: List[int] = None) -> Dict[str, Any]:
        """
        检查回溯一致性（前后数据的关联性）
        
        Args:
            data: 输入数据
            lookback_periods: 要检查的回溯周期列表
            
        Returns:
            回溯一致性检查结果
        """
        logger.info("开始检查回溯一致性...")
        
        if lookback_periods is None:
            lookback_periods = [1, 5, 10, 24]  # 1小时、5小时、10小时、24小时
        
        consistency_results = {}
        
        # 获取列名映射
        if len(self.price_columns) >= 4:
            open_col, high_col, low_col, close_col = self.price_columns[:4]
        else:
            # 如果列数不足，尝试按名称匹配
            open_col = next((col for col in self.price_columns if 'open' in col.lower() or col.lower() == 'o'), None)
            close_col = next((col for col in self.price_columns if 'close' in col.lower() or col.lower() == 'c'), None)
        
        for period in lookback_periods:
            if len(data) <= period:
                continue
                
            # 检查价格的连续性（相邻收盘价和开盘价的关系）
            if period == 1 and close_col and open_col and close_col in data.columns and open_col in data.columns:
                # 检查收盘价和下一个开盘价的差异
                close_open_diff = []
                for i in range(len(data) - 1):
                    curr_close = data.iloc[i][close_col]
                    next_open = data.iloc[i + 1][open_col]
                    if curr_close > 0:
                        diff_ratio = abs(next_open - curr_close) / curr_close
                        close_open_diff.append(diff_ratio)
                
                # 检查是否有异常的跳跃（超过5%）
                abnormal_gaps = [i for i, diff in enumerate(close_open_diff) if diff > 0.05]
                
                consistency_results[f'period_{period}'] = {
                    'type': 'close_open_consistency',
                    'abnormal_gaps': abnormal_gaps,
                    'max_gap_ratio': max(close_open_diff) if close_open_diff else 0,
                    'avg_gap_ratio': np.mean(close_open_diff) if close_open_diff else 0
                }
            
            # 检查移动平均的单调性
            if close_col and close_col in data.columns:
                ma_short = data[close_col].rolling(window=min(period, 5)).mean()
                ma_long = data[close_col].rolling(window=period).mean()
                
                # 检查短期均线和长期均线的关系异常
                ma_violations = []
                for i in range(period, len(data)):
                    if not pd.isna(ma_short.iloc[i]) and not pd.isna(ma_long.iloc[i]):
                        # 检查是否有异常的均线交叉
                        short_val = ma_short.iloc[i]
                        long_val = ma_long.iloc[i]
                        if abs(short_val - long_val) / long_val > 0.1:  # 10%的差异
                            ma_violations.append(i)
                
                consistency_results[f'ma_consistency_{period}'] = {
                    'type': 'moving_average_consistency',
                    'violations': ma_violations,
                    'violation_count': len(ma_violations)
                }
        
        total_violations = sum(
            len(result.get('abnormal_gaps', [])) + len(result.get('violations', []))
            for result in consistency_results.values()
        )
        
        result = {
            'is_consistent': total_violations == 0,
            'total_violations': total_violations,
            'consistency_results': consistency_results,
            'checked_periods': lookback_periods
        }
        
        logger.info(f"回溯一致性检查完成: {total_violations} 个违规项")
        
        return result
    
    def comprehensive_integrity_check(self, 
                                    data: pd.DataFrame,
                                    expected_freq: str = '1H',
                                    tolerance: float = 0.1,
                                    lookback_periods: List[int] = None) -> Dict[str, Any]:
        """
        综合数据完整性检查
        
        Args:
            data: 输入数据
            expected_freq: 期望的时间频率
            tolerance: 容忍度
            lookback_periods: 回溯检查周期
            
        Returns:
            综合检查结果
        """
        logger.info("开始综合数据完整性检查...")
        
        results = {
            'timestamp': datetime.now(),
            'data_shape': data.shape,
            'checks': {}
        }
        
        # 1. 日期连续性检查
        try:
            results['checks']['date_continuity'] = self.check_date_continuity(
                data, expected_freq, tolerance
            )
        except Exception as e:
            results['checks']['date_continuity'] = {
                'error': str(e),
                'is_continuous': False
            }
        
        # 2. OHLC逻辑检查
        try:
            results['checks']['ohlc_logic'] = self.check_ohlc_logic(data)
        except Exception as e:
            results['checks']['ohlc_logic'] = {
                'error': str(e),
                'is_valid': False
            }
        
        # 3. 成交量逻辑检查
        try:
            results['checks']['volume_logic'] = self.check_volume_logic(data)
        except Exception as e:
            results['checks']['volume_logic'] = {
                'error': str(e),
                'is_valid': False
            }
        
        # 4. 回溯一致性检查
        try:
            results['checks']['lookback_consistency'] = self.check_lookback_consistency(
                data, lookback_periods
            )
        except Exception as e:
            results['checks']['lookback_consistency'] = {
                'error': str(e),
                'is_consistent': False
            }
        
        # 计算总体评分
        checks = results['checks']
        score_components = []
        
        if 'date_continuity' in checks and 'continuity_ratio' in checks['date_continuity']:
            score_components.append(checks['date_continuity']['continuity_ratio'])
        
        if 'ohlc_logic' in checks and 'violation_ratio' in checks['ohlc_logic']:
            score_components.append(1 - checks['ohlc_logic']['violation_ratio'])
        
        if 'volume_logic' in checks and checks['volume_logic'].get('is_valid'):
            score_components.append(1.0)
        elif 'volume_logic' in checks:
            score_components.append(0.5)
        
        if 'lookback_consistency' in checks and checks['lookback_consistency'].get('is_consistent'):
            score_components.append(1.0)
        elif 'lookback_consistency' in checks:
            score_components.append(0.5)
        
        overall_score = np.mean(score_components) if score_components else 0
        
        results['overall_score'] = overall_score
        results['overall_status'] = 'PASS' if overall_score >= 0.8 else 'FAIL'
        
        logger.info(f"综合数据完整性检查完成: 总体评分 {overall_score:.2%}")
        
        return results
    
    def generate_integrity_report(self, integrity_results: Dict[str, Any]) -> str:
        """
        生成数据完整性检查报告
        
        Args:
            integrity_results: 完整性检查结果
            
        Returns:
            格式化的报告字符串
        """
        report = []
        report.append("=" * 60)
        report.append("数据完整性检查报告")
        report.append("=" * 60)
        
        # 基本信息
        report.append(f"检查时间: {integrity_results['timestamp']}")
        report.append(f"数据形状: {integrity_results['data_shape']}")
        report.append(f"总体评分: {integrity_results['overall_score']:.2%}")
        report.append(f"总体状态: {integrity_results['overall_status']}")
        report.append("")
        
        # 详细检查结果
        checks = integrity_results['checks']
        
        # 日期连续性
        if 'date_continuity' in checks:
            dc = checks['date_continuity']
            report.append("1. 日期连续性检查")
            report.append("-" * 30)
            if 'error' in dc:
                report.append(f"   错误: {dc['error']}")
            else:
                report.append(f"   连续性: {'通过' if dc['is_continuous'] else '失败'}")
                report.append(f"   连续性比例: {dc['continuity_ratio']:.2%}")
                report.append(f"   缺失时间点: {dc['missing_points']}")
                report.append(f"   重复时间点: {dc['duplicate_points']}")
            report.append("")
        
        # OHLC逻辑
        if 'ohlc_logic' in checks:
            ohlc = checks['ohlc_logic']
            report.append("2. OHLC价格逻辑检查")
            report.append("-" * 30)
            if 'error' in ohlc:
                report.append(f"   错误: {ohlc['error']}")
            else:
                report.append(f"   逻辑正确性: {'通过' if ohlc['is_valid'] else '失败'}")
                report.append(f"   总违规项: {ohlc['total_violations']}")
                if 'summary' in ohlc:
                    summary = ohlc['summary']
                    report.append(f"   高低价逻辑错误: {summary['high_low_logic_errors']}")
                    report.append(f"   负价格错误: {summary['negative_price_errors']}")
                    report.append(f"   零价格错误: {summary['zero_price_errors']}")
                    report.append(f"   价格跳跃错误: {summary['price_jump_errors']}")
            report.append("")
        
        # 成交量逻辑
        if 'volume_logic' in checks:
            vol = checks['volume_logic']
            report.append("3. 成交量逻辑检查")
            report.append("-" * 30)
            if 'error' in vol:
                report.append(f"   错误: {vol['error']}")
            else:
                report.append(f"   逻辑正确性: {'通过' if vol['is_valid'] else '失败'}")
                report.append(f"   总违规项: {vol['total_violations']}")
            report.append("")
        
        # 回溯一致性
        if 'lookback_consistency' in checks:
            lc = checks['lookback_consistency']
            report.append("4. 回溯一致性检查")
            report.append("-" * 30)
            if 'error' in lc:
                report.append(f"   错误: {lc['error']}")
            else:
                report.append(f"   一致性: {'通过' if lc['is_consistent'] else '失败'}")
                report.append(f"   总违规项: {lc['total_violations']}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # 测试代码
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    test_data = pd.DataFrame({
        'open': np.random.normal(50000, 1000, 100),
        'high': np.random.normal(51000, 1000, 100),
        'low': np.random.normal(49000, 1000, 100),
        'close': np.random.normal(50000, 1000, 100),
        'volume': np.random.normal(1000, 200, 100)
    }, index=dates)
    
    # 确保OHLC逻辑正确
    test_data['high'] = test_data[['open', 'close']].max(axis=1) + np.random.normal(100, 50, 100)
    test_data['low'] = test_data[['open', 'close']].min(axis=1) - np.random.normal(100, 50, 100)
    
    # 添加一些问题数据进行测试
    test_data.loc[test_data.index[10], 'high'] = test_data.loc[test_data.index[10], 'low'] - 100  # 高价低于低价
    test_data.loc[test_data.index[20], 'volume'] = -100  # 负成交量
    
    checker = DataIntegrityChecker()
    results = checker.comprehensive_integrity_check(test_data)
    print(checker.generate_integrity_report(results))