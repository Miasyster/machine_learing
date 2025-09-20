#!/usr/bin/env python3
"""
测试集成了数据完整性检查的数据清洗流水线
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from src.etl.data_cleaner import DataCleaner

def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建时间序列
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    
    # 创建基础价格数据
    np.random.seed(42)
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 创建OHLC数据
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # 生成合理的OHLC数据
        volatility = 0.01
        high = close_price * (1 + np.random.uniform(0, volatility))
        low = close_price * (1 - np.random.uniform(0, volatility))
        
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i-1]  # 前一天的收盘价
        
        # 确保价格关系正确
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # 故意引入一些问题来测试完整性检查
    # 1. 添加一些缺失值
    df.loc[df.index[10:15], 'volume'] = np.nan
    
    # 2. 添加一个价格逻辑错误（low > high）
    df.loc[df.index[20], 'low'] = df.loc[df.index[20], 'high'] * 1.1
    
    # 3. 添加负价格
    df.loc[df.index[30], 'open'] = -100
    
    print(f"测试数据创建完成，形状: {df.shape}")
    print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
    
    return df

def test_integrated_pipeline():
    """测试集成的数据清洗流水线"""
    print("=" * 60)
    print("测试集成了数据完整性检查的数据清洗流水线")
    print("=" * 60)
    
    # 创建测试数据
    test_data = create_test_data()
    
    print("\n原始数据概览:")
    print(f"形状: {test_data.shape}")
    print(f"缺失值: {test_data.isnull().sum().sum()}")
    print(f"负值数量: {(test_data < 0).sum().sum()}")
    
    # 初始化数据清洗器
    cleaner = DataCleaner()
    
    # 执行完整的数据清洗流水线（包含完整性检查）
    print("\n开始执行数据清洗流水线...")
    cleaned_data, report = cleaner.clean_data_pipeline(
        data=test_data,
        missing_method='forward',
        outlier_detection='iqr',
        outlier_handling='clip',
        enable_integrity_check=True,
        price_columns=['open', 'high', 'low', 'close']
    )
    
    print("\n清洗后数据概览:")
    print(f"形状: {cleaned_data.shape}")
    print(f"缺失值: {cleaned_data.isnull().sum().sum()}")
    print(f"负值数量: {(cleaned_data < 0).sum().sum()}")
    
    # 生成并显示清洗摘要
    summary = cleaner.generate_cleaning_summary(report)
    print("\n" + summary)
    
    # 详细显示完整性检查结果
    if 'integrity_check' in report:
        print("\n详细完整性检查结果:")
        integrity = report['integrity_check']
        if 'error' not in integrity:
            print(f"总体评分: {integrity['overall_score']:.2f}%")
            print(f"总体状态: {integrity['overall_status']}")
            
            print("\n各项检查详情:")
            print(f"1. 日期连续性: {integrity['date_continuity']}")
            print(f"2. OHLC逻辑: {integrity['ohlc_logic']}")
            print(f"3. 成交量逻辑: {integrity['volume_logic']}")
            print(f"4. 回溯一致性: {integrity['lookback_consistency']}")
    
    return cleaned_data, report

def test_custom_columns():
    """测试自定义列名的完整性检查"""
    print("\n" + "=" * 60)
    print("测试自定义列名的完整性检查")
    print("=" * 60)
    
    # 创建自定义列名的测试数据
    test_data = create_test_data()
    test_data = test_data.rename(columns={
        'open': 'price_open',
        'high': 'price_high', 
        'low': 'price_low',
        'close': 'price_close'
    })
    
    print(f"自定义列名数据: {list(test_data.columns)}")
    
    # 初始化数据清洗器
    cleaner = DataCleaner()
    
    # 使用自定义列名执行清洗流水线
    cleaned_data, report = cleaner.clean_data_pipeline(
        data=test_data,
        enable_integrity_check=True,
        price_columns=['price_open', 'price_high', 'price_low', 'price_close']
    )
    
    # 显示结果
    if 'integrity_check' in report:
        integrity = report['integrity_check']
        if 'error' not in integrity:
            print(f"自定义列名完整性检查 - 总体评分: {integrity['overall_score']:.2f}%")
            print(f"自定义列名完整性检查 - 总体状态: {integrity['overall_status']}")
        else:
            print(f"自定义列名完整性检查失败: {integrity['error']}")
    
    return cleaned_data, report

def main():
    """主测试函数"""
    print("🚀 开始测试集成数据完整性检查的数据清洗流水线")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 测试1: 标准流水线
        cleaned_data1, report1 = test_integrated_pipeline()
        
        # 测试2: 自定义列名
        cleaned_data2, report2 = test_custom_columns()
        
        print("\n" + "=" * 60)
        print("测试结果摘要")
        print("=" * 60)
        
        # 检查测试结果
        test1_success = 'integrity_check' in report1 and 'error' not in report1['integrity_check']
        test2_success = 'integrity_check' in report2 and 'error' not in report2['integrity_check']
        
        print(f"标准流水线测试: {'✅ 通过' if test1_success else '❌ 失败'}")
        print(f"自定义列名测试: {'✅ 通过' if test2_success else '❌ 失败'}")
        
        if test1_success and test2_success:
            print("\n🎉 所有测试通过！数据完整性检查已成功集成到ETL流水线中。")
        else:
            print("\n⚠️ 部分测试失败，请检查错误信息。")
            
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()