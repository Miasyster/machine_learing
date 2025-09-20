"""
数据获取和存储流程测试脚本
测试从币安获取1小时K线数据并保存的完整流程
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# 导入测试配置
from .test_config import setup_test_environment, TEST_CONFIG

# 设置测试环境
setup_test_environment()

from src.etl.binance_data_fetcher import BinanceDataFetcher, get_default_symbols
from src.etl.data_storage import DataStorage
from src.utils.config import get_config


def test_single_symbol_data_pipeline():
    """测试单个交易对的数据获取和存储流程"""
    print("=" * 60)
    print("测试单个交易对数据获取和存储流程")
    print("=" * 60)
    
    try:
        # 初始化组件
        fetcher = BinanceDataFetcher()
        storage = DataStorage()
        
        # 测试参数
        symbol = 'BTCUSDT'
        interval = '1h'
        limit = 100  # 获取最近100条数据
        
        print(f"正在获取 {symbol} 的 {interval} 数据...")
        
        # 获取数据
        data = fetcher.get_klines(
            symbol=symbol,
            interval=fetcher.client.KLINE_INTERVAL_1HOUR,
            limit=limit
        )
        
        if data.empty:
            print(f"❌ 未获取到 {symbol} 的数据")
            return False
        
        print(f"✅ 成功获取 {symbol} 数据，共 {len(data)} 条记录")
        print(f"数据时间范围: {data.index.min()} 到 {data.index.max()}")
        print(f"数据列: {list(data.columns)}")
        print("\n前5条数据:")
        print(data.head())
        
        # 保存数据
        print(f"\n正在保存 {symbol} 数据...")
        saved_path = storage.save_klines_data(
            data=data,
            symbol=symbol,
            interval=interval,
            file_format='csv'
        )
        
        print(f"✅ 数据已保存到: {saved_path}")
        
        # 验证保存的数据
        print(f"\n正在验证保存的数据...")
        loaded_data = storage.load_klines_data(symbol=symbol, interval=interval)
        
        if loaded_data.empty:
            print("❌ 加载保存的数据失败")
            return False
        
        print(f"✅ 成功加载保存的数据，共 {len(loaded_data)} 条记录")
        
        # 数据一致性检查
        if len(data) == len(loaded_data):
            print("✅ 数据条数一致")
        else:
            print(f"⚠️ 数据条数不一致: 原始 {len(data)}, 加载 {len(loaded_data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 单个交易对测试失败: {e}")
        return False


def test_multiple_symbols_data_pipeline():
    """测试多个交易对的数据获取和存储流程"""
    print("\n" + "=" * 60)
    print("测试多个交易对数据获取和存储流程")
    print("=" * 60)
    
    try:
        # 初始化组件
        fetcher = BinanceDataFetcher()
        storage = DataStorage()
        
        # 获取配置中的交易对（取前5个进行测试）
        config = get_config()
        symbols = config.asset_universe.primary_pairs[:5]
        interval = '1h'
        limit = 50  # 每个交易对获取50条数据
        
        print(f"正在获取 {len(symbols)} 个交易对的数据: {symbols}")
        
        # 批量获取数据
        data_dict = fetcher.get_multiple_symbols_data(
            symbols=symbols,
            interval=fetcher.client.KLINE_INTERVAL_1HOUR,
            limit=limit,
            delay=0.2  # 200ms延迟避免限流
        )
        
        if not data_dict:
            print("❌ 未获取到任何数据")
            return False
        
        print(f"✅ 成功获取 {len(data_dict)} 个交易对的数据")
        
        # 显示每个交易对的数据摘要
        for symbol, data in data_dict.items():
            print(f"  {symbol}: {len(data)} 条记录, 时间范围: {data.index.min()} 到 {data.index.max()}")
        
        # 批量保存数据
        print(f"\n正在批量保存数据...")
        saved_paths = storage.save_multiple_symbols_data(
            data_dict=data_dict,
            interval=interval,
            file_format='csv'
        )
        
        print(f"✅ 成功保存 {len(saved_paths)} 个文件")
        for symbol, path in saved_paths.items():
            print(f"  {symbol}: {path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 多个交易对测试失败: {e}")
        return False


def test_data_management():
    """测试数据管理功能"""
    print("\n" + "=" * 60)
    print("测试数据管理功能")
    print("=" * 60)
    
    try:
        storage = DataStorage()
        
        # 获取可用数据列表
        print("获取可用数据列表...")
        available_data = storage.get_available_data()
        print(f"✅ 找到 {len(available_data)} 个数据文件")
        
        # 获取数据摘要
        print("\n获取数据摘要...")
        summary = storage.get_data_summary()
        
        if summary.empty:
            print("⚠️ 暂无数据摘要")
        else:
            print("✅ 数据摘要:")
            print(summary.to_string(index=False))
            
            # 计算总文件大小
            total_size_mb = summary['file_size_mb'].sum()
            print(f"\n总文件大小: {total_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据管理测试失败: {e}")
        return False


def test_latest_price_fetching():
    """测试最新价格获取功能"""
    print("\n" + "=" * 60)
    print("测试最新价格获取功能")
    print("=" * 60)
    
    try:
        fetcher = BinanceDataFetcher()
        
        # 测试几个主要交易对的最新价格
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        print("获取最新价格...")
        for symbol in test_symbols:
            try:
                price = fetcher.get_latest_price(symbol)
                print(f"  {symbol}: ${price:,.2f}")
            except Exception as e:
                print(f"  {symbol}: 获取失败 - {e}")
        
        # 测试24小时统计
        print(f"\n获取 BTCUSDT 24小时统计...")
        ticker = fetcher.get_24hr_ticker('BTCUSDT')
        print(f"  开盘价: ${float(ticker['openPrice']):,.2f}")
        print(f"  最高价: ${float(ticker['highPrice']):,.2f}")
        print(f"  最低价: ${float(ticker['lowPrice']):,.2f}")
        print(f"  收盘价: ${float(ticker['lastPrice']):,.2f}")
        print(f"  涨跌幅: {float(ticker['priceChangePercent']):.2f}%")
        print(f"  成交量: {float(ticker['volume']):,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 最新价格测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始数据获取和存储流程测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行所有测试
    tests = [
        ("单个交易对数据流程", test_single_symbol_data_pipeline),
        ("多个交易对数据流程", test_multiple_symbols_data_pipeline),
        ("数据管理功能", test_data_management),
        ("最新价格获取", test_latest_price_fetching)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！数据获取和存储流程工作正常。")
    else:
        print("⚠️ 部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()