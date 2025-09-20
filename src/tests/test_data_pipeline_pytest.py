"""
使用pytest框架的数据管道测试
可以使用 pytest 命令运行
"""

import pytest
import sys
import os
from datetime import datetime
import pandas as pd

# 导入测试配置
from .test_config import setup_test_environment, TEST_CONFIG

# 设置测试环境
setup_test_environment()

from src.etl.binance_data_fetcher import BinanceDataFetcher, get_default_symbols
from src.etl.data_storage import DataStorage
from src.utils.config import get_config


class TestDataPipeline:
    """数据管道测试类"""
    
    @pytest.fixture(scope="class")
    def fetcher(self):
        """创建数据获取器实例"""
        return BinanceDataFetcher()
    
    @pytest.fixture(scope="class")
    def storage(self):
        """创建数据存储器实例"""
        return DataStorage()
    
    def test_single_symbol_data_pipeline(self, fetcher, storage):
        """测试单个交易对的数据获取和存储流程"""
        symbol = 'BTCUSDT'
        interval = '1h'
        limit = 50
        
        # 获取数据
        data = fetcher.get_klines(
            symbol=symbol,
            interval=fetcher.client.KLINE_INTERVAL_1HOUR,
            limit=limit
        )
        
        # 验证数据
        assert not data.empty, f"未获取到 {symbol} 的数据"
        assert len(data) > 0, "数据条数应该大于0"
        assert 'open' in data.columns, "数据应包含开盘价列"
        assert 'close' in data.columns, "数据应包含收盘价列"
        
        # 保存数据
        saved_path = storage.save_klines_data(
            data=data,
            symbol=symbol,
            interval=interval,
            file_format='csv'
        )
        
        assert os.path.exists(saved_path), "保存的文件应该存在"
        
        # 验证保存的数据
        loaded_data = storage.load_klines_data(symbol=symbol, interval=interval)
        assert not loaded_data.empty, "加载保存的数据不应为空"
        assert len(loaded_data) >= len(data), "加载的数据条数应该不少于原始数据"
    
    def test_multiple_symbols_data_pipeline(self, fetcher, storage):
        """测试多个交易对的数据获取和存储流程"""
        symbols = TEST_CONFIG['symbols']
        interval = '1h'
        limit = TEST_CONFIG['limit']
        
        # 批量获取数据
        data_dict = fetcher.get_multiple_symbols_data(
            symbols=symbols,
            interval=fetcher.client.KLINE_INTERVAL_1HOUR,
            limit=limit,
            delay=TEST_CONFIG['delay']
        )
        
        assert data_dict, "应该获取到数据字典"
        assert len(data_dict) > 0, "数据字典不应为空"
        
        for symbol in symbols:
            assert symbol in data_dict, f"应该包含 {symbol} 的数据"
            assert not data_dict[symbol].empty, f"{symbol} 的数据不应为空"
        
        # 批量保存数据
        saved_paths = storage.save_multiple_symbols_data(
            data_dict=data_dict,
            interval=interval,
            file_format='csv'
        )
        
        assert saved_paths, "应该返回保存路径字典"
        assert len(saved_paths) == len(symbols), "保存路径数量应该等于交易对数量"
        
        for symbol, path in saved_paths.items():
            assert os.path.exists(path), f"{symbol} 的保存文件应该存在"
    
    def test_data_management(self, storage):
        """测试数据管理功能"""
        # 获取可用数据列表
        available_data = storage.get_available_data()
        assert isinstance(available_data, (list, dict)), "可用数据应该是列表或字典"
        
        # 获取数据摘要
        summary = storage.get_data_summary()
        assert isinstance(summary, pd.DataFrame), "数据摘要应该是DataFrame"
        
        if not summary.empty:
            required_columns = ['symbol', 'interval', 'record_count', 'file_size_mb']
            for col in required_columns:
                assert col in summary.columns, f"摘要应该包含 {col} 列"
    
    def test_latest_price_fetching(self, fetcher):
        """测试最新价格获取功能"""
        test_symbols = ['BTCUSDT', 'ETHUSDT']
        
        for symbol in test_symbols:
            price = fetcher.get_latest_price(symbol)
            assert isinstance(price, (int, float)), f"{symbol} 价格应该是数字"
            assert price > 0, f"{symbol} 价格应该大于0"
        
        # 测试24小时统计
        ticker = fetcher.get_24hr_ticker('BTCUSDT')
        assert isinstance(ticker, dict), "24小时统计应该是字典"
        
        required_fields = ['openPrice', 'highPrice', 'lowPrice', 'lastPrice', 'priceChangePercent']
        for field in required_fields:
            assert field in ticker, f"24小时统计应该包含 {field} 字段"
    
    def test_data_quality(self, fetcher):
        """测试数据质量"""
        symbol = 'BTCUSDT'
        data = fetcher.get_klines(
            symbol=symbol,
            interval=fetcher.client.KLINE_INTERVAL_1HOUR,
            limit=10
        )
        
        # 检查数据完整性
        assert not data.isnull().all().any(), "数据不应该有全空列"
        
        # 检查价格数据的合理性
        assert (data['open'] > 0).all(), "开盘价应该都大于0"
        assert (data['high'] > 0).all(), "最高价应该都大于0"
        assert (data['low'] > 0).all(), "最低价应该都大于0"
        assert (data['close'] > 0).all(), "收盘价应该都大于0"
        assert (data['volume'] >= 0).all(), "成交量应该都大于等于0"
        
        # 检查价格关系
        assert (data['high'] >= data['low']).all(), "最高价应该大于等于最低价"
        assert (data['high'] >= data['open']).all(), "最高价应该大于等于开盘价"
        assert (data['high'] >= data['close']).all(), "最高价应该大于等于收盘价"
        assert (data['low'] <= data['open']).all(), "最低价应该小于等于开盘价"
        assert (data['low'] <= data['close']).all(), "最低价应该小于等于收盘价"


if __name__ == "__main__":
    # 如果直接运行此文件，使用pytest运行测试
    pytest.main([__file__, "-v"])