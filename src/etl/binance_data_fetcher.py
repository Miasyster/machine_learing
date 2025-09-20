"""
币安数据获取模块
用于从币安交易所获取K线数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import time
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """币安数据获取器"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        初始化币安客户端
        
        Args:
            api_key: API密钥（可选，获取公开数据不需要）
            api_secret: API密钥（可选，获取公开数据不需要）
        """
        try:
            # 对于获取K线数据，不需要API密钥
            self.client = Client(api_key, api_secret)
            logger.info("币安客户端初始化成功")
        except Exception as e:
            logger.error(f"币安客户端初始化失败: {e}")
            raise
    
    def get_klines(self, 
                   symbol: str, 
                   interval: str = Client.KLINE_INTERVAL_1HOUR,
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   limit: int = 1000) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易对符号，如 'BTCUSDT'
            interval: 时间间隔，默认1小时
            start_time: 开始时间，格式 'YYYY-MM-DD' 或时间戳
            end_time: 结束时间，格式 'YYYY-MM-DD' 或时间戳
            limit: 数据条数限制，最大1000
            
        Returns:
            包含K线数据的DataFrame
        """
        try:
            logger.info(f"开始获取 {symbol} 的K线数据，间隔: {interval}")
            
            # 获取K线数据
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time,
                end_str=end_time,
                limit=limit
            )
            
            if not klines:
                logger.warning(f"未获取到 {symbol} 的数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = self._klines_to_dataframe(klines, symbol)
            logger.info(f"成功获取 {symbol} 数据，共 {len(df)} 条记录")
            
            return df
            
        except BinanceAPIException as e:
            logger.error(f"币安API异常: {e}")
            raise
        except BinanceRequestException as e:
            logger.error(f"币安请求异常: {e}")
            raise
        except Exception as e:
            logger.error(f"获取K线数据时发生未知错误: {e}")
            raise
    
    def _klines_to_dataframe(self, klines: List[List], symbol: str) -> pd.DataFrame:
        """
        将K线数据转换为DataFrame
        
        Args:
            klines: 原始K线数据
            symbol: 交易对符号
            
        Returns:
            格式化的DataFrame
        """
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(klines, columns=columns)
        
        # 数据类型转换
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 时间转换
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # 添加交易对信息
        df['symbol'] = symbol
        
        # 重新排列列顺序
        df = df[['symbol', 'open_time', 'close_time', 'open', 'high', 'low', 'close', 
                'volume', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]
        
        # 设置索引
        df.set_index('open_time', inplace=True)
        
        return df
    
    def get_multiple_symbols_data(self, 
                                 symbols: List[str],
                                 interval: str = Client.KLINE_INTERVAL_1HOUR,
                                 start_time: Optional[str] = None,
                                 end_time: Optional[str] = None,
                                 limit: int = 1000,
                                 delay: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个交易对的数据
        
        Args:
            symbols: 交易对列表
            interval: 时间间隔
            start_time: 开始时间
            end_time: 结束时间
            limit: 数据条数限制
            delay: 请求间隔（秒），避免触发限流
            
        Returns:
            字典，键为交易对，值为对应的DataFrame
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"正在获取 {symbol} 数据 ({i+1}/{len(symbols)})")
                
                df = self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                
                if not df.empty:
                    results[symbol] = df
                
                # 添加延迟避免限流
                if i < len(symbols) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")
                continue
        
        logger.info(f"批量获取完成，成功获取 {len(results)} 个交易对的数据")
        return results
    
    def get_latest_price(self, symbol: str) -> float:
        """
        获取最新价格
        
        Args:
            symbol: 交易对符号
            
        Returns:
            最新价格
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"获取 {symbol} 最新价格失败: {e}")
            raise
    
    def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        获取24小时价格变动统计
        
        Args:
            symbol: 交易对符号
            
        Returns:
            24小时统计数据
        """
        try:
            return self.client.get_ticker(symbol=symbol)
        except Exception as e:
            logger.error(f"获取 {symbol} 24小时统计失败: {e}")
            raise


def get_default_symbols() -> List[str]:
    """获取默认的交易对列表"""
    return [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
        'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT'
    ]


if __name__ == "__main__":
    # 测试代码
    fetcher = BinanceDataFetcher()
    
    # 获取BTC 1小时数据（最近100条）
    btc_data = fetcher.get_klines('BTCUSDT', limit=100)
    print(f"BTC数据形状: {btc_data.shape}")
    print(btc_data.head())
    
    # 获取最新价格
    latest_price = fetcher.get_latest_price('BTCUSDT')
    print(f"BTC最新价格: {latest_price}")