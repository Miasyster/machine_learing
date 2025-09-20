"""
数据存储管理模块
用于保存和管理从交易所获取的数据
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import logging
from pathlib import Path
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStorage:
    """数据存储管理器"""
    
    def __init__(self, base_path: str = "data"):
        """
        初始化数据存储管理器
        
        Args:
            base_path: 数据存储基础路径
        """
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "raw"
        self.processed_data_path = self.base_path / "processed"
        self.external_data_path = self.base_path / "external"
        
        # 创建目录结构
        self._create_directories()
        
        logger.info(f"数据存储管理器初始化完成，基础路径: {self.base_path}")
    
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.raw_data_path,
            self.processed_data_path,
            self.external_data_path,
            self.raw_data_path / "klines",
            self.raw_data_path / "tickers",
            self.processed_data_path / "features",
            self.processed_data_path / "signals"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"创建目录: {directory}")
    
    def save_klines_data(self, 
                        data: pd.DataFrame, 
                        symbol: str,
                        interval: str = "1h",
                        file_format: str = "csv",
                        append: bool = False) -> str:
        """
        保存K线数据
        
        Args:
            data: K线数据DataFrame
            symbol: 交易对符号
            interval: 时间间隔
            file_format: 文件格式 ('csv' 或 'parquet')
            append: 是否追加到现有文件
            
        Returns:
            保存的文件路径
        """
        try:
            # 构建文件路径
            filename = f"{symbol}_{interval}.{file_format}"
            file_path = self.raw_data_path / "klines" / filename
            
            if data.empty:
                logger.warning(f"数据为空，跳过保存: {symbol}")
                return str(file_path)
            
            # 数据预处理
            data_to_save = data.copy()
            
            # 确保索引是时间类型
            if not isinstance(data_to_save.index, pd.DatetimeIndex):
                if 'open_time' in data_to_save.columns:
                    data_to_save.set_index('open_time', inplace=True)
            
            # 排序数据
            data_to_save.sort_index(inplace=True)
            
            # 保存数据
            if file_format.lower() == "csv":
                if append and file_path.exists():
                    # 读取现有数据
                    existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    # 合并数据并去重
                    combined_data = pd.concat([existing_data, data_to_save])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data.sort_index(inplace=True)
                    combined_data.to_csv(file_path)
                else:
                    data_to_save.to_csv(file_path)
                    
            elif file_format.lower() == "parquet":
                if append and file_path.exists():
                    # 读取现有数据
                    existing_data = pd.read_parquet(file_path)
                    # 合并数据并去重
                    combined_data = pd.concat([existing_data, data_to_save])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data.sort_index(inplace=True)
                    combined_data.to_parquet(file_path)
                else:
                    data_to_save.to_parquet(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_format}")
            
            logger.info(f"成功保存 {symbol} 数据到 {file_path}，共 {len(data_to_save)} 条记录")
            
            # 保存元数据
            self._save_metadata(symbol, interval, file_path, len(data_to_save))
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"保存 {symbol} 数据失败: {e}")
            raise
    
    def load_klines_data(self, 
                        symbol: str,
                        interval: str = "1h",
                        file_format: str = "csv",
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        加载K线数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            file_format: 文件格式
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            K线数据DataFrame
        """
        try:
            filename = f"{symbol}_{interval}.{file_format}"
            file_path = self.raw_data_path / "klines" / filename
            
            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}")
                return pd.DataFrame()
            
            # 加载数据
            if file_format.lower() == "csv":
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif file_format.lower() == "parquet":
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_format}")
            
            # 日期过滤
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            logger.info(f"成功加载 {symbol} 数据，共 {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"加载 {symbol} 数据失败: {e}")
            raise
    
    def save_multiple_symbols_data(self, 
                                  data_dict: Dict[str, pd.DataFrame],
                                  interval: str = "1h",
                                  file_format: str = "csv",
                                  append: bool = False) -> Dict[str, str]:
        """
        批量保存多个交易对的数据
        
        Args:
            data_dict: 数据字典，键为交易对，值为DataFrame
            interval: 时间间隔
            file_format: 文件格式
            append: 是否追加
            
        Returns:
            保存路径字典
        """
        saved_paths = {}
        
        for symbol, data in data_dict.items():
            try:
                path = self.save_klines_data(
                    data=data,
                    symbol=symbol,
                    interval=interval,
                    file_format=file_format,
                    append=append
                )
                saved_paths[symbol] = path
            except Exception as e:
                logger.error(f"保存 {symbol} 数据失败: {e}")
                continue
        
        logger.info(f"批量保存完成，成功保存 {len(saved_paths)} 个文件")
        return saved_paths
    
    def _save_metadata(self, symbol: str, interval: str, file_path: str, record_count: int):
        """
        保存数据元信息
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            file_path: 文件路径
            record_count: 记录数量
        """
        try:
            metadata = {
                "symbol": symbol,
                "interval": interval,
                "file_path": str(file_path),
                "record_count": record_count,
                "last_updated": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
            
            metadata_file = self.raw_data_path / "metadata.json"
            
            # 读取现有元数据
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {}
            
            # 更新元数据
            key = f"{symbol}_{interval}"
            all_metadata[key] = metadata
            
            # 保存元数据
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def get_available_data(self) -> Dict[str, Any]:
        """
        获取可用数据列表
        
        Returns:
            可用数据信息
        """
        try:
            metadata_file = self.raw_data_path / "metadata.json"
            
            if not metadata_file.exists():
                return {}
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            logger.error(f"获取可用数据列表失败: {e}")
            return {}
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        获取数据摘要
        
        Returns:
            数据摘要DataFrame
        """
        try:
            metadata = self.get_available_data()
            
            if not metadata:
                return pd.DataFrame()
            
            summary_data = []
            for key, info in metadata.items():
                summary_data.append({
                    'symbol': info['symbol'],
                    'interval': info['interval'],
                    'record_count': info['record_count'],
                    'file_size_mb': round(info['file_size'] / (1024 * 1024), 2),
                    'last_updated': info['last_updated']
                })
            
            df = pd.DataFrame(summary_data)
            return df.sort_values(['symbol', 'interval'])
            
        except Exception as e:
            logger.error(f"获取数据摘要失败: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        清理旧数据
        
        Args:
            days_to_keep: 保留天数
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            metadata = self.get_available_data()
            
            removed_count = 0
            for key, info in metadata.items():
                last_updated = datetime.fromisoformat(info['last_updated'])
                if last_updated < cutoff_date:
                    file_path = Path(info['file_path'])
                    if file_path.exists():
                        file_path.unlink()
                        removed_count += 1
                        logger.info(f"删除旧文件: {file_path}")
            
            logger.info(f"清理完成，删除了 {removed_count} 个旧文件")
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")


if __name__ == "__main__":
    # 测试代码
    storage = DataStorage()
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'symbol': ['BTCUSDT'] * 5,
        'open': [50000, 50100, 50200, 50300, 50400],
        'high': [50200, 50300, 50400, 50500, 50600],
        'low': [49900, 50000, 50100, 50200, 50300],
        'close': [50100, 50200, 50300, 50400, 50500],
        'volume': [100, 110, 120, 130, 140]
    }, index=pd.date_range('2024-01-01', periods=5, freq='H'))
    
    # 保存测试数据
    storage.save_klines_data(test_data, 'BTCUSDT')
    
    # 加载测试数据
    loaded_data = storage.load_klines_data('BTCUSDT')
    print(f"加载的数据形状: {loaded_data.shape}")
    
    # 获取数据摘要
    summary = storage.get_data_summary()
    print("数据摘要:")
    print(summary)