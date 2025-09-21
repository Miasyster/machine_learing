"""
执行一致性模块

该模块实现交易执行的一致性管理，包括：
1. 持仓期管理 - 跟踪持仓时间，强制到期平仓
2. 止损平仓规则 - 基于亏损阈值的自动平仓
3. 执行一致性管理器 - 统一管理持仓和平仓逻辑

主要功能：
- 3天持仓期限制
- 1%止损平仓规则
- 持仓状态跟踪
- 自动平仓触发
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """持仓状态枚举"""
    ACTIVE = "active"           # 活跃持仓
    EXPIRED = "expired"         # 到期平仓
    STOP_LOSS = "stop_loss"     # 止损平仓
    CLOSED = "closed"           # 已平仓


@dataclass
class Position:
    """持仓信息数据类"""
    symbol: str                 # 交易标的
    entry_time: datetime        # 开仓时间
    entry_price: float          # 开仓价格
    quantity: float             # 持仓数量
    direction: str              # 持仓方向 ('long' 或 'short')
    position_id: str            # 持仓ID
    status: PositionStatus = PositionStatus.ACTIVE
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


class HoldingPeriodManager:
    """
    持仓期管理器
    
    负责跟踪持仓时间，确保在指定期限内强制平仓
    """
    
    def __init__(self, holding_days: int = 3):
        """
        初始化持仓期管理器
        
        Parameters:
        -----------
        holding_days : int, default=3
            最大持仓天数
        """
        self.holding_days = holding_days
        self.positions: Dict[str, Position] = {}
        
    def add_position(self, position: Position) -> None:
        """
        添加新持仓
        
        Parameters:
        -----------
        position : Position
            持仓信息
        """
        self.positions[position.position_id] = position
        logger.info(f"添加持仓: {position.position_id}, 标的: {position.symbol}")
        
    def check_expired_positions(self, current_time: datetime, 
                               current_prices: Dict[str, float] = None) -> List[Position]:
        """
        检查到期持仓
        
        Parameters:
        -----------
        current_time : datetime
            当前时间
        current_prices : Dict[str, float], optional
            当前价格字典
            
        Returns:
        --------
        List[Position]
            到期需要平仓的持仓列表
        """
        expired_positions = []
        
        for position in self.positions.values():
            if position.status == PositionStatus.ACTIVE:
                holding_duration = current_time - position.entry_time
                if holding_duration.days >= self.holding_days:
                    position.status = PositionStatus.EXPIRED
                    position.exit_time = current_time
                    position.exit_reason = f"持仓期到期({self.holding_days}天)"
                    
                    # 设置退出价格
                    if current_prices and position.symbol in current_prices:
                        position.exit_price = current_prices[position.symbol]
                    else:
                        # 如果没有当前价格，使用入场价格作为退出价格
                        position.exit_price = position.entry_price
                        
                    expired_positions.append(position)
                    
        return expired_positions
    
    def get_active_positions(self) -> List[Position]:
        """获取所有活跃持仓"""
        return [pos for pos in self.positions.values() 
                if pos.status == PositionStatus.ACTIVE]
    
    def close_position(self, position_id: str, exit_price: float, 
                      exit_time: datetime, reason: str) -> bool:
        """
        平仓操作
        
        Parameters:
        -----------
        position_id : str
            持仓ID
        exit_price : float
            平仓价格
        exit_time : datetime
            平仓时间
        reason : str
            平仓原因
            
        Returns:
        --------
        bool
            是否成功平仓
        """
        if position_id in self.positions:
            position = self.positions[position_id]
            position.exit_price = exit_price
            position.exit_time = exit_time
            position.exit_reason = reason
            position.status = PositionStatus.CLOSED
            logger.info(f"平仓成功: {position_id}, 原因: {reason}")
            return True
        return False


class StopLossManager:
    """
    止损管理器
    
    负责监控持仓盈亏，触发止损平仓
    """
    
    def __init__(self, stop_loss_threshold: float = 0.01):
        """
        初始化止损管理器
        
        Parameters:
        -----------
        stop_loss_threshold : float, default=0.01
            止损阈值（1%）
        """
        self.stop_loss_threshold = stop_loss_threshold
        
    def calculate_pnl(self, position: Position, current_price: float) -> float:
        """
        计算持仓盈亏
        
        Parameters:
        -----------
        position : Position
            持仓信息
        current_price : float
            当前价格
            
        Returns:
        --------
        float
            盈亏比例
        """
        if position.direction == 'long':
            return (current_price - position.entry_price) / position.entry_price
        else:  # short
            return (position.entry_price - current_price) / position.entry_price
    
    def check_stop_loss(self, positions: List[Position], 
                       current_prices: Dict[str, float],
                       current_time: datetime) -> List[Position]:
        """
        检查止损条件
        
        Parameters:
        -----------
        positions : List[Position]
            持仓列表
        current_prices : Dict[str, float]
            当前价格字典 {symbol: price}
        current_time : datetime
            当前时间
            
        Returns:
        --------
        List[Position]
            需要止损的持仓列表
        """
        stop_loss_positions = []
        
        for position in positions:
            if (position.status == PositionStatus.ACTIVE and 
                position.symbol in current_prices):
                
                current_price = current_prices[position.symbol]
                pnl = self.calculate_pnl(position, current_price)
                
                # 检查是否触发止损
                if pnl <= -self.stop_loss_threshold:
                    position.status = PositionStatus.STOP_LOSS
                    position.exit_time = current_time
                    position.exit_price = current_price
                    position.exit_reason = f"止损平仓(亏损{pnl:.2%})"
                    stop_loss_positions.append(position)
                    
        return stop_loss_positions


class ExecutionConsistencyManager:
    """
    执行一致性管理器
    
    统一管理持仓期和止损规则，确保交易执行的一致性
    """
    
    def __init__(self, holding_days: int = 3, stop_loss_threshold: float = 0.01):
        """
        初始化执行一致性管理器
        
        Parameters:
        -----------
        holding_days : int, default=3
            最大持仓天数
        stop_loss_threshold : float, default=0.01
            止损阈值
        """
        self.holding_manager = HoldingPeriodManager(holding_days)
        self.stop_loss_manager = StopLossManager(stop_loss_threshold)
        self.execution_log: List[Dict] = []
        
    def open_position(self, symbol: str, entry_price: float, quantity: float,
                     direction: str, entry_time: datetime, 
                     position_id: Optional[str] = None) -> str:
        """
        开仓操作
        
        Parameters:
        -----------
        symbol : str
            交易标的
        entry_price : float
            开仓价格
        quantity : float
            持仓数量
        direction : str
            持仓方向
        entry_time : datetime
            开仓时间
        position_id : str, optional
            持仓ID，如果不提供则自动生成
            
        Returns:
        --------
        str
            持仓ID
        """
        if position_id is None:
            position_id = f"{symbol}_{entry_time.strftime('%Y%m%d_%H%M%S')}"
            
        position = Position(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            quantity=quantity,
            direction=direction,
            position_id=position_id
        )
        
        self.holding_manager.add_position(position)
        
        # 记录执行日志
        self.execution_log.append({
            'timestamp': entry_time,
            'action': 'open',
            'position_id': position_id,
            'symbol': symbol,
            'price': entry_price,
            'quantity': quantity,
            'direction': direction
        })
        
        return position_id
    
    def update_positions(self, current_prices: Dict[str, float], 
                        current_time: datetime) -> Dict[str, List[Position]]:
        """
        更新持仓状态，检查平仓条件
        
        Parameters:
        -----------
        current_prices : Dict[str, float]
            当前价格字典
        current_time : datetime
            当前时间
            
        Returns:
        --------
        Dict[str, List[Position]]
            平仓结果 {'expired': [...], 'stop_loss': [...]}
        """
        results = {'expired': [], 'stop_loss': []}
        
        # 检查到期持仓
        expired_positions = self.holding_manager.check_expired_positions(current_time, current_prices)
        results['expired'] = expired_positions
        
        # 检查止损持仓
        active_positions = self.holding_manager.get_active_positions()
        stop_loss_positions = self.stop_loss_manager.check_stop_loss(
            active_positions, current_prices, current_time
        )
        results['stop_loss'] = stop_loss_positions
        
        # 记录平仓日志
        for position in expired_positions + stop_loss_positions:
            self.execution_log.append({
                'timestamp': current_time,
                'action': 'close',
                'position_id': position.position_id,
                'symbol': position.symbol,
                'price': position.exit_price,
                'reason': position.exit_reason,
                'pnl': self._calculate_position_pnl(position)
            })
            
        return results
    
    def _calculate_position_pnl(self, position: Position) -> float:
        """计算持仓总盈亏"""
        if position.exit_price is None:
            return 0.0
            
        if position.direction == 'long':
            return (position.exit_price - position.entry_price) * position.quantity
        else:
            return (position.entry_price - position.exit_price) * position.quantity
    
    def get_position_summary(self) -> pd.DataFrame:
        """
        获取持仓汇总信息
        
        Returns:
        --------
        pd.DataFrame
            持仓汇总表
        """
        positions_data = []
        
        for position in self.holding_manager.positions.values():
            pnl = self._calculate_position_pnl(position)
            
            positions_data.append({
                'position_id': position.position_id,
                'symbol': position.symbol,
                'direction': position.direction,
                'entry_time': position.entry_time,
                'entry_price': position.entry_price,
                'quantity': position.quantity,
                'exit_time': position.exit_time,
                'exit_price': position.exit_price,
                'status': position.status.value,
                'exit_reason': position.exit_reason,
                'pnl': pnl
            })
            
        return pd.DataFrame(positions_data)
    
    def get_execution_log(self) -> pd.DataFrame:
        """
        获取执行日志
        
        Returns:
        --------
        pd.DataFrame
            执行日志表
        """
        return pd.DataFrame(self.execution_log)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        获取执行绩效指标
        
        Returns:
        --------
        Dict[str, float]
            绩效指标字典
        """
        summary = self.get_position_summary()
        
        if summary.empty:
            return {}
            
        closed_positions = summary[summary['status'].isin(['expired', 'stop_loss', 'closed'])]
        
        if closed_positions.empty:
            return {'total_positions': len(summary)}
            
        total_pnl = closed_positions['pnl'].sum()
        win_positions = closed_positions[closed_positions['pnl'] > 0]
        loss_positions = closed_positions[closed_positions['pnl'] <= 0]
        
        metrics = {
            'total_positions': len(summary),
            'closed_positions': len(closed_positions),
            'total_pnl': total_pnl,
            'win_rate': len(win_positions) / len(closed_positions) if len(closed_positions) > 0 else 0,
            'avg_win': win_positions['pnl'].mean() if len(win_positions) > 0 else 0,
            'avg_loss': loss_positions['pnl'].mean() if len(loss_positions) > 0 else 0,
            'stop_loss_rate': len(closed_positions[closed_positions['exit_reason'].str.contains('止损', na=False)]) / len(closed_positions) if len(closed_positions) > 0 else 0,
            'expiry_rate': len(closed_positions[closed_positions['exit_reason'].str.contains('到期', na=False)]) / len(closed_positions) if len(closed_positions) > 0 else 0
        }
        
        return metrics


# 便捷函数
def create_execution_manager(holding_days: int = 3, 
                           stop_loss_threshold: float = 0.01) -> ExecutionConsistencyManager:
    """
    创建执行一致性管理器的便捷函数
    
    Parameters:
    -----------
    holding_days : int, default=3
        最大持仓天数
    stop_loss_threshold : float, default=0.01
        止损阈值
        
    Returns:
    --------
    ExecutionConsistencyManager
        执行一致性管理器实例
    """
    return ExecutionConsistencyManager(holding_days, stop_loss_threshold)


def simulate_trading_execution(data: pd.DataFrame, 
                             signals: pd.Series,
                             holding_days: int = 3,
                             stop_loss_threshold: float = 0.01,
                             initial_capital: float = 100000) -> Dict:
    """
    模拟交易执行的便捷函数
    
    Parameters:
    -----------
    data : pd.DataFrame
        价格数据，需包含 'close' 列
    signals : pd.Series
        交易信号，1为买入，-1为卖出，0为持有
    holding_days : int, default=3
        最大持仓天数
    stop_loss_threshold : float, default=0.01
        止损阈值
    initial_capital : float, default=100000
        初始资金
        
    Returns:
    --------
    Dict
        模拟结果，包含管理器实例和绩效指标
    """
    manager = create_execution_manager(holding_days, stop_loss_threshold)
    
    # 确保数据和信号对齐
    aligned_data = data.reindex(signals.index).dropna()
    aligned_signals = signals.reindex(aligned_data.index)
    
    current_capital = initial_capital
    position_size = 0.1  # 每次交易使用10%资金
    
    for timestamp, signal in aligned_signals.items():
        current_price = aligned_data.loc[timestamp, 'close']
        current_time = pd.to_datetime(timestamp)
        
        # 更新现有持仓
        current_prices = {'default': current_price}
        manager.update_positions(current_prices, current_time)
        
        # 处理新信号
        if signal == 1:  # 买入信号
            quantity = (current_capital * position_size) / current_price
            manager.open_position(
                symbol='default',
                entry_price=current_price,
                quantity=quantity,
                direction='long',
                entry_time=current_time
            )
        elif signal == -1:  # 卖出信号
            quantity = (current_capital * position_size) / current_price
            manager.open_position(
                symbol='default',
                entry_price=current_price,
                quantity=quantity,
                direction='short',
                entry_time=current_time
            )
    
    return {
        'manager': manager,
        'summary': manager.get_position_summary(),
        'execution_log': manager.get_execution_log(),
        'metrics': manager.get_performance_metrics()
    }