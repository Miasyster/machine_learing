"""
策略配置管理模块
用于加载和管理策略配置参数
"""

import yaml
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TransactionCosts:
    """交易成本配置"""
    spot_fee_rate: float = 0.001
    spot_fee_rate_with_bnb: float = 0.00075
    futures_maker_fee: float = 0.0002
    futures_taker_fee: float = 0.0004
    futures_maker_fee_with_bnb: float = 0.00018
    futures_taker_fee_with_bnb: float = 0.00036
    slippage_model: str = "linear"
    slippage_rate: float = 0.0005
    min_trade_amount: float = 10.0


@dataclass
class RiskLimits:
    """风险限制配置"""
    max_drawdown: float = 0.20
    max_daily_loss: float = 0.05
    max_position_per_asset: float = 0.30
    max_leverage: float = 3.0
    stop_loss: float = 0.10
    take_profit: float = 0.20


@dataclass
class AssetUniverse:
    """资产池配置"""
    primary_pairs: List[str]
    min_market_cap: float = 1000000000
    min_daily_volume: float = 100000000


@dataclass
class PositionManagement:
    """仓位管理配置"""
    sizing_method: str = "volatility_target"
    target_volatility: float = 0.15
    max_total_position: float = 0.95
    cash_reserve: float = 0.05
    rebalance_frequency: str = "daily"


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100000
    data_frequency: str = "1d"


@dataclass
class RiskMetrics:
    """风险指标目标"""
    target_sharpe: float = 1.5
    target_annual_return: float = 0.30
    max_tolerable_drawdown: float = 0.25
    target_win_rate: float = 0.55
    target_profit_loss_ratio: float = 1.5


@dataclass
class MonitoringConfig:
    """监控配置"""
    real_time_metrics: List[str]
    alerts: Dict[str, float]


@dataclass
class StrategyConfig:
    """策略主配置类"""
    # 策略基本信息
    name: str
    description: str
    version: str
    type: str
    frequency: str
    asset_class: str
    objective: str
    benchmark: str
    
    # 子配置
    transaction_costs: TransactionCosts
    risk_limits: RiskLimits
    asset_universe: AssetUniverse
    position_management: PositionManagement
    backtest: BacktestConfig
    risk_metrics: RiskMetrics
    monitoring: MonitoringConfig


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "strategy_config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Optional[StrategyConfig] = None
    
    def load_config(self) -> StrategyConfig:
        """
        加载配置文件
        
        Returns:
            StrategyConfig: 策略配置对象
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 解析配置
        strategy_info = config_dict['strategy']
        constraints = config_dict['constraints']
        
        # 创建子配置对象
        transaction_costs = TransactionCosts(**constraints['transaction_costs'])
        risk_limits = RiskLimits(**constraints['risk_limits'])
        asset_universe = AssetUniverse(**config_dict['asset_universe'])
        position_management = PositionManagement(**config_dict['position_management'])
        backtest = BacktestConfig(**config_dict['backtest'])
        risk_metrics = RiskMetrics(**config_dict['risk_metrics'])
        monitoring = MonitoringConfig(**config_dict['monitoring'])
        
        # 创建主配置对象
        self._config = StrategyConfig(
            name=strategy_info['name'],
            description=strategy_info['description'],
            version=strategy_info['version'],
            type=strategy_info['type'],
            frequency=strategy_info['frequency'],
            asset_class=strategy_info['asset_class'],
            objective=strategy_info['objective'],
            benchmark=strategy_info['benchmark'],
            transaction_costs=transaction_costs,
            risk_limits=risk_limits,
            asset_universe=asset_universe,
            position_management=position_management,
            backtest=backtest,
            risk_metrics=risk_metrics,
            monitoring=monitoring
        )
        
        return self._config
    
    @property
    def config(self) -> StrategyConfig:
        """
        获取配置对象，如果未加载则自动加载
        
        Returns:
            StrategyConfig: 策略配置对象
        """
        if self._config is None:
            self.load_config()
        return self._config
    
    def get_trading_pairs(self) -> List[str]:
        """
        获取交易对列表
        
        Returns:
            List[str]: 交易对列表
        """
        return self.config.asset_universe.primary_pairs
    
    def get_transaction_cost(self, use_bnb: bool = False, is_futures: bool = False, 
                           is_maker: bool = True) -> float:
        """
        获取交易成本
        
        Args:
            use_bnb: 是否使用BNB抵扣
            is_futures: 是否为合约交易
            is_maker: 是否为挂单（仅合约有效）
            
        Returns:
            float: 手续费率
        """
        costs = self.config.transaction_costs
        
        if is_futures:
            if is_maker:
                return costs.futures_maker_fee_with_bnb if use_bnb else costs.futures_maker_fee
            else:
                return costs.futures_taker_fee_with_bnb if use_bnb else costs.futures_taker_fee
        else:
            return costs.spot_fee_rate_with_bnb if use_bnb else costs.spot_fee_rate
    
    def validate_config(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            bool: 配置是否有效
        """
        config = self.config
        
        # 验证基本参数
        if config.risk_limits.max_drawdown <= 0 or config.risk_limits.max_drawdown > 1:
            raise ValueError("最大回撤必须在0-1之间")
        
        if config.position_management.target_volatility <= 0:
            raise ValueError("目标波动率必须大于0")
        
        if not config.asset_universe.primary_pairs:
            raise ValueError("必须指定至少一个交易对")
        
        # 验证费率
        if config.transaction_costs.spot_fee_rate < 0:
            raise ValueError("手续费率不能为负数")
        
        return True
    
    def save_config(self, config: StrategyConfig, save_path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            config: 要保存的配置对象
            save_path: 保存路径，如果为None则覆盖原文件
        """
        if save_path is None:
            save_path = self.config_path
        
        # 将配置对象转换为字典
        config_dict = {
            'strategy': {
                'name': config.name,
                'description': config.description,
                'version': config.version,
                'type': config.type,
                'frequency': config.frequency,
                'asset_class': config.asset_class,
                'objective': config.objective,
                'benchmark': config.benchmark
            },
            'constraints': {
                'transaction_costs': config.transaction_costs.__dict__,
                'risk_limits': config.risk_limits.__dict__
            },
            'asset_universe': config.asset_universe.__dict__,
            'position_management': config.position_management.__dict__,
            'backtest': config.backtest.__dict__,
            'risk_metrics': config.risk_metrics.__dict__,
            'monitoring': config.monitoring.__dict__
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config() -> StrategyConfig:
    """
    获取全局配置对象
    
    Returns:
        StrategyConfig: 策略配置对象
    """
    return config_manager.config


if __name__ == "__main__":
    # 测试配置加载
    try:
        config = get_config()
        print(f"策略名称: {config.name}")
        print(f"策略类型: {config.type}")
        print(f"交易频率: {config.frequency}")
        print(f"资产类别: {config.asset_class}")
        print(f"目标函数: {config.objective}")
        print(f"现货手续费率: {config.transaction_costs.spot_fee_rate}")
        print(f"最大回撤: {config.risk_limits.max_drawdown}")
        print(f"交易对数量: {len(config.asset_universe.primary_pairs)}")
        print("配置加载成功！")
    except Exception as e:
        print(f"配置加载失败: {e}")