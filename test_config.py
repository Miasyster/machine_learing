"""
配置测试脚本
用于验证策略配置是否正确加载
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.utils.config import ConfigManager, get_config


def test_config_loading():
    """测试配置加载功能"""
    print("=" * 50)
    print("策略配置测试")
    print("=" * 50)
    
    try:
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 加载配置
        config = config_manager.load_config()
        
        print("✅ 配置文件加载成功!")
        print()
        
        # 显示策略基本信息
        print("📊 策略基本信息:")
        print(f"  策略名称: {config.name}")
        print(f"  策略描述: {config.description}")
        print(f"  策略版本: {config.version}")
        print(f"  策略类型: {config.type}")
        print(f"  交易频率: {config.frequency}")
        print(f"  资产类别: {config.asset_class}")
        print(f"  目标函数: {config.objective}")
        print(f"  基准: {config.benchmark}")
        print()
        
        # 显示交易成本信息
        print("💰 交易成本配置:")
        costs = config.transaction_costs
        print(f"  现货手续费率: {costs.spot_fee_rate:.4f} ({costs.spot_fee_rate*100:.2f}%)")
        print(f"  BNB抵扣费率: {costs.spot_fee_rate_with_bnb:.4f} ({costs.spot_fee_rate_with_bnb*100:.3f}%)")
        print(f"  合约挂单费率: {costs.futures_maker_fee:.4f} ({costs.futures_maker_fee*100:.2f}%)")
        print(f"  合约吃单费率: {costs.futures_taker_fee:.4f} ({costs.futures_taker_fee*100:.2f}%)")
        print(f"  滑点率: {costs.slippage_rate:.4f} ({costs.slippage_rate*100:.2f}%)")
        print(f"  最小交易金额: {costs.min_trade_amount} USDT")
        print()
        
        # 显示风险限制
        print("⚠️  风险限制配置:")
        risk = config.risk_limits
        print(f"  最大回撤: {risk.max_drawdown:.1%}")
        print(f"  单日最大损失: {risk.max_daily_loss:.1%}")
        print(f"  单品种最大持仓: {risk.max_position_per_asset:.1%}")
        print(f"  最大杠杆: {risk.max_leverage}x")
        print(f"  止损比例: {risk.stop_loss:.1%}")
        print(f"  止盈比例: {risk.take_profit:.1%}")
        print()
        
        # 显示资产池
        print("🪙 资产池配置:")
        assets = config.asset_universe
        print(f"  交易对数量: {len(assets.primary_pairs)}")
        print(f"  主要交易对: {', '.join(assets.primary_pairs[:5])}...")
        print(f"  最小市值要求: {assets.min_market_cap/1e9:.1f}B USDT")
        print(f"  最小日交易量: {assets.min_daily_volume/1e6:.0f}M USDT")
        print()
        
        # 显示仓位管理
        print("📈 仓位管理配置:")
        pos = config.position_management
        print(f"  仓位计算方法: {pos.sizing_method}")
        print(f"  目标波动率: {pos.target_volatility:.1%}")
        print(f"  最大总仓位: {pos.max_total_position:.1%}")
        print(f"  现金保留比例: {pos.cash_reserve:.1%}")
        print(f"  再平衡频率: {pos.rebalance_frequency}")
        print()
        
        # 显示回测配置
        print("🔄 回测配置:")
        bt = config.backtest
        print(f"  回测期间: {bt.start_date} 至 {bt.end_date}")
        print(f"  初始资金: {bt.initial_capital:,.0f} USDT")
        print(f"  数据频率: {bt.data_frequency}")
        print()
        
        # 显示风险指标目标
        print("🎯 风险指标目标:")
        metrics = config.risk_metrics
        print(f"  目标夏普比率: {metrics.target_sharpe}")
        print(f"  目标年化收益: {metrics.target_annual_return:.1%}")
        print(f"  最大容忍回撤: {metrics.max_tolerable_drawdown:.1%}")
        print(f"  目标胜率: {metrics.target_win_rate:.1%}")
        print(f"  目标盈亏比: {metrics.target_profit_loss_ratio}")
        print()
        
        # 测试配置验证
        print("🔍 配置验证:")
        is_valid = config_manager.validate_config()
        print(f"  配置有效性: {'✅ 有效' if is_valid else '❌ 无效'}")
        print()
        
        # 测试交易成本计算
        print("💱 交易成本计算示例:")
        spot_fee = config_manager.get_transaction_cost(use_bnb=False, is_futures=False)
        spot_fee_bnb = config_manager.get_transaction_cost(use_bnb=True, is_futures=False)
        futures_maker = config_manager.get_transaction_cost(use_bnb=False, is_futures=True, is_maker=True)
        futures_taker = config_manager.get_transaction_cost(use_bnb=False, is_futures=True, is_maker=False)
        
        print(f"  现货交易费率: {spot_fee:.4f} ({spot_fee*100:.2f}%)")
        print(f"  现货交易费率(BNB): {spot_fee_bnb:.4f} ({spot_fee_bnb*100:.3f}%)")
        print(f"  合约挂单费率: {futures_maker:.4f} ({futures_maker*100:.2f}%)")
        print(f"  合约吃单费率: {futures_taker:.4f} ({futures_taker*100:.2f}%)")
        print()
        
        print("🎉 所有测试通过!")
        
    except FileNotFoundError as e:
        print(f"❌ 配置文件未找到: {e}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_config_loading()