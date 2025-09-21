# 回测引擎使用示例

本目录包含了回测引擎的完整使用示例，展示了如何使用我们的回测框架进行量化策略开发和测试。

## 📁 文件结构

```
examples/backtest/
├── README.md                          # 本文件
├── basic_backtest_example.py          # 基础回测示例
├── vectorbt_example.py                # VectorBT集成示例
└── advanced_strategies_example.py     # 高级策略示例
```

## 🚀 快速开始

### 1. 基础回测示例 (`basic_backtest_example.py`)

这是最简单的入门示例，展示了：
- 如何生成示例数据
- 如何配置回测参数
- 如何实现简单的移动平均策略
- 如何运行回测并分析结果

```python
# 运行基础示例
python basic_backtest_example.py
```

**包含的策略：**
- 简单移动平均交叉策略
- 买入持有策略（基准）

**主要功能：**
- 数据生成和处理
- 策略实现和回测
- 结果分析和可视化
- 报告生成

### 2. VectorBT集成示例 (`vectorbt_example.py`)

展示如何使用VectorBT进行高性能向量化回测：
- 多策略并行回测
- 参数优化
- 性能对比分析

```python
# 运行VectorBT示例
python vectorbt_example.py
```

**包含的策略：**
- 移动平均交叉策略（多参数组合）
- RSI策略
- 均值回归策略
- 动量策略

**主要功能：**
- 向量化回测引擎
- 多策略性能对比
- 参数优化
- 高性能计算

### 3. 高级策略示例 (`advanced_strategies_example.py`)

展示复杂量化策略的实现：
- 多因子策略
- 配对交易策略
- 机器学习策略

```python
# 运行高级策略示例
python advanced_strategies_example.py
```

**包含的策略：**
- 多因子选股策略
- 统计套利配对交易
- 机器学习预测策略
- 等权重基准策略

**主要功能：**
- 复杂策略实现
- 多资产相关性建模
- 机器学习集成
- 高级风险管理

## 📊 示例特性

### 数据生成
所有示例都包含数据生成功能，无需外部数据源：
- 模拟真实市场数据
- 支持多资产相关性
- 可配置的市场参数

### 策略类型
- **趋势跟踪**：移动平均、动量策略
- **均值回归**：RSI、统计套利
- **多因子**：基于多个技术和基本面因子
- **机器学习**：使用ML模型进行预测

### 分析功能
- 完整的绩效指标计算
- 风险分析（VaR、最大回撤等）
- 可视化图表生成
- 详细报告输出

## 🛠️ 依赖要求

### 基础依赖
```bash
pip install pandas numpy matplotlib
```

### VectorBT示例额外依赖
```bash
pip install vectorbt
```

### 机器学习示例额外依赖
```bash
pip install scikit-learn
```

### 完整依赖安装
```bash
pip install pandas numpy matplotlib vectorbt scikit-learn
```

## 📈 运行示例

### 方法1：直接运行
```bash
cd examples/backtest
python basic_backtest_example.py
```

### 方法2：作为模块导入
```python
import sys
sys.path.append('path/to/examples/backtest')

from basic_backtest_example import run_basic_backtest
results = run_basic_backtest()
```

### 方法3：交互式运行
```python
# 在Jupyter Notebook中
%run basic_backtest_example.py
```

## 📋 示例输出

### 控制台输出
每个示例都会在控制台输出详细的运行信息：
- 数据生成进度
- 策略执行日志
- 性能指标对比
- 分析结果总结

### 图表输出
自动生成多种分析图表：
- 权益曲线对比
- 回撤分析
- 收益率分布
- 风险收益散点图
- 滚动指标图表

### 文件输出
- 详细的Markdown格式报告
- 可选的CSV数据导出
- 图表文件保存

## 🎯 学习路径

### 初学者
1. 从 `basic_backtest_example.py` 开始
2. 理解基本的回测流程
3. 学习策略实现方法
4. 掌握结果分析技巧

### 进阶用户
1. 学习 `vectorbt_example.py` 的向量化方法
2. 理解参数优化技术
3. 掌握多策略对比分析

### 高级用户
1. 研究 `advanced_strategies_example.py` 的复杂策略
2. 学习多因子模型构建
3. 掌握机器学习在量化中的应用
4. 理解风险管理技术

## 🔧 自定义开发

### 创建新策略
```python
class MyStrategy:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
        self.engine = None
        
    def set_engine(self, engine):
        self.engine = engine
        
    def on_bar(self, timestamp, bar_data):
        # 实现你的策略逻辑
        pass
```

### 添加新指标
```python
def calculate_custom_indicator(prices, window):
    # 实现自定义技术指标
    return indicator_values
```

### 扩展分析功能
```python
def custom_analysis(backtest_result):
    # 实现自定义分析
    return analysis_result
```

## 📚 相关文档

- [回测引擎API文档](../../src/backtest/README.md)
- [策略开发指南](../../docs/strategy_development.md)
- [性能优化指南](../../docs/performance_optimization.md)
- [风险管理指南](../../docs/risk_management.md)

## ❓ 常见问题

### Q: 如何添加真实数据？
A: 替换示例中的数据生成函数，使用你的数据源：
```python
# 替换生成的数据
data = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
engine.add_data(data, symbol)
```

### Q: 如何调整回测参数？
A: 修改BacktestConfig中的参数：
```python
config = BacktestConfig(
    initial_capital=100000.0,
    commission_model=PercentageCommissionModel(commission_rate=0.001),
    slippage_model=FixedSlippageModel(slippage_bps=5.0)
)
```

### Q: 如何保存回测结果？
A: 使用分析器的导出功能：
```python
analyzer = analyze_backtest_result(result)
analyzer.export_to_csv('backtest_results.csv')
```

### Q: 如何进行参数优化？
A: 参考vectorbt_example.py中的参数优化部分，或使用网格搜索：
```python
for param1 in param1_range:
    for param2 in param2_range:
        strategy = MyStrategy(param1, param2)
        result = run_backtest(strategy)
        # 记录结果
```

## 🤝 贡献

欢迎提交新的示例和改进：
1. Fork项目
2. 创建新的示例文件
3. 添加相应的文档
4. 提交Pull Request

## 📄 许可证

本示例代码遵循项目的开源许可证。

---

**开始你的量化交易之旅！** 🚀