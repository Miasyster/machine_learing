# 集成方法使用指南

本目录包含了Stacking、Blending和模型校准的完整实现和使用示例。

## 📁 文件结构

```
examples/
├── README.md              # 本文档
├── quick_start.py         # 快速入门示例
└── ensemble_examples.py   # 详细使用示例
```

## 🚀 快速开始

### 1. 运行快速入门示例

```bash
python examples/quick_start.py
```

这个示例展示了：
- 基础的Stacking和Blending使用
- 模型校准
- 性能比较

### 2. 查看详细示例

```bash
python examples/ensemble_examples.py
```

这个示例包含：
- 多种集成策略
- 参数调优
- 性能分析
- 错误处理演示

## 📚 主要功能

### Stacking集成
- **多层堆叠**: 支持多层模型堆叠
- **交叉验证**: 避免过拟合
- **概率融合**: 使用预测概率而非硬预测
- **特征组合**: 可选择是否在元学习中使用原始特征

### Blending集成
- **保留集验证**: 使用独立的验证集生成元特征
- **动态权重**: 支持动态调整模型权重
- **自适应策略**: 根据模型性能自动调整

### 模型校准
- **Platt校准**: 使用sigmoid函数校准
- **Isotonic校准**: 使用保序回归校准
- **集成校准**: 对整个集成模型进行校准

## 🔧 基础用法

### Stacking示例

```python
from src.ensemble.stacking import StackingEnsemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 定义基础模型
base_models = [
    RandomForestClassifier(n_estimators=100),
    LogisticRegression()
]

# 创建Stacking集成
stacking = StackingEnsemble(
    base_models=base_models,
    meta_model=LogisticRegression(),
    cv=5
)

# 训练和预测
stacking.fit(X_train, y_train)
result = stacking.predict(X_test)
```

### Blending示例

```python
from src.ensemble.blending import BlendingEnsemble

# 创建Blending集成
blending = BlendingEnsemble(
    base_models=base_models,
    meta_model=LogisticRegression(),
    holdout_size=0.2
)

# 训练和预测
blending.fit(X_train, y_train)
result = blending.predict(X_test)
```

### 模型校准示例

```python
from src.ensemble.calibration import CalibratedEnsemble, ModelCalibrator

# 创建校准器
calibrator = ModelCalibrator(method='platt')

# 创建校准集成
calibrated = CalibratedEnsemble(
    base_ensemble=stacking,
    calibrator=calibrator
)

# 训练和预测
calibrated.fit(X_train, y_train)
result = calibrated.predict(X_test)
```

## ⚙️ 高级配置

### 参数说明

#### StackingEnsemble参数
- `base_models`: 基础模型列表
- `meta_model`: 元学习器
- `cv`: 交叉验证折数
- `use_probas`: 是否使用概率预测
- `use_features_in_secondary`: 是否在元学习中使用原始特征
- `model_names`: 模型名称列表
- `verbose`: 是否显示详细信息

#### BlendingEnsemble参数
- `base_models`: 基础模型列表
- `meta_model`: 元学习器
- `holdout_size`: 保留集大小比例
- `random_state`: 随机种子
- `verbose`: 是否显示详细信息

#### CalibratedEnsemble参数
- `base_ensemble`: 基础集成模型
- `calibrator`: 校准器
- `cv`: 交叉验证折数

### 性能优化建议

1. **模型选择**
   - 选择多样性强的基础模型
   - 避免过于相似的模型

2. **参数调优**
   - 使用网格搜索或随机搜索
   - 注意交叉验证的折数设置

3. **数据处理**
   - 确保数据预处理的一致性
   - 处理缺失值和异常值

4. **计算效率**
   - 对于大数据集，考虑使用Blending
   - 合理设置交叉验证折数

## 🧪 测试

运行测试脚本：

```bash
python test_ensemble_methods.py
```

测试包括：
- 分类和回归任务
- 错误处理
- 性能基准测试
- 边界情况测试

## 📊 性能比较

一般情况下的性能表现：

1. **Stacking**: 通常性能最好，但计算成本较高
2. **Blending**: 性能良好，计算效率较高
3. **校准后**: 提供更可靠的概率预测

## ❗ 注意事项

1. **数据泄露**: 确保正确使用交叉验证
2. **过拟合**: 监控验证集性能
3. **计算成本**: 集成方法需要更多计算资源
4. **模型解释性**: 集成模型的可解释性较差

## 🔍 故障排除

### 常见问题

1. **内存不足**
   - 减少基础模型数量
   - 使用更小的数据集进行调试

2. **训练时间过长**
   - 减少交叉验证折数
   - 使用更简单的基础模型

3. **性能不佳**
   - 检查基础模型的多样性
   - 调整元学习器参数

### 错误处理

所有集成方法都包含完善的错误处理：
- 自动跳过失败的模型
- 记录详细的错误信息
- 提供降级策略

## 📞 支持

如果遇到问题，请：
1. 查看错误日志
2. 运行测试脚本验证环境
3. 检查数据格式和模型兼容性

## 🔄 更新日志

- v1.0: 初始版本，包含基础Stacking和Blending
- v1.1: 添加模型校准功能
- v1.2: 完善错误处理和性能优化
- v1.3: 添加详细示例和文档