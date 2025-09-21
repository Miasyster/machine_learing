# 模型解释性文档

本文档介绍机器学习框架中的模型解释性功能，包括SHAP分析和特征重要性分析。

## 概述

模型解释性是现代机器学习的重要组成部分，特别是在需要理解模型决策过程的应用场景中。我们的框架提供了全面的解释性工具，支持：

- **SHAP (SHapley Additive exPlanations)** - 基于博弈论的特征贡献分析
- **特征重要性分析** - 多种方法计算特征重要性
- **可视化工具** - 丰富的图表展示解释结果
- **集成学习支持** - 与集成模型无缝集成

## 快速开始

### 基本使用

```python
from src.ensemble import VotingEnsemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 创建集成模型
models = [RandomForestClassifier(), LogisticRegression()]
ensemble = VotingEnsemble(models=models)
ensemble.fit(X_train, y_train)

# 解释预测
explanation = ensemble.explain_predictions(
    X_test, 
    method='both',  # 同时使用SHAP和特征重要性
    feature_names=feature_names
)

# 可视化结果
fig = ensemble.visualize_explanations(
    explanation['feature_importance'],
    plot_type='feature_importance'
)
```

### 单独使用解释器

```python
from src.explainability import SHAPExplainer, FeatureImportanceExplainer

# SHAP解释器
shap_explainer = SHAPExplainer(
    model=ensemble,
    background_data=X_train[:100],
    feature_names=feature_names
)
shap_result = shap_explainer.explain(X_test)

# 特征重要性解释器
fi_explainer = FeatureImportanceExplainer(
    model=ensemble,
    feature_names=feature_names
)
fi_result = fi_explainer.explain(X_test)
```

## 核心组件

### 1. ExplanationResult

解释结果的统一数据结构：

```python
@dataclass
class ExplanationResult:
    feature_names: List[str]
    feature_importance: Optional[np.ndarray] = None
    permutation_importance: Optional[np.ndarray] = None
    feature_importance_std: Optional[np.ndarray] = None
    shap_values: Optional[np.ndarray] = None
    shap_expected_value: Optional[float] = None
    shap_interaction_values: Optional[np.ndarray] = None
    instance_explanations: Optional[List[Dict]] = None
    global_explanation: Optional[Dict] = None
    visualization_data: Optional[Dict] = None
```

### 2. SHAPExplainer

SHAP值计算和分析：

```python
class SHAPExplainer(BaseExplainer):
    def __init__(self, 
                 model, 
                 background_data, 
                 feature_names,
                 explainer_type='auto'):
        # 自动选择合适的SHAP解释器类型
        
    def explain(self, X, **kwargs):
        # 计算SHAP值
        
    def explain_instance(self, X, instance_idx):
        # 解释单个实例
        
    def calculate_feature_importance(self, shap_values):
        # 从SHAP值计算特征重要性
```

**支持的SHAP解释器类型：**
- `TreeExplainer` - 适用于树模型
- `LinearExplainer` - 适用于线性模型
- `KernelExplainer` - 通用解释器
- `DeepExplainer` - 适用于深度学习模型

### 3. FeatureImportanceExplainer

特征重要性分析：

```python
class FeatureImportanceExplainer(BaseExplainer):
    def explain(self, X, methods=['builtin', 'permutation']):
        # 计算特征重要性
        
    def analyze_stability(self, X, n_iterations=10):
        # 分析特征重要性的稳定性
        
    def rank_features(self, importance_scores):
        # 特征排名
```

**支持的重要性计算方法：**
- `builtin` - 使用模型内置的特征重要性
- `permutation` - 排列重要性
- `drop_column` - 删除列重要性

### 4. ExplanationVisualizer

可视化工具：

```python
class ExplanationVisualizer:
    def plot_feature_importance(self, result, top_n=20):
        # 特征重要性条形图
        
    def plot_shap_summary(self, result, plot_type='bar'):
        # SHAP摘要图
        
    def plot_shap_waterfall(self, result, instance_idx):
        # SHAP瀑布图（单个实例）
        
    def create_explanation_dashboard(self, result):
        # 综合解释仪表板
```

## 详细功能

### SHAP分析

SHAP (SHapley Additive exPlanations) 基于博弈论中的Shapley值，为每个特征分配一个重要性分数。

#### 特点：
- **可加性**: 所有特征的SHAP值之和等于预测值与期望值的差
- **对称性**: 相同贡献的特征具有相同的SHAP值
- **虚拟性**: 不影响预测的特征SHAP值为0
- **效率性**: 所有特征的SHAP值之和等于总贡献

#### 使用示例：

```python
# 全局解释
shap_result = shap_explainer.explain(X_test)
print(f"平均绝对SHAP值: {np.abs(shap_result.shap_values).mean(axis=0)}")

# 局部解释
instance_explanation = shap_explainer.explain_instance(X_test, instance_idx=0)
print(f"实例0的特征贡献: {instance_explanation}")

# 交互效应
if shap_result.shap_interaction_values is not None:
    print("特征交互效应已计算")
```

### 特征重要性分析

提供多种特征重要性计算方法，适用于不同的模型和场景。

#### 内置重要性 (Builtin)
使用模型自带的特征重要性计算方法：

```python
fi_result = fi_explainer.explain(X_test, methods=['builtin'])
```

#### 排列重要性 (Permutation)
通过随机打乱特征值来评估特征重要性：

```python
fi_result = fi_explainer.explain(X_test, methods=['permutation'])
```

#### 删除列重要性 (Drop Column)
通过删除特征来评估其重要性：

```python
fi_result = fi_explainer.explain(X_test, methods=['drop_column'])
```

#### 稳定性分析
评估特征重要性的稳定性：

```python
stability_data = fi_explainer.analyze_stability(X_test, n_iterations=20)
print(f"最稳定的特征: {stability_data['stability_ranking'][:5]}")
```

### 可视化功能

#### 1. 特征重要性图

```python
visualizer = ExplanationVisualizer()
fig = visualizer.plot_feature_importance(
    result=fi_result,
    top_n=15,
    show_std=True,
    save_path='feature_importance.png'
)
```

#### 2. SHAP摘要图

```python
# 条形图
fig = visualizer.plot_shap_summary(
    result=shap_result,
    plot_type='bar',
    max_display=20
)

# 蜂群图
fig = visualizer.plot_shap_summary(
    result=shap_result,
    plot_type='beeswarm',
    max_display=20
)
```

#### 3. SHAP瀑布图

```python
fig = visualizer.plot_shap_waterfall(
    result=shap_result,
    instance_idx=0,
    save_path='waterfall.png'
)
```

#### 4. 特征依赖图

```python
fig = visualizer.plot_feature_dependence(
    X=X_test,
    shap_values=shap_result.shap_values,
    feature_idx=0,
    feature_names=feature_names,
    interaction_idx=1
)
```

#### 5. 综合仪表板

```python
fig = visualizer.create_explanation_dashboard(
    result=shap_result,
    X=X_test,
    instance_idx=0,
    save_path='dashboard.png'
)
```

## 与集成学习的集成

### 自动解释

集成模型提供了内置的解释方法：

```python
# 创建集成模型
ensemble = VotingEnsemble(models=[rf, lr, svm])
ensemble.fit(X_train, y_train)

# 一键解释
explanation = ensemble.explain_predictions(
    X_test,
    method='both',
    feature_names=feature_names
)

# 一键可视化
fig = ensemble.visualize_explanations(
    explanation['shap'],
    plot_type='dashboard'
)
```

### 支持的集成方法

- **VotingEnsemble** - 投票集成
- **StackingEnsemble** - 堆叠集成
- **BaggingEnsemble** - 装袋集成
- **BoostingEnsemble** - 提升集成
- **CalibratedEnsemble** - 校准集成

## 最佳实践

### 1. 选择合适的解释方法

- **SHAP**: 适用于需要精确特征贡献的场景
- **特征重要性**: 适用于快速特征筛选和排序
- **两者结合**: 获得最全面的解释

### 2. 背景数据选择

SHAP解释器需要背景数据来计算基准值：

```python
# 推荐使用训练数据的子集
background_data = X_train[:100]

# 或使用聚类中心
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=50)
background_data = kmeans.fit(X_train).cluster_centers_
```

### 3. 性能优化

- 对于大数据集，使用数据子集进行解释
- 选择合适的SHAP解释器类型
- 使用并行计算加速

```python
# 使用数据子集
sample_indices = np.random.choice(len(X_test), size=100, replace=False)
X_sample = X_test[sample_indices]

# 并行计算
explanation = ensemble.explain_predictions(
    X_sample,
    method='shap',
    n_jobs=4  # 使用4个进程
)
```

### 4. 结果解释

- 关注top特征的稳定性
- 结合业务知识验证解释结果
- 使用多种方法交叉验证

## 依赖库

解释性功能需要以下依赖库：

```bash
pip install shap matplotlib seaborn pandas numpy scikit-learn
```

可选依赖：
```bash
pip install plotly  # 交互式可视化
pip install ipywidgets  # Jupyter notebook支持
```

## 故障排除

### 常见问题

1. **SHAP安装问题**
   ```bash
   pip install shap --upgrade
   ```

2. **内存不足**
   - 减少样本数量
   - 使用更简单的SHAP解释器

3. **模型不兼容**
   - 检查模型是否支持predict方法
   - 使用KernelExplainer作为通用解决方案

### 错误处理

框架提供了完善的错误处理机制：

```python
try:
    explanation = ensemble.explain_predictions(X_test, method='shap')
except ImportError:
    print("SHAP库未安装，请安装后重试")
except ValueError as e:
    print(f"输入数据错误: {e}")
except Exception as e:
    print(f"解释过程出错: {e}")
```

## 示例代码

完整的使用示例请参考：
- `examples/explainability_example.py` - 综合示例
- `examples/quick_start.py` - 快速开始示例

## 扩展功能

### 自定义解释器

可以继承BaseExplainer创建自定义解释器：

```python
class CustomExplainer(BaseExplainer):
    def explain(self, X, **kwargs):
        # 实现自定义解释逻辑
        pass
```

### 自定义可视化

可以扩展ExplanationVisualizer添加新的图表类型：

```python
class CustomVisualizer(ExplanationVisualizer):
    def plot_custom_chart(self, result):
        # 实现自定义可视化
        pass
```

## 参考资料

- [SHAP官方文档](https://shap.readthedocs.io/)
- [Shapley值理论](https://en.wikipedia.org/wiki/Shapley_value)
- [可解释AI综述](https://arxiv.org/abs/1702.08608)