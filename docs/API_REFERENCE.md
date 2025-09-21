# API 参考文档

## 概述

本文档提供了机器学习框架的完整API参考，包括所有模块、类和函数的详细说明。

## 目录

- [数据处理模块 (src.data)](#数据处理模块)
- [模型训练模块 (src.training)](#模型训练模块)
- [模型部署模块 (src.deployment)](#模型部署模块)
- [监控系统模块 (src.monitoring)](#监控系统模块)

---

## 数据处理模块

### DataLoader

数据加载器，支持多种数据源的加载。

#### 类定义

```python
class DataLoader:
    def __init__(self, config: Optional[Dict] = None)
```

#### 方法

##### load_from_file()

```python
def load_from_file(self, file_path: str, **kwargs) -> pd.DataFrame
```

从文件加载数据。

**参数:**
- `file_path` (str): 文件路径
- `**kwargs`: 额外的加载参数

**返回:**
- `pd.DataFrame`: 加载的数据

**支持的文件格式:**
- CSV (.csv)
- JSON (.json)
- Parquet (.parquet)
- Excel (.xlsx, .xls)

**示例:**
```python
loader = DataLoader()
data = loader.load_from_file("data.csv", sep=",", header=0)
```

##### load_from_database()

```python
def load_from_database(self, connection_string: str, query: str) -> pd.DataFrame
```

从数据库加载数据。

**参数:**
- `connection_string` (str): 数据库连接字符串
- `query` (str): SQL查询语句

**返回:**
- `pd.DataFrame`: 查询结果

**示例:**
```python
loader = DataLoader()
data = loader.load_from_database(
    "sqlite:///example.db",
    "SELECT * FROM users"
)
```

##### load_from_api()

```python
def load_from_api(self, url: str, headers: Optional[Dict] = None, 
                  params: Optional[Dict] = None) -> pd.DataFrame
```

从API加载数据。

**参数:**
- `url` (str): API端点URL
- `headers` (Dict, optional): HTTP请求头
- `params` (Dict, optional): 请求参数

**返回:**
- `pd.DataFrame`: API响应数据

### DataPreprocessor

数据预处理器，提供数据清洗和转换功能。

#### 类定义

```python
class DataPreprocessor:
    def __init__(self, config: Optional[Dict] = None)
```

#### 方法

##### add_step()

```python
def add_step(self, step_name: str, method: str, **kwargs) -> None
```

添加预处理步骤。

**参数:**
- `step_name` (str): 步骤名称
- `method` (str): 预处理方法
- `**kwargs`: 方法参数

**支持的方法:**
- `handle_missing`: 处理缺失值
- `remove_outliers`: 移除异常值
- `normalize`: 数据标准化
- `encode_categorical`: 分类变量编码

**示例:**
```python
preprocessor = DataPreprocessor()
preprocessor.add_step('normalize', method='standard')
preprocessor.add_step('handle_missing', method='mean', columns=['age'])
```

##### fit_transform()

```python
def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame
```

拟合并转换数据。

**参数:**
- `data` (pd.DataFrame): 输入数据

**返回:**
- `pd.DataFrame`: 预处理后的数据

### FeatureEngineer

特征工程器，提供特征创建和选择功能。

#### 类定义

```python
class FeatureEngineer:
    def __init__(self, config: Optional[Dict] = None)
```

#### 方法

##### add_transformer()

```python
def add_transformer(self, transformer_name: str, **kwargs) -> None
```

添加特征转换器。

**参数:**
- `transformer_name` (str): 转换器名称
- `**kwargs`: 转换器参数

**支持的转换器:**
- `polynomial`: 多项式特征
- `interaction`: 交互特征
- `binning`: 特征分箱
- `scaling`: 特征缩放

##### add_selector()

```python
def add_selector(self, selector_name: str, **kwargs) -> None
```

添加特征选择器。

**参数:**
- `selector_name` (str): 选择器名称
- `**kwargs`: 选择器参数

**支持的选择器:**
- `variance_threshold`: 方差阈值选择
- `correlation_threshold`: 相关性阈值选择
- `univariate_selection`: 单变量选择
- `recursive_elimination`: 递归特征消除

---

## 模型训练模块

### ModelTrainer

模型训练器，支持多种机器学习算法。

#### 类定义

```python
class ModelTrainer:
    def __init__(self, model_type: str, hyperparameters: Optional[Dict] = None)
```

**参数:**
- `model_type` (str): 模型类型
- `hyperparameters` (Dict, optional): 超参数

**支持的模型类型:**
- `linear_regression`: 线性回归
- `logistic_regression`: 逻辑回归
- `random_forest`: 随机森林
- `gradient_boosting`: 梯度提升
- `svm`: 支持向量机
- `neural_network`: 神经网络

#### 方法

##### train()

```python
def train(self, X: pd.DataFrame, y: pd.Series, 
          validation_data: Optional[Tuple] = None) -> Any
```

训练模型。

**参数:**
- `X` (pd.DataFrame): 特征数据
- `y` (pd.Series): 目标变量
- `validation_data` (Tuple, optional): 验证数据 (X_val, y_val)

**返回:**
- `Any`: 训练好的模型

**示例:**
```python
trainer = ModelTrainer(
    model_type='random_forest',
    hyperparameters={'n_estimators': 100, 'max_depth': 10}
)
model = trainer.train(X_train, y_train)
```

### HyperparameterOptimizer

超参数优化器，支持多种优化算法。

#### 类定义

```python
class HyperparameterOptimizer:
    def __init__(self, model_type: str, optimization_method: str = 'grid_search')
```

**参数:**
- `model_type` (str): 模型类型
- `optimization_method` (str): 优化方法

**支持的优化方法:**
- `grid_search`: 网格搜索
- `random_search`: 随机搜索
- `bayesian`: 贝叶斯优化

#### 方法

##### optimize()

```python
def optimize(self, X: pd.DataFrame, y: pd.Series, 
             search_space: Dict, cv_folds: int = 5,
             n_trials: int = 100, scoring: str = 'accuracy') -> Tuple[Dict, float, List]
```

执行超参数优化。

**参数:**
- `X` (pd.DataFrame): 特征数据
- `y` (pd.Series): 目标变量
- `search_space` (Dict): 搜索空间
- `cv_folds` (int): 交叉验证折数
- `n_trials` (int): 试验次数
- `scoring` (str): 评分指标

**返回:**
- `Tuple[Dict, float, List]`: (最佳参数, 最佳得分, 优化历史)

### ModelValidator

模型验证器，提供模型评估功能。

#### 类定义

```python
class ModelValidator:
    def __init__(self, config: Optional[Dict] = None)
```

#### 方法

##### evaluate()

```python
def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series,
             metrics: Optional[List[str]] = None) -> Dict[str, float]
```

评估模型性能。

**参数:**
- `model` (Any): 训练好的模型
- `X` (pd.DataFrame): 测试特征
- `y` (pd.Series): 测试目标
- `metrics` (List[str], optional): 评估指标

**返回:**
- `Dict[str, float]`: 评估结果

**支持的指标:**
- 分类: `accuracy`, `precision`, `recall`, `f1`, `auc`
- 回归: `mse`, `rmse`, `mae`, `r2`

---

## 模型部署模块

### ModelSerializer

模型序列化器，支持多种序列化格式。

#### 类定义

```python
class ModelSerializer:
    def __init__(self, config: Optional[Dict] = None)
```

#### 方法

##### save_model()

```python
def save_model(self, model: Any, file_path: str, format: str = 'pickle',
               compress: bool = False, metadata: Optional[Dict] = None) -> None
```

保存模型。

**参数:**
- `model` (Any): 要保存的模型
- `file_path` (str): 保存路径
- `format` (str): 序列化格式
- `compress` (bool): 是否压缩
- `metadata` (Dict, optional): 元数据

**支持的格式:**
- `pickle`: Python pickle格式
- `joblib`: Joblib格式
- `json`: JSON格式（仅支持简单模型）

### ModelInferenceEngine

模型推理引擎，提供高效的模型推理服务。

#### 类定义

```python
class ModelInferenceEngine:
    def __init__(self, model_path: str, enable_caching: bool = True,
                 cache_size: int = 1000)
```

**参数:**
- `model_path` (str): 模型文件路径
- `enable_caching` (bool): 是否启用缓存
- `cache_size` (int): 缓存大小

#### 方法

##### predict()

```python
def predict(self, data: pd.DataFrame) -> np.ndarray
```

单次预测。

**参数:**
- `data` (pd.DataFrame): 输入数据

**返回:**
- `np.ndarray`: 预测结果

##### batch_predict()

```python
def batch_predict(self, data: pd.DataFrame, batch_size: int = 32) -> np.ndarray
```

批量预测。

**参数:**
- `data` (pd.DataFrame): 输入数据
- `batch_size` (int): 批次大小

**返回:**
- `np.ndarray`: 预测结果

##### predict_async()

```python
async def predict_async(self, data: pd.DataFrame) -> np.ndarray
```

异步预测。

**参数:**
- `data` (pd.DataFrame): 输入数据

**返回:**
- `np.ndarray`: 预测结果

### ModelVersionManager

模型版本管理器，提供模型版本控制功能。

#### 类定义

```python
class ModelVersionManager:
    def __init__(self, models_dir: str, use_git: bool = False)
```

**参数:**
- `models_dir` (str): 模型存储目录
- `use_git` (bool): 是否使用Git版本控制

#### 方法

##### create_version()

```python
def create_version(self, model: Any, version: str, 
                   metadata: Optional[Dict] = None) -> VersionInfo
```

创建模型版本。

**参数:**
- `model` (Any): 模型对象
- `version` (str): 版本号
- `metadata` (Dict, optional): 版本元数据

**返回:**
- `VersionInfo`: 版本信息

##### load_model()

```python
def load_model(self, version: str) -> Any
```

加载指定版本的模型。

**参数:**
- `version` (str): 版本号

**返回:**
- `Any`: 模型对象

---

## 监控系统模块

### PerformanceMonitor

性能监控器，监控模型和系统性能。

#### 类定义

```python
class PerformanceMonitor:
    def __init__(self, monitoring_interval: float = 1.0,
                 enable_system_monitoring: bool = True,
                 enable_model_monitoring: bool = True)
```

**参数:**
- `monitoring_interval` (float): 监控间隔（秒）
- `enable_system_monitoring` (bool): 是否启用系统监控
- `enable_model_monitoring` (bool): 是否启用模型监控

#### 方法

##### start_monitoring()

```python
def start_monitoring() -> None
```

开始监控。

##### stop_monitoring()

```python
def stop_monitoring() -> None
```

停止监控。

##### record_inference_metrics()

```python
def record_inference_metrics(self, inference_time: float,
                           memory_usage: float, cpu_usage: float) -> None
```

记录推理指标。

**参数:**
- `inference_time` (float): 推理时间
- `memory_usage` (float): 内存使用量
- `cpu_usage` (float): CPU使用率

##### get_statistics()

```python
def get_statistics() -> Dict[str, Any]
```

获取监控统计信息。

**返回:**
- `Dict[str, Any]`: 统计信息

### AlertManager

告警管理器，提供异常检测和告警功能。

#### 类定义

```python
class AlertManager:
    def __init__(self, config: Optional[Dict] = None)
```

#### 方法

##### add_rule()

```python
def add_rule(self, rule_id: str, metric_name: str, threshold: float,
             comparison: str, severity: str, description: str) -> None
```

添加告警规则。

**参数:**
- `rule_id` (str): 规则ID
- `metric_name` (str): 指标名称
- `threshold` (float): 阈值
- `comparison` (str): 比较操作符
- `severity` (str): 严重级别
- `description` (str): 描述

**比较操作符:**
- `greater_than`: 大于
- `less_than`: 小于
- `equal_to`: 等于

**严重级别:**
- `info`: 信息
- `warning`: 警告
- `critical`: 严重

##### check_metric()

```python
def check_metric(self, metric_name: str, value: float) -> None
```

检查指标值是否触发告警。

**参数:**
- `metric_name` (str): 指标名称
- `value` (float): 指标值

### MetricCollector

指标收集器，收集和存储各种指标数据。

#### 类定义

```python
class MetricCollector:
    def __init__(self, storage_backend: str = 'memory',
                 retention_days: int = 30)
```

**参数:**
- `storage_backend` (str): 存储后端
- `retention_days` (int): 数据保留天数

**支持的存储后端:**
- `memory`: 内存存储
- `sqlite`: SQLite数据库
- `postgresql`: PostgreSQL数据库

#### 方法

##### register_metric()

```python
def register_metric(self, name: str, metric_type: str,
                   description: str = "") -> None
```

注册指标。

**参数:**
- `name` (str): 指标名称
- `metric_type` (str): 指标类型
- `description` (str): 描述

**指标类型:**
- `gauge`: 仪表盘（瞬时值）
- `counter`: 计数器（累积值）
- `histogram`: 直方图（分布）

##### record_metric()

```python
def record_metric(self, name: str, value: float,
                 timestamp: Optional[datetime] = None,
                 tags: Optional[Dict] = None) -> None
```

记录指标值。

**参数:**
- `name` (str): 指标名称
- `value` (float): 指标值
- `timestamp` (datetime, optional): 时间戳
- `tags` (Dict, optional): 标签

##### query_metrics()

```python
def query_metrics(self, metric_name: str, start_time: datetime,
                 end_time: datetime, tags: Optional[Dict] = None) -> List[Dict]
```

查询指标数据。

**参数:**
- `metric_name` (str): 指标名称
- `start_time` (datetime): 开始时间
- `end_time` (datetime): 结束时间
- `tags` (Dict, optional): 标签过滤

**返回:**
- `List[Dict]`: 指标数据列表

---

## 工具函数

### 数据处理工具

#### validate_data()

```python
def validate_data(data: pd.DataFrame, schema: Dict) -> Tuple[bool, List[str]]
```

验证数据格式。

**参数:**
- `data` (pd.DataFrame): 数据
- `schema` (Dict): 数据模式

**返回:**
- `Tuple[bool, List[str]]`: (是否有效, 错误列表)

#### split_data()

```python
def split_data(data: pd.DataFrame, target_column: str,
               test_size: float = 0.2, random_state: int = 42) -> Tuple
```

分割数据集。

**参数:**
- `data` (pd.DataFrame): 数据
- `target_column` (str): 目标列名
- `test_size` (float): 测试集比例
- `random_state` (int): 随机种子

**返回:**
- `Tuple`: (X_train, X_test, y_train, y_test)

### 模型工具

#### save_model_config()

```python
def save_model_config(config: Dict, file_path: str) -> None
```

保存模型配置。

#### load_model_config()

```python
def load_model_config(file_path: str) -> Dict
```

加载模型配置。

### 监控工具

#### setup_monitoring_logging()

```python
def setup_monitoring_logging(log_level: str = 'INFO',
                           log_file: Optional[str] = None) -> logging.Logger
```

设置监控日志。

#### create_health_check()

```python
def create_health_check(components: List[str]) -> Callable
```

创建健康检查函数。

---

## 异常处理

### 自定义异常

#### DataError

数据处理相关异常。

```python
class DataError(Exception):
    pass
```

#### ModelError

模型相关异常。

```python
class ModelError(Exception):
    pass
```

#### DeploymentError

部署相关异常。

```python
class DeploymentError(Exception):
    pass
```

#### MonitoringError

监控相关异常。

```python
class MonitoringError(Exception):
    pass
```

---

## 配置

### 全局配置

框架支持通过配置文件或环境变量进行配置。

#### 配置文件示例

```yaml
# config.yaml
data:
  default_format: "csv"
  cache_enabled: true
  cache_size: 1000

training:
  default_cv_folds: 5
  random_state: 42
  n_jobs: -1

deployment:
  default_format: "pickle"
  compression: true
  model_registry: "local"

monitoring:
  log_level: "INFO"
  metrics_retention_days: 30
  alert_cooldown: 300
```

#### 环境变量

```bash
ML_FRAMEWORK_LOG_LEVEL=INFO
ML_FRAMEWORK_CACHE_SIZE=1000
ML_FRAMEWORK_N_JOBS=4
```

---

## 版本兼容性

- Python: >= 3.8
- pandas: >= 1.3.0
- scikit-learn: >= 1.0.0
- numpy: >= 1.21.0

---

## 更新日志

### v1.0.0 (当前版本)

- 初始版本发布
- 完整的数据处理、训练、部署和监控功能
- 支持多种机器学习算法
- 完整的API文档和示例