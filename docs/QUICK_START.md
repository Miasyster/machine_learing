# 快速开始指南

## 概述

本指南将帮助您快速上手机器学习框架，在几分钟内完成第一个机器学习项目。

## 安装

### 环境要求

- Python 3.8+
- pip 或 conda

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd machine_learning

# 安装依赖
pip install -r requirements.txt

# 或使用conda
conda env create -f environment.yml
conda activate ml-framework
```

## 5分钟快速开始

### 1. 准备数据

```python
import pandas as pd
from src.data import DataLoader, DataPreprocessor

# 加载示例数据（或使用您自己的数据）
loader = DataLoader()

# 从CSV文件加载
data = loader.load_from_file('examples/data/sample_data.csv')

# 或创建示例数据
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
data['target'] = y

print(f"数据形状: {data.shape}")
print(data.head())
```

### 2. 数据预处理

```python
# 创建预处理器
preprocessor = DataPreprocessor()

# 添加预处理步骤
preprocessor.add_step('handle_missing', method='mean')
preprocessor.add_step('normalize', method='standard')

# 分离特征和目标
X = data.drop('target', axis=1)
y = data['target']

# 预处理数据
X_processed = preprocessor.fit_transform(X)

print(f"预处理后数据形状: {X_processed.shape}")
```

### 3. 训练模型

```python
from src.training import ModelTrainer
from sklearn.model_selection import train_test_split

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 创建并训练模型
trainer = ModelTrainer(
    model_type='random_forest',
    hyperparameters={'n_estimators': 100, 'random_state': 42}
)

model = trainer.train(X_train, y_train)
print("模型训练完成！")
```

### 4. 评估模型

```python
from src.training import ModelValidator

# 评估模型
validator = ModelValidator()
metrics = validator.evaluate(
    model, X_test, y_test,
    metrics=['accuracy', 'precision', 'recall', 'f1']
)

print("模型评估结果:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

### 5. 保存和部署模型

```python
from src.deployment import ModelSerializer, ModelInferenceEngine

# 保存模型
serializer = ModelSerializer()
serializer.save_model(
    model, 
    'my_first_model.joblib',
    format='joblib',
    metadata={'accuracy': metrics['accuracy']}
)

# 创建推理引擎
engine = ModelInferenceEngine('my_first_model.joblib')

# 进行预测
sample_data = X_test[:5]
predictions = engine.predict(sample_data)
print(f"预测结果: {predictions}")
```

🎉 **恭喜！** 您已经完成了第一个机器学习项目！

## 常见使用场景

### 场景1: 分类任务

```python
# 完整的分类任务示例
from src.data import DataLoader, DataPreprocessor
from src.training import ModelTrainer, ModelValidator
from src.deployment import ModelSerializer

def classification_pipeline(data_path, target_column):
    # 1. 加载数据
    loader = DataLoader()
    data = loader.load_from_file(data_path)
    
    # 2. 预处理
    preprocessor = DataPreprocessor()
    preprocessor.add_step('handle_missing', method='mode')
    preprocessor.add_step('encode_categorical', method='onehot')
    preprocessor.add_step('normalize', method='standard')
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_processed = preprocessor.fit_transform(X)
    
    # 3. 分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 4. 训练模型
    trainer = ModelTrainer('random_forest', {'n_estimators': 100})
    model = trainer.train(X_train, y_train)
    
    # 5. 评估
    validator = ModelValidator()
    metrics = validator.evaluate(model, X_test, y_test)
    
    # 6. 保存
    serializer = ModelSerializer()
    serializer.save_model(model, 'classification_model.joblib')
    
    return model, metrics

# 使用示例
# model, results = classification_pipeline('data.csv', 'target')
```

### 场景2: 回归任务

```python
def regression_pipeline(data_path, target_column):
    # 类似分类任务，但使用回归模型和指标
    loader = DataLoader()
    data = loader.load_from_file(data_path)
    
    preprocessor = DataPreprocessor()
    preprocessor.add_step('handle_missing', method='mean')
    preprocessor.add_step('remove_outliers', method='iqr')
    preprocessor.add_step('normalize', method='standard')
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_processed = preprocessor.fit_transform(X)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # 使用回归模型
    trainer = ModelTrainer('linear_regression')
    model = trainer.train(X_train, y_train)
    
    # 使用回归指标
    validator = ModelValidator()
    metrics = validator.evaluate(
        model, X_test, y_test,
        metrics=['mse', 'rmse', 'mae', 'r2']
    )
    
    return model, metrics
```

### 场景3: 超参数优化

```python
from src.training import HyperparameterOptimizer

def optimized_training(X_train, y_train):
    # 定义搜索空间
    search_space = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # 创建优化器
    optimizer = HyperparameterOptimizer(
        model_type='random_forest',
        optimization_method='grid_search'
    )
    
    # 执行优化
    best_params, best_score, history = optimizer.optimize(
        X_train, y_train,
        search_space=search_space,
        cv_folds=5,
        scoring='accuracy'
    )
    
    print(f"最佳参数: {best_params}")
    print(f"最佳得分: {best_score}")
    
    # 使用最佳参数训练最终模型
    trainer = ModelTrainer('random_forest', best_params)
    final_model = trainer.train(X_train, y_train)
    
    return final_model, best_params
```

## 监控和部署

### 添加监控

```python
from src.monitoring import PerformanceMonitor, AlertManager

# 创建性能监控
monitor = PerformanceMonitor(
    monitoring_interval=1.0,
    enable_system_monitoring=True
)

# 创建告警管理器
alert_manager = AlertManager()
alert_manager.add_rule(
    rule_id='high_latency',
    metric_name='inference_latency',
    threshold=1000,
    comparison='greater_than',
    severity='warning',
    description='推理延迟过高'
)

# 启动监控
monitor.start_monitoring()

# 在推理时记录指标
@monitor.track_inference
def predict_with_monitoring(data):
    return engine.predict(data)
```

### 创建API服务

```python
from flask import Flask, request, jsonify
from src.deployment import ModelInferenceEngine

app = Flask(__name__)
engine = ModelInferenceEngine('my_first_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取输入数据
        data = request.json
        input_df = pd.DataFrame([data])
        
        # 进行预测
        prediction = engine.predict(input_df)
        
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## 下一步

现在您已经掌握了基础用法，可以探索更高级的功能：

### 1. 查看完整示例

```bash
# 运行基础示例
python examples/basic_usage.py

# 运行高级训练示例
python examples/advanced_training.py

# 运行部署示例
python examples/deployment_example.py

# 运行监控示例
python examples/monitoring_example.py
```

### 2. 阅读详细文档

- [API参考文档](API_REFERENCE.md) - 完整的API说明
- [最佳实践指南](BEST_PRACTICES.md) - 生产环境最佳实践
- [部署指南](DEPLOYMENT.md) - 详细的部署说明

### 3. 自定义配置

创建您自己的配置文件：

```yaml
# config.yaml
data:
  default_format: "csv"
  cache_enabled: true
  
training:
  default_cv_folds: 5
  random_state: 42
  
deployment:
  model_format: "joblib"
  enable_caching: true
  
monitoring:
  log_level: "INFO"
  metrics_retention_days: 30
```

### 4. 集成到现有项目

```python
# 在现有项目中使用框架
from src import MLFramework

# 创建框架实例
ml = MLFramework(config_path='config.yaml')

# 使用框架功能
data = ml.load_data('data.csv')
model = ml.train_model(data, 'target', model_type='random_forest')
ml.deploy_model(model, 'production')
```

## 常见问题

### Q: 如何处理大数据集？

A: 使用分块处理和并行计算：

```python
# 分块处理大文件
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    processed_chunk = preprocessor.transform(chunk)
    # 处理每个块
```

### Q: 如何添加自定义模型？

A: 扩展ModelTrainer类：

```python
from src.training import ModelTrainer

class CustomModelTrainer(ModelTrainer):
    def _create_model(self):
        if self.model_type == 'my_custom_model':
            return MyCustomModel(**self.hyperparameters)
        return super()._create_model()
```

### Q: 如何集成外部数据源？

A: 扩展DataLoader类：

```python
from src.data import DataLoader

class CustomDataLoader(DataLoader):
    def load_from_custom_source(self, connection_params):
        # 实现自定义数据源加载逻辑
        pass
```

## 获取帮助

- 查看 [examples/](../examples/) 目录中的完整示例
- 阅读 [docs/](.) 目录中的详细文档
- 检查 [tests/](../tests/) 目录中的测试用例

祝您使用愉快！🚀