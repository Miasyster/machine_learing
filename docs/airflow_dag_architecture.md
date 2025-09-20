# Airflow DAG架构设计

## 📊 现有模块分析

### 1. 数据获取模块 (BinanceDataFetcher)
- **功能**: 从币安交易所获取K线数据
- **主要方法**: 
  - `get_klines()`: 获取历史K线数据
  - `get_latest_price()`: 获取最新价格
- **输出**: 标准化的DataFrame格式数据
- **特点**: 支持多种时间间隔，内置错误处理和重试机制

### 2. 数据存储模块 (DataStorage)
- **功能**: 管理数据的保存和读取
- **主要方法**:
  - `save_klines_data()`: 保存K线数据
  - `load_klines_data()`: 加载K线数据
  - `append_data()`: 增量数据追加
- **特点**: 支持CSV和Parquet格式，自动目录管理

### 3. 数据清洗模块 (DataCleaner)
- **功能**: 数据质量处理和清洗
- **主要方法**:
  - `clean_data_pipeline()`: 完整清洗流水线
  - `fill_missing_values()`: 缺失值处理
  - `detect_outliers()`: 异常值检测
- **特点**: 集成数据完整性检查，支持多种清洗策略

### 4. 数据完整性检查模块 (DataIntegrityChecker)
- **功能**: 数据质量验证和完整性检查
- **主要方法**:
  - `comprehensive_integrity_check()`: 综合完整性检查
  - `check_date_continuity()`: 日期连续性检查
  - `check_ohlc_logic()`: OHLC价格逻辑检查
- **特点**: 多维度数据质量评估，生成详细报告

### 5. 特征工程模块 (FeatureEngineer)
- **功能**: 技术指标计算和特征生成
- **主要方法**:
  - `calculate_feature()`: 单个特征计算
  - `calculate_multiple_features()`: 批量特征计算
- **特点**: 支持15+种技术指标，可扩展架构

## 🏗️ Airflow DAG架构设计

### DAG依赖关系图
```
数据获取DAG (data_ingestion_dag)
    ↓
数据质量检查DAG (data_quality_dag)
    ↓
数据清洗DAG (data_cleaning_dag)
    ↓
特征工程DAG (feature_engineering_dag)
    ↓
数据验证DAG (data_validation_dag)
```

### 1. 数据获取DAG (data_ingestion_dag)
**调度频率**: 每小时执行一次
**任务流程**:
```
start_task
    ↓
check_api_connection
    ↓
fetch_btc_data → fetch_eth_data → fetch_bnb_data (并行)
    ↓
validate_raw_data
    ↓
save_raw_data
    ↓
end_task
```

**任务详情**:
- `check_api_connection`: 验证Binance API连接
- `fetch_*_data`: 获取各币种的K线数据
- `validate_raw_data`: 原始数据基础验证
- `save_raw_data`: 保存到存储系统

### 2. 数据质量检查DAG (data_quality_dag)
**调度频率**: 数据获取完成后触发
**任务流程**:
```
start_task
    ↓
load_latest_data
    ↓
date_continuity_check → ohlc_logic_check → volume_logic_check (并行)
    ↓
generate_quality_report
    ↓
quality_alert_check
    ↓
end_task
```

### 3. 数据清洗DAG (data_cleaning_dag)
**调度频率**: 数据质量检查通过后触发
**任务流程**:
```
start_task
    ↓
load_raw_data
    ↓
missing_value_handling
    ↓
outlier_detection
    ↓
time_series_alignment
    ↓
integrity_check
    ↓
save_cleaned_data
    ↓
end_task
```

### 4. 特征工程DAG (feature_engineering_dag)
**调度频率**: 数据清洗完成后触发
**任务流程**:
```
start_task
    ↓
load_cleaned_data
    ↓
price_features → volume_features → volatility_features (并行)
    ↓
technical_indicators
    ↓
feature_validation
    ↓
save_features
    ↓
end_task
```

### 5. 数据验证DAG (data_validation_dag)
**调度频率**: 特征工程完成后触发
**任务流程**:
```
start_task
    ↓
load_feature_data
    ↓
feature_quality_check
    ↓
data_drift_detection
    ↓
generate_final_report
    ↓
send_notifications
    ↓
end_task
```

## 🔧 技术实现要点

### 1. 任务包装器设计
- 为每个现有模块创建Airflow任务包装器
- 统一的错误处理和日志记录
- 参数化配置支持

### 2. 数据传递机制
- 使用XCom传递小量元数据
- 文件系统传递大量数据
- 数据路径标准化

### 3. 错误处理策略
- 任务级别重试机制
- 数据质量阈值告警
- 失败任务通知

### 4. 监控和告警
- 数据质量指标监控
- 任务执行时间监控
- 异常情况自动告警

## 📋 配置管理

### 1. DAG配置
- 调度时间配置
- 重试策略配置
- 依赖关系配置

### 2. 数据源配置
- API连接配置
- 数据存储路径配置
- 数据格式配置

### 3. 质量阈值配置
- 数据完整性阈值
- 特征质量阈值
- 告警触发条件

## 🚀 部署考虑

### 1. 资源需求
- CPU: 中等计算需求
- 内存: 数据处理需要足够内存
- 存储: 历史数据存储需求

### 2. 扩展性
- 支持新增数据源
- 支持新增特征指标
- 支持多环境部署

### 3. 可维护性
- 模块化设计
- 标准化接口
- 完善的文档和测试