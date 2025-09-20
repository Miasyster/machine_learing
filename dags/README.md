# Airflow DAGs

这个目录包含所有的Airflow DAG定义文件。

## 📁 目录结构

```
dags/
├── README.md                    # 本文件
├── config/                      # DAG配置文件
│   ├── dag_config.yaml         # DAG通用配置
│   ├── data_sources.yaml       # 数据源配置
│   └── quality_thresholds.yaml # 数据质量阈值配置
├── operators/                   # 自定义操作符
│   ├── __init__.py
│   ├── data_ingestion_operator.py
│   ├── data_cleaning_operator.py
│   └── feature_engineering_operator.py
├── utils/                       # DAG工具函数
│   ├── __init__.py
│   ├── task_helpers.py
│   └── notification_helpers.py
├── data_ingestion_dag.py        # 数据获取DAG
├── data_quality_dag.py          # 数据质量检查DAG
├── data_cleaning_dag.py         # 数据清洗DAG
├── feature_engineering_dag.py   # 特征工程DAG
└── data_validation_dag.py       # 数据验证DAG
```

## 🔄 DAG执行顺序

1. **data_ingestion_dag**: 从Binance获取原始数据
2. **data_quality_dag**: 检查原始数据质量
3. **data_cleaning_dag**: 清洗和预处理数据
4. **feature_engineering_dag**: 生成技术指标特征
5. **data_validation_dag**: 最终数据验证和报告

## ⚙️ 配置说明

### DAG通用配置 (dag_config.yaml)
- 调度时间设置
- 重试策略
- 超时设置
- 邮件通知配置

### 数据源配置 (data_sources.yaml)
- Binance API配置
- 支持的交易对列表
- 数据获取参数

### 质量阈值配置 (quality_thresholds.yaml)
- 数据完整性阈值
- 异常值检测参数
- 告警触发条件

## 🚀 使用方法

1. 确保Airflow已正确安装和配置
2. 将此目录设置为Airflow的DAG目录
3. 配置相关的连接和变量
4. 启动Airflow调度器和Web服务器

## 📊 监控和告警

- 所有DAG都配置了失败告警
- 数据质量指标会自动监控
- 异常情况会发送邮件通知