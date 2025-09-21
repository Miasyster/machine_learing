"""模型评估系统

提供全面的模型评估功能，包括性能指标计算、可视化、模型比较和报告生成"""

# 基础类
from .base import EvaluationConfig, EvaluationResult, BaseEvaluator

# 评估器
from .classification import ClassificationEvaluator
from .regression import RegressionEvaluator

# 可视化
from .visualization import EvaluationVisualizer, create_summary_table, save_evaluation_report

# 比较分析
from .comparison import ModelComparator, quick_model_comparison

# 报告生成
from .reports import EvaluationReportGenerator, create_quick_report, export_results_to_json, export_results_to_csv

# 工具函数
from .utils import (
    bootstrap_confidence_interval, get_metric_function, cross_validate_with_groups,
    calculate_learning_curve, calculate_validation_curve, evaluate_model_stability,
    benchmark_prediction_time, calculate_model_complexity, save_evaluation_results,
    load_evaluation_results, create_evaluation_summary, detect_overfitting,
    calculate_feature_stability, generate_evaluation_config
)

__all__ = [
    # 基础类
    'EvaluationConfig', 'EvaluationResult', 'BaseEvaluator',
    
    # 评估器
    'ClassificationEvaluator', 'RegressionEvaluator',
    
    # 可视化
    'EvaluationVisualizer', 'create_summary_table', 'save_evaluation_report',
    
    # 比较分析
    'ModelComparator', 'quick_model_comparison',
    
    # 报告生成
    'EvaluationReportGenerator', 'create_quick_report', 'export_results_to_json', 'export_results_to_csv',
    
    # 工具函数
    'bootstrap_confidence_interval', 'get_metric_function', 'cross_validate_with_groups',
    'calculate_learning_curve', 'calculate_validation_curve', 'evaluate_model_stability',
    'benchmark_prediction_time', 'calculate_model_complexity', 'save_evaluation_results',
    'load_evaluation_results', 'create_evaluation_summary', 'detect_overfitting',
    'calculate_feature_stability', 'generate_evaluation_config'
]

__version__ = "1.0.0"