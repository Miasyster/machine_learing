"""模型训练框架

提供统一的模型训练接口和各种训练策略
"""

# 基础类
from .base import BaseTrainer, TrainingConfig, TrainingResult

# 训练器
from .supervised import SupervisedTrainer, ClassificationTrainer, RegressionTrainer
from .unsupervised import UnsupervisedTrainer, ClusteringTrainer, DimensionalityReductionTrainer, AnomalyDetectionTrainer
from .ensemble_trainer import EnsembleTrainer

# 时间序列训练
from .time_series_trainer import TimeSeriesTrainer, TimeSeriesTrainingConfig, train_time_series_models
from .time_series_validation import (
    TimeSeriesConfig,
    TimeSeriesDataSplitter,
    TimeSeriesCrossValidator,
    WalkForwardValidator,
    create_time_series_splits
)

# 模型选择
from .model_selection import (
    AutoMLTrainer,
    ModelSelector,
    HyperparameterOptimizer,
    ModelAnalyzer
)

# 训练策略
from .strategies import (
    EarlyStoppingStrategy,
    LearningRateScheduler,
    DataAugmentationStrategy,
    RegularizationStrategy,
    CrossValidationStrategy,
    FeatureSelectionStrategy,
    TrainingStrategy
)

# 工具函数
from .utils import (
    save_model,
    load_model,
    load_model_metadata,
    compare_models,
    plot_training_history,
    plot_validation_curve,
    plot_learning_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    generate_classification_report,
    calculate_model_complexity,
    estimate_memory_usage,
    benchmark_prediction_time,
    create_training_summary,
    setup_logging
)

__all__ = [
    # 基础类
    'BaseTrainer',
    'TrainingConfig', 
    'TrainingResult',
    
    # 训练器
    'SupervisedTrainer',
    'ClassificationTrainer',
    'RegressionTrainer',
    'UnsupervisedTrainer',
    'ClusteringTrainer',
    'DimensionalityReductionTrainer',
    'AnomalyDetectionTrainer',
    'EnsembleTrainer',
    
    # 模型选择
    'AutoMLTrainer',
    'ModelSelector',
    'HyperparameterOptimizer',
    'ModelAnalyzer',
    
    # 训练策略
    'EarlyStoppingStrategy',
    'LearningRateScheduler',
    'DataAugmentationStrategy',
    'RegularizationStrategy',
    'CrossValidationStrategy',
    'FeatureSelectionStrategy',
    'TrainingStrategy',
    
    # 工具函数
    'save_model',
    'load_model',
    'load_model_metadata',
    'compare_models',
    'plot_training_history',
    'plot_validation_curve',
    'plot_learning_curve',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'generate_classification_report',
    'calculate_model_complexity',
    'estimate_memory_usage',
    'benchmark_prediction_time',
    'create_training_summary',
    'setup_logging'
]

__version__ = "0.1.0"