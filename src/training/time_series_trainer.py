"""
时间序列训练器

专门用于时间序列数据的训练器，集成了：
1. 时间序列数据划分
2. 时间序列交叉验证
3. Walk-forward验证
4. 模型性能评估和比较
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path
import joblib
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .base import BaseTrainer, TrainingConfig, TrainingResult
from .time_series_validation import (
    TimeSeriesConfig, TimeSeriesDataSplitter, 
    TimeSeriesCrossValidator, WalkForwardValidator,
    create_time_series_splits
)

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesTrainingConfig(TrainingConfig):
    """时间序列训练配置"""
    
    # 时间序列特定配置
    time_series_config: Optional[TimeSeriesConfig] = None
    
    # 验证策略
    use_time_series_cv: bool = True
    use_walk_forward: bool = True
    
    # 性能评估
    regression_metrics: List[str] = field(default_factory=lambda: ['mse', 'mae', 'r2'])
    classification_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'precision', 'recall', 'f1'])
    
    # 模型保存
    save_cv_models: bool = False
    save_predictions: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        if self.time_series_config is None:
            self.time_series_config = TimeSeriesConfig()


class TimeSeriesTrainer(BaseTrainer):
    """时间序列训练器"""
    
    def __init__(self, 
                 config: Optional[TimeSeriesTrainingConfig] = None,
                 models: Optional[Dict[str, BaseEstimator]] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        初始化时间序列训练器
        
        Args:
            config: 训练配置
            models: 模型字典
            random_state: 随机种子
            verbose: 是否显示详细信息
        """
        if config is None:
            config = TimeSeriesTrainingConfig()
            
        super().__init__(config, random_state, verbose)
        
        self.models = models or {}
        self.ts_config = config.time_series_config
        self.data_splits = None
        self.cv_results = {}
        self.walk_forward_results = {}
        
    def add_model(self, name: str, model: BaseEstimator) -> 'TimeSeriesTrainer':
        """添加模型"""
        self.models[name] = model
        return self
        
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            time_index: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]] = None,
            **kwargs) -> 'TimeSeriesTrainer':
        """
        训练所有模型
        
        Args:
            X: 特征数据
            y: 目标变量
            time_index: 时间索引
            **kwargs: 其他参数
            
        Returns:
            self
        """
        if not self.models:
            raise ValueError("没有添加任何模型，请先使用add_model()添加模型")
            
        logger.info(f"开始时间序列训练，数据形状: {X.shape}")
        
        # 1. 数据划分
        self._split_data(X, y, time_index)
        
        # 2. 训练每个模型
        for model_name, model in self.models.items():
            logger.info(f"训练模型: {model_name}")
            self._train_single_model(model_name, model)
            
        logger.info("所有模型训练完成")
        return self
    
    def _split_data(self, 
                   X: Union[np.ndarray, pd.DataFrame], 
                   y: Union[np.ndarray, pd.Series],
                   time_index: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]] = None):
        """划分数据"""
        splitter = TimeSeriesDataSplitter(self.ts_config)
        
        # 基础划分
        if self.ts_config.val_ratio > 0:
            X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(X, y, time_index)
            self.data_splits = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
        else:
            X_train, X_test, y_train, y_test = splitter.split_data(X, y, time_index)
            self.data_splits = {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test
            }
        
        # 交叉验证划分
        if self.config.use_time_series_cv:
            cv = TimeSeriesCrossValidator(self.ts_config)
            cv_splits = list(cv.split(X_train))
            self.data_splits['cv_splits'] = cv_splits
            self.data_splits['n_cv_splits'] = len(cv_splits)
        
        # 记录划分信息
        split_info = splitter.get_split_info(len(X))
        self.data_splits['split_info'] = split_info
        
        if self.verbose:
            logger.info(f"数据划分完成: {split_info}")
    
    def _train_single_model(self, model_name: str, model: BaseEstimator):
        """训练单个模型"""
        start_time = time.time()
        
        # 基础训练
        X_train = self.data_splits['X_train']
        y_train = self.data_splits['y_train']
        
        # 克隆模型进行训练
        trained_model = clone(model)
        trained_model.fit(X_train, y_train)
        
        # 创建训练结果
        result = TrainingResult(
            model_name=model_name,
            task_type=self._get_task_type(),
            training_time=time.time() - start_time,
            best_model=trained_model,
            config=self.config
        )
        
        # 评估模型
        self._evaluate_model_performance(result)
        
        # 时间序列交叉验证
        if self.config.use_time_series_cv and 'cv_splits' in self.data_splits:
            self._perform_time_series_cv(model_name, model, result)
        
        # Walk-forward验证
        if self.config.use_walk_forward:
            self._perform_walk_forward(model_name, model, result)
        
        # 保存结果
        self.training_results_[model_name] = result
        
        if self.verbose:
            logger.info(f"模型 {model_name} 训练完成，用时 {result.training_time:.2f}秒")
    
    def _evaluate_model_performance(self, result: TrainingResult):
        """评估模型性能"""
        model = result.best_model
        
        # 训练集评估
        X_train = self.data_splits['X_train']
        y_train = self.data_splits['y_train']
        train_pred = model.predict(X_train)
        result.train_predictions = train_pred
        result.train_scores = self._calculate_metrics(y_train, train_pred)
        
        # 验证集评估（如果存在）
        if 'X_val' in self.data_splits:
            X_val = self.data_splits['X_val']
            y_val = self.data_splits['y_val']
            val_pred = model.predict(X_val)
            result.val_predictions = val_pred
            result.val_scores = self._calculate_metrics(y_val, val_pred)
        
        # 测试集评估
        X_test = self.data_splits['X_test']
        y_test = self.data_splits['y_test']
        test_pred = model.predict(X_test)
        result.test_predictions = test_pred
        result.test_scores = self._calculate_metrics(y_test, test_pred)
    
    def _perform_time_series_cv(self, model_name: str, model: BaseEstimator, result: TrainingResult):
        """执行时间序列交叉验证"""
        cv_scores = {}
        cv_predictions = []
        
        X_train = self.data_splits['X_train']
        y_train = self.data_splits['y_train']
        cv_splits = self.data_splits['cv_splits']
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            # 获取折叠数据
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # 训练模型
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            
            # 预测和评估
            fold_pred = fold_model.predict(X_fold_val)
            fold_scores = self._calculate_metrics(y_fold_val, fold_pred)
            
            # 保存结果
            for metric, score in fold_scores.items():
                if metric not in cv_scores:
                    cv_scores[metric] = []
                cv_scores[metric].append(score)
            
            cv_predictions.extend(fold_pred)
            
            if self.config.save_cv_models:
                if 'cv_models' not in result.metadata:
                    result.metadata['cv_models'] = []
                result.metadata['cv_models'].append(fold_model)
        
        # 保存交叉验证结果
        result.cv_scores = cv_scores
        self.cv_results[model_name] = {
            'scores': cv_scores,
            'predictions': cv_predictions,
            'mean_scores': {metric: np.mean(scores) for metric, scores in cv_scores.items()},
            'std_scores': {metric: np.std(scores) for metric, scores in cv_scores.items()}
        }
        
        if self.verbose:
            for metric, scores in cv_scores.items():
                logger.info(f"CV {metric}: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
    
    def _perform_walk_forward(self, model_name: str, model: BaseEstimator, result: TrainingResult):
        """执行walk-forward验证"""
        validator = WalkForwardValidator(self.ts_config)
        
        X_train = self.data_splits['X_train']
        y_train = self.data_splits['y_train']
        
        # 执行验证
        wf_result = validator.validate(model, X_train, y_train)
        
        # 保存结果
        self.walk_forward_results[model_name] = wf_result
        result.metadata['walk_forward'] = wf_result
        
        if self.verbose:
            logger.info(f"Walk-forward验证: 平均得分 {wf_result['mean_score']:.4f} "
                       f"(+/- {wf_result['std_score']:.4f})")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        if self._is_classification():
            # 分类指标
            if 'accuracy' in self.config.classification_metrics:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
            if 'precision' in self.config.classification_metrics:
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            if 'recall' in self.config.classification_metrics:
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            if 'f1' in self.config.classification_metrics:
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:
            # 回归指标
            if 'mse' in self.config.regression_metrics:
                metrics['mse'] = mean_squared_error(y_true, y_pred)
            if 'mae' in self.config.regression_metrics:
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
            if 'r2' in self.config.regression_metrics:
                metrics['r2'] = r2_score(y_true, y_pred)
            if 'rmse' in self.config.regression_metrics:
                metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        return metrics
    
    def _get_task_type(self) -> str:
        """获取任务类型"""
        return 'classification' if self._is_classification() else 'regression'
    
    def _is_classification(self) -> bool:
        """判断是否为分类任务"""
        # 简单判断：如果目标变量是整数且唯一值较少，则认为是分类
        y_train = self.data_splits['y_train']
        if hasattr(y_train, 'dtype') and np.issubdtype(y_train.dtype, np.integer):
            unique_values = len(np.unique(y_train))
            return unique_values <= max(10, len(y_train) * 0.05)
        return False
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """预测"""
        if model_name is None:
            # 使用最佳模型
            model_name = self.get_best_model_name()
        
        if model_name not in self.training_results_:
            raise ValueError(f"模型 {model_name} 未找到")
        
        model = self.training_results_[model_name].best_model
        return model.predict(X)
    
    def get_best_model_name(self, metric: str = 'auto') -> str:
        """获取最佳模型名称"""
        if not self.training_results_:
            raise ValueError("没有训练结果")
        
        if metric == 'auto':
            metric = 'accuracy' if self._is_classification() else 'r2'
        
        best_score = float('-inf') if metric in ['accuracy', 'r2', 'f1', 'precision', 'recall'] else float('inf')
        best_model = None
        
        for model_name, result in self.training_results_.items():
            if 'X_val' in self.data_splits and metric in result.val_scores:
                score = result.val_scores[metric]
            elif metric in result.test_scores:
                score = result.test_scores[metric]
            else:
                continue
            
            if metric in ['accuracy', 'r2', 'f1', 'precision', 'recall']:
                if score > best_score:
                    best_score = score
                    best_model = model_name
            else:  # 损失指标，越小越好
                if score < best_score:
                    best_score = score
                    best_model = model_name
        
        if best_model is None:
            return list(self.training_results_.keys())[0]
        
        return best_model
    
    def get_model_comparison(self) -> pd.DataFrame:
        """获取模型比较结果"""
        if not self.training_results_:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, result in self.training_results_.items():
            row = {'model': model_name}
            
            # 基础性能指标
            for dataset in ['train', 'val', 'test']:
                scores_attr = f'{dataset}_scores'
                if hasattr(result, scores_attr):
                    scores = getattr(result, scores_attr)
                    for metric, score in scores.items():
                        row[f'{dataset}_{metric}'] = score
            
            # 交叉验证结果
            if model_name in self.cv_results:
                cv_result = self.cv_results[model_name]
                for metric, mean_score in cv_result['mean_scores'].items():
                    row[f'cv_mean_{metric}'] = mean_score
                    row[f'cv_std_{metric}'] = cv_result['std_scores'][metric]
            
            # Walk-forward结果
            if model_name in self.walk_forward_results:
                wf_result = self.walk_forward_results[model_name]
                row['wf_mean_score'] = wf_result['mean_score']
                row['wf_std_score'] = wf_result['std_score']
            
            # 训练时间
            row['training_time'] = result.training_time
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def save_models(self, save_dir: str):
        """保存所有模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, result in self.training_results_.items():
            model_path = save_path / f"{model_name}_model.joblib"
            joblib.dump(result.best_model, model_path)
            
            # 保存训练结果
            result_path = save_path / f"{model_name}_result.joblib"
            joblib.dump(result, result_path)
        
        # 保存比较结果
        comparison = self.get_model_comparison()
        comparison.to_csv(save_path / "model_comparison.csv", index=False)
        
        logger.info(f"模型保存到: {save_path}")
    
    def load_models(self, load_dir: str):
        """加载模型"""
        load_path = Path(load_dir)
        
        for model_file in load_path.glob("*_model.joblib"):
            model_name = model_file.stem.replace("_model", "")
            model = joblib.load(model_file)
            
            # 加载训练结果
            result_file = load_path / f"{model_name}_result.joblib"
            if result_file.exists():
                result = joblib.load(result_file)
                self.training_results_[model_name] = result
        
        logger.info(f"模型从 {load_path} 加载完成")


# 便捷函数
def train_time_series_models(X: Union[np.ndarray, pd.DataFrame],
                           y: Union[np.ndarray, pd.Series],
                           models: Dict[str, BaseEstimator],
                           config: Optional[TimeSeriesTrainingConfig] = None,
                           time_index: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]] = None
                           ) -> TimeSeriesTrainer:
    """
    训练时间序列模型的便捷函数
    
    Args:
        X: 特征数据
        y: 目标变量
        models: 模型字典
        config: 训练配置
        time_index: 时间索引
        
    Returns:
        训练好的时间序列训练器
    """
    trainer = TimeSeriesTrainer(config=config)
    
    for name, model in models.items():
        trainer.add_model(name, model)
    
    trainer.fit(X, y, time_index)
    
    return trainer