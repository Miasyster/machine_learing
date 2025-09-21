"""
训练策略模块

提供各种训练策略和技术
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import time
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

from .base import BaseTrainer, TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)


class EarlyStoppingStrategy:
    """早停策略"""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.001,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        初始化早停策略
        
        Args:
            patience: 耐心值
            min_delta: 最小改进量
            monitor: 监控指标
            mode: 模式 ('min' 或 'max')
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: BaseEstimator) -> bool:
        """
        在每个epoch结束时调用
        
        Args:
            epoch: 当前epoch
            logs: 日志信息
            model: 模型
            
        Returns:
            是否应该停止训练
        """
        current = logs.get(self.monitor)
        if current is None:
            return False
        
        if self.best_score is None:
            self.best_score = current
            if self.restore_best_weights and hasattr(model, 'get_params'):
                self.best_weights = model.get_params()
        
        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            self.wait = 0
            if self.restore_best_weights and hasattr(model, 'get_params'):
                self.best_weights = model.get_params()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights:
                    model.set_params(**self.best_weights)
                return True
        
        return False


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, 
                 schedule: str = 'step',
                 initial_lr: float = 0.01,
                 **kwargs):
        """
        初始化学习率调度器
        
        Args:
            schedule: 调度策略
            initial_lr: 初始学习率
            **kwargs: 其他参数
        """
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.kwargs = kwargs
        
        self.current_lr = initial_lr
    
    def get_lr(self, epoch: int) -> float:
        """获取当前学习率"""
        if self.schedule == 'step':
            step_size = self.kwargs.get('step_size', 10)
            gamma = self.kwargs.get('gamma', 0.1)
            return self.initial_lr * (gamma ** (epoch // step_size))
        
        elif self.schedule == 'exponential':
            gamma = self.kwargs.get('gamma', 0.95)
            return self.initial_lr * (gamma ** epoch)
        
        elif self.schedule == 'cosine':
            T_max = self.kwargs.get('T_max', 100)
            eta_min = self.kwargs.get('eta_min', 0)
            return eta_min + (self.initial_lr - eta_min) * \
                   (1 + np.cos(np.pi * epoch / T_max)) / 2
        
        elif self.schedule == 'plateau':
            # 需要外部调用update方法
            return self.current_lr
        
        else:
            return self.initial_lr
    
    def update(self, metric: float):
        """更新学习率（用于plateau调度）"""
        if self.schedule == 'plateau':
            factor = self.kwargs.get('factor', 0.1)
            patience = self.kwargs.get('patience', 10)
            threshold = self.kwargs.get('threshold', 1e-4)
            
            # 简化的plateau逻辑
            # 实际实现需要跟踪历史指标
            pass


class DataAugmentationStrategy:
    """数据增强策略"""
    
    def __init__(self, 
                 augmentation_type: str = 'noise',
                 augmentation_factor: float = 0.1,
                 **kwargs):
        """
        初始化数据增强策略
        
        Args:
            augmentation_type: 增强类型
            augmentation_factor: 增强因子
            **kwargs: 其他参数
        """
        self.augmentation_type = augmentation_type
        self.augmentation_factor = augmentation_factor
        self.kwargs = kwargs
    
    def augment(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        数据增强
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            增强后的数据
        """
        if self.augmentation_type == 'noise':
            return self._add_noise(X, y)
        elif self.augmentation_type == 'rotation':
            return self._rotation(X, y)
        elif self.augmentation_type == 'scaling':
            return self._scaling(X, y)
        elif self.augmentation_type == 'smote':
            return self._smote(X, y)
        else:
            return X, y
    
    def _add_noise(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """添加噪声"""
        noise = np.random.normal(0, self.augmentation_factor * np.std(X), X.shape)
        X_aug = X + noise
        return X_aug, y
    
    def _rotation(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """旋转变换（适用于2D数据）"""
        if X.shape[1] < 2:
            return X, y
        
        angle = np.random.uniform(-self.augmentation_factor, self.augmentation_factor)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                   [sin_angle, cos_angle]])
        
        X_aug = X.copy()
        X_aug[:, :2] = X[:, :2] @ rotation_matrix.T
        
        return X_aug, y
    
    def _scaling(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """缩放变换"""
        scale_factor = 1 + np.random.uniform(-self.augmentation_factor, self.augmentation_factor)
        X_aug = X * scale_factor
        return X_aug, y
    
    def _smote(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """SMOTE过采样"""
        if y is None:
            return X, y
        
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_aug, y_aug = smote.fit_resample(X, y)
            return X_aug, y_aug
        except ImportError:
            logger.warning("imbalanced-learn not available, skipping SMOTE")
            return X, y


class RegularizationStrategy:
    """正则化策略"""
    
    def __init__(self, 
                 l1_ratio: float = 0.0,
                 l2_ratio: float = 0.01,
                 dropout_rate: float = 0.0,
                 **kwargs):
        """
        初始化正则化策略
        
        Args:
            l1_ratio: L1正则化比例
            l2_ratio: L2正则化比例
            dropout_rate: Dropout比例
            **kwargs: 其他参数
        """
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs
    
    def apply_to_model(self, model: BaseEstimator) -> BaseEstimator:
        """
        将正则化应用到模型
        
        Args:
            model: 模型
            
        Returns:
            应用正则化后的模型
        """
        model_params = model.get_params()
        
        # 应用L1/L2正则化
        if 'alpha' in model_params:
            model.set_params(alpha=self.l2_ratio)
        elif 'C' in model_params:
            model.set_params(C=1.0 / self.l2_ratio if self.l2_ratio > 0 else 1.0)
        
        # 应用L1比例
        if 'l1_ratio' in model_params:
            model.set_params(l1_ratio=self.l1_ratio)
        
        return model


class CrossValidationStrategy:
    """交叉验证策略"""
    
    def __init__(self, 
                 cv_type: str = 'kfold',
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: int = 42,
                 **kwargs):
        """
        初始化交叉验证策略
        
        Args:
            cv_type: 交叉验证类型
            n_splits: 折数
            shuffle: 是否打乱
            random_state: 随机种子
            **kwargs: 其他参数
        """
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.cv_splitter = self._create_cv_splitter()
    
    def _create_cv_splitter(self):
        """创建交叉验证分割器"""
        if self.cv_type == 'kfold':
            return KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.cv_type == 'stratified':
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        else:
            return KFold(n_splits=self.n_splits)
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """分割数据"""
        return self.cv_splitter.split(X, y)


class FeatureSelectionStrategy:
    """特征选择策略"""
    
    def __init__(self, 
                 selection_method: str = 'variance',
                 n_features: Optional[int] = None,
                 threshold: Optional[float] = None,
                 **kwargs):
        """
        初始化特征选择策略
        
        Args:
            selection_method: 选择方法
            n_features: 特征数量
            threshold: 阈值
            **kwargs: 其他参数
        """
        self.selection_method = selection_method
        self.n_features = n_features
        self.threshold = threshold
        self.kwargs = kwargs
        
        self.selector = None
        self.selected_features_ = None
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        拟合并转换特征
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            选择后的特征
        """
        if self.selection_method == 'variance':
            return self._variance_threshold(X)
        elif self.selection_method == 'univariate':
            return self._univariate_selection(X, y)
        elif self.selection_method == 'rfe':
            return self._recursive_feature_elimination(X, y)
        elif self.selection_method == 'lasso':
            return self._lasso_selection(X, y)
        else:
            return X
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换特征"""
        if self.selector is None:
            return X
        
        if hasattr(self.selector, 'transform'):
            return self.selector.transform(X)
        elif self.selected_features_ is not None:
            return X[:, self.selected_features_]
        else:
            return X
    
    def _variance_threshold(self, X: np.ndarray) -> np.ndarray:
        """方差阈值选择"""
        from sklearn.feature_selection import VarianceThreshold
        
        threshold = self.threshold if self.threshold is not None else 0.0
        self.selector = VarianceThreshold(threshold=threshold)
        return self.selector.fit_transform(X)
    
    def _univariate_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """单变量特征选择"""
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        
        if y is None:
            return X
        
        # 判断任务类型
        if len(np.unique(y)) < 10:  # 分类任务
            score_func = f_classif
        else:  # 回归任务
            score_func = f_regression
        
        k = self.n_features if self.n_features is not None else 'all'
        self.selector = SelectKBest(score_func=score_func, k=k)
        return self.selector.fit_transform(X, y)
    
    def _recursive_feature_elimination(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """递归特征消除"""
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        if y is None:
            return X
        
        # 选择基础估计器
        if len(np.unique(y)) < 10:  # 分类任务
            estimator = LogisticRegression(random_state=42)
        else:  # 回归任务
            estimator = LinearRegression()
        
        n_features = self.n_features if self.n_features is not None else X.shape[1] // 2
        self.selector = RFE(estimator=estimator, n_features_to_select=n_features)
        return self.selector.fit_transform(X, y)
    
    def _lasso_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Lasso特征选择"""
        from sklearn.feature_selection import SelectFromModel
        from sklearn.linear_model import Lasso, LassoCV
        
        if y is None:
            return X
        
        # 使用交叉验证选择alpha
        lasso = LassoCV(cv=5, random_state=42)
        self.selector = SelectFromModel(lasso)
        return self.selector.fit_transform(X, y)


class TrainingStrategy:
    """综合训练策略"""
    
    def __init__(self, 
                 config: Optional[TrainingConfig] = None,
                 early_stopping: Optional[EarlyStoppingStrategy] = None,
                 lr_scheduler: Optional[LearningRateScheduler] = None,
                 data_augmentation: Optional[DataAugmentationStrategy] = None,
                 regularization: Optional[RegularizationStrategy] = None,
                 cv_strategy: Optional[CrossValidationStrategy] = None,
                 feature_selection: Optional[FeatureSelectionStrategy] = None,
                 **kwargs):
        """
        初始化训练策略
        
        Args:
            config: 训练配置
            early_stopping: 早停策略
            lr_scheduler: 学习率调度器
            data_augmentation: 数据增强策略
            regularization: 正则化策略
            cv_strategy: 交叉验证策略
            feature_selection: 特征选择策略
            **kwargs: 其他参数
        """
        self.config = config or TrainingConfig()
        self.early_stopping = early_stopping
        self.lr_scheduler = lr_scheduler
        self.data_augmentation = data_augmentation
        self.regularization = regularization
        self.cv_strategy = cv_strategy or CrossValidationStrategy()
        self.feature_selection = feature_selection
        self.kwargs = kwargs
        
        self.verbose = kwargs.get('verbose', True)
        
        if self.verbose:
            logger.info("Initialized TrainingStrategy")
    
    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        准备数据
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            处理后的数据
        """
        # 特征选择
        if self.feature_selection:
            X = self.feature_selection.fit_transform(X, y)
        
        # 数据增强
        if self.data_augmentation:
            X, y = self.data_augmentation.augment(X, y)
        
        return X, y
    
    def prepare_model(self, model: BaseEstimator) -> BaseEstimator:
        """
        准备模型
        
        Args:
            model: 原始模型
            
        Returns:
            处理后的模型
        """
        # 应用正则化
        if self.regularization:
            model = self.regularization.apply_to_model(model)
        
        return model
    
    def should_stop_training(self, epoch: int, logs: Dict[str, float], model: BaseEstimator) -> bool:
        """
        判断是否应该停止训练
        
        Args:
            epoch: 当前epoch
            logs: 日志信息
            model: 模型
            
        Returns:
            是否应该停止
        """
        if self.early_stopping:
            return self.early_stopping.on_epoch_end(epoch, logs, model)
        return False
    
    def get_learning_rate(self, epoch: int) -> float:
        """
        获取学习率
        
        Args:
            epoch: 当前epoch
            
        Returns:
            学习率
        """
        if self.lr_scheduler:
            return self.lr_scheduler.get_lr(epoch)
        return 0.01
    
    def get_cv_splitter(self):
        """获取交叉验证分割器"""
        return self.cv_strategy.cv_splitter
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        转换特征（用于预测时）
        
        Args:
            X: 特征数据
            
        Returns:
            转换后的特征
        """
        if self.feature_selection:
            X = self.feature_selection.transform(X)
        
        return X