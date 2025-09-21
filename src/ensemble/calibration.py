"""
模型校准方法

实现Platt Scaling、Isotonic Regression等概率校准技术
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt

from .base import BaseEnsemble, EnsembleResult

logger = logging.getLogger(__name__)


class ModelCalibrator:
    """模型校准器"""
    
    def __init__(self,
                 method: str = 'platt',
                 cv: Union[int, object] = 3,
                 ensemble_method: str = 'average'):
        """
        初始化模型校准器
        
        Args:
            method: 校准方法 ('platt', 'isotonic', 'both')
            cv: 交叉验证折数
            ensemble_method: 多方法集成策略 ('average', 'weighted', 'stacking')
        """
        self.method = method
        self.cv = cv
        self.ensemble_method = ensemble_method
        self.calibrators_ = {}
        self.is_fitted_ = False
        
    def fit(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> 'ModelCalibrator':
        """
        训练校准器
        
        Args:
            model: 待校准的模型
            X: 训练特征
            y: 训练标签
            
        Returns:
            校准器实例
        """
        if self.method == 'platt':
            self.calibrators_['platt'] = CalibratedClassifierCV(
                clone(model), method='sigmoid', cv=self.cv
            )
            self.calibrators_['platt'].fit(X, y)
            
        elif self.method == 'isotonic':
            self.calibrators_['isotonic'] = CalibratedClassifierCV(
                clone(model), method='isotonic', cv=self.cv
            )
            self.calibrators_['isotonic'].fit(X, y)
            
        elif self.method == 'both':
            self.calibrators_['platt'] = CalibratedClassifierCV(
                clone(model), method='sigmoid', cv=self.cv
            )
            self.calibrators_['isotonic'] = CalibratedClassifierCV(
                clone(model), method='isotonic', cv=self.cv
            )
            self.calibrators_['platt'].fit(X, y)
            self.calibrators_['isotonic'].fit(X, y)
            
            # 如果使用stacking集成，训练元学习器
            if self.ensemble_method == 'stacking':
                self._fit_meta_learner(X, y)
                
        self.is_fitted_ = True
        return self
    
    def _fit_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """训练元学习器用于集成多个校准方法"""
        # 获取各校准方法的交叉验证预测
        platt_probs = cross_val_predict(
            self.calibrators_['platt'], X, y, cv=self.cv, method='predict_proba'
        )
        isotonic_probs = cross_val_predict(
            self.calibrators_['isotonic'], X, y, cv=self.cv, method='predict_proba'
        )
        
        # 构建元特征
        meta_features = np.column_stack([
            platt_probs[:, 1],  # 正类概率
            isotonic_probs[:, 1]
        ])
        
        # 训练元学习器
        self.meta_learner_ = LogisticRegression()
        self.meta_learner_.fit(meta_features, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测校准后的概率
        
        Args:
            X: 测试特征
            
        Returns:
            校准后的概率
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before prediction")
        
        if len(self.calibrators_) == 1:
            # 单一校准方法
            method_name = list(self.calibrators_.keys())[0]
            return self.calibrators_[method_name].predict_proba(X)
        
        else:
            # 多校准方法集成
            platt_probs = self.calibrators_['platt'].predict_proba(X)
            isotonic_probs = self.calibrators_['isotonic'].predict_proba(X)
            
            if self.ensemble_method == 'average':
                return (platt_probs + isotonic_probs) / 2
            
            elif self.ensemble_method == 'weighted':
                # 基于训练时的性能加权
                weights = self._get_method_weights()
                return weights[0] * platt_probs + weights[1] * isotonic_probs
            
            elif self.ensemble_method == 'stacking':
                meta_features = np.column_stack([
                    platt_probs[:, 1],
                    isotonic_probs[:, 1]
                ])
                return self.meta_learner_.predict_proba(meta_features)
    
    def _get_method_weights(self) -> np.ndarray:
        """获取方法权重（基于性能）"""
        # 简化实现：等权重
        return np.array([0.5, 0.5])
    
    def evaluate_calibration(self, X: np.ndarray, y: np.ndarray, 
                           n_bins: int = 10) -> Dict[str, Any]:
        """
        评估校准效果
        
        Args:
            X: 测试特征
            y: 真实标签
            n_bins: 校准曲线的分箱数
            
        Returns:
            校准评估结果
        """
        probs = self.predict_proba(X)[:, 1]  # 正类概率
        
        # 计算校准曲线
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, probs, n_bins=n_bins
        )
        
        # 计算校准指标
        brier_score = brier_score_loss(y, probs)
        log_loss_score = log_loss(y, probs)
        
        # 计算可靠性图的面积（ECE - Expected Calibration Error）
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        return {
            'brier_score': brier_score,
            'log_loss': log_loss_score,
            'expected_calibration_error': ece,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
        }
    
    def plot_calibration_curve(self, X: np.ndarray, y: np.ndarray, 
                              n_bins: int = 10, save_path: Optional[str] = None):
        """
        绘制校准曲线
        
        Args:
            X: 测试特征
            y: 真实标签
            n_bins: 分箱数
            save_path: 保存路径
        """
        probs = self.predict_proba(X)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, probs, n_bins=n_bins
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f"Calibrated ({self.method})")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Plot")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class CalibratedEnsemble(BaseEnsemble):
    """校准集成器"""
    
    def __init__(self,
                 models: List[BaseEstimator],
                 calibration_method: str = 'platt',
                 cv: Union[int, object] = 3,
                 model_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        初始化校准集成器
        
        Args:
            models: 基础模型列表
            calibration_method: 校准方法
            cv: 交叉验证折数
            model_names: 模型名称
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__(models, model_names, random_state, n_jobs, verbose)
        self.calibration_method = calibration_method
        self.cv = cv
        self.calibrators_ = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'CalibratedEnsemble':
        """
        训练校准集成器
        
        Args:
            X: 训练特征
            y: 训练标签
            
        Returns:
            集成器实例
        """
        self.calibrators_ = []
        
        for i, model in enumerate(self.models):
            if self.verbose:
                logger.info(f"Training calibrator for model {i+1}/{self.n_models_}")
            
            # 创建并训练校准器
            calibrator = ModelCalibrator(
                method=self.calibration_method,
                cv=self.cv
            )
            calibrator.fit(model, X, y)
            self.calibrators_.append(calibrator)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """
        预测
        
        Args:
            X: 测试特征
            
        Returns:
            集成预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # 获取各模型的校准概率
        individual_probabilities = []
        individual_predictions = []
        
        for calibrator in self.calibrators_:
            probs = calibrator.predict_proba(X)
            individual_probabilities.append(probs)
            individual_predictions.append(np.argmax(probs, axis=1))
        
        # 集成预测（简单平均）
        ensemble_probs = np.mean(individual_probabilities, axis=0)
        ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        
        return EnsembleResult(
            predictions=ensemble_predictions,
            prediction_probabilities=ensemble_probs,
            individual_predictions=individual_predictions,
            individual_probabilities=individual_probabilities
        )
    
    def evaluate_ensemble_calibration(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        评估集成模型的校准效果
        
        Args:
            X: 测试特征
            y: 真实标签
            
        Returns:
            校准评估结果
        """
        result = self.predict(X)
        probs = result.prediction_probabilities[:, 1]
        
        # 计算校准指标
        brier_score = brier_score_loss(y, probs)
        log_loss_score = log_loss(y, probs)
        
        # 计算各个模型的校准效果
        individual_calibrations = []
        for i, calibrator in enumerate(self.calibrators_):
            individual_probs = calibrator.predict_proba(X)[:, 1]
            individual_brier = brier_score_loss(y, individual_probs)
            individual_log_loss = log_loss(y, individual_probs)
            
            individual_calibrations.append({
                'model_name': self.model_names_[i],
                'brier_score': individual_brier,
                'log_loss': individual_log_loss
            })
        
        return {
            'ensemble_brier_score': brier_score,
            'ensemble_log_loss': log_loss_score,
            'individual_calibrations': individual_calibrations,
            'calibration_improvement': {
                'brier_score_reduction': np.mean([
                    cal['brier_score'] for cal in individual_calibrations
                ]) - brier_score,
                'log_loss_reduction': np.mean([
                    cal['log_loss'] for cal in individual_calibrations
                ]) - log_loss_score
            }
        }


class TemperatureScaling:
    """温度缩放校准方法"""
    
    def __init__(self):
        self.temperature_ = 1.0
        self.is_fitted_ = False
    
    def fit(self, logits: np.ndarray, y: np.ndarray) -> 'TemperatureScaling':
        """
        训练温度参数
        
        Args:
            logits: 模型输出的logits
            y: 真实标签
            
        Returns:
            温度缩放实例
        """
        from scipy.optimize import minimize_scalar
        
        def temperature_loss(temp):
            scaled_logits = logits / temp
            probs = self._softmax(scaled_logits)
            return log_loss(y, probs)
        
        # 优化温度参数
        result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature_ = result.x
        self.is_fitted_ = True
        
        return self
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        预测校准后的概率
        
        Args:
            logits: 模型输出的logits
            
        Returns:
            校准后的概率
        """
        if not self.is_fitted_:
            raise ValueError("Temperature scaling must be fitted before prediction")
        
        scaled_logits = logits / self.temperature_
        return self._softmax(scaled_logits)
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """计算softmax概率"""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)