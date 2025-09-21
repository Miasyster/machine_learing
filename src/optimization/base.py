"""
超参数优化基础类

定义优化器接口和结果类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果类"""
    
    best_params: Dict[str, Any]
    best_score: float
    best_trial: int
    n_trials: int
    optimization_time: float
    history: List[Dict[str, Any]]
    convergence_curve: List[float]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """后处理初始化"""
        if self.metadata is None:
            self.metadata = {}
        
        # 添加优化统计信息
        self.metadata.update({
            'optimization_date': datetime.now().isoformat(),
            'improvement_ratio': self._calculate_improvement_ratio(),
            'convergence_trial': self._find_convergence_trial(),
            'score_std': np.std([trial['score'] for trial in self.history]) if self.history else 0.0
        })
    
    def _calculate_improvement_ratio(self) -> float:
        """计算改进比例"""
        if not self.history or len(self.history) < 2:
            return 0.0
        
        first_score = self.history[0]['score']
        if first_score == 0:
            return float('inf') if self.best_score > 0 else 0.0
        
        return (self.best_score - first_score) / abs(first_score)
    
    def _find_convergence_trial(self) -> Optional[int]:
        """找到收敛的试验编号"""
        if not self.convergence_curve or len(self.convergence_curve) < 10:
            return None
        
        # 简单的收敛检测：最后10个试验的改进小于1%
        recent_scores = self.convergence_curve[-10:]
        max_recent = max(recent_scores)
        min_recent = min(recent_scores)
        
        if max_recent == 0:
            return None
        
        improvement = (max_recent - min_recent) / abs(max_recent)
        
        if improvement < 0.01:  # 1%的改进阈值
            return len(self.convergence_curve) - 10
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        return {
            'best_score': self.best_score,
            'best_params': self.best_params,
            'n_trials': self.n_trials,
            'optimization_time': self.optimization_time,
            'improvement_ratio': self.metadata.get('improvement_ratio', 0.0),
            'convergence_trial': self.metadata.get('convergence_trial'),
            'score_std': self.metadata.get('score_std', 0.0)
        }
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """绘制收敛曲线"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.convergence_curve, 'b-', linewidth=2, label='Best Score')
            plt.xlabel('Trial')
            plt.ylabel('Score')
            plt.title('Hyperparameter Optimization Convergence')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Convergence plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available, cannot plot convergence curve")
    
    def save_results(self, filepath: str):
        """保存优化结果"""
        import json
        
        # 准备可序列化的数据
        data = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'best_trial': int(self.best_trial),
            'n_trials': int(self.n_trials),
            'optimization_time': float(self.optimization_time),
            'history': self.history,
            'convergence_curve': [float(x) for x in self.convergence_curve],
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    @classmethod
    def load_results(cls, filepath: str) -> 'OptimizationResult':
        """加载优化结果"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(**data)


class BaseOptimizer(ABC):
    """超参数优化器基类"""
    
    def __init__(self, 
                 objective_function: Callable,
                 hyperparameter_space: Dict[str, Any],
                 config: Optional[Dict] = None):
        """
        初始化优化器
        
        Args:
            objective_function: 目标函数，接受超参数字典，返回分数
            hyperparameter_space: 超参数空间定义
            config: 优化器配置
        """
        self.objective_function = objective_function
        self.hyperparameter_space = hyperparameter_space
        self.config = config or {}
        
        # 优化参数
        self.n_trials = self.config.get('n_trials', 100)
        self.timeout = self.config.get('timeout', None)
        self.n_jobs = self.config.get('n_jobs', 1)
        self.random_state = self.config.get('random_state', None)
        self.verbose = self.config.get('verbose', True)
        
        # 早停参数
        self.early_stopping = self.config.get('early_stopping', False)
        self.patience = self.config.get('patience', 10)
        self.min_improvement = self.config.get('min_improvement', 1e-6)
        
        # 状态变量
        self.trials_history = []
        self.best_score = float('-inf')
        self.best_params = None
        self.best_trial = -1
        self.convergence_curve = []
        self.start_time = None
        
        # 早停状态
        self.no_improvement_count = 0
        self.should_stop = False
        
        # 设置随机种子
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        logger.info(f"Initialized {self.__class__.__name__} with {self.n_trials} trials")
    
    @abstractmethod
    def _suggest_hyperparameters(self, trial_number: int) -> Dict[str, Any]:
        """
        建议下一组超参数
        
        Args:
            trial_number: 试验编号
            
        Returns:
            超参数字典
        """
        pass
    
    def _evaluate_hyperparameters(self, params: Dict[str, Any]) -> float:
        """
        评估超参数
        
        Args:
            params: 超参数字典
            
        Returns:
            评估分数
        """
        try:
            score = self.objective_function(params)
            
            # 确保分数是数值类型
            if not isinstance(score, (int, float)):
                raise ValueError(f"Objective function must return a number, got {type(score)}")
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error evaluating hyperparameters {params}: {e}")
            return float('-inf')
    
    def _update_best(self, params: Dict[str, Any], score: float, trial_number: int):
        """
        更新最佳结果
        
        Args:
            params: 超参数
            score: 分数
            trial_number: 试验编号
        """
        if score > self.best_score:
            improvement = score - self.best_score
            self.best_score = score
            self.best_params = params.copy()
            self.best_trial = trial_number
            self.no_improvement_count = 0
            
            if self.verbose:
                logger.info(f"Trial {trial_number}: New best score {score:.6f} "
                          f"(improvement: {improvement:.6f})")
        else:
            self.no_improvement_count += 1
    
    def _check_early_stopping(self) -> bool:
        """
        检查是否应该早停
        
        Returns:
            是否应该停止
        """
        if not self.early_stopping:
            return False
        
        if self.no_improvement_count >= self.patience:
            if self.verbose:
                logger.info(f"Early stopping triggered after {self.patience} trials without improvement")
            return True
        
        return False
    
    def _check_timeout(self) -> bool:
        """
        检查是否超时
        
        Returns:
            是否超时
        """
        if self.timeout is None:
            return False
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.timeout:
            if self.verbose:
                logger.info(f"Optimization stopped due to timeout ({self.timeout}s)")
            return True
        
        return False
    
    def optimize(self) -> OptimizationResult:
        """
        执行超参数优化
        
        Returns:
            优化结果
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        self.start_time = time.time()
        
        # 重置状态
        self.trials_history = []
        self.best_score = float('-inf')
        self.best_params = None
        self.best_trial = -1
        self.convergence_curve = []
        self.no_improvement_count = 0
        self.should_stop = False
        
        try:
            for trial in range(self.n_trials):
                # 检查停止条件
                if self._check_early_stopping() or self._check_timeout():
                    break
                
                # 建议超参数
                params = self._suggest_hyperparameters(trial)
                
                # 评估超参数
                score = self._evaluate_hyperparameters(params)
                
                # 记录试验
                trial_info = {
                    'trial': trial,
                    'params': params.copy(),
                    'score': score,
                    'timestamp': time.time() - self.start_time
                }
                self.trials_history.append(trial_info)
                
                # 更新最佳结果
                self._update_best(params, score, trial)
                
                # 更新收敛曲线
                self.convergence_curve.append(self.best_score)
                
                # 记录进度
                if self.verbose and (trial + 1) % max(1, self.n_trials // 10) == 0:
                    logger.info(f"Progress: {trial + 1}/{self.n_trials} trials completed, "
                              f"best score: {self.best_score:.6f}")
        
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
        
        finally:
            optimization_time = time.time() - self.start_time
            
            # 创建结果对象
            result = OptimizationResult(
                best_params=self.best_params or {},
                best_score=self.best_score,
                best_trial=self.best_trial,
                n_trials=len(self.trials_history),
                optimization_time=optimization_time,
                history=self.trials_history,
                convergence_curve=self.convergence_curve,
                metadata={
                    'optimizer_type': self.__class__.__name__,
                    'hyperparameter_space': self.hyperparameter_space,
                    'config': self.config
                }
            )
            
            if self.verbose:
                logger.info(f"Optimization completed in {optimization_time:.2f}s")
                logger.info(f"Best score: {self.best_score:.6f}")
                logger.info(f"Best parameters: {self.best_params}")
            
            return result
    
    def get_hyperparameter_importance(self) -> Dict[str, float]:
        """
        计算超参数重要性
        
        Returns:
            超参数重要性字典
        """
        if not self.trials_history:
            return {}
        
        # 简单的相关性分析
        importance = {}
        scores = [trial['score'] for trial in self.trials_history]
        
        for param_name in self.hyperparameter_space.keys():
            param_values = []
            
            for trial in self.trials_history:
                if param_name in trial['params']:
                    value = trial['params'][param_name]
                    # 处理分类参数
                    if isinstance(value, str):
                        # 为字符串参数分配数值
                        unique_values = list(set(trial['params'][param_name] 
                                               for trial in self.trials_history 
                                               if param_name in trial['params']))
                        value = unique_values.index(value)
                    param_values.append(value)
                else:
                    param_values.append(0)  # 默认值
            
            if len(set(param_values)) > 1:  # 参数有变化
                correlation = np.corrcoef(param_values, scores)[0, 1]
                importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                importance[param_name] = 0.0
        
        return importance
    
    def get_best_trials(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        获取最佳的n个试验
        
        Args:
            n: 返回的试验数量
            
        Returns:
            最佳试验列表
        """
        if not self.trials_history:
            return []
        
        sorted_trials = sorted(self.trials_history, 
                             key=lambda x: x['score'], 
                             reverse=True)
        
        return sorted_trials[:n]