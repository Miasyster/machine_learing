"""
随机搜索优化器

实现随机搜索超参数优化
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm

from .base import BaseOptimizer, OptimizationResult
from .hyperparameter_space import HyperparameterSpace

logger = logging.getLogger(__name__)


class RandomSearchOptimizer(BaseOptimizer):
    """随机搜索优化器"""
    
    def __init__(self,
                 hyperparameter_space: HyperparameterSpace,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        初始化随机搜索优化器
        
        Args:
            hyperparameter_space: 超参数空间
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示进度
        """
        self.hyperparameter_space = hyperparameter_space
        self.random_state = np.random.RandomState(random_state)
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # 初始化结果存储
        self.results = []
        self.best_score = float('-inf')
        self.best_params = {}
    
    def _suggest_hyperparameters(self, trial_number: int) -> Dict[str, Any]:
        """建议下一组超参数（抽象方法实现）"""
        return self.suggest(trial_number)
    
    def suggest(self, trial_id: Optional[int] = None) -> Dict[str, Any]:
        """
        建议下一组超参数
        
        Args:
            trial_id: 试验ID（未使用，保持接口一致性）
            
        Returns:
            超参数字典
        """
        return self.hyperparameter_space.sample(self.random_state)
    
    def _evaluate_trial(self, objective_function: Callable[[Dict[str, Any]], float],
                       params: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
        """评估单个试验"""
        import time
        
        start_time = time.time()
        try:
            score = objective_function(params)
            success = True
            error = None
        except Exception as e:
            score = float('-inf')
            success = False
            error = str(e)
            if self.verbose:
                print(f"Trial {trial_id} failed: {error}")
        
        duration = time.time() - start_time
        
        # 存储结果
        result = {
            'trial_id': trial_id,
            'params': params.copy(),
            'score': score,
            'duration': duration,
            'success': success,
            'error': error
        }
        
        self.results.append(result)
        
        # 更新最佳结果
        if success and score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
        
        return result
    
    def optimize(self,
                 objective_function: Callable[[Dict[str, Any]], float],
                 n_trials: int = 100,
                 timeout: Optional[float] = None,
                 early_stopping_rounds: Optional[int] = None,
                 early_stopping_threshold: float = 1e-6) -> OptimizationResult:
        """
        执行随机搜索优化
        
        Args:
            objective_function: 目标函数
            n_trials: 试验次数
            timeout: 超时时间（秒）
            early_stopping_rounds: 早停轮数
            early_stopping_threshold: 早停阈值
            
        Returns:
            优化结果
        """
        logger.info(f"Starting random search with {n_trials} trials")
        
        if self.n_jobs == 1:
            # 串行执行
            results = self._optimize_serial(
                objective_function, n_trials, timeout,
                early_stopping_rounds, early_stopping_threshold
            )
        else:
            # 并行执行
            results = self._optimize_parallel(
                objective_function, n_trials, timeout
            )
        
        # 创建优化结果
        optimization_result = OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            best_trial=0,
            n_trials=len(results),
            optimization_time=sum(r['duration'] for r in results),
            history=results,
            convergence_curve=[r['score'] for r in results],
            metadata={}
        )
        
        logger.info(f"Random search completed. Best score: {self.best_score:.6f}")
        
        return optimization_result
    
    def _optimize_serial(self,
                        objective_function: Callable[[Dict[str, Any]], float],
                        n_trials: int,
                        timeout: Optional[float],
                        early_stopping_rounds: Optional[int],
                        early_stopping_threshold: float) -> List[Dict[str, Any]]:
        """串行优化"""
        results = []
        no_improvement_count = 0
        last_best_score = float('-inf')
        
        progress_bar = tqdm(range(n_trials), disable=not self.verbose,
                           desc="Random Search")
        
        for trial_id in progress_bar:
            # 检查超时
            if timeout and sum(r['duration'] for r in results) >= timeout:
                logger.info(f"Timeout reached after {len(results)} trials")
                break
            
            # 检查早停
            if early_stopping_rounds and no_improvement_count >= early_stopping_rounds:
                logger.info(f"Early stopping after {no_improvement_count} rounds without improvement")
                break
            
            params = self.suggest(trial_id)
            result = self._evaluate_trial(objective_function, params, trial_id)
            results.append(result)
            
            # 检查是否有改进
            if self.best_score - last_best_score > early_stopping_threshold:
                no_improvement_count = 0
                last_best_score = self.best_score
            else:
                no_improvement_count += 1
            
            # 更新进度条
            if self.verbose:
                progress_bar.set_postfix({
                    'best_score': f"{self.best_score:.6f}",
                    'current_score': f"{result['score']:.6f}",
                    'no_improve': no_improvement_count
                })
        
        return results
    
    def _optimize_parallel(self,
                          objective_function: Callable[[Dict[str, Any]], float],
                          n_trials: int,
                          timeout: Optional[float]) -> List[Dict[str, Any]]:
        """并行优化"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # 提交所有任务
            future_to_trial = {}
            for trial_id in range(n_trials):
                params = self.suggest(trial_id)
                future = executor.submit(self._evaluate_objective, 
                                       objective_function, params, trial_id)
                future_to_trial[future] = trial_id
            
            # 收集结果
            progress_bar = tqdm(total=n_trials, disable=not self.verbose,
                               desc="Random Search (Parallel)")
            
            for future in as_completed(future_to_trial):
                if timeout and sum(r['duration'] for r in results) >= timeout:
                    logger.info(f"Timeout reached after {len(results)} trials")
                    break
                
                trial_id = future_to_trial[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 更新最佳结果
                    if result['score'] > self.best_score:
                        self.best_score = result['score']
                        self.best_params = result['params'].copy()
                    
                    # 更新进度条
                    if self.verbose:
                        progress_bar.update(1)
                        progress_bar.set_postfix({
                            'best_score': f"{self.best_score:.6f}",
                            'current_score': f"{result['score']:.6f}"
                        })
                
                except Exception as e:
                    logger.error(f"Trial {trial_id} failed: {e}")
            
            progress_bar.close()
        
        # 按trial_id排序
        results.sort(key=lambda x: x['trial_id'])
        
        return results
    
    @staticmethod
    def _evaluate_objective(objective_function: Callable[[Dict[str, Any]], float],
                           params: Dict[str, Any],
                           trial_id: int) -> Dict[str, Any]:
        """评估目标函数（用于并行执行）"""
        start_time = time.time()
        try:
            score = objective_function(params)
            success = True
            error = None
        except Exception as e:
            score = float('-inf')
            success = False
            error = str(e)
        
        duration = time.time() - start_time
        
        return {
            'trial_id': trial_id,
            'params': params,
            'score': score,
            'duration': duration,
            'success': success,
            'error': error
        }


class AdaptiveRandomSearchOptimizer(RandomSearchOptimizer):
    """自适应随机搜索优化器"""
    
    def __init__(self,
                 hyperparameter_space: HyperparameterSpace,
                 exploration_ratio: float = 0.8,
                 exploitation_ratio: float = 0.2,
                 adaptation_frequency: int = 10,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        初始化自适应随机搜索优化器
        
        Args:
            hyperparameter_space: 超参数空间
            exploration_ratio: 探索比例
            exploitation_ratio: 利用比例
            adaptation_frequency: 适应频率
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示进度
        """
        super().__init__(hyperparameter_space, random_state, n_jobs, verbose)
        self.exploration_ratio = exploration_ratio
        self.exploitation_ratio = exploitation_ratio
        self.adaptation_frequency = adaptation_frequency
        self.promising_regions = []
        self.region_weights = []
    
    def suggest(self, trial_id: Optional[int] = None) -> Dict[str, Any]:
        """
        建议下一组超参数（自适应策略）
        
        Args:
            trial_id: 试验ID
            
        Returns:
            超参数字典
        """
        if trial_id is None:
            trial_id = len(self.results)
        
        # 每隔一定频率更新有希望的区域
        if trial_id > 0 and trial_id % self.adaptation_frequency == 0:
            self._update_promising_regions()
        
        # 决定是探索还是利用
        if (not self.promising_regions or 
            self.random_state.random() < self.exploration_ratio):
            # 探索：从整个空间随机采样
            return self.hyperparameter_space.sample(self.random_state)
        else:
            # 利用：从有希望的区域采样
            return self._sample_from_promising_regions()
    
    def _update_promising_regions(self):
        """更新有希望的区域"""
        if len(self.results) < 3:
            return
        
        # 选择前25%的最佳结果
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        top_k = max(1, len(sorted_results) // 4)
        top_results = sorted_results[:top_k]
        
        # 更新有希望的区域
        self.promising_regions = []
        self.region_weights = []
        
        for result in top_results:
            region = self._create_region_around_point(result['params'])
            weight = result['score'] - min(r['score'] for r in self.results)
            
            self.promising_regions.append(region)
            self.region_weights.append(weight)
        
        # 归一化权重
        total_weight = sum(self.region_weights)
        if total_weight > 0:
            self.region_weights = [w / total_weight for w in self.region_weights]
        
        logger.debug(f"Updated {len(self.promising_regions)} promising regions")
    
    def _create_region_around_point(self, center_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        在给定点周围创建区域
        
        Args:
            center_params: 中心参数
            
        Returns:
            区域定义
        """
        region = {}
        
        for name, param in self.hyperparameter_space.parameters.items():
            center_value = center_params[name]
            
            if param.param_type.value in ['float', 'int']:
                # 计算区域范围（原范围的20%）
                param_range = param.high - param.low
                region_size = param_range * 0.2
                
                low = max(param.low, center_value - region_size / 2)
                high = min(param.high, center_value + region_size / 2)
                
                region[name] = {
                    'type': param.param_type.value,
                    'low': low,
                    'high': high,
                    'log': param.log
                }
            
            else:  # categorical or boolean
                # 分类参数保持原值
                region[name] = {
                    'type': param.param_type.value,
                    'value': center_value
                }
        
        return region
    
    def _sample_from_promising_regions(self) -> Dict[str, Any]:
        """从有希望的区域采样"""
        # 根据权重选择区域
        region_idx = self.random_state.choice(
            len(self.promising_regions),
            p=self.region_weights
        )
        region = self.promising_regions[region_idx]
        
        # 从选定区域采样
        params = {}
        for name, region_def in region.items():
            if region_def['type'] == 'float':
                if region_def.get('log', False):
                    log_low = np.log(region_def['low'])
                    log_high = np.log(region_def['high'])
                    log_value = self.random_state.uniform(log_low, log_high)
                    params[name] = np.exp(log_value)
                else:
                    params[name] = self.random_state.uniform(
                        region_def['low'], region_def['high']
                    )
            
            elif region_def['type'] == 'int':
                if region_def.get('log', False):
                    log_low = np.log(max(1, region_def['low']))
                    log_high = np.log(region_def['high'])
                    log_value = self.random_state.uniform(log_low, log_high)
                    params[name] = int(np.exp(log_value))
                else:
                    params[name] = self.random_state.randint(
                        int(region_def['low']), int(region_def['high']) + 1
                    )
            
            else:  # categorical or boolean
                params[name] = region_def['value']
        
        return params


class QuasiRandomSearchOptimizer(RandomSearchOptimizer):
    """准随机搜索优化器（使用低差异序列）"""
    
    def __init__(self,
                 hyperparameter_space: HyperparameterSpace,
                 sequence_type: str = 'sobol',
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        初始化准随机搜索优化器
        
        Args:
            hyperparameter_space: 超参数空间
            sequence_type: 序列类型 ('sobol', 'halton')
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示进度
        """
        super().__init__(hyperparameter_space, random_state, n_jobs, verbose)
        self.sequence_type = sequence_type
        self.sequence_index = 0
        
        # 初始化低差异序列生成器
        self._init_sequence_generator()
    
    def _init_sequence_generator(self):
        """初始化序列生成器"""
        # 计算连续参数的维度
        self.continuous_params = []
        self.categorical_params = []
        
        for name, param in self.hyperparameter_space.parameters.items():
            if param.param_type.value in ['float', 'int']:
                self.continuous_params.append(name)
            else:
                self.categorical_params.append(name)
        
        self.n_continuous_dims = len(self.continuous_params)
        
        if self.sequence_type == 'sobol':
            try:
                from scipy.stats import qmc
                self.sequence_generator = qmc.Sobol(
                    d=self.n_continuous_dims,
                    scramble=True,
                    seed=self.random_state.randint(0, 2**31-1) if self.random_state else None
                )
            except ImportError:
                logger.warning("scipy.stats.qmc not available, falling back to random sampling")
                self.sequence_generator = None
        
        elif self.sequence_type == 'halton':
            try:
                from scipy.stats import qmc
                self.sequence_generator = qmc.Halton(
                    d=self.n_continuous_dims,
                    scramble=True,
                    seed=self.random_state.randint(0, 2**31-1) if self.random_state else None
                )
            except ImportError:
                logger.warning("scipy.stats.qmc not available, falling back to random sampling")
                self.sequence_generator = None
        
        else:
            raise ValueError(f"Unknown sequence type: {self.sequence_type}")
    
    def suggest(self, trial_id: Optional[int] = None) -> Dict[str, Any]:
        """
        建议下一组超参数（使用低差异序列）
        
        Args:
            trial_id: 试验ID
            
        Returns:
            超参数字典
        """
        params = {}
        
        # 为连续参数使用低差异序列
        if self.sequence_generator and self.n_continuous_dims > 0:
            # 生成低差异序列点
            quasi_random_point = self.sequence_generator.random(1)[0]
            
            for i, param_name in enumerate(self.continuous_params):
                param = self.hyperparameter_space.parameters[param_name]
                unit_value = quasi_random_point[i]
                
                # 将[0,1]映射到参数范围
                if param.param_type.value == 'float':
                    if param.log:
                        log_low = np.log(param.low)
                        log_high = np.log(param.high)
                        log_value = log_low + unit_value * (log_high - log_low)
                        params[param_name] = np.exp(log_value)
                    else:
                        params[param_name] = param.low + unit_value * (param.high - param.low)
                
                elif param.param_type.value == 'int':
                    if param.log:
                        log_low = np.log(max(1, param.low))
                        log_high = np.log(param.high)
                        log_value = log_low + unit_value * (log_high - log_low)
                        value = int(np.exp(log_value))
                    else:
                        value = int(param.low + unit_value * (param.high - param.low + 1))
                    
                    params[param_name] = np.clip(value, param.low, param.high)
        
        # 为分类参数使用随机采样
        for param_name in self.categorical_params:
            param = self.hyperparameter_space.parameters[param_name]
            params[param_name] = self.random_state.choice(param.choices)
        
        self.sequence_index += 1
        
        return params