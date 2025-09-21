"""
网格搜索优化器

实现网格搜索超参数优化
"""

import numpy as np
import itertools
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm

from .base import BaseOptimizer, OptimizationResult
from .hyperparameter_space import HyperparameterSpace, Parameter, ParameterType

logger = logging.getLogger(__name__)


class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""
    
    def __init__(self,
                 hyperparameter_space: HyperparameterSpace,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        初始化网格搜索优化器
        
        Args:
            hyperparameter_space: 超参数空间
            n_jobs: 并行作业数
            verbose: 是否显示进度
        """
        self.hyperparameter_space = hyperparameter_space
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # 初始化结果存储
        self.results = []
        self.best_score = float('-inf')
        self.best_params = {}
        
        self.grid_points = []
        self._generate_grid()
    
    def _generate_grid(self):
        """生成网格点"""
        param_grids = {}
        
        for name, param in self.hyperparameter_space.parameters.items():
            if param.param_type == ParameterType.FLOAT:
                # 对于浮点数，生成等间距的点
                if param.log:
                    points = np.logspace(np.log10(param.low), np.log10(param.high), 10)
                else:
                    points = np.linspace(param.low, param.high, 10)
                param_grids[name] = points.tolist()
            
            elif param.param_type == ParameterType.INT:
                # 对于整数，生成所有可能的值（如果范围不太大）
                range_size = param.high - param.low + 1
                if range_size <= 20:
                    points = list(range(param.low, param.high + 1))
                else:
                    if param.log:
                        points = np.logspace(np.log10(max(1, param.low)), 
                                           np.log10(param.high), 10)
                        points = [int(p) for p in points]
                    else:
                        points = np.linspace(param.low, param.high, 10)
                        points = [int(p) for p in points]
                    # 去重并排序
                    points = sorted(list(set(points)))
                
                # 应用步长
                if param.step is not None:
                    points = [p for p in points if (p - param.low) % param.step == 0]
                
                param_grids[name] = points
            
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                param_grids[name] = param.choices
        
        # 生成所有组合
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        self.grid_points = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            self.grid_points.append(params)
        
        logger.info(f"Generated {len(self.grid_points)} grid points")
    
    def _suggest_hyperparameters(self, trial_number: int) -> Dict[str, Any]:
        """建议下一组超参数（抽象方法实现）"""
        return self.suggest(trial_number)
    
    def suggest(self, trial_id: Optional[int] = None) -> Dict[str, Any]:
        """
        建议下一组超参数
        
        Args:
            trial_id: 试验ID
            
        Returns:
            超参数字典
        """
        if trial_id is None:
            trial_id = len(self.results)
        
        if trial_id >= len(self.grid_points):
            raise StopIteration("All grid points have been evaluated")
        
        return self.grid_points[trial_id].copy()
    
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
                 n_trials: Optional[int] = None,
                 timeout: Optional[float] = None) -> OptimizationResult:
        """
        执行网格搜索优化
        
        Args:
            objective_function: 目标函数
            n_trials: 试验次数（None表示搜索所有网格点）
            timeout: 超时时间（秒）
            
        Returns:
            优化结果
        """
        if n_trials is None:
            n_trials = len(self.grid_points)
        else:
            n_trials = min(n_trials, len(self.grid_points))
        
        logger.info(f"Starting grid search with {n_trials} trials")
        
        if self.n_jobs == 1:
            # 串行执行
            results = self._optimize_serial(objective_function, n_trials, timeout)
        else:
            # 并行执行
            results = self._optimize_parallel(objective_function, n_trials, timeout)
        
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
        
        logger.info(f"Grid search completed. Best score: {self.best_score:.6f}")
        
        return optimization_result
    
    def _optimize_serial(self,
                        objective_function: Callable[[Dict[str, Any]], float],
                        n_trials: int,
                        timeout: Optional[float]) -> List[Dict[str, Any]]:
        """串行优化"""
        results = []
        
        progress_bar = tqdm(range(n_trials), disable=not self.verbose,
                           desc="Grid Search")
        
        for trial_id in progress_bar:
            if timeout and sum(r['duration'] for r in results) >= timeout:
                logger.info(f"Timeout reached after {len(results)} trials")
                break
            
            params = self.suggest(trial_id)
            result = self._evaluate_trial(objective_function, params, trial_id)
            results.append(result)
            
            # 更新进度条
            if self.verbose:
                progress_bar.set_postfix({
                    'best_score': f"{self.best_score:.6f}",
                    'current_score': f"{result['score']:.6f}"
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
                               desc="Grid Search (Parallel)")
            
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
        
        duration = time.time() - start_time
        
        return {
            'trial_id': trial_id,
            'params': params,
            'score': score,
            'duration': duration,
            'success': success,
            'error': error
        }
    
    def get_search_space_size(self) -> int:
        """
        获取搜索空间大小
        
        Returns:
            搜索空间大小
        """
        return len(self.grid_points)
    
    def get_remaining_trials(self) -> int:
        """
        获取剩余试验次数
        
        Returns:
            剩余试验次数
        """
        return max(0, len(self.grid_points) - len(self.results))


class AdaptiveGridSearchOptimizer(GridSearchOptimizer):
    """自适应网格搜索优化器"""
    
    def __init__(self,
                 hyperparameter_space: HyperparameterSpace,
                 initial_grid_size: int = 5,
                 refinement_factor: int = 3,
                 n_refinements: int = 2,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        初始化自适应网格搜索优化器
        
        Args:
            hyperparameter_space: 超参数空间
            initial_grid_size: 初始网格大小
            refinement_factor: 细化因子
            n_refinements: 细化次数
            n_jobs: 并行作业数
            verbose: 是否显示进度
        """
        self.initial_grid_size = initial_grid_size
        self.refinement_factor = refinement_factor
        self.n_refinements = n_refinements
        
        # 不调用父类的__init__，因为我们要自定义网格生成
        BaseOptimizer.__init__(self, hyperparameter_space)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.current_refinement = 0
        self.refinement_regions = []
        
        # 生成初始粗糙网格
        self._generate_initial_grid()
    
    def _generate_initial_grid(self):
        """生成初始粗糙网格"""
        param_grids = {}
        
        for name, param in self.hyperparameter_space.parameters.items():
            if param.param_type == ParameterType.FLOAT:
                if param.log:
                    points = np.logspace(np.log10(param.low), np.log10(param.high), 
                                       self.initial_grid_size)
                else:
                    points = np.linspace(param.low, param.high, self.initial_grid_size)
                param_grids[name] = points.tolist()
            
            elif param.param_type == ParameterType.INT:
                range_size = param.high - param.low + 1
                if range_size <= self.initial_grid_size:
                    points = list(range(param.low, param.high + 1))
                else:
                    if param.log:
                        points = np.logspace(np.log10(max(1, param.low)), 
                                           np.log10(param.high), self.initial_grid_size)
                        points = [int(p) for p in points]
                    else:
                        points = np.linspace(param.low, param.high, self.initial_grid_size)
                        points = [int(p) for p in points]
                    points = sorted(list(set(points)))
                
                if param.step is not None:
                    points = [p for p in points if (p - param.low) % param.step == 0]
                
                param_grids[name] = points
            
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                param_grids[name] = param.choices
        
        # 生成所有组合
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        self.grid_points = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            self.grid_points.append(params)
        
        logger.info(f"Generated initial grid with {len(self.grid_points)} points")
    
    def _refine_grid(self, top_k: int = 3):
        """
        细化网格
        
        Args:
            top_k: 选择前k个最佳点进行细化
        """
        if not self.results:
            return
        
        # 选择最佳的k个点
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        top_results = sorted_results[:top_k]
        
        logger.info(f"Refining grid around top {len(top_results)} points")
        
        # 为每个最佳点生成细化区域
        new_grid_points = []
        
        for result in top_results:
            best_params = result['params']
            refined_points = self._generate_refined_region(best_params)
            new_grid_points.extend(refined_points)
        
        # 去重
        seen = set()
        unique_points = []
        for point in new_grid_points:
            point_tuple = tuple(sorted(point.items()))
            if point_tuple not in seen:
                seen.add(point_tuple)
                unique_points.append(point)
        
        self.grid_points = unique_points
        logger.info(f"Generated refined grid with {len(self.grid_points)} points")
    
    def _generate_refined_region(self, center_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        在给定中心点周围生成细化区域
        
        Args:
            center_params: 中心参数
            
        Returns:
            细化区域的参数列表
        """
        param_grids = {}
        
        for name, param in self.hyperparameter_space.parameters.items():
            center_value = center_params[name]
            
            if param.param_type == ParameterType.FLOAT:
                # 计算细化范围
                if param.log:
                    log_center = np.log(center_value)
                    log_range = (np.log(param.high) - np.log(param.low)) / (2 ** self.current_refinement)
                    log_low = max(np.log(param.low), log_center - log_range/2)
                    log_high = min(np.log(param.high), log_center + log_range/2)
                    points = np.logspace(log_low, log_high, self.refinement_factor)
                else:
                    param_range = (param.high - param.low) / (2 ** self.current_refinement)
                    low = max(param.low, center_value - param_range/2)
                    high = min(param.high, center_value + param_range/2)
                    points = np.linspace(low, high, self.refinement_factor)
                
                param_grids[name] = points.tolist()
            
            elif param.param_type == ParameterType.INT:
                # 计算细化范围
                param_range = max(1, int((param.high - param.low) / (2 ** self.current_refinement)))
                low = max(param.low, center_value - param_range//2)
                high = min(param.high, center_value + param_range//2)
                
                points = list(range(low, high + 1))
                
                if param.step is not None:
                    points = [p for p in points if (p - param.low) % param.step == 0]
                
                param_grids[name] = points
            
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                # 分类参数不细化，保持原值
                param_grids[name] = [center_value]
        
        # 生成所有组合
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        refined_points = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            refined_points.append(params)
        
        return refined_points
    
    def optimize(self,
                 objective_function: Callable[[Dict[str, Any]], float],
                 n_trials: Optional[int] = None,
                 timeout: Optional[float] = None) -> OptimizationResult:
        """
        执行自适应网格搜索优化
        
        Args:
            objective_function: 目标函数
            n_trials: 试验次数
            timeout: 超时时间（秒）
            
        Returns:
            优化结果
        """
        all_results = []
        total_time = 0
        
        logger.info("Starting adaptive grid search optimization")
        
        # 初始网格搜索
        logger.info("Phase 1: Initial coarse grid search")
        initial_results = super().optimize(objective_function, n_trials, timeout)
        all_results.extend(initial_results.all_results)
        total_time += initial_results.optimization_time
        
        # 细化搜索
        for refinement in range(self.n_refinements):
            if timeout and total_time >= timeout:
                logger.info(f"Timeout reached after {refinement} refinements")
                break
            
            self.current_refinement = refinement + 1
            logger.info(f"Phase {refinement + 2}: Refinement {self.current_refinement}")
            
            # 细化网格
            self._refine_grid()
            
            # 重置结果以进行新的搜索
            self.results = []
            
            # 执行细化搜索
            remaining_time = timeout - total_time if timeout else None
            refinement_results = super().optimize(objective_function, n_trials, remaining_time)
            all_results.extend(refinement_results.all_results)
            total_time += refinement_results.optimization_time
        
        # 创建最终优化结果
        optimization_result = OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            best_trial=0,
            n_trials=len(all_results),
            optimization_time=total_time,
            history=all_results,
            convergence_curve=[r['score'] for r in all_results],
            metadata={}
        )
        
        logger.info(f"Adaptive grid search completed. Best score: {self.best_score:.6f}")
        
        return optimization_result