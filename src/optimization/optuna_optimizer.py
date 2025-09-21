"""
Optuna优化器

基于Optuna库的超参数优化
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
import logging
from tqdm import tqdm

from .base import OptimizationResult
from .hyperparameter_space import HyperparameterSpace, ParameterType

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Optuna优化器"""
    
    def __init__(self,
                 hyperparameter_space: HyperparameterSpace,
                 sampler: str = 'tpe',
                 sampler_params: Optional[Dict[str, Any]] = None,
                 pruner: Optional[str] = None,
                 pruner_params: Optional[Dict[str, Any]] = None,
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None,
                 direction: str = 'maximize',
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        初始化Optuna优化器
        
        Args:
            hyperparameter_space: 超参数空间
            sampler: 采样器类型 ('tpe', 'random', 'cmaes', 'grid')
            sampler_params: 采样器参数
            pruner: 剪枝器类型 ('median', 'percentile', 'hyperband')
            pruner_params: 剪枝器参数
            study_name: 研究名称
            storage: 存储后端
            direction: 优化方向 ('maximize', 'minimize')
            random_state: 随机种子
            verbose: 是否显示进度
        """
        self.hyperparameter_space = hyperparameter_space
        
        try:
            import optuna
            self.optuna = optuna
        except ImportError:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        
        self.sampler_type = sampler
        self.sampler_params = sampler_params or {}
        self.pruner_type = pruner
        self.pruner_params = pruner_params or {}
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.random_state = random_state
        self.verbose = verbose
        
        # 初始化结果存储
        self.results = []
        self.best_score = float('-inf')
        self.best_params = {}
        
        # 创建study
        self.study = None
        self._create_study()
    
    def _create_study(self):
        """创建Optuna study"""
        # 创建采样器
        sampler = self._create_sampler()
        
        # 创建剪枝器
        pruner = self._create_pruner()
        
        # 创建study
        self.study = self.optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.direction,
            load_if_exists=True
        )
        
        logger.info(f"Created Optuna study with {self.sampler_type} sampler")
    
    def _create_sampler(self):
        """创建采样器"""
        if self.sampler_type == 'tpe':
            return self.optuna.samplers.TPESampler(
                seed=self.random_state,
                **self.sampler_params
            )
        
        elif self.sampler_type == 'random':
            return self.optuna.samplers.RandomSampler(
                seed=self.random_state,
                **self.sampler_params
            )
        
        elif self.sampler_type == 'cmaes':
            return self.optuna.samplers.CmaEsSampler(
                seed=self.random_state,
                **self.sampler_params
            )
        
        elif self.sampler_type == 'grid':
            # 需要预先定义搜索空间
            search_space = self._create_grid_search_space()
            return self.optuna.samplers.GridSampler(
                search_space=search_space,
                **self.sampler_params
            )
        
        elif self.sampler_type == 'nsgaii':
            return self.optuna.samplers.NSGAIISampler(
                seed=self.random_state,
                **self.sampler_params
            )
        
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")
    
    def _create_pruner(self):
        """创建剪枝器"""
        if self.pruner_type is None:
            return self.optuna.pruners.NopPruner()
        
        elif self.pruner_type == 'median':
            return self.optuna.pruners.MedianPruner(**self.pruner_params)
        
        elif self.pruner_type == 'percentile':
            return self.optuna.pruners.PercentilePruner(**self.pruner_params)
        
        elif self.pruner_type == 'hyperband':
            return self.optuna.pruners.HyperbandPruner(**self.pruner_params)
        
        elif self.pruner_type == 'successive_halving':
            return self.optuna.pruners.SuccessiveHalvingPruner(**self.pruner_params)
        
        else:
            raise ValueError(f"Unknown pruner type: {self.pruner_type}")
    
    def _create_grid_search_space(self) -> Dict[str, List[Any]]:
        """为网格搜索创建搜索空间"""
        search_space = {}
        
        for name, param in self.hyperparameter_space.parameters.items():
            if param.param_type == ParameterType.FLOAT:
                if param.log:
                    values = np.logspace(np.log10(param.low), np.log10(param.high), 10)
                else:
                    values = np.linspace(param.low, param.high, 10)
                search_space[name] = values.tolist()
            
            elif param.param_type == ParameterType.INT:
                range_size = param.high - param.low + 1
                if range_size <= 20:
                    values = list(range(param.low, param.high + 1))
                else:
                    if param.log:
                        values = np.logspace(np.log10(max(1, param.low)), 
                                           np.log10(param.high), 10)
                        values = [int(v) for v in values]
                    else:
                        values = np.linspace(param.low, param.high, 10)
                        values = [int(v) for v in values]
                    values = sorted(list(set(values)))
                
                search_space[name] = values
            
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                search_space[name] = param.choices
        
        return search_space
    
    def suggest(self, trial_id: Optional[int] = None) -> Dict[str, Any]:
        """
        建议下一组超参数
        
        Args:
            trial_id: 试验ID（未使用，保持接口一致性）
            
        Returns:
            超参数字典
        """
        # 创建trial
        trial = self.study.ask()
        
        # 根据超参数空间建议参数
        params = {}
        for name, param in self.hyperparameter_space.parameters.items():
            if param.param_type == ParameterType.FLOAT:
                if param.log:
                    params[name] = trial.suggest_float(
                        name, param.low, param.high, log=True
                    )
                else:
                    params[name] = trial.suggest_float(
                        name, param.low, param.high
                    )
            
            elif param.param_type == ParameterType.INT:
                if param.log:
                    params[name] = trial.suggest_int(
                        name, param.low, param.high, log=True
                    )
                else:
                    params[name] = trial.suggest_int(
                        name, param.low, param.high, step=param.step
                    )
            
            elif param.param_type == ParameterType.CATEGORICAL:
                params[name] = trial.suggest_categorical(name, param.choices)
            
            elif param.param_type == ParameterType.BOOLEAN:
                params[name] = trial.suggest_categorical(name, [True, False])
        
        # 存储trial以便后续使用
        self._current_trial = trial
        
        return params
    
    def tell(self, params: Dict[str, Any], score: float, 
             intermediate_values: Optional[List[float]] = None):
        """
        告知优化器评估结果
        
        Args:
            params: 参数字典
            score: 评估分数
            intermediate_values: 中间值（用于剪枝）
        """
        if hasattr(self, '_current_trial'):
            trial = self._current_trial
            
            # 报告中间值
            if intermediate_values:
                for step, value in enumerate(intermediate_values):
                    trial.report(value, step)
                    
                    # 检查是否应该剪枝
                    if trial.should_prune():
                        raise self.optuna.TrialPruned()
            
            # 告知最终结果
            self.study.tell(trial, score)
            
            # 更新最佳结果
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
    
    def optimize(self,
                 objective_function: Callable[[Dict[str, Any]], float],
                 n_trials: int = 100,
                 timeout: Optional[float] = None,
                 callbacks: Optional[List[Callable]] = None) -> OptimizationResult:
        """
        执行Optuna优化
        
        Args:
            objective_function: 目标函数
            n_trials: 试验次数
            timeout: 超时时间（秒）
            callbacks: 回调函数列表
            
        Returns:
            优化结果
        """
        logger.info(f"Starting Optuna optimization with {n_trials} trials")
        
        # 包装目标函数
        def optuna_objective(trial):
            params = {}
            
            # 根据超参数空间建议参数
            for name, param in self.hyperparameter_space.parameters.items():
                if param.param_type == ParameterType.FLOAT:
                    if param.log:
                        params[name] = trial.suggest_float(
                            name, param.low, param.high, log=True
                        )
                    else:
                        params[name] = trial.suggest_float(
                            name, param.low, param.high
                        )
                
                elif param.param_type == ParameterType.INT:
                    if param.log:
                        params[name] = trial.suggest_int(
                            name, param.low, param.high, log=True
                        )
                    else:
                        params[name] = trial.suggest_int(
                            name, param.low, param.high, step=param.step
                        )
                
                elif param.param_type == ParameterType.CATEGORICAL:
                    params[name] = trial.suggest_categorical(name, param.choices)
                
                elif param.param_type == ParameterType.BOOLEAN:
                    params[name] = trial.suggest_categorical(name, [True, False])
            
            # 评估目标函数
            try:
                score = objective_function(params)
                
                # 记录结果
                result = {
                    'trial_id': trial.number,
                    'params': params,
                    'score': score,
                    'success': True,
                    'error': None
                }
                self.results.append(result)
                
                # 更新最佳结果
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                return score
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                
                # 记录失败结果
                result = {
                    'trial_id': trial.number,
                    'params': params,
                    'score': float('-inf'),
                    'success': False,
                    'error': str(e)
                }
                self.results.append(result)
                
                raise e
        
        # 执行优化
        if self.verbose:
            # 添加进度条回调
            progress_callback = self._create_progress_callback(n_trials)
            callbacks = callbacks or []
            callbacks.append(progress_callback)
        
        self.study.optimize(
            optuna_objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks
        )
        
        # 获取最佳结果
        best_trial = self.study.best_trial
        self.best_params = best_trial.params
        self.best_score = best_trial.value
        
        # 创建优化结果
        all_results = []
        for trial in self.study.trials:
            result = {
                'trial_id': trial.number,
                'params': trial.params,
                'score': trial.value if trial.value is not None else float('-inf'),
                'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() 
                          if trial.datetime_complete and trial.datetime_start else 0,
                'success': trial.state == self.optuna.trial.TrialState.COMPLETE,
                'error': None if trial.state == self.optuna.trial.TrialState.COMPLETE else str(trial.state)
            }
            all_results.append(result)
        
        optimization_result = OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            best_trial=0,
            n_trials=len(all_results),
            optimization_time=sum(r['duration'] for r in all_results),
            history=all_results,
            convergence_curve=[r['score'] for r in all_results],
            metadata={}
        )
        
        logger.info(f"Optuna optimization completed. Best score: {self.best_score:.6f}")
        
        return optimization_result
    
    def _create_progress_callback(self, n_trials: int):
        """创建进度条回调"""
        progress_bar = tqdm(total=n_trials, desc="Optuna Optimization")
        
        def callback(study, trial):
            progress_bar.update(1)
            progress_bar.set_postfix({
                'best_score': f"{study.best_value:.6f}" if study.best_value else "N/A",
                'current_score': f"{trial.value:.6f}" if trial.value else "N/A"
            })
            
            if progress_bar.n >= n_trials:
                progress_bar.close()
        
        return callback
    
    def get_study_statistics(self) -> Dict[str, Any]:
        """
        获取study统计信息
        
        Returns:
            统计信息字典
        """
        if not self.study:
            return {}
        
        trials = self.study.trials
        completed_trials = [t for t in trials if t.state == self.optuna.trial.TrialState.COMPLETE]
        
        stats = {
            'n_trials': len(trials),
            'n_completed_trials': len(completed_trials),
            'n_pruned_trials': len([t for t in trials if t.state == self.optuna.trial.TrialState.PRUNED]),
            'n_failed_trials': len([t for t in trials if t.state == self.optuna.trial.TrialState.FAIL]),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial_number': self.study.best_trial.number if self.study.best_trial else None
        }
        
        if completed_trials:
            values = [t.value for t in completed_trials]
            stats.update({
                'mean_value': np.mean(values),
                'std_value': np.std(values),
                'median_value': np.median(values),
                'min_value': np.min(values),
                'max_value': np.max(values)
            })
        
        return stats
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        获取参数重要性
        
        Returns:
            参数重要性字典
        """
        try:
            importance = self.optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            logger.warning(f"Failed to calculate parameter importance: {e}")
            return {}
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        绘制优化历史
        
        Args:
            save_path: 保存路径
        """
        try:
            from optuna.visualization import plot_optimization_history
            
            fig = plot_optimization_history(self.study)
            
            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
                
        except ImportError:
            logger.warning("Plotly is required for visualization. Install with: pip install plotly")
        except Exception as e:
            logger.error(f"Failed to plot optimization history: {e}")
    
    def plot_parameter_importances(self, save_path: Optional[str] = None):
        """
        绘制参数重要性
        
        Args:
            save_path: 保存路径
        """
        try:
            from optuna.visualization import plot_param_importances
            
            fig = plot_param_importances(self.study)
            
            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
                
        except ImportError:
            logger.warning("Plotly is required for visualization. Install with: pip install plotly")
        except Exception as e:
            logger.error(f"Failed to plot parameter importances: {e}")
    
    def plot_parallel_coordinate(self, save_path: Optional[str] = None):
        """
        绘制平行坐标图
        
        Args:
            save_path: 保存路径
        """
        try:
            from optuna.visualization import plot_parallel_coordinate
            
            fig = plot_parallel_coordinate(self.study)
            
            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
                
        except ImportError:
            logger.warning("Plotly is required for visualization. Install with: pip install plotly")
        except Exception as e:
            logger.error(f"Failed to plot parallel coordinate: {e}")


class MultiObjectiveOptunaOptimizer(OptunaOptimizer):
    """多目标Optuna优化器"""
    
    def __init__(self,
                 hyperparameter_space: HyperparameterSpace,
                 directions: List[str],
                 sampler: str = 'nsgaii',
                 sampler_params: Optional[Dict[str, Any]] = None,
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        初始化多目标Optuna优化器
        
        Args:
            hyperparameter_space: 超参数空间
            directions: 优化方向列表
            sampler: 采样器类型
            sampler_params: 采样器参数
            study_name: 研究名称
            storage: 存储后端
            random_state: 随机种子
            verbose: 是否显示进度
        """
        self.directions = directions
        
        # 不调用父类的__init__，因为我们需要多目标study
        BaseOptimizer.__init__(self, hyperparameter_space)
        
        try:
            import optuna
            self.optuna = optuna
        except ImportError:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        
        self.sampler_type = sampler
        self.sampler_params = sampler_params or {}
        self.study_name = study_name
        self.storage = storage
        self.random_state = random_state
        self.verbose = verbose
        
        # 创建多目标study
        self.study = None
        self._create_multi_objective_study()
    
    def _create_multi_objective_study(self):
        """创建多目标study"""
        # 创建采样器
        sampler = self._create_sampler()
        
        # 创建多目标study
        self.study = self.optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
            directions=self.directions,
            load_if_exists=True
        )
        
        logger.info(f"Created multi-objective Optuna study with {len(self.directions)} objectives")
    
    def optimize(self,
                 objective_function: Callable[[Dict[str, Any]], List[float]],
                 n_trials: int = 100,
                 timeout: Optional[float] = None,
                 callbacks: Optional[List[Callable]] = None) -> OptimizationResult:
        """
        执行多目标优化
        
        Args:
            objective_function: 多目标函数（返回目标值列表）
            n_trials: 试验次数
            timeout: 超时时间（秒）
            callbacks: 回调函数列表
            
        Returns:
            优化结果
        """
        logger.info(f"Starting multi-objective Optuna optimization with {n_trials} trials")
        
        # 包装目标函数
        def optuna_objective(trial):
            params = {}
            
            # 根据超参数空间建议参数
            for name, param in self.hyperparameter_space.parameters.items():
                if param.param_type == ParameterType.FLOAT:
                    if param.log:
                        params[name] = trial.suggest_float(
                            name, param.low, param.high, log=True
                        )
                    else:
                        params[name] = trial.suggest_float(
                            name, param.low, param.high
                        )
                
                elif param.param_type == ParameterType.INT:
                    if param.log:
                        params[name] = trial.suggest_int(
                            name, param.low, param.high, log=True
                        )
                    else:
                        params[name] = trial.suggest_int(
                            name, param.low, param.high, step=param.step
                        )
                
                elif param.param_type == ParameterType.CATEGORICAL:
                    params[name] = trial.suggest_categorical(name, param.choices)
                
                elif param.param_type == ParameterType.BOOLEAN:
                    params[name] = trial.suggest_categorical(name, [True, False])
            
            # 评估目标函数
            try:
                scores = objective_function(params)
                
                # 记录结果
                result = {
                    'trial_id': trial.number,
                    'params': params,
                    'scores': scores,
                    'success': True,
                    'error': None
                }
                self.results.append(result)
                
                return scores
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                
                # 记录失败结果
                result = {
                    'trial_id': trial.number,
                    'params': params,
                    'scores': [float('-inf')] * len(self.directions),
                    'success': False,
                    'error': str(e)
                }
                self.results.append(result)
                
                raise e
        
        # 执行优化
        if self.verbose:
            # 添加进度条回调
            progress_callback = self._create_progress_callback(n_trials)
            callbacks = callbacks or []
            callbacks.append(progress_callback)
        
        self.study.optimize(
            optuna_objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks
        )
        
        # 获取Pareto最优解
        pareto_trials = self.study.best_trials
        
        # 创建优化结果
        all_results = []
        for trial in self.study.trials:
            result = {
                'trial_id': trial.number,
                'params': trial.params,
                'scores': trial.values if trial.values is not None else [float('-inf')] * len(self.directions),
                'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() 
                          if trial.datetime_complete and trial.datetime_start else 0,
                'success': trial.state == self.optuna.trial.TrialState.COMPLETE,
                'error': None if trial.state == self.optuna.trial.TrialState.COMPLETE else str(trial.state)
            }
            all_results.append(result)
        
        # 对于多目标，best_params和best_score使用第一个Pareto最优解
        if pareto_trials:
            best_trial = pareto_trials[0]
            self.best_params = best_trial.params
            self.best_score = best_trial.values[0] if best_trial.values else float('-inf')
        
        optimization_result = OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            best_trial=0,
            n_trials=len(all_results),
            optimization_time=sum(r['duration'] for r in all_results),
            history=all_results,
            convergence_curve=[r['score'] for r in all_results],
            metadata={}
        )
        
        # 添加Pareto最优解信息
        optimization_result.pareto_trials = [
            {
                'params': trial.params,
                'scores': trial.values,
                'trial_id': trial.number
            }
            for trial in pareto_trials
        ]
        
        logger.info(f"Multi-objective Optuna optimization completed. Found {len(pareto_trials)} Pareto optimal solutions")
        
        return optimization_result