"""
贝叶斯优化器

实现基于高斯过程的贝叶斯优化
"""

import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import logging
from tqdm import tqdm

from .base import OptimizationResult
from .hyperparameter_space import HyperparameterSpace, ParameterType

logger = logging.getLogger(__name__)


# 忽略sklearn的警告
warnings.filterwarnings('ignore', category=UserWarning)


class AcquisitionFunction(ABC):
    """采集函数基类"""
    
    @abstractmethod
    def __call__(self, X: np.ndarray, gp_model, y_best: float, **kwargs) -> np.ndarray:
        """
        计算采集函数值
        
        Args:
            X: 候选点
            gp_model: 高斯过程模型
            y_best: 当前最佳值
            **kwargs: 其他参数
            
        Returns:
            采集函数值
        """
        pass


class ExpectedImprovement(AcquisitionFunction):
    """期望改进采集函数"""
    
    def __init__(self, xi: float = 0.01):
        """
        初始化期望改进
        
        Args:
            xi: 探索参数
        """
        self.xi = xi
    
    def __call__(self, X: np.ndarray, gp_model, y_best: float, **kwargs) -> np.ndarray:
        """计算期望改进"""
        try:
            mu, sigma = gp_model.predict(X, return_std=True)
        except:
            # 如果预测失败，返回随机值
            return np.random.random(len(X))
        
        # 确保mu和sigma都是一维数组
        mu = mu.flatten()
        sigma = sigma.flatten()
        
        with np.errstate(divide='warn'):
            imp = mu - y_best - self.xi
            Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma!=0)
            ei = imp * self._norm_cdf(Z) + sigma * self._norm_pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    @staticmethod
    def _norm_cdf(x):
        """标准正态分布累积分布函数"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    @staticmethod
    def _norm_pdf(x):
        """标准正态分布概率密度函数"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


class UpperConfidenceBound(AcquisitionFunction):
    """置信上界采集函数"""
    
    def __init__(self, kappa: float = 2.576):
        """
        初始化置信上界
        
        Args:
            kappa: 探索参数
        """
        self.kappa = kappa
    
    def __call__(self, X: np.ndarray, gp_model, y_best: float, **kwargs) -> np.ndarray:
        """计算置信上界"""
        try:
            mu, sigma = gp_model.predict(X, return_std=True)
        except:
            return np.random.random(len(X))
        
        return mu + self.kappa * sigma


class ProbabilityOfImprovement(AcquisitionFunction):
    """改进概率采集函数"""
    
    def __init__(self, xi: float = 0.01):
        """
        初始化改进概率
        
        Args:
            xi: 探索参数
        """
        self.xi = xi
    
    def __call__(self, X: np.ndarray, gp_model, y_best: float, **kwargs) -> np.ndarray:
        """计算改进概率"""
        try:
            mu, sigma = gp_model.predict(X, return_std=True)
        except:
            return np.random.random(len(X))
        
        sigma = sigma.reshape(-1, 1) if sigma.ndim == 1 else sigma
        
        with np.errstate(divide='warn'):
            Z = (mu - y_best - self.xi) / sigma
            pi = self._norm_cdf(Z)
            pi[sigma == 0.0] = 0.0
        
        return pi.flatten()
    
    @staticmethod
    def _norm_cdf(x):
        """标准正态分布累积分布函数"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))


class BayesianOptimizer:
    """贝叶斯优化器"""
    
    def __init__(self,
                 hyperparameter_space: HyperparameterSpace,
                 acquisition_function: str = 'ei',
                 acquisition_params: Optional[Dict[str, Any]] = None,
                 n_initial_points: int = 5,
                 n_candidates: int = 1000,
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        初始化贝叶斯优化器
        
        Args:
            hyperparameter_space: 超参数空间
            acquisition_function: 采集函数类型 ('ei', 'ucb', 'pi')
            acquisition_params: 采集函数参数
            n_initial_points: 初始随机点数量
            n_candidates: 候选点数量
            random_state: 随机种子
            verbose: 是否显示进度
        """
        self.hyperparameter_space = hyperparameter_space
        self.n_initial_points = n_initial_points
        self.n_candidates = n_candidates
        self.random_state = np.random.RandomState(random_state)
        self.verbose = verbose
        
        # 初始化结果存储
        self.results = []
        
        # 初始化采集函数
        acquisition_params = acquisition_params or {}
        if acquisition_function == 'ei':
            self.acquisition_function = ExpectedImprovement(**acquisition_params)
        elif acquisition_function == 'ucb':
            self.acquisition_function = UpperConfidenceBound(**acquisition_params)
        elif acquisition_function == 'pi':
            self.acquisition_function = ProbabilityOfImprovement(**acquisition_params)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")
        
        # 初始化高斯过程
        self.gp_model = None
        self._init_gaussian_process()
        
        # 参数编码器
        self.param_encoder = ParameterEncoder(hyperparameter_space)
    
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
        
        return result
    
    @property
    def best_score(self) -> float:
        """获取最佳分数"""
        if not self.results:
            return float('-inf')
        
        successful_results = [r for r in self.results if r['success']]
        if not successful_results:
            return float('-inf')
        
        return max(r['score'] for r in successful_results)
    
    @property
    def best_params(self) -> Dict[str, Any]:
        """获取最佳参数"""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r['success']]
        if not successful_results:
            return {}
        
        best_result = max(successful_results, key=lambda x: x['score'])
        return best_result['params']
    
    def _init_gaussian_process(self):
        """初始化高斯过程模型"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel
            
            # 使用Matern核
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self.random_state.randint(0, 2**31-1)
            )
            
        except ImportError:
            logger.error("scikit-learn is required for Bayesian optimization")
            raise ImportError("Please install scikit-learn: pip install scikit-learn")
    
    def suggest(self, trial_id: Optional[int] = None) -> Dict[str, Any]:
        """
        建议下一组超参数
        
        Args:
            trial_id: 试验ID
            
        Returns:
            超参数字典
        """
        if len(self.results) < self.n_initial_points:
            # 初始随机采样
            return self.hyperparameter_space.sample(self.random_state)
        else:
            # 基于采集函数选择下一个点
            return self._suggest_via_acquisition()
    
    def _suggest_via_acquisition(self) -> Dict[str, Any]:
        """通过采集函数建议下一个点"""
        # 训练高斯过程
        self._fit_gaussian_process()
        
        # 生成候选点
        candidates = self._generate_candidates()
        
        # 计算采集函数值
        acquisition_values = self.acquisition_function(
            candidates, self.gp_model, self.best_score
        )
        
        # 选择最佳候选点
        best_idx = np.argmax(acquisition_values)
        best_candidate = candidates[best_idx]
        
        # 解码参数
        return self.param_encoder.decode(best_candidate)
    
    def _fit_gaussian_process(self):
        """训练高斯过程模型"""
        if not self.results:
            return
        
        # 准备训练数据
        X = []
        y = []
        
        for result in self.results:
            if result['success']:
                encoded_params = self.param_encoder.encode(result['params'])
                X.append(encoded_params)
                y.append(result['score'])
        
        if len(X) == 0:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        try:
            self.gp_model.fit(X, y)
        except Exception as e:
            logger.warning(f"Failed to fit Gaussian process: {e}")
    
    def _generate_candidates(self) -> np.ndarray:
        """生成候选点"""
        candidates = []
        
        for _ in range(self.n_candidates):
            params = self.hyperparameter_space.sample(self.random_state)
            encoded_params = self.param_encoder.encode(params)
            candidates.append(encoded_params)
        
        return np.array(candidates)
    
    def optimize(self,
                 objective_function: Callable[[Dict[str, Any]], float],
                 n_trials: int = 100,
                 timeout: Optional[float] = None) -> OptimizationResult:
        """
        执行贝叶斯优化
        
        Args:
            objective_function: 目标函数
            n_trials: 试验次数
            timeout: 超时时间（秒）
            
        Returns:
            优化结果
        """
        logger.info(f"Starting Bayesian optimization with {n_trials} trials")
        
        results = []
        
        progress_bar = tqdm(range(n_trials), disable=not self.verbose,
                           desc="Bayesian Optimization")
        
        for trial_id in progress_bar:
            # 检查超时
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
        
        logger.info(f"Bayesian optimization completed. Best score: {self.best_score:.6f}")
        
        return optimization_result


class ParameterEncoder:
    """参数编码器，将超参数转换为数值向量"""
    
    def __init__(self, hyperparameter_space: HyperparameterSpace):
        """
        初始化参数编码器
        
        Args:
            hyperparameter_space: 超参数空间
        """
        self.hyperparameter_space = hyperparameter_space
        self.param_info = {}
        self.total_dims = 0
        
        self._analyze_parameters()
    
    def _analyze_parameters(self):
        """分析参数结构"""
        current_dim = 0
        
        for name, param in self.hyperparameter_space.parameters.items():
            if param.param_type in [ParameterType.FLOAT, ParameterType.INT]:
                self.param_info[name] = {
                    'type': 'continuous',
                    'dim_start': current_dim,
                    'dim_end': current_dim + 1,
                    'low': param.low,
                    'high': param.high,
                    'log': param.log,
                    'param_type': param.param_type
                }
                current_dim += 1
            
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                n_choices = len(param.choices)
                self.param_info[name] = {
                    'type': 'categorical',
                    'dim_start': current_dim,
                    'dim_end': current_dim + n_choices,
                    'choices': param.choices,
                    'n_choices': n_choices
                }
                current_dim += n_choices
        
        self.total_dims = current_dim
    
    def encode(self, params: Dict[str, Any]) -> np.ndarray:
        """
        编码参数
        
        Args:
            params: 参数字典
            
        Returns:
            编码后的向量
        """
        encoded = np.zeros(self.total_dims)
        
        for name, value in params.items():
            if name not in self.param_info:
                continue
            
            info = self.param_info[name]
            
            if info['type'] == 'continuous':
                # 归一化到[0, 1]
                if info['log']:
                    log_low = np.log(info['low'])
                    log_high = np.log(info['high'])
                    log_value = np.log(value)
                    normalized = (log_value - log_low) / (log_high - log_low)
                else:
                    normalized = (value - info['low']) / (info['high'] - info['low'])
                
                encoded[info['dim_start']] = np.clip(normalized, 0, 1)
            
            elif info['type'] == 'categorical':
                # One-hot编码
                try:
                    choice_idx = info['choices'].index(value)
                    encoded[info['dim_start'] + choice_idx] = 1.0
                except ValueError:
                    # 如果值不在选择列表中，随机选择一个
                    choice_idx = 0
                    encoded[info['dim_start'] + choice_idx] = 1.0
        
        return encoded
    
    def decode(self, encoded: np.ndarray) -> Dict[str, Any]:
        """
        解码参数
        
        Args:
            encoded: 编码后的向量
            
        Returns:
            参数字典
        """
        params = {}
        
        for name, info in self.param_info.items():
            if info['type'] == 'continuous':
                normalized = encoded[info['dim_start']]
                normalized = np.clip(normalized, 0, 1)
                
                if info['log']:
                    log_low = np.log(info['low'])
                    log_high = np.log(info['high'])
                    log_value = log_low + normalized * (log_high - log_low)
                    value = np.exp(log_value)
                else:
                    value = info['low'] + normalized * (info['high'] - info['low'])
                
                if info['param_type'] == ParameterType.INT:
                    value = int(round(value))
                    value = np.clip(value, info['low'], info['high'])
                
                params[name] = value
            
            elif info['type'] == 'categorical':
                # 选择概率最大的类别
                probs = encoded[info['dim_start']:info['dim_end']]
                choice_idx = np.argmax(probs)
                params[name] = info['choices'][choice_idx]
        
        return params


class TPEOptimizer:
    """Tree-structured Parzen Estimator优化器"""
    
    def __init__(self,
                 hyperparameter_space: HyperparameterSpace,
                 n_initial_points: int = 10,
                 n_ei_candidates: int = 24,
                 gamma: float = 0.25,
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        初始化TPE优化器
        
        Args:
            hyperparameter_space: 超参数空间
            n_initial_points: 初始随机点数量
            n_ei_candidates: EI候选点数量
            gamma: 分位数参数
            random_state: 随机种子
            verbose: 是否显示进度
        """
        self.hyperparameter_space = hyperparameter_space
        self.n_initial_points = n_initial_points
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.random_state = np.random.RandomState(random_state)
        self.verbose = verbose
        
        # 初始化结果存储
        self.results = []
    
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
        
        return result
    
    @property
    def best_score(self) -> float:
        """获取最佳分数"""
        if not self.results:
            return float('-inf')
        
        successful_results = [r for r in self.results if r['success']]
        if not successful_results:
            return float('-inf')
        
        return max(r['score'] for r in successful_results)
    
    @property
    def best_params(self) -> Dict[str, Any]:
        """获取最佳参数"""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r['success']]
        if not successful_results:
            return {}
        
        best_result = max(successful_results, key=lambda x: x['score'])
        return best_result['params']
    
    def suggest(self, trial_id: Optional[int] = None) -> Dict[str, Any]:
        """
        建议下一组超参数
        
        Args:
            trial_id: 试验ID
            
        Returns:
            超参数字典
        """
        if len(self.results) < self.n_initial_points:
            # 初始随机采样
            return self.hyperparameter_space.sample(self.random_state)
        else:
            # 基于TPE选择下一个点
            return self._suggest_via_tpe()
    
    def _suggest_via_tpe(self) -> Dict[str, Any]:
        """通过TPE建议下一个点"""
        # 获取成功的结果
        successful_results = [r for r in self.results if r['success']]
        if not successful_results:
            return self.hyperparameter_space.sample(self.random_state)
        
        # 按分数排序
        sorted_results = sorted(successful_results, key=lambda x: x['score'], reverse=True)
        
        # 计算分位数
        n_below = max(1, int(self.gamma * len(sorted_results)))
        
        # 分割为好的和坏的结果
        good_results = sorted_results[:n_below]
        bad_results = sorted_results[n_below:]
        
        # 生成候选点并计算EI
        best_ei = float('-inf')
        best_params = None
        
        for _ in range(self.n_ei_candidates):
            # 从好的结果分布中采样
            candidate = self._sample_from_good_distribution(good_results)
            
            # 计算EI
            ei = self._calculate_tpe_ei(candidate, good_results, bad_results)
            
            if ei > best_ei:
                best_ei = ei
                best_params = candidate
        
        return best_params if best_params else self.hyperparameter_space.sample(self.random_state)
    
    def _sample_from_good_distribution(self, good_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从好的结果分布中采样"""
        if not good_results:
            return self.hyperparameter_space.sample(self.random_state)
        
        # 随机选择一个好的结果作为基础
        base_result = self.random_state.choice(good_results)
        base_params = base_result['params']
        
        # 在其周围添加噪声
        params = {}
        for name, param in self.hyperparameter_space.parameters.items():
            base_value = base_params[name]
            
            if param.param_type in [ParameterType.FLOAT, ParameterType.INT]:
                # 添加高斯噪声
                param_range = param.high - param.low
                noise_std = param_range * 0.1  # 10%的标准差
                
                if param.log:
                    log_base = np.log(base_value)
                    log_noise = self.random_state.normal(0, noise_std / base_value)
                    new_value = np.exp(log_base + log_noise)
                else:
                    new_value = base_value + self.random_state.normal(0, noise_std)
                
                # 裁剪到有效范围
                new_value = np.clip(new_value, param.low, param.high)
                
                if param.param_type == ParameterType.INT:
                    new_value = int(round(new_value))
                
                params[name] = new_value
            
            else:  # categorical or boolean
                # 有80%概率保持原值，20%概率随机选择
                if self.random_state.random() < 0.8:
                    params[name] = base_value
                else:
                    params[name] = self.random_state.choice(param.choices)
        
        return params
    
    def _calculate_tpe_ei(self, candidate: Dict[str, Any], 
                         good_results: List[Dict[str, Any]], 
                         bad_results: List[Dict[str, Any]]) -> float:
        """计算TPE期望改进"""
        # 简化的EI计算：好的分布密度 / 坏的分布密度
        good_density = self._estimate_density(candidate, good_results)
        bad_density = self._estimate_density(candidate, bad_results)
        
        if bad_density == 0:
            return float('inf') if good_density > 0 else 0
        
        return good_density / bad_density
    
    def _estimate_density(self, candidate: Dict[str, Any], 
                         results: List[Dict[str, Any]]) -> float:
        """估计候选点在结果分布中的密度"""
        if not results:
            return 1e-10
        
        densities = []
        
        for result in results:
            density = 1.0
            
            for name, value in candidate.items():
                param = self.hyperparameter_space.parameters[name]
                result_value = result['params'][name]
                
                if param.param_type in [ParameterType.FLOAT, ParameterType.INT]:
                    # 高斯核密度估计
                    param_range = param.high - param.low
                    bandwidth = param_range * 0.1
                    
                    if param.log:
                        log_diff = np.log(value) - np.log(result_value)
                        param_density = np.exp(-0.5 * (log_diff / bandwidth)**2)
                    else:
                        diff = value - result_value
                        param_density = np.exp(-0.5 * (diff / bandwidth)**2)
                    
                    density *= param_density
                
                else:  # categorical or boolean
                    # 离散密度
                    if value == result_value:
                        density *= 1.0
                    else:
                        density *= 0.1  # 小的非零值
            
            densities.append(density)
        
        return np.mean(densities)
    
    def optimize(self,
                 objective_function: Callable[[Dict[str, Any]], float],
                 n_trials: int = 100,
                 timeout: Optional[float] = None) -> OptimizationResult:
        """
        执行TPE优化
        
        Args:
            objective_function: 目标函数
            n_trials: 试验次数
            timeout: 超时时间（秒）
            
        Returns:
            优化结果
        """
        logger.info(f"Starting TPE optimization with {n_trials} trials")
        
        results = []
        
        progress_bar = tqdm(range(n_trials), disable=not self.verbose,
                           desc="TPE Optimization")
        
        for trial_id in progress_bar:
            # 检查超时
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
        
        logger.info(f"TPE optimization completed. Best score: {self.best_score:.6f}")
        
        return optimization_result