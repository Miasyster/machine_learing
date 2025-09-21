"""
超参数优化工具函数

提供各种优化相关的工具函数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import json
import pickle
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .base import OptimizationResult
from .hyperparameter_space import HyperparameterSpace

logger = logging.getLogger(__name__)


def create_objective_function(model_class,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: Optional[np.ndarray] = None,
                            y_val: Optional[np.ndarray] = None,
                            cv: int = 5,
                            scoring: str = 'accuracy',
                            fit_params: Optional[Dict[str, Any]] = None,
                            random_state: Optional[int] = None) -> Callable[[Dict[str, Any]], float]:
    """
    创建目标函数
    
    Args:
        model_class: 模型类
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        cv: 交叉验证折数
        scoring: 评分方法
        fit_params: 拟合参数
        random_state: 随机种子
        
    Returns:
        目标函数
    """
    fit_params = fit_params or {}
    
    def objective(params: Dict[str, Any]) -> float:
        try:
            # 创建模型
            if hasattr(model_class, '__call__'):
                model = model_class(**params)
            else:
                model = model_class
                for key, value in params.items():
                    setattr(model, key, value)
            
            # 如果提供了验证集，使用验证集评估
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, **fit_params)
                y_pred = model.predict(X_val)
                
                if scoring == 'accuracy':
                    score = accuracy_score(y_val, y_pred)
                elif scoring == 'precision':
                    score = precision_score(y_val, y_pred, average='weighted')
                elif scoring == 'recall':
                    score = recall_score(y_val, y_pred, average='weighted')
                elif scoring == 'f1':
                    score = f1_score(y_val, y_pred, average='weighted')
                elif scoring == 'roc_auc':
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_val)
                        if y_proba.shape[1] == 2:
                            score = roc_auc_score(y_val, y_proba[:, 1])
                        else:
                            score = roc_auc_score(y_val, y_proba, multi_class='ovr')
                    else:
                        score = roc_auc_score(y_val, y_pred)
                else:
                    raise ValueError(f"Unknown scoring method: {scoring}")
            
            # 否则使用交叉验证
            else:
                if len(np.unique(y_train)) > 1:
                    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
                else:
                    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
                
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_splitter,
                    scoring=scoring,
                    fit_params=fit_params
                )
                score = np.mean(scores)
            
            return score
            
        except Exception as e:
            logger.warning(f"Objective function failed with params {params}: {e}")
            return float('-inf')
    
    return objective


def create_multi_objective_function(model_class,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  objectives: List[str],
                                  X_val: Optional[np.ndarray] = None,
                                  y_val: Optional[np.ndarray] = None,
                                  cv: int = 5,
                                  fit_params: Optional[Dict[str, Any]] = None,
                                  random_state: Optional[int] = None) -> Callable[[Dict[str, Any]], List[float]]:
    """
    创建多目标函数
    
    Args:
        model_class: 模型类
        X_train: 训练特征
        y_train: 训练标签
        objectives: 目标列表
        X_val: 验证特征
        y_val: 验证标签
        cv: 交叉验证折数
        fit_params: 拟合参数
        random_state: 随机种子
        
    Returns:
        多目标函数
    """
    fit_params = fit_params or {}
    
    def multi_objective(params: Dict[str, Any]) -> List[float]:
        try:
            # 创建模型
            if hasattr(model_class, '__call__'):
                model = model_class(**params)
            else:
                model = model_class
                for key, value in params.items():
                    setattr(model, key, value)
            
            scores = []
            
            # 如果提供了验证集，使用验证集评估
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, **fit_params)
                y_pred = model.predict(X_val)
                
                for objective in objectives:
                    if objective == 'accuracy':
                        score = accuracy_score(y_val, y_pred)
                    elif objective == 'precision':
                        score = precision_score(y_val, y_pred, average='weighted')
                    elif objective == 'recall':
                        score = recall_score(y_val, y_pred, average='weighted')
                    elif objective == 'f1':
                        score = f1_score(y_val, y_pred, average='weighted')
                    elif objective == 'roc_auc':
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_val)
                            if y_proba.shape[1] == 2:
                                score = roc_auc_score(y_val, y_proba[:, 1])
                            else:
                                score = roc_auc_score(y_val, y_proba, multi_class='ovr')
                        else:
                            score = roc_auc_score(y_val, y_pred)
                    elif objective == 'model_complexity':
                        # 模型复杂度（负值，因为我们想要最小化）
                        if hasattr(model, 'n_estimators'):
                            score = -model.n_estimators
                        elif hasattr(model, 'max_depth'):
                            score = -model.max_depth if model.max_depth else -10
                        else:
                            score = -len(params)
                    else:
                        raise ValueError(f"Unknown objective: {objective}")
                    
                    scores.append(score)
            
            # 否则使用交叉验证
            else:
                if len(np.unique(y_train)) > 1:
                    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
                else:
                    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
                
                for objective in objectives:
                    if objective == 'model_complexity':
                        # 模型复杂度（负值，因为我们想要最小化）
                        if hasattr(model, 'n_estimators'):
                            score = -model.n_estimators
                        elif hasattr(model, 'max_depth'):
                            score = -model.max_depth if model.max_depth else -10
                        else:
                            score = -len(params)
                    else:
                        cv_scores = cross_val_score(
                            model, X_train, y_train,
                            cv=cv_splitter,
                            scoring=objective,
                            fit_params=fit_params
                        )
                        score = np.mean(cv_scores)
                    
                    scores.append(score)
            
            return scores
            
        except Exception as e:
            logger.warning(f"Multi-objective function failed with params {params}: {e}")
            return [float('-inf')] * len(objectives)
    
    return multi_objective


def compare_optimizers(optimizers: Dict[str, Any],
                      objective_function: Callable[[Dict[str, Any]], float],
                      n_trials: int = 100,
                      n_runs: int = 5,
                      random_state: Optional[int] = None) -> pd.DataFrame:
    """
    比较不同优化器的性能
    
    Args:
        optimizers: 优化器字典
        objective_function: 目标函数
        n_trials: 每次运行的试验次数
        n_runs: 运行次数
        random_state: 随机种子
        
    Returns:
        比较结果DataFrame
    """
    results = []
    
    for optimizer_name, optimizer in optimizers.items():
        logger.info(f"Testing optimizer: {optimizer_name}")
        
        run_results = []
        
        for run in range(n_runs):
            # 设置随机种子
            if random_state is not None:
                np.random.seed(random_state + run)
            
            # 运行优化
            result = optimizer.optimize(objective_function, n_trials=n_trials)
            
            run_results.append({
                'optimizer': optimizer_name,
                'run': run,
                'best_score': result.best_score,
                'n_trials': result.n_trials,
                'optimization_time': result.optimization_time,
                'convergence_trial': result.get_convergence_trial()
            })
        
        results.extend(run_results)
    
    df = pd.DataFrame(results)
    
    # 计算统计信息
    summary = df.groupby('optimizer').agg({
        'best_score': ['mean', 'std', 'min', 'max'],
        'optimization_time': ['mean', 'std'],
        'convergence_trial': ['mean', 'std']
    }).round(6)
    
    logger.info("Optimizer comparison summary:")
    logger.info(f"\n{summary}")
    
    return df


def plot_optimization_comparison(results_df: pd.DataFrame,
                               save_path: Optional[str] = None):
    """
    绘制优化器比较图
    
    Args:
        results_df: 比较结果DataFrame
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 最佳分数分布
    sns.boxplot(data=results_df, x='optimizer', y='best_score', ax=axes[0, 0])
    axes[0, 0].set_title('Best Score Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 优化时间分布
    sns.boxplot(data=results_df, x='optimizer', y='optimization_time', ax=axes[0, 1])
    axes[0, 1].set_title('Optimization Time Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 收敛试验分布
    sns.boxplot(data=results_df, x='optimizer', y='convergence_trial', ax=axes[1, 0])
    axes[1, 0].set_title('Convergence Trial Distribution')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 平均性能对比
    summary = results_df.groupby('optimizer')['best_score'].mean().sort_values(ascending=False)
    summary.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Average Best Score by Optimizer')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_convergence_curves(optimization_results: Dict[str, OptimizationResult],
                          save_path: Optional[str] = None):
    """
    绘制收敛曲线
    
    Args:
        optimization_results: 优化结果字典
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    
    for name, result in optimization_results.items():
        # 计算累积最佳分数
        scores = [r['score'] for r in result.all_results if r['success']]
        if not scores:
            continue
        
        cumulative_best = []
        current_best = float('-inf')
        
        for score in scores:
            if score > current_best:
                current_best = score
            cumulative_best.append(current_best)
        
        plt.plot(range(1, len(cumulative_best) + 1), cumulative_best, 
                label=name, linewidth=2, marker='o', markersize=3)
    
    plt.xlabel('Trial Number')
    plt.ylabel('Best Score')
    plt.title('Optimization Convergence Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_parameter_importance(optimization_result: OptimizationResult,
                            hyperparameter_space: HyperparameterSpace,
                            method: str = 'correlation',
                            save_path: Optional[str] = None):
    """
    绘制参数重要性
    
    Args:
        optimization_result: 优化结果
        hyperparameter_space: 超参数空间
        method: 计算方法 ('correlation', 'mutual_info', 'permutation')
        save_path: 保存路径
    """
    # 提取成功的试验
    successful_results = [r for r in optimization_result.all_results if r['success']]
    
    if len(successful_results) < 10:
        logger.warning("Not enough successful trials for parameter importance analysis")
        return
    
    # 构建数据
    param_names = list(hyperparameter_space.parameters.keys())
    X = []
    y = []
    
    for result in successful_results:
        params = result['params']
        param_values = []
        
        for name in param_names:
            value = params.get(name, 0)
            # 对分类参数进行编码
            if isinstance(value, (str, bool)):
                param = hyperparameter_space.parameters[name]
                if hasattr(param, 'choices'):
                    value = param.choices.index(value) if value in param.choices else 0
                else:
                    value = int(value) if isinstance(value, bool) else 0
            param_values.append(value)
        
        X.append(param_values)
        y.append(result['score'])
    
    X = np.array(X)
    y = np.array(y)
    
    # 计算重要性
    if method == 'correlation':
        importances = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            importances.append(abs(corr) if not np.isnan(corr) else 0)
    
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        importances = mutual_info_regression(X, y)
    
    elif method == 'permutation':
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 绘制重要性
    importance_df = pd.DataFrame({
        'parameter': param_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, max(6, len(param_names) * 0.5)))
    plt.barh(importance_df['parameter'], importance_df['importance'])
    plt.xlabel(f'Importance ({method})')
    plt.title('Parameter Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def save_optimization_results(results: Dict[str, OptimizationResult],
                            save_dir: Union[str, Path]):
    """
    保存优化结果
    
    Args:
        results: 优化结果字典
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for name, result in results.items():
        # 保存为JSON
        json_path = save_dir / f"{name}_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                'best_params': result.best_params,
                'best_score': result.best_score,
                'n_trials': result.n_trials,
                'optimization_time': result.optimization_time,
                'all_results': result.all_results
            }, f, indent=2)
        
        # 保存为pickle
        pickle_path = save_dir / f"{name}_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        
        logger.info(f"Saved {name} results to {save_dir}")


def load_optimization_results(save_dir: Union[str, Path]) -> Dict[str, OptimizationResult]:
    """
    加载优化结果
    
    Args:
        save_dir: 保存目录
        
    Returns:
        优化结果字典
    """
    save_dir = Path(save_dir)
    results = {}
    
    for pickle_path in save_dir.glob("*_results.pkl"):
        name = pickle_path.stem.replace('_results', '')
        
        with open(pickle_path, 'rb') as f:
            result = pickle.load(f)
        
        results[name] = result
        logger.info(f"Loaded {name} results from {pickle_path}")
    
    return results


def calculate_optimization_efficiency(optimization_result: OptimizationResult,
                                    baseline_score: float = 0.0) -> Dict[str, float]:
    """
    计算优化效率指标
    
    Args:
        optimization_result: 优化结果
        baseline_score: 基线分数
        
    Returns:
        效率指标字典
    """
    successful_results = [r for r in optimization_result.all_results if r['success']]
    
    if not successful_results:
        return {}
    
    scores = [r['score'] for r in successful_results]
    times = [r.get('duration', 0) for r in successful_results]
    
    # 计算指标
    metrics = {
        'improvement_ratio': optimization_result.get_improvement_ratio(baseline_score),
        'convergence_trial': optimization_result.get_convergence_trial(),
        'success_rate': len(successful_results) / len(optimization_result.all_results),
        'score_std': np.std(scores),
        'score_range': np.max(scores) - np.min(scores),
        'avg_trial_time': np.mean(times) if times else 0,
        'total_time': optimization_result.optimization_time,
        'trials_per_second': len(successful_results) / optimization_result.optimization_time if optimization_result.optimization_time > 0 else 0
    }
    
    return metrics


def suggest_optimization_strategy(hyperparameter_space: HyperparameterSpace,
                                n_trials: int,
                                time_budget: Optional[float] = None) -> Dict[str, Any]:
    """
    建议优化策略
    
    Args:
        hyperparameter_space: 超参数空间
        n_trials: 试验次数
        time_budget: 时间预算（秒）
        
    Returns:
        建议策略字典
    """
    n_params = len(hyperparameter_space.parameters)
    space_size = hyperparameter_space.get_space_size()
    
    suggestions = {
        'recommended_optimizer': None,
        'recommended_params': {},
        'reasoning': []
    }
    
    # 根据参数数量和空间大小选择优化器
    if space_size <= 1000:
        suggestions['recommended_optimizer'] = 'grid_search'
        suggestions['reasoning'].append("Small search space, grid search is feasible")
    
    elif n_params <= 5 and n_trials >= 100:
        suggestions['recommended_optimizer'] = 'bayesian_optimization'
        suggestions['reasoning'].append("Low-dimensional space with sufficient trials, Bayesian optimization is efficient")
    
    elif n_trials >= 200:
        suggestions['recommended_optimizer'] = 'optuna'
        suggestions['recommended_params'] = {'sampler': 'tpe'}
        suggestions['reasoning'].append("Large number of trials, TPE sampler in Optuna is recommended")
    
    else:
        suggestions['recommended_optimizer'] = 'random_search'
        suggestions['reasoning'].append("Limited trials or high-dimensional space, random search is robust")
    
    # 时间预算考虑
    if time_budget:
        if time_budget < 300:  # 5分钟
            suggestions['recommended_optimizer'] = 'random_search'
            suggestions['reasoning'].append("Limited time budget, random search has low overhead")
        
        elif time_budget > 3600:  # 1小时
            if suggestions['recommended_optimizer'] != 'grid_search':
                suggestions['recommended_optimizer'] = 'bayesian_optimization'
                suggestions['reasoning'].append("Sufficient time budget, Bayesian optimization can be beneficial")
    
    # 参数类型考虑
    categorical_params = sum(1 for p in hyperparameter_space.parameters.values() 
                           if p.param_type.name in ['CATEGORICAL', 'BOOLEAN'])
    
    if categorical_params / n_params > 0.5:
        suggestions['recommended_optimizer'] = 'optuna'
        suggestions['recommended_params'] = {'sampler': 'tpe'}
        suggestions['reasoning'].append("Many categorical parameters, TPE handles them well")
    
    return suggestions