#!/usr/bin/env python3
"""
测试所有优化器的功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from optimization.hyperparameter_space import HyperparameterSpace, ParameterType
from optimization.grid_search import GridSearchOptimizer
from optimization.random_search import RandomSearchOptimizer
from optimization.bayesian_optimization import BayesianOptimizer, TPEOptimizer
from optimization.optuna_optimizer import OptunaOptimizer

def test_objective_function(params):
    """简单的测试目标函数"""
    x = params['x']
    y = params['y']
    return -(x**2 + y**2)  # 最大化问题，所以返回负值

def create_test_hyperparameter_space():
    """创建测试用的超参数空间"""
    space = HyperparameterSpace()
    space.add_float('x', -5.0, 5.0)
    space.add_float('y', -5.0, 5.0)
    return space

def test_optimizer(optimizer_class, optimizer_name, **kwargs):
    """测试单个优化器"""
    print(f"\n{'='*50}")
    print(f"测试 {optimizer_name}")
    print(f"{'='*50}")
    
    try:
        # 创建超参数空间
        space = create_test_hyperparameter_space()
        
        # 创建优化器
        optimizer = optimizer_class(space, **kwargs)
        
        # 运行优化
        result = optimizer.optimize(test_objective_function, n_trials=10)
        
        print(f"✅ {optimizer_name} 测试成功!")
        print(f"最佳参数: {result.best_params}")
        print(f"最佳值: {result.best_score}")
        print(f"总试验次数: {result.n_trials}")
        
        return True
        
    except Exception as e:
        print(f"❌ {optimizer_name} 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试所有优化器...")
    
    test_results = {}
    
    # 测试GridSearchOptimizer
    test_results['GridSearch'] = test_optimizer(
        GridSearchOptimizer, 
        "GridSearchOptimizer",
        verbose=False
    )
    
    # 测试RandomSearchOptimizer
    test_results['RandomSearch'] = test_optimizer(
        RandomSearchOptimizer, 
        "RandomSearchOptimizer",
        random_state=42,
        verbose=False
    )
    
    # 测试BayesianOptimizer
    test_results['BayesianOptimizer'] = test_optimizer(
        BayesianOptimizer, 
        "BayesianOptimizer",
        n_initial_points=3,
        random_state=42,
        verbose=False
    )
    
    # 测试TPEOptimizer
    test_results['TPEOptimizer'] = test_optimizer(
        TPEOptimizer, 
        "TPEOptimizer",
        n_initial_points=3,
        random_state=42,
        verbose=False
    )
    
    # 测试OptunaOptimizer (跳过，需要安装optuna)
    # test_results['OptunaOptimizer'] = test_optimizer(
    #     OptunaOptimizer, 
    #     "OptunaOptimizer",
    #     sampler='tpe',
    #     random_state=42,
    #     verbose=False
    # )
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    
    passed = 0
    total = len(test_results)
    
    for optimizer_name, success in test_results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{optimizer_name:20} : {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个优化器测试通过")
    print("注意: OptunaOptimizer 测试已跳过（需要安装 optuna 库）")
    
    if passed == total:
        print("🎉 所有优化器测试通过!")
        return True
    else:
        print("⚠️  部分优化器测试失败，请检查错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)