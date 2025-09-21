"""
Optuna和贝叶斯优化综合示例

演示如何使用Optuna和贝叶斯优化进行超参数调优，包括：
1. 单目标优化
2. 多目标优化
3. 不同优化器比较
4. 结果可视化和分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 导入我们的优化模块
from src.optimization import (
    OptunaOptimizer, MultiObjectiveOptunaOptimizer,
    BayesianOptimizer, TPEOptimizer,
    HyperparameterSpace, ParameterType, Distribution,
    create_objective_function, create_multi_objective_function,
    compare_optimizers, plot_optimization_comparison,
    plot_convergence_curves, plot_parameter_importance,
    save_optimization_results, calculate_optimization_efficiency
)


def create_sample_datasets():
    """创建示例数据集"""
    print("=== 创建示例数据集 ===")
    
    datasets = {}
    
    # 1. 合成分类数据集
    print("1. 生成合成分类数据集...")
    X_synthetic, y_synthetic = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=3,
        class_sep=0.8,
        random_state=42
    )
    
    X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
        X_synthetic, y_synthetic, test_size=0.2, random_state=42, stratify=y_synthetic
    )
    
    datasets['synthetic'] = {
        'X_train': X_train_syn, 'X_test': X_test_syn,
        'y_train': y_train_syn, 'y_test': y_test_syn,
        'name': '合成分类数据集'
    }
    
    # 2. 乳腺癌数据集
    print("2. 加载乳腺癌数据集...")
    cancer = load_breast_cancer()
    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
    )
    
    datasets['cancer'] = {
        'X_train': X_train_cancer, 'X_test': X_test_cancer,
        'y_train': y_train_cancer, 'y_test': y_test_cancer,
        'name': '乳腺癌数据集'
    }
    
    # 3. 红酒数据集
    print("3. 加载红酒数据集...")
    wine = load_wine()
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
    )
    
    datasets['wine'] = {
        'X_train': X_train_wine, 'X_test': X_test_wine,
        'y_train': y_train_wine, 'y_test': y_test_wine,
        'name': '红酒数据集'
    }
    
    print(f"创建了 {len(datasets)} 个数据集")
    for name, data in datasets.items():
        print(f"  {data['name']}: 训练集 {data['X_train'].shape}, 测试集 {data['X_test'].shape}")
    
    return datasets


def create_hyperparameter_spaces():
    """创建不同模型的超参数空间"""
    print("\n=== 创建超参数空间 ===")
    
    spaces = {}
    
    # 1. 随机森林超参数空间
    print("1. 创建随机森林超参数空间...")
    rf_space = HyperparameterSpace()
    rf_space.add_int('n_estimators', 50, 300)
    rf_space.add_int('max_depth', 3, 20)
    rf_space.add_int('min_samples_split', 2, 20)
    rf_space.add_int('min_samples_leaf', 1, 10)
    rf_space.add_categorical('max_features', ['sqrt', 'log2', None])
    rf_space.add_float('max_samples', 0.5, 1.0)
    
    spaces['random_forest'] = rf_space
    
    # 2. 梯度提升超参数空间
    print("2. 创建梯度提升超参数空间...")
    gb_space = HyperparameterSpace()
    gb_space.add_int('n_estimators', 50, 200)
    gb_space.add_float('learning_rate', 0.01, 0.3, log=True)
    gb_space.add_int('max_depth', 3, 10)
    gb_space.add_float('subsample', 0.6, 1.0)
    gb_space.add_int('min_samples_split', 2, 20)
    gb_space.add_int('min_samples_leaf', 1, 10)
    
    spaces['gradient_boosting'] = gb_space
    
    # 3. SVM超参数空间
    print("3. 创建SVM超参数空间...")
    svm_space = HyperparameterSpace()
    svm_space.add_float('C', 0.1, 100.0, log=True)
    svm_space.add_float('gamma', 1e-4, 1.0, log=True)
    svm_space.add_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    
    spaces['svm'] = svm_space
    
    # 4. 神经网络超参数空间
    print("4. 创建神经网络超参数空间...")
    mlp_space = HyperparameterSpace()
    mlp_space.add_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50), (100, 100)])
    mlp_space.add_float('alpha', 1e-5, 1e-1, log=True)
    mlp_space.add_float('learning_rate_init', 1e-4, 1e-1, log=True)
    mlp_space.add_categorical('activation', ['relu', 'tanh', 'logistic'])
    mlp_space.add_categorical('solver', ['adam', 'lbfgs'])
    
    spaces['mlp'] = mlp_space
    
    print(f"创建了 {len(spaces)} 个超参数空间")
    for name, space in spaces.items():
        print(f"  {name}: {len(space)} 个参数")
    
    return spaces


def single_objective_optimization_example(datasets, spaces):
    """单目标优化示例"""
    print("\n=== 单目标优化示例 ===")
    
    # 选择数据集和模型
    dataset = datasets['synthetic']
    space = spaces['random_forest']
    
    print(f"使用数据集: {dataset['name']}")
    print(f"优化模型: 随机森林")
    
    # 创建目标函数
    def objective_function(params):
        model = RandomForestClassifier(random_state=42, **params)
        scores = cross_val_score(
            model, dataset['X_train'], dataset['y_train'],
            cv=5, scoring='f1_macro'
        )
        return scores.mean()
    
    # 1. Optuna优化
    print("\n1. 使用Optuna优化器...")
    optuna_optimizer = OptunaOptimizer(
        hyperparameter_space=space,
        sampler_type='tpe',
        random_state=42
    )
    
    optuna_result = optuna_optimizer.optimize(
        objective_function=objective_function,
        n_trials=50
    )
    
    print(f"Optuna最佳得分: {optuna_result.best_score:.4f}")
    print(f"Optuna最佳参数: {optuna_result.best_params}")
    
    # 2. 贝叶斯优化
    print("\n2. 使用贝叶斯优化器...")
    bayes_optimizer = BayesianOptimizer(
        hyperparameter_space=space,
        acquisition_function='ei',
        n_initial_points=10,
        random_state=42
    )
    
    bayes_result = bayes_optimizer.optimize(
        objective_function=objective_function,
        n_trials=50
    )
    
    print(f"贝叶斯优化最佳得分: {bayes_result.best_score:.4f}")
    print(f"贝叶斯优化最佳参数: {bayes_result.best_params}")
    
    # 3. TPE优化
    print("\n3. 使用TPE优化器...")
    tpe_optimizer = TPEOptimizer(
        hyperparameter_space=space,
        n_initial_points=10,
        random_state=42
    )
    
    tpe_result = tpe_optimizer.optimize(
        objective_function=objective_function,
        n_trials=50
    )
    
    print(f"TPE优化最佳得分: {tpe_result.best_score:.4f}")
    print(f"TPE优化最佳参数: {tpe_result.best_params}")
    
    return {
        'optuna': optuna_result,
        'bayesian': bayes_result,
        'tpe': tpe_result
    }


def multi_objective_optimization_example(datasets, spaces):
    """多目标优化示例"""
    print("\n=== 多目标优化示例 ===")
    
    dataset = datasets['cancer']
    space = spaces['gradient_boosting']
    
    print(f"使用数据集: {dataset['name']}")
    print(f"优化模型: 梯度提升")
    print("优化目标: 准确率 + F1分数")
    
    # 创建多目标函数
    def multi_objective_function(params):
        model = GradientBoostingClassifier(random_state=42, **params)
        
        # 计算准确率
        acc_scores = cross_val_score(
            model, dataset['X_train'], dataset['y_train'],
            cv=5, scoring='accuracy'
        )
        
        # 计算F1分数
        f1_scores = cross_val_score(
            model, dataset['X_train'], dataset['y_train'],
            cv=5, scoring='f1_macro'
        )
        
        return [acc_scores.mean(), f1_scores.mean()]
    
    # 使用多目标Optuna优化
    print("\n使用多目标Optuna优化器...")
    multi_optimizer = MultiObjectiveOptunaOptimizer(
        hyperparameter_space=space,
        sampler_type='nsga2',
        random_state=42
    )
    
    multi_result = multi_optimizer.optimize(
        objective_function=multi_objective_function,
        n_trials=100
    )
    
    print(f"找到 {len(multi_result.pareto_front)} 个帕累托最优解")
    print("前5个帕累托最优解:")
    for i, (params, scores) in enumerate(zip(multi_result.pareto_front, multi_result.pareto_scores)):
        if i >= 5:
            break
        print(f"  解{i+1}: 准确率={scores[0]:.4f}, F1={scores[1]:.4f}")
    
    return multi_result


def optimizer_comparison_example(datasets, spaces):
    """优化器比较示例"""
    print("\n=== 优化器比较示例 ===")
    
    dataset = datasets['wine']
    space = spaces['svm']
    
    print(f"使用数据集: {dataset['name']}")
    print(f"优化模型: SVM")
    
    # 创建目标函数
    def objective_function(params):
        model = SVC(random_state=42, **params)
        scores = cross_val_score(
            model, dataset['X_train'], dataset['y_train'],
            cv=5, scoring='accuracy'
        )
        return scores.mean()
    
    # 创建不同的优化器
    optimizers = {
        'Optuna-TPE': OptunaOptimizer(
            hyperparameter_space=space,
            sampler_type='tpe',
            random_state=42
        ),
        'Optuna-Random': OptunaOptimizer(
            hyperparameter_space=space,
            sampler_type='random',
            random_state=42
        ),
        'Bayesian-EI': BayesianOptimizer(
            hyperparameter_space=space,
            acquisition_function='ei',
            random_state=42
        ),
        'Bayesian-UCB': BayesianOptimizer(
            hyperparameter_space=space,
            acquisition_function='ucb',
            random_state=42
        ),
        'TPE': TPEOptimizer(
            hyperparameter_space=space,
            random_state=42
        )
    }
    
    # 比较优化器
    print(f"\n比较 {len(optimizers)} 个优化器...")
    comparison_results = compare_optimizers(
        optimizers=optimizers,
        objective_function=objective_function,
        n_trials=30,
        n_runs=3,
        random_state=42
    )
    
    print("\n优化器比较结果:")
    print(comparison_results.round(4))
    
    return comparison_results, optimizers


def visualization_example(single_results, multi_result, comparison_results):
    """可视化示例"""
    print("\n=== 创建可视化图表 ===")
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 收敛曲线比较
    print("1. 绘制收敛曲线...")
    ax1 = plt.subplot(2, 3, 1)
    plot_convergence_curves(single_results, save_path=None)
    plt.title('单目标优化收敛曲线比较')
    
    # 2. 优化器性能比较
    print("2. 绘制优化器性能比较...")
    ax2 = plt.subplot(2, 3, 2)
    plot_optimization_comparison(comparison_results, save_path=None)
    plt.title('优化器性能比较')
    
    # 3. 多目标优化帕累托前沿
    print("3. 绘制帕累托前沿...")
    ax3 = plt.subplot(2, 3, 3)
    if hasattr(multi_result, 'pareto_scores') and len(multi_result.pareto_scores) > 0:
        pareto_scores = np.array(multi_result.pareto_scores)
        plt.scatter(pareto_scores[:, 0], pareto_scores[:, 1], alpha=0.7, s=50)
        plt.xlabel('准确率')
        plt.ylabel('F1分数')
        plt.title('多目标优化帕累托前沿')
        plt.grid(True, alpha=0.3)
    
    # 4. 参数重要性分析
    print("4. 绘制参数重要性...")
    ax4 = plt.subplot(2, 3, 4)
    if 'optuna' in single_results:
        try:
            from src.optimization.hyperparameter_space import HyperparameterSpace
            space = HyperparameterSpace()
            space.add_int('n_estimators', 50, 300)
            space.add_int('max_depth', 3, 20)
            space.add_int('min_samples_split', 2, 20)
            space.add_int('min_samples_leaf', 1, 10)
            space.add_categorical('max_features', ['sqrt', 'log2', None])
            space.add_float('max_samples', 0.5, 1.0)
            
            plot_parameter_importance(
                single_results['optuna'], 
                space, 
                method='correlation',
                save_path=None
            )
            plt.title('参数重要性分析')
        except Exception as e:
            plt.text(0.5, 0.5, f'参数重要性分析\n(需要更多数据)', 
                    ha='center', va='center', transform=ax4.transAxes)
    
    # 5. 优化历史分布
    print("5. 绘制优化历史分布...")
    ax5 = plt.subplot(2, 3, 5)
    if 'optuna' in single_results and hasattr(single_results['optuna'], 'trial_scores'):
        scores = single_results['optuna'].trial_scores
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(single_results['optuna'].best_score, color='red', 
                   linestyle='--', label=f'最佳得分: {single_results["optuna"].best_score:.4f}')
        plt.xlabel('目标函数值')
        plt.ylabel('频次')
        plt.title('Optuna优化历史分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 6. 优化效率比较
    print("6. 绘制优化效率比较...")
    ax6 = plt.subplot(2, 3, 6)
    
    # 计算优化效率
    efficiency_data = []
    for name, result in single_results.items():
        if hasattr(result, 'trial_scores') and len(result.trial_scores) > 0:
            efficiency = calculate_optimization_efficiency(result, baseline_score=0.5)
            efficiency_data.append({
                'Optimizer': name,
                'Convergence_Speed': efficiency.get('convergence_speed', 0),
                'Final_Performance': efficiency.get('final_performance', 0),
                'Stability': efficiency.get('stability', 0)
            })
    
    if efficiency_data:
        eff_df = pd.DataFrame(efficiency_data)
        eff_df.set_index('Optimizer').plot(kind='bar', ax=ax6)
        plt.title('优化效率比较')
        plt.ylabel('效率指标')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('optuna_bayesopt_results.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存到 optuna_bayesopt_results.png")
    
    return fig


def generate_comprehensive_report(single_results, multi_result, comparison_results, datasets):
    """生成综合报告"""
    print("\n=== 生成综合报告 ===")
    
    report = []
    report.append("# Optuna和贝叶斯优化综合报告\n")
    report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 数据集信息
    report.append("## 数据集信息\n")
    for name, data in datasets.items():
        report.append(f"### {data['name']}\n")
        report.append(f"- 训练集大小: {data['X_train'].shape}\n")
        report.append(f"- 测试集大小: {data['X_test'].shape}\n")
        report.append(f"- 特征数量: {data['X_train'].shape[1]}\n")
        report.append(f"- 类别数量: {len(np.unique(data['y_train']))}\n\n")
    
    # 单目标优化结果
    report.append("## 单目标优化结果\n")
    for name, result in single_results.items():
        report.append(f"### {name.upper()}优化器\n")
        report.append(f"- 最佳得分: {result.best_score:.6f}\n")
        report.append(f"- 最佳参数:\n")
        for param, value in result.best_params.items():
            report.append(f"  - {param}: {value}\n")
        
        if hasattr(result, 'trial_scores') and len(result.trial_scores) > 0:
            scores = result.trial_scores
            report.append(f"- 优化统计:\n")
            report.append(f"  - 试验次数: {len(scores)}\n")
            report.append(f"  - 平均得分: {np.mean(scores):.6f}\n")
            report.append(f"  - 标准差: {np.std(scores):.6f}\n")
            report.append(f"  - 最小得分: {np.min(scores):.6f}\n")
            report.append(f"  - 最大得分: {np.max(scores):.6f}\n")
        report.append("\n")
    
    # 多目标优化结果
    report.append("## 多目标优化结果\n")
    if hasattr(multi_result, 'pareto_front'):
        report.append(f"- 帕累托最优解数量: {len(multi_result.pareto_front)}\n")
        report.append("- 前5个帕累托最优解:\n")
        for i, (params, scores) in enumerate(zip(multi_result.pareto_front, multi_result.pareto_scores)):
            if i >= 5:
                break
            report.append(f"  {i+1}. 准确率={scores[0]:.4f}, F1={scores[1]:.4f}\n")
        report.append("\n")
    
    # 优化器比较结果
    report.append("## 优化器比较结果\n")
    report.append("```\n")
    report.append(comparison_results.to_string())
    report.append("\n```\n\n")
    
    # 主要发现和建议
    report.append("## 主要发现和建议\n")
    
    # 找出最佳优化器
    best_optimizer = max(single_results.items(), key=lambda x: x[1].best_score)
    report.append(f"### 最佳单目标优化器: {best_optimizer[0].upper()}\n")
    report.append(f"- 最佳得分: {best_optimizer[1].best_score:.6f}\n")
    
    # 优化器特点分析
    report.append("### 优化器特点分析\n")
    report.append("- **Optuna-TPE**: 基于树形结构的Parzen估计器，适合中等维度的超参数空间\n")
    report.append("- **贝叶斯优化**: 基于高斯过程，适合昂贵的目标函数评估\n")
    report.append("- **TPE**: Tree-structured Parzen Estimator，在离散和连续参数混合时表现良好\n\n")
    
    # 使用建议
    report.append("### 使用建议\n")
    report.append("1. **小规模问题**: 使用贝叶斯优化，能够快速收敛\n")
    report.append("2. **大规模问题**: 使用Optuna-TPE，具有更好的可扩展性\n")
    report.append("3. **多目标问题**: 使用多目标Optuna，支持帕累托前沿分析\n")
    report.append("4. **混合参数类型**: 使用TPE，对离散和连续参数都有良好支持\n\n")
    
    # 保存报告
    report_content = ''.join(report)
    with open('optuna_bayesopt_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("综合报告已保存到 optuna_bayesopt_report.md")
    return report_content


def main():
    """主函数"""
    print("开始Optuna和贝叶斯优化综合示例")
    print("=" * 50)
    
    try:
        # 1. 创建数据集
        datasets = create_sample_datasets()
        
        # 2. 创建超参数空间
        spaces = create_hyperparameter_spaces()
        
        # 3. 单目标优化示例
        single_results = single_objective_optimization_example(datasets, spaces)
        
        # 4. 多目标优化示例
        multi_result = multi_objective_optimization_example(datasets, spaces)
        
        # 5. 优化器比较示例
        comparison_results, optimizers = optimizer_comparison_example(datasets, spaces)
        
        # 6. 创建可视化
        visualization_example(single_results, multi_result, comparison_results)
        
        # 7. 生成综合报告
        report = generate_comprehensive_report(single_results, multi_result, comparison_results, datasets)
        
        # 8. 保存优化结果
        print("\n=== 保存优化结果 ===")
        all_results = {**single_results, 'multi_objective': multi_result}
        save_optimization_results(all_results, 'optimization_results')
        print("优化结果已保存到 optimization_results 目录")
        
        print("\n" + "=" * 50)
        print("Optuna和贝叶斯优化综合示例完成！")
        print("生成的文件:")
        print("- optuna_bayesopt_results.png: 可视化结果")
        print("- optuna_bayesopt_report.md: 综合报告")
        print("- optimization_results/: 详细优化结果")
        
        return {
            'single_results': single_results,
            'multi_result': multi_result,
            'comparison_results': comparison_results,
            'datasets': datasets,
            'spaces': spaces
        }
        
    except Exception as e:
        print(f"示例运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    results = main()