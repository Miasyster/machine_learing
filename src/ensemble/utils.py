"""
集成学习工具函数

提供集成模型的评估、可视化、比较和优化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.model_selection import cross_val_score, learning_curve
import logging
import pickle
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)


def evaluate_ensemble_performance(ensemble, X_test: np.ndarray, y_test: np.ndarray,
                                 task_type: str = 'auto') -> Dict[str, Any]:
    """
    评估集成模型性能
    
    Args:
        ensemble: 集成模型
        X_test: 测试特征
        y_test: 测试标签
        task_type: 任务类型 ('classification', 'regression', 'auto')
        
    Returns:
        性能评估结果
    """
    if task_type == 'auto':
        task_type = 'classification' if hasattr(ensemble, '_is_classifier') and ensemble._is_classifier() else 'regression'
    
    # 获取预测结果
    result = ensemble.predict(X_test)
    predictions = result.predictions
    
    performance = {
        'task_type': task_type,
        'n_samples': len(y_test),
        'prediction_time': 0
    }
    
    # 计算预测时间
    start_time = time.time()
    _ = ensemble.predict(X_test[:min(100, len(X_test))])
    prediction_time = (time.time() - start_time) / min(100, len(X_test))
    performance['prediction_time'] = prediction_time
    
    if task_type == 'classification':
        # 分类任务指标
        performance.update({
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
        })
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, predictions)
        performance['confusion_matrix'] = cm.tolist()
        
        # 分类报告
        try:
            performance['classification_report'] = classification_report(y_test, predictions, output_dict=True)
        except:
            performance['classification_report'] = None
        
        # ROC和AUC（二分类）
        if len(np.unique(y_test)) == 2 and hasattr(result, 'prediction_probabilities') and result.prediction_probabilities is not None:
            try:
                fpr, tpr, _ = roc_curve(y_test, result.prediction_probabilities[:, 1])
                performance['roc_auc'] = auc(fpr, tpr)
                performance['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            except:
                performance['roc_auc'] = None
                performance['roc_curve'] = None
    
    else:
        # 回归任务指标
        performance.update({
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2_score': r2_score(y_test, predictions)
        })
        
        # 残差统计
        residuals = y_test - predictions
        performance.update({
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_min': np.min(residuals),
            'residual_max': np.max(residuals)
        })
    
    # 个体模型性能
    if hasattr(result, 'individual_predictions') and result.individual_predictions:
        individual_performance = []
        for i, pred in enumerate(result.individual_predictions):
            if task_type == 'classification':
                acc = accuracy_score(y_test, pred)
                individual_performance.append({'model_index': i, 'accuracy': acc})
            else:
                mse = mean_squared_error(y_test, pred)
                individual_performance.append({'model_index': i, 'mse': mse})
        
        performance['individual_performance'] = individual_performance
    
    return performance


def compare_ensembles(ensembles: List[Any], ensemble_names: List[str],
                     X_test: np.ndarray, y_test: np.ndarray,
                     cv_folds: int = 5) -> Dict[str, Any]:
    """
    比较多个集成模型
    
    Args:
        ensembles: 集成模型列表
        ensemble_names: 集成模型名称列表
        X_test: 测试特征
        y_test: 测试标签
        cv_folds: 交叉验证折数
        
    Returns:
        比较结果
    """
    comparison_results = {
        'ensemble_names': ensemble_names,
        'performance_comparison': [],
        'statistical_tests': {},
        'ranking': {}
    }
    
    # 评估每个集成模型
    for i, (ensemble, name) in enumerate(zip(ensembles, ensemble_names)):
        logger.info(f"Evaluating ensemble {i+1}/{len(ensembles)}: {name}")
        
        # 单次评估
        performance = evaluate_ensemble_performance(ensemble, X_test, y_test)
        performance['ensemble_name'] = name
        comparison_results['performance_comparison'].append(performance)
        
        # 交叉验证评估
        try:
            if performance['task_type'] == 'classification':
                cv_scores = cross_val_score(ensemble, X_test, y_test, cv=cv_folds, scoring='accuracy')
                metric_name = 'accuracy'
            else:
                cv_scores = cross_val_score(ensemble, X_test, y_test, cv=cv_folds, scoring='neg_mean_squared_error')
                cv_scores = -cv_scores  # 转换为正值
                metric_name = 'mse'
            
            performance[f'cv_{metric_name}_mean'] = np.mean(cv_scores)
            performance[f'cv_{metric_name}_std'] = np.std(cv_scores)
            performance['cv_scores'] = cv_scores.tolist()
        
        except Exception as e:
            logger.warning(f"Cross-validation failed for {name}: {e}")
            performance[f'cv_{metric_name}_mean'] = None
            performance[f'cv_{metric_name}_std'] = None
    
    # 排名
    if comparison_results['performance_comparison']:
        task_type = comparison_results['performance_comparison'][0]['task_type']
        
        if task_type == 'classification':
            # 按准确率排名
            sorted_results = sorted(comparison_results['performance_comparison'], 
                                  key=lambda x: x.get('accuracy', 0), reverse=True)
            ranking_metric = 'accuracy'
        else:
            # 按MSE排名（越小越好）
            sorted_results = sorted(comparison_results['performance_comparison'], 
                                  key=lambda x: x.get('mse', float('inf')))
            ranking_metric = 'mse'
        
        comparison_results['ranking'] = {
            'metric': ranking_metric,
            'ranked_ensembles': [result['ensemble_name'] for result in sorted_results],
            'ranked_scores': [result.get(ranking_metric, None) for result in sorted_results]
        }
    
    return comparison_results


def visualize_ensemble_performance(performance_data: Dict[str, Any], 
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    可视化集成模型性能
    
    Args:
        performance_data: 性能数据
        save_path: 保存路径
        figsize: 图形大小
    """
    task_type = performance_data.get('task_type', 'classification')
    
    if task_type == 'classification':
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Ensemble Classification Performance', fontsize=16)
        
        # 混淆矩阵
        if 'confusion_matrix' in performance_data:
            cm = np.array(performance_data['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
        
        # ROC曲线
        if 'roc_curve' in performance_data and performance_data['roc_curve']:
            roc_data = performance_data['roc_curve']
            axes[0, 1].plot(roc_data['fpr'], roc_data['tpr'], 
                           label=f"AUC = {performance_data.get('roc_auc', 0):.3f}")
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
        
        # 性能指标
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = [performance_data.get(metric, 0) for metric in metrics]
        axes[1, 0].bar(metrics, metric_values)
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 个体模型性能
        if 'individual_performance' in performance_data:
            individual_perf = performance_data['individual_performance']
            model_indices = [p['model_index'] for p in individual_perf]
            accuracies = [p['accuracy'] for p in individual_perf]
            axes[1, 1].bar(model_indices, accuracies)
            axes[1, 1].set_title('Individual Model Performance')
            axes[1, 1].set_xlabel('Model Index')
            axes[1, 1].set_ylabel('Accuracy')
    
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Ensemble Regression Performance', fontsize=16)
        
        # 预测vs实际（需要额外数据）
        axes[0, 0].text(0.5, 0.5, 'Prediction vs Actual\n(Requires additional data)', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Prediction vs Actual')
        
        # 残差分布
        if all(key in performance_data for key in ['residual_mean', 'residual_std']):
            # 模拟残差分布
            residuals = np.random.normal(performance_data['residual_mean'], 
                                       performance_data['residual_std'], 1000)
            axes[0, 1].hist(residuals, bins=30, alpha=0.7)
            axes[0, 1].set_title('Residual Distribution')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
        
        # 性能指标
        metrics = ['mse', 'rmse', 'mae', 'r2_score']
        metric_values = [performance_data.get(metric, 0) for metric in metrics]
        axes[1, 0].bar(metrics, metric_values)
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 个体模型性能
        if 'individual_performance' in performance_data:
            individual_perf = performance_data['individual_performance']
            model_indices = [p['model_index'] for p in individual_perf]
            mses = [p['mse'] for p in individual_perf]
            axes[1, 1].bar(model_indices, mses)
            axes[1, 1].set_title('Individual Model Performance')
            axes[1, 1].set_xlabel('Model Index')
            axes[1, 1].set_ylabel('MSE')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance visualization saved to {save_path}")
    
    plt.show()


def visualize_ensemble_comparison(comparison_data: Dict[str, Any],
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    可视化集成模型比较
    
    Args:
        comparison_data: 比较数据
        save_path: 保存路径
        figsize: 图形大小
    """
    ensemble_names = comparison_data['ensemble_names']
    performance_comparison = comparison_data['performance_comparison']
    
    if not performance_comparison:
        logger.warning("No performance data to visualize")
        return
    
    task_type = performance_comparison[0]['task_type']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Ensemble Model Comparison', fontsize=16)
    
    if task_type == 'classification':
        # 准确率比较
        accuracies = [p.get('accuracy', 0) for p in performance_comparison]
        axes[0].bar(ensemble_names, accuracies)
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        
        # F1分数比较
        f1_scores = [p.get('f1_score', 0) for p in performance_comparison]
        axes[1].bar(ensemble_names, f1_scores)
        axes[1].set_title('F1 Score Comparison')
        axes[1].set_ylabel('F1 Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 交叉验证结果
        cv_means = [p.get('cv_accuracy_mean', 0) for p in performance_comparison]
        cv_stds = [p.get('cv_accuracy_std', 0) for p in performance_comparison]
        axes[2].bar(ensemble_names, cv_means, yerr=cv_stds, capsize=5)
        axes[2].set_title('Cross-Validation Accuracy')
        axes[2].set_ylabel('CV Accuracy')
        axes[2].tick_params(axis='x', rotation=45)
    
    else:
        # MSE比较
        mses = [p.get('mse', 0) for p in performance_comparison]
        axes[0].bar(ensemble_names, mses)
        axes[0].set_title('MSE Comparison')
        axes[0].set_ylabel('MSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R²分数比较
        r2_scores = [p.get('r2_score', 0) for p in performance_comparison]
        axes[1].bar(ensemble_names, r2_scores)
        axes[1].set_title('R² Score Comparison')
        axes[1].set_ylabel('R² Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 交叉验证结果
        cv_means = [p.get('cv_mse_mean', 0) for p in performance_comparison]
        cv_stds = [p.get('cv_mse_std', 0) for p in performance_comparison]
        axes[2].bar(ensemble_names, cv_means, yerr=cv_stds, capsize=5)
        axes[2].set_title('Cross-Validation MSE')
        axes[2].set_ylabel('CV MSE')
        axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison visualization saved to {save_path}")
    
    plt.show()


def analyze_model_diversity(ensemble, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    分析模型多样性
    
    Args:
        ensemble: 集成模型
        X: 特征数据
        y: 标签数据
        
    Returns:
        多样性分析结果
    """
    if not hasattr(ensemble, 'models') or not ensemble.models:
        raise ValueError("Ensemble must have individual models for diversity analysis")
    
    # 获取个体模型预测
    result = ensemble.predict(X)
    if not hasattr(result, 'individual_predictions') or not result.individual_predictions:
        raise ValueError("Ensemble must provide individual predictions for diversity analysis")
    
    individual_predictions = result.individual_predictions
    n_models = len(individual_predictions)
    
    diversity_metrics = {
        'n_models': n_models,
        'pairwise_diversity': {},
        'overall_diversity': {},
        'agreement_matrix': None,
        'disagreement_rate': None
    }
    
    # 计算成对多样性
    pairwise_diversities = []
    agreement_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pred_i = individual_predictions[i]
            pred_j = individual_predictions[j]
            
            if ensemble._is_classifier():
                # 分类任务：计算不一致率
                diversity = np.mean(pred_i != pred_j)
                agreement = np.mean(pred_i == pred_j)
            else:
                # 回归任务：计算相关系数的补
                correlation = np.corrcoef(pred_i, pred_j)[0, 1]
                diversity = 1 - abs(correlation)
                agreement = abs(correlation)
            
            pairwise_diversities.append(diversity)
            agreement_matrix[i, j] = agreement
            agreement_matrix[j, i] = agreement
    
    # 对角线设为1（自己与自己完全一致）
    np.fill_diagonal(agreement_matrix, 1.0)
    
    diversity_metrics['pairwise_diversity'] = {
        'mean': np.mean(pairwise_diversities),
        'std': np.std(pairwise_diversities),
        'min': np.min(pairwise_diversities),
        'max': np.max(pairwise_diversities),
        'values': pairwise_diversities
    }
    
    diversity_metrics['agreement_matrix'] = agreement_matrix.tolist()
    
    # 整体多样性指标
    if ensemble._is_classifier():
        # Q统计量
        q_statistics = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = individual_predictions[i]
                pred_j = individual_predictions[j]
                
                # 计算2x2列联表
                n11 = np.sum((pred_i == y) & (pred_j == y))
                n10 = np.sum((pred_i == y) & (pred_j != y))
                n01 = np.sum((pred_i != y) & (pred_j == y))
                n00 = np.sum((pred_i != y) & (pred_j != y))
                
                # Q统计量
                if (n11 * n00 + n10 * n01) != 0:
                    q = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
                    q_statistics.append(q)
        
        if q_statistics:
            diversity_metrics['overall_diversity']['q_statistic'] = {
                'mean': np.mean(q_statistics),
                'std': np.std(q_statistics)
            }
        
        # 不一致率
        ensemble_pred = result.predictions
        individual_correct = [pred == y for pred in individual_predictions]
        ensemble_correct = ensemble_pred == y
        
        # 计算个体模型正确但集成错误的比例
        individual_correct_ensemble_wrong = []
        for correct in individual_correct:
            rate = np.mean(correct & ~ensemble_correct) / np.mean(correct) if np.mean(correct) > 0 else 0
            individual_correct_ensemble_wrong.append(rate)
        
        diversity_metrics['disagreement_rate'] = {
            'individual_vs_ensemble': individual_correct_ensemble_wrong,
            'mean_disagreement': np.mean(individual_correct_ensemble_wrong)
        }
    
    else:
        # 回归任务的多样性指标
        # 计算预测方差
        predictions_array = np.array(individual_predictions)
        prediction_variance = np.var(predictions_array, axis=0)
        
        diversity_metrics['overall_diversity']['prediction_variance'] = {
            'mean': np.mean(prediction_variance),
            'std': np.std(prediction_variance),
            'min': np.min(prediction_variance),
            'max': np.max(prediction_variance)
        }
        
        # 计算偏差-方差分解
        ensemble_pred = result.predictions
        bias_squared = np.mean((np.mean(predictions_array, axis=0) - y) ** 2)
        variance = np.mean(prediction_variance)
        
        diversity_metrics['bias_variance_decomposition'] = {
            'bias_squared': bias_squared,
            'variance': variance,
            'total_error': bias_squared + variance
        }
    
    return diversity_metrics


def save_ensemble_results(ensemble, results: Dict[str, Any], save_dir: str) -> str:
    """
    保存集成模型和结果
    
    Args:
        ensemble: 集成模型
        results: 结果数据
        save_dir: 保存目录
        
    Returns:
        保存路径
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_path = save_path / f"ensemble_model_{timestamp}.pkl"
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        logger.info(f"Ensemble model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save ensemble model: {e}")
    
    # 保存结果
    results_path = save_path / f"ensemble_results_{timestamp}.json"
    try:
        # 转换numpy数组为列表
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"Results saved to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    return str(save_path)


def load_ensemble_results(load_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    加载集成模型和结果
    
    Args:
        load_path: 加载路径（目录或具体文件）
        
    Returns:
        (集成模型, 结果数据)
    """
    load_path = Path(load_path)
    
    if load_path.is_dir():
        # 如果是目录，找最新的文件
        model_files = list(load_path.glob("ensemble_model_*.pkl"))
        result_files = list(load_path.glob("ensemble_results_*.json"))
        
        if not model_files:
            raise FileNotFoundError(f"No ensemble model files found in {load_path}")
        
        model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        result_file = max(result_files, key=lambda x: x.stat().st_mtime) if result_files else None
    
    else:
        # 具体文件路径
        model_file = load_path
        result_file = None
    
    # 加载模型
    try:
        with open(model_file, 'rb') as f:
            ensemble = pickle.load(f)
        logger.info(f"Ensemble model loaded from {model_file}")
    except Exception as e:
        logger.error(f"Failed to load ensemble model: {e}")
        raise
    
    # 加载结果
    results = {}
    if result_file and result_file.exists():
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Results loaded from {result_file}")
        except Exception as e:
            logger.warning(f"Failed to load results: {e}")
    
    return ensemble, results


def generate_ensemble_report(ensemble, X_test: np.ndarray, y_test: np.ndarray,
                           save_path: Optional[str] = None) -> str:
    """
    生成集成模型报告
    
    Args:
        ensemble: 集成模型
        X_test: 测试特征
        y_test: 测试标签
        save_path: 保存路径
        
    Returns:
        报告内容
    """
    # 评估性能
    performance = evaluate_ensemble_performance(ensemble, X_test, y_test)
    
    # 分析多样性
    try:
        diversity = analyze_model_diversity(ensemble, X_test, y_test)
    except Exception as e:
        logger.warning(f"Diversity analysis failed: {e}")
        diversity = {}
    
    # 生成报告
    report_lines = [
        "# Ensemble Model Report",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model Information",
        f"- Task Type: {performance.get('task_type', 'Unknown')}",
        f"- Number of Base Models: {getattr(ensemble, 'n_models_', 'Unknown')}",
        f"- Test Samples: {performance.get('n_samples', 'Unknown')}",
        "",
        "## Performance Metrics"
    ]
    
    if performance['task_type'] == 'classification':
        report_lines.extend([
            f"- Accuracy: {performance.get('accuracy', 0):.4f}",
            f"- Precision: {performance.get('precision', 0):.4f}",
            f"- Recall: {performance.get('recall', 0):.4f}",
            f"- F1 Score: {performance.get('f1_score', 0):.4f}",
        ])
        
        if 'roc_auc' in performance and performance['roc_auc']:
            report_lines.append(f"- ROC AUC: {performance['roc_auc']:.4f}")
    
    else:
        report_lines.extend([
            f"- MSE: {performance.get('mse', 0):.4f}",
            f"- RMSE: {performance.get('rmse', 0):.4f}",
            f"- MAE: {performance.get('mae', 0):.4f}",
            f"- R² Score: {performance.get('r2_score', 0):.4f}",
        ])
    
    # 个体模型性能
    if 'individual_performance' in performance:
        report_lines.extend([
            "",
            "## Individual Model Performance"
        ])
        
        for i, perf in enumerate(performance['individual_performance']):
            if performance['task_type'] == 'classification':
                report_lines.append(f"- Model {i}: Accuracy = {perf['accuracy']:.4f}")
            else:
                report_lines.append(f"- Model {i}: MSE = {perf['mse']:.4f}")
    
    # 多样性分析
    if diversity:
        report_lines.extend([
            "",
            "## Diversity Analysis",
            f"- Number of Models: {diversity.get('n_models', 'Unknown')}"
        ])
        
        if 'pairwise_diversity' in diversity:
            pd = diversity['pairwise_diversity']
            report_lines.extend([
                f"- Mean Pairwise Diversity: {pd.get('mean', 0):.4f}",
                f"- Diversity Standard Deviation: {pd.get('std', 0):.4f}",
                f"- Min Diversity: {pd.get('min', 0):.4f}",
                f"- Max Diversity: {pd.get('max', 0):.4f}"
            ])
        
        if 'overall_diversity' in diversity and 'q_statistic' in diversity['overall_diversity']:
            q_stat = diversity['overall_diversity']['q_statistic']
            report_lines.extend([
                f"- Mean Q Statistic: {q_stat.get('mean', 0):.4f}",
                f"- Q Statistic Std: {q_stat.get('std', 0):.4f}"
            ])
    
    # 预测时间
    if 'prediction_time' in performance:
        report_lines.extend([
            "",
            "## Performance",
            f"- Average Prediction Time per Sample: {performance['prediction_time']:.6f} seconds"
        ])
    
    report_content = "\n".join(report_lines)
    
    # 保存报告
    if save_path:
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Report saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    return report_content


def suggest_ensemble_strategy(X: np.ndarray, y: np.ndarray, 
                            task_type: str = 'auto',
                            dataset_size: str = 'auto') -> Dict[str, Any]:
    """
    建议集成策略
    
    Args:
        X: 特征数据
        y: 标签数据
        task_type: 任务类型 ('classification', 'regression', 'auto')
        dataset_size: 数据集大小 ('small', 'medium', 'large', 'auto')
        
    Returns:
        策略建议
    """
    n_samples, n_features = X.shape
    
    # 自动检测任务类型
    if task_type == 'auto':
        unique_values = len(np.unique(y))
        if unique_values <= 20 and unique_values < n_samples * 0.1:
            task_type = 'classification'
        else:
            task_type = 'regression'
    
    # 自动检测数据集大小
    if dataset_size == 'auto':
        if n_samples < 1000:
            dataset_size = 'small'
        elif n_samples < 10000:
            dataset_size = 'medium'
        else:
            dataset_size = 'large'
    
    suggestions = {
        'task_type': task_type,
        'dataset_size': dataset_size,
        'n_samples': n_samples,
        'n_features': n_features,
        'recommended_strategies': [],
        'base_models': [],
        'ensemble_methods': [],
        'considerations': []
    }
    
    # 基础模型建议
    if task_type == 'classification':
        if dataset_size == 'small':
            suggestions['base_models'] = [
                'LogisticRegression', 'SVM', 'KNeighborsClassifier', 
                'DecisionTreeClassifier', 'GaussianNB'
            ]
        elif dataset_size == 'medium':
            suggestions['base_models'] = [
                'RandomForestClassifier', 'GradientBoostingClassifier',
                'SVM', 'LogisticRegression', 'ExtraTreesClassifier'
            ]
        else:
            suggestions['base_models'] = [
                'RandomForestClassifier', 'GradientBoostingClassifier',
                'XGBClassifier', 'LGBMClassifier', 'ExtraTreesClassifier'
            ]
    
    else:  # regression
        if dataset_size == 'small':
            suggestions['base_models'] = [
                'LinearRegression', 'Ridge', 'Lasso', 
                'DecisionTreeRegressor', 'KNeighborsRegressor'
            ]
        elif dataset_size == 'medium':
            suggestions['base_models'] = [
                'RandomForestRegressor', 'GradientBoostingRegressor',
                'SVR', 'Ridge', 'ExtraTreesRegressor'
            ]
        else:
            suggestions['base_models'] = [
                'RandomForestRegressor', 'GradientBoostingRegressor',
                'XGBRegressor', 'LGBMRegressor', 'ExtraTreesRegressor'
            ]
    
    # 集成方法建议
    if dataset_size == 'small':
        suggestions['ensemble_methods'] = ['Voting', 'Averaging']
        suggestions['considerations'].append("Small dataset: Use simple ensemble methods to avoid overfitting")
    
    elif dataset_size == 'medium':
        suggestions['ensemble_methods'] = ['Voting', 'Averaging', 'Stacking']
        suggestions['considerations'].append("Medium dataset: Stacking can be effective with proper cross-validation")
    
    else:
        suggestions['ensemble_methods'] = ['Voting', 'Averaging', 'Stacking', 'Blending']
        suggestions['considerations'].append("Large dataset: All ensemble methods are viable")
    
    # 特征维度考虑
    if n_features > n_samples:
        suggestions['considerations'].append("High-dimensional data: Consider feature selection and regularization")
        suggestions['base_models'] = [model for model in suggestions['base_models'] 
                                    if 'Ridge' in model or 'Lasso' in model or 'SVM' in model]
    
    # 具体策略建议
    if task_type == 'classification':
        if dataset_size == 'small':
            suggestions['recommended_strategies'] = [
                {
                    'name': 'Simple Voting Ensemble',
                    'method': 'VotingEnsemble',
                    'parameters': {'voting': 'soft'},
                    'reason': 'Robust for small datasets, reduces overfitting'
                }
            ]
        else:
            suggestions['recommended_strategies'] = [
                {
                    'name': 'Stacking Ensemble',
                    'method': 'StackingEnsemble',
                    'parameters': {'cv': 5},
                    'reason': 'Can capture complex patterns with sufficient data'
                },
                {
                    'name': 'Weighted Voting',
                    'method': 'WeightedVotingEnsemble',
                    'parameters': {'optimize_weights': True},
                    'reason': 'Balances simplicity and performance'
                }
            ]
    
    else:  # regression
        suggestions['recommended_strategies'] = [
            {
                'name': 'Weighted Averaging',
                'method': 'AveragingEnsemble',
                'parameters': {'averaging_method': 'weighted'},
                'reason': 'Effective for regression tasks'
            }
        ]
        
        if dataset_size != 'small':
            suggestions['recommended_strategies'].append({
                'name': 'Stacking Ensemble',
                'method': 'StackingEnsemble',
                'parameters': {'cv': 5},
                'reason': 'Can learn complex combinations of predictions'
            })
    
    # 额外考虑
    if n_features > 100:
        suggestions['considerations'].append("High-dimensional features: Consider dimensionality reduction")
    
    if len(np.unique(y)) > 10 and task_type == 'classification':
        suggestions['considerations'].append("Multi-class problem: Ensure base models handle multi-class well")
    
    return suggestions