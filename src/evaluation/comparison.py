"""
模型比较分析模块

提供多个模型之间的详细比较分析功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from .base import EvaluationResult

logger = logging.getLogger(__name__)


class ModelComparator:
    """模型比较器"""
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 correction_method: str = 'bonferroni'):
        """
        初始化模型比较器
        
        Args:
            significance_level: 显著性水平
            correction_method: 多重比较校正方法
        """
        self.significance_level = significance_level
        self.correction_method = correction_method
        
        logger.info("Initialized ModelComparator")
    
    def compare_models(self, 
                      results: List[EvaluationResult],
                      metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        比较多个模型
        
        Args:
            results: 评估结果列表
            metrics: 要比较的指标
            
        Returns:
            比较结果
        """
        if len(results) < 2:
            raise ValueError("At least 2 models are required for comparison")
        
        # 确定要比较的指标
        if metrics is None:
            all_metrics = set()
            for result in results:
                all_metrics.update(result.test_scores.keys())
            metrics = list(all_metrics)
        
        comparison_results = {
            'models': [result.model_name for result in results],
            'metrics': metrics,
            'statistical_tests': {},
            'rankings': {},
            'summary': {}
        }
        
        # 对每个指标进行比较
        for metric in metrics:
            comparison_results['statistical_tests'][metric] = self._compare_metric(
                results, metric
            )
            comparison_results['rankings'][metric] = self._rank_models(
                results, metric
            )
        
        # 生成总体排名
        comparison_results['overall_ranking'] = self._calculate_overall_ranking(
            results, metrics
        )
        
        # 生成比较摘要
        comparison_results['summary'] = self._generate_comparison_summary(
            comparison_results
        )
        
        return comparison_results
    
    def _compare_metric(self, 
                       results: List[EvaluationResult], 
                       metric: str) -> Dict[str, Any]:
        """
        比较特定指标
        
        Args:
            results: 评估结果列表
            metric: 指标名称
            
        Returns:
            统计测试结果
        """
        # 收集交叉验证分数
        cv_scores = []
        model_names = []
        
        for result in results:
            if result.cv_scores and metric in result.cv_scores:
                cv_scores.append(result.cv_scores[metric])
                model_names.append(result.model_name)
        
        if len(cv_scores) < 2:
            logger.warning(f"Insufficient CV data for metric '{metric}'")
            return {'error': 'Insufficient data for statistical testing'}
        
        test_results = {
            'metric': metric,
            'models': model_names,
            'cv_scores': cv_scores,
            'descriptive_stats': {},
            'statistical_tests': {}
        }
        
        # 描述性统计
        for i, (name, scores) in enumerate(zip(model_names, cv_scores)):
            test_results['descriptive_stats'][name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'q25': np.percentile(scores, 25),
                'q75': np.percentile(scores, 75)
            }
        
        # 统计测试
        if len(cv_scores) == 2:
            # 两个模型：配对t检验
            test_results['statistical_tests'] = self._paired_t_test(
                cv_scores[0], cv_scores[1], model_names[0], model_names[1]
            )
        else:
            # 多个模型：方差分析 + 事后检验
            test_results['statistical_tests'] = self._anova_with_posthoc(
                cv_scores, model_names
            )
        
        return test_results
    
    def _paired_t_test(self, 
                      scores1: List[float], 
                      scores2: List[float],
                      name1: str, 
                      name2: str) -> Dict[str, Any]:
        """
        执行配对t检验
        
        Args:
            scores1: 模型1的分数
            scores2: 模型2的分数
            name1: 模型1名称
            name2: 模型2名称
            
        Returns:
            t检验结果
        """
        try:
            # 确保长度相同
            min_len = min(len(scores1), len(scores2))
            scores1 = scores1[:min_len]
            scores2 = scores2[:min_len]
            
            # 配对t检验
            statistic, p_value = stats.ttest_rel(scores1, scores2)
            
            # 效应大小 (Cohen's d)
            diff = np.array(scores1) - np.array(scores2)
            cohens_d = np.mean(diff) / np.std(diff, ddof=1)
            
            return {
                'test_type': 'paired_t_test',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'cohens_d': cohens_d,
                'effect_size': self._interpret_effect_size(abs(cohens_d)),
                'winner': name1 if np.mean(scores1) > np.mean(scores2) else name2,
                'mean_difference': np.mean(scores1) - np.mean(scores2)
            }
        except Exception as e:
            logger.error(f"Failed to perform paired t-test: {e}")
            return {'error': str(e)}
    
    def _anova_with_posthoc(self, 
                           cv_scores: List[List[float]], 
                           model_names: List[str]) -> Dict[str, Any]:
        """
        执行方差分析和事后检验
        
        Args:
            cv_scores: 各模型的交叉验证分数
            model_names: 模型名称列表
            
        Returns:
            ANOVA和事后检验结果
        """
        try:
            # 单因素方差分析
            f_statistic, p_value = stats.f_oneway(*cv_scores)
            
            results = {
                'test_type': 'one_way_anova',
                'f_statistic': f_statistic,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'posthoc_tests': {}
            }
            
            # 如果ANOVA显著，进行事后检验
            if p_value < self.significance_level:
                results['posthoc_tests'] = self._tukey_hsd_posthoc(
                    cv_scores, model_names
                )
            
            return results
        except Exception as e:
            logger.error(f"Failed to perform ANOVA: {e}")
            return {'error': str(e)}
    
    def _tukey_hsd_posthoc(self, 
                          cv_scores: List[List[float]], 
                          model_names: List[str]) -> Dict[str, Any]:
        """
        执行Tukey HSD事后检验
        
        Args:
            cv_scores: 各模型的交叉验证分数
            model_names: 模型名称列表
            
        Returns:
            事后检验结果
        """
        from scipy.stats import tukey_hsd
        
        try:
            # 执行Tukey HSD检验
            res = tukey_hsd(*cv_scores)
            
            posthoc_results = {
                'test_type': 'tukey_hsd',
                'pairwise_comparisons': {}
            }
            
            # 解析成对比较结果
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    comparison_key = f"{model_names[i]}_vs_{model_names[j]}"
                    
                    # 计算均值差异
                    mean_diff = np.mean(cv_scores[i]) - np.mean(cv_scores[j])
                    
                    # 获取p值（如果可用）
                    try:
                        p_val = res.pvalue[i, j] if hasattr(res, 'pvalue') else None
                    except:
                        p_val = None
                    
                    posthoc_results['pairwise_comparisons'][comparison_key] = {
                        'mean_difference': mean_diff,
                        'p_value': p_val,
                        'significant': p_val < self.significance_level if p_val else None,
                        'winner': model_names[i] if mean_diff > 0 else model_names[j]
                    }
            
            return posthoc_results
        except Exception as e:
            logger.warning(f"Tukey HSD failed, using pairwise t-tests: {e}")
            return self._pairwise_t_tests(cv_scores, model_names)
    
    def _pairwise_t_tests(self, 
                         cv_scores: List[List[float]], 
                         model_names: List[str]) -> Dict[str, Any]:
        """
        执行成对t检验（作为Tukey HSD的备选）
        
        Args:
            cv_scores: 各模型的交叉验证分数
            model_names: 模型名称列表
            
        Returns:
            成对t检验结果
        """
        posthoc_results = {
            'test_type': 'pairwise_t_tests',
            'pairwise_comparisons': {}
        }
        
        # 计算所有成对比较
        p_values = []
        comparisons = []
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                comparison_key = f"{model_names[i]}_vs_{model_names[j]}"
                comparisons.append(comparison_key)
                
                # 独立t检验
                statistic, p_value = stats.ttest_ind(cv_scores[i], cv_scores[j])
                p_values.append(p_value)
                
                mean_diff = np.mean(cv_scores[i]) - np.mean(cv_scores[j])
                
                posthoc_results['pairwise_comparisons'][comparison_key] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'mean_difference': mean_diff,
                    'winner': model_names[i] if mean_diff > 0 else model_names[j]
                }
        
        # 多重比较校正
        if self.correction_method == 'bonferroni':
            corrected_alpha = self.significance_level / len(p_values)
            for comparison_key, p_val in zip(comparisons, p_values):
                posthoc_results['pairwise_comparisons'][comparison_key]['significant'] = (
                    p_val < corrected_alpha
                )
        
        return posthoc_results
    
    def _rank_models(self, 
                    results: List[EvaluationResult], 
                    metric: str) -> List[Dict[str, Any]]:
        """
        对模型进行排名
        
        Args:
            results: 评估结果列表
            metric: 指标名称
            
        Returns:
            排名列表
        """
        model_scores = []
        
        for result in results:
            score = result.test_scores.get(metric)
            if score is not None:
                model_scores.append({
                    'model': result.model_name,
                    'score': score,
                    'cv_mean': np.mean(result.cv_scores[metric]) if result.cv_scores and metric in result.cv_scores else None,
                    'cv_std': np.std(result.cv_scores[metric]) if result.cv_scores and metric in result.cv_scores else None
                })
        
        # 根据测试分数排序（假设越高越好，对于损失函数需要特殊处理）
        is_loss_metric = metric.lower() in ['mse', 'mae', 'rmse', 'log_loss', 'loss']
        model_scores.sort(key=lambda x: x['score'], reverse=not is_loss_metric)
        
        # 添加排名
        for i, model_score in enumerate(model_scores):
            model_score['rank'] = i + 1
        
        return model_scores
    
    def _calculate_overall_ranking(self, 
                                 results: List[EvaluationResult], 
                                 metrics: List[str]) -> List[Dict[str, Any]]:
        """
        计算总体排名
        
        Args:
            results: 评估结果列表
            metrics: 指标列表
            
        Returns:
            总体排名
        """
        model_rankings = {}
        
        # 初始化
        for result in results:
            model_rankings[result.model_name] = {
                'model': result.model_name,
                'ranks': {},
                'average_rank': 0,
                'rank_std': 0
            }
        
        # 计算每个指标的排名
        for metric in metrics:
            rankings = self._rank_models(results, metric)
            for ranking in rankings:
                model_name = ranking['model']
                if model_name in model_rankings:
                    model_rankings[model_name]['ranks'][metric] = ranking['rank']
        
        # 计算平均排名
        for model_name, data in model_rankings.items():
            ranks = list(data['ranks'].values())
            if ranks:
                data['average_rank'] = np.mean(ranks)
                data['rank_std'] = np.std(ranks)
        
        # 按平均排名排序
        overall_ranking = list(model_rankings.values())
        overall_ranking.sort(key=lambda x: x['average_rank'])
        
        # 添加总体排名
        for i, model_data in enumerate(overall_ranking):
            model_data['overall_rank'] = i + 1
        
        return overall_ranking
    
    def _generate_comparison_summary(self, 
                                   comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成比较摘要
        
        Args:
            comparison_results: 比较结果
            
        Returns:
            比较摘要
        """
        summary = {
            'best_models': {},
            'significant_differences': {},
            'recommendations': []
        }
        
        # 找出每个指标的最佳模型
        for metric in comparison_results['metrics']:
            rankings = comparison_results['rankings'].get(metric, [])
            if rankings:
                best_model = rankings[0]
                summary['best_models'][metric] = {
                    'model': best_model['model'],
                    'score': best_model['score'],
                    'rank': best_model['rank']
                }
        
        # 统计显著差异
        for metric, test_results in comparison_results['statistical_tests'].items():
            if 'statistical_tests' in test_results:
                stat_tests = test_results['statistical_tests']
                
                if stat_tests.get('significant', False):
                    summary['significant_differences'][metric] = True
                    
                    # 添加具体的显著差异信息
                    if 'posthoc_tests' in stat_tests:
                        posthoc = stat_tests['posthoc_tests']
                        significant_pairs = []
                        
                        for comparison, result in posthoc.get('pairwise_comparisons', {}).items():
                            if result.get('significant', False):
                                significant_pairs.append({
                                    'comparison': comparison,
                                    'winner': result.get('winner'),
                                    'p_value': result.get('p_value')
                                })
                        
                        summary['significant_differences'][f'{metric}_details'] = significant_pairs
                else:
                    summary['significant_differences'][metric] = False
        
        # 生成推荐
        overall_ranking = comparison_results.get('overall_ranking', [])
        if overall_ranking:
            best_overall = overall_ranking[0]
            summary['recommendations'].append(
                f"Overall best model: {best_overall['model']} "
                f"(average rank: {best_overall['average_rank']:.2f})"
            )
            
            # 检查是否有明显的赢家
            if len(overall_ranking) > 1:
                rank_diff = overall_ranking[1]['average_rank'] - overall_ranking[0]['average_rank']
                if rank_diff > 1.0:
                    summary['recommendations'].append(
                        f"{best_overall['model']} shows clear superiority with "
                        f"average rank difference of {rank_diff:.2f}"
                    )
        
        return summary
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """
        解释效应大小
        
        Args:
            cohens_d: Cohen's d值
            
        Returns:
            效应大小解释
        """
        if cohens_d < 0.2:
            return 'negligible'
        elif cohens_d < 0.5:
            return 'small'
        elif cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def generate_comparison_report(self, 
                                 comparison_results: Dict[str, Any]) -> str:
        """
        生成比较报告
        
        Args:
            comparison_results: 比较结果
            
        Returns:
            比较报告文本
        """
        report = []
        report.append("Model Comparison Report")
        report.append("=" * 50)
        report.append("")
        
        # 基本信息
        models = comparison_results['models']
        metrics = comparison_results['metrics']
        
        report.append(f"Models compared: {', '.join(models)}")
        report.append(f"Metrics evaluated: {', '.join(metrics)}")
        report.append(f"Significance level: {self.significance_level}")
        report.append("")
        
        # 总体排名
        overall_ranking = comparison_results.get('overall_ranking', [])
        if overall_ranking:
            report.append("Overall Ranking:")
            report.append("-" * 20)
            for i, model_data in enumerate(overall_ranking):
                report.append(
                    f"{i+1}. {model_data['model']} "
                    f"(avg rank: {model_data['average_rank']:.2f} ± {model_data['rank_std']:.2f})"
                )
            report.append("")
        
        # 每个指标的详细结果
        for metric in metrics:
            report.append(f"Metric: {metric}")
            report.append("-" * 30)
            
            # 排名
            rankings = comparison_results['rankings'].get(metric, [])
            if rankings:
                report.append("Rankings:")
                for ranking in rankings:
                    cv_info = ""
                    if ranking['cv_mean'] is not None:
                        cv_info = f" (CV: {ranking['cv_mean']:.4f} ± {ranking['cv_std']:.4f})"
                    report.append(
                        f"  {ranking['rank']}. {ranking['model']}: {ranking['score']:.4f}{cv_info}"
                    )
            
            # 统计测试结果
            test_results = comparison_results['statistical_tests'].get(metric, {})
            if 'statistical_tests' in test_results:
                stat_tests = test_results['statistical_tests']
                report.append(f"\nStatistical Test: {stat_tests.get('test_type', 'Unknown')}")
                
                if stat_tests.get('significant', False):
                    report.append("Result: Significant differences detected")
                    
                    # 详细的成对比较
                    if 'posthoc_tests' in stat_tests:
                        posthoc = stat_tests['posthoc_tests']
                        report.append("Pairwise comparisons:")
                        
                        for comparison, result in posthoc.get('pairwise_comparisons', {}).items():
                            significance = "significant" if result.get('significant', False) else "not significant"
                            p_val = result.get('p_value', 'N/A')
                            winner = result.get('winner', 'N/A')
                            
                            report.append(f"  {comparison}: {significance} (p={p_val}, winner: {winner})")
                else:
                    report.append("Result: No significant differences")
            
            report.append("")
        
        # 摘要和推荐
        summary = comparison_results.get('summary', {})
        if summary.get('recommendations'):
            report.append("Recommendations:")
            report.append("-" * 20)
            for rec in summary['recommendations']:
                report.append(f"• {rec}")
            report.append("")
        
        return "\n".join(report)
    
    def plot_comparison_results(self, 
                              comparison_results: Dict[str, Any],
                              save_path: Optional[str] = None) -> None:
        """
        绘制比较结果
        
        Args:
            comparison_results: 比较结果
            save_path: 保存路径
        """
        metrics = comparison_results['metrics']
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            logger.warning("No metrics to plot")
            return
        
        # 创建子图
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # 获取排名数据
            rankings = comparison_results['rankings'].get(metric, [])
            if rankings:
                models = [r['model'] for r in rankings]
                scores = [r['score'] for r in rankings]
                
                # 条形图
                bars = ax.bar(models, scores)
                
                # 标记最佳模型
                best_idx = 0
                bars[best_idx].set_color('gold')
                
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for j, (model, score) in enumerate(zip(models, scores)):
                    ax.text(j, score + 0.01 * ax.get_ylim()[1], f'{score:.3f}', 
                           ha='center', va='bottom')
        
        # 隐藏多余的子图
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()


def quick_model_comparison(results: List[EvaluationResult], 
                          metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    快速模型比较
    
    Args:
        results: 评估结果列表
        metrics: 要比较的指标
        
    Returns:
        比较结果DataFrame
    """
    comparator = ModelComparator()
    comparison_results = comparator.compare_models(results, metrics)
    
    # 转换为DataFrame
    data = []
    for metric in comparison_results['metrics']:
        rankings = comparison_results['rankings'].get(metric, [])
        for ranking in rankings:
            data.append({
                'Model': ranking['model'],
                'Metric': metric,
                'Score': ranking['score'],
                'Rank': ranking['rank'],
                'CV_Mean': ranking.get('cv_mean'),
                'CV_Std': ranking.get('cv_std')
            })
    
    return pd.DataFrame(data)