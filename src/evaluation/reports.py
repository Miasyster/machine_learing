"""
评估报告生成模块

提供各种格式的评估报告生成功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
from jinja2 import Template
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, output_dir: str = "evaluation_reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_comprehensive_report(self,
                                    evaluation_results: List[Any],
                                    dataset_info: Dict[str, Any],
                                    experiment_config: Dict[str, Any],
                                    output_format: str = 'html') -> str:
        """
        生成综合评估报告
        
        Args:
            evaluation_results: 评估结果列表
            dataset_info: 数据集信息
            experiment_config: 实验配置
            output_format: 输出格式 ('html', 'pdf', 'markdown')
            
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备报告数据
        report_data = {
            'timestamp': timestamp,
            'dataset_info': dataset_info,
            'experiment_config': experiment_config,
            'evaluation_results': evaluation_results,
            'summary_table': self._create_summary_table(evaluation_results),
            'best_model': self._find_best_model(evaluation_results),
            'plots': self._generate_report_plots(evaluation_results)
        }
        
        if output_format == 'html':
            return self._generate_html_report(report_data, timestamp)
        elif output_format == 'pdf':
            return self._generate_pdf_report(report_data, timestamp)
        elif output_format == 'markdown':
            return self._generate_markdown_report(report_data, timestamp)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def generate_model_comparison_report(self,
                                       comparison_results: Dict[str, Any],
                                       output_format: str = 'html') -> str:
        """
        生成模型比较报告
        
        Args:
            comparison_results: 模型比较结果
            output_format: 输出格式
            
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备报告数据
        report_data = {
            'timestamp': timestamp,
            'comparison_results': comparison_results,
            'statistical_tests': comparison_results.get('statistical_tests', {}),
            'rankings': comparison_results.get('rankings', {}),
            'plots': self._generate_comparison_plots(comparison_results)
        }
        
        if output_format == 'html':
            return self._generate_comparison_html_report(report_data, timestamp)
        elif output_format == 'markdown':
            return self._generate_comparison_markdown_report(report_data, timestamp)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def generate_performance_dashboard(self,
                                     evaluation_results: List[Any],
                                     save_path: Optional[str] = None) -> str:
        """
        生成性能仪表板
        
        Args:
            evaluation_results: 评估结果列表
            save_path: 保存路径
            
        Returns:
            仪表板文件路径
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"performance_dashboard_{timestamp}.html"
        
        # 创建仪表板图表
        dashboard_plots = self._create_dashboard_plots(evaluation_results)
        
        # 生成HTML仪表板
        html_content = self._create_dashboard_html(dashboard_plots, evaluation_results)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Performance dashboard saved to {save_path}")
        return str(save_path)
    
    def _create_summary_table(self, evaluation_results: List[Any]) -> pd.DataFrame:
        """创建摘要表"""
        summary_data = []
        
        for result in evaluation_results:
            if hasattr(result, 'model_name') and hasattr(result, 'test_scores'):
                row = {'Model': result.model_name}
                
                # 添加测试分数
                for metric, score in result.test_scores.items():
                    row[f'Test_{metric}'] = f"{score:.4f}"
                
                # 添加交叉验证分数
                if hasattr(result, 'cv_scores') and result.cv_scores:
                    for metric, scores in result.cv_scores.items():
                        row[f'CV_{metric}'] = f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"
                
                # 添加训练时间
                if hasattr(result, 'training_time'):
                    row['Training_Time'] = f"{result.training_time:.2f}s"
                
                # 添加预测时间
                if hasattr(result, 'prediction_time'):
                    row['Prediction_Time'] = f"{result.prediction_time:.4f}s"
                
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def _find_best_model(self, evaluation_results: List[Any]) -> Dict[str, Any]:
        """找到最佳模型"""
        best_model = None
        best_score = -np.inf
        primary_metric = 'accuracy'  # 默认主要指标
        
        for result in evaluation_results:
            if hasattr(result, 'test_scores') and primary_metric in result.test_scores:
                score = result.test_scores[primary_metric]
                if score > best_score:
                    best_score = score
                    best_model = result
        
        if best_model is None:
            return {}
        
        return {
            'model_name': getattr(best_model, 'model_name', 'Unknown'),
            'best_score': best_score,
            'metric': primary_metric,
            'all_scores': getattr(best_model, 'test_scores', {})
        }
    
    def _generate_report_plots(self, evaluation_results: List[Any]) -> Dict[str, str]:
        """生成报告图表"""
        plots = {}
        
        # 性能比较图
        plots['performance_comparison'] = self._create_performance_comparison_plot(evaluation_results)
        
        # 指标雷达图
        plots['radar_chart'] = self._create_radar_chart(evaluation_results)
        
        # 训练时间比较
        plots['training_time'] = self._create_training_time_plot(evaluation_results)
        
        # 交叉验证分数分布
        plots['cv_distribution'] = self._create_cv_distribution_plot(evaluation_results)
        
        return plots
    
    def _create_performance_comparison_plot(self, evaluation_results: List[Any]) -> str:
        """创建性能比较图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = []
        metrics_data = {}
        
        for result in evaluation_results:
            if hasattr(result, 'model_name') and hasattr(result, 'test_scores'):
                models.append(result.model_name)
                for metric, score in result.test_scores.items():
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    metrics_data[metric].append(score)
        
        # 创建分组柱状图
        x = np.arange(len(models))
        width = 0.8 / len(metrics_data)
        
        for i, (metric, scores) in enumerate(metrics_data.items()):
            ax.bar(x + i * width, scores, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * (len(metrics_data) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 转换为base64字符串
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def _create_radar_chart(self, evaluation_results: List[Any]) -> str:
        """创建雷达图"""
        if not evaluation_results:
            return ""
        
        # 获取所有指标
        all_metrics = set()
        for result in evaluation_results:
            if hasattr(result, 'test_scores'):
                all_metrics.update(result.test_scores.keys())
        
        all_metrics = list(all_metrics)
        if len(all_metrics) < 3:
            return ""  # 雷达图至少需要3个指标
        
        # 准备数据
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(all_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for result in evaluation_results:
            if hasattr(result, 'model_name') and hasattr(result, 'test_scores'):
                values = []
                for metric in all_metrics:
                    values.append(result.test_scores.get(metric, 0))
                values += values[:1]  # 闭合图形
                
                ax.plot(angles, values, 'o-', linewidth=2, label=result.model_name)
                ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        # 转换为base64字符串
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def _create_training_time_plot(self, evaluation_results: List[Any]) -> str:
        """创建训练时间比较图"""
        models = []
        times = []
        
        for result in evaluation_results:
            if hasattr(result, 'model_name') and hasattr(result, 'training_time'):
                models.append(result.model_name)
                times.append(result.training_time)
        
        if not models:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(models, times, alpha=0.7, color='skyblue')
        ax.set_xlabel('Models')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Model Training Time Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 转换为base64字符串
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def _create_cv_distribution_plot(self, evaluation_results: List[Any]) -> str:
        """创建交叉验证分数分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # 获取所有指标
        all_metrics = set()
        for result in evaluation_results:
            if hasattr(result, 'cv_scores'):
                all_metrics.update(result.cv_scores.keys())
        
        all_metrics = list(all_metrics)[:4]  # 最多显示4个指标
        
        for i, metric in enumerate(all_metrics):
            ax = axes[i]
            
            for result in evaluation_results:
                if (hasattr(result, 'model_name') and 
                    hasattr(result, 'cv_scores') and 
                    metric in result.cv_scores):
                    
                    scores = result.cv_scores[metric]
                    ax.hist(scores, alpha=0.6, label=result.model_name, bins=10)
            
            ax.set_xlabel(f'{metric} Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Cross-Validation {metric} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏未使用的子图
        for i in range(len(all_metrics), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 转换为base64字符串
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def _generate_comparison_plots(self, comparison_results: Dict[str, Any]) -> Dict[str, str]:
        """生成比较图表"""
        plots = {}
        
        # 统计显著性热图
        if 'statistical_tests' in comparison_results:
            plots['statistical_heatmap'] = self._create_statistical_heatmap(
                comparison_results['statistical_tests']
            )
        
        # 模型排名图
        if 'rankings' in comparison_results:
            plots['ranking_plot'] = self._create_ranking_plot(
                comparison_results['rankings']
            )
        
        return plots
    
    def _create_statistical_heatmap(self, statistical_tests: Dict[str, Any]) -> str:
        """创建统计显著性热图"""
        if 'pairwise_tests' not in statistical_tests:
            return ""
        
        pairwise_tests = statistical_tests['pairwise_tests']
        models = list(set([test['model1'] for test in pairwise_tests] + 
                         [test['model2'] for test in pairwise_tests]))
        
        # 创建p值矩阵
        p_matrix = np.ones((len(models), len(models)))
        
        for test in pairwise_tests:
            i = models.index(test['model1'])
            j = models.index(test['model2'])
            p_matrix[i, j] = test['p_value']
            p_matrix[j, i] = test['p_value']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建热图
        im = ax.imshow(p_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # 设置标签
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels(models)
        
        # 添加数值标签
        for i in range(len(models)):
            for j in range(len(models)):
                text = ax.text(j, i, f'{p_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black")
        
        ax.set_title('Statistical Significance (p-values)')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        # 转换为base64字符串
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def _create_ranking_plot(self, rankings: Dict[str, Any]) -> str:
        """创建排名图"""
        if 'average_rank' not in rankings:
            return ""
        
        models = list(rankings['average_rank'].keys())
        ranks = list(rankings['average_rank'].values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(models, ranks, alpha=0.7, color='lightcoral')
        ax.set_xlabel('Models')
        ax.set_ylabel('Average Rank')
        ax.set_title('Model Average Ranking (Lower is Better)')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, rank in zip(bars, ranks):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rank:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 转换为base64字符串
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def _generate_html_report(self, report_data: Dict[str, Any], timestamp: str) -> str:
        """生成HTML报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 30px; }
                .plot { text-align: center; margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .best-model { background-color: #e8f5e8; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Evaluation Report</h1>
                <p>Generated on: {{ timestamp }}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Information</h2>
                <ul>
                    {% for key, value in dataset_info.items() %}
                    <li><strong>{{ key }}:</strong> {{ value }}</li>
                    {% endfor %}
                </ul>
            </div>
            
            <div class="section">
                <h2>Best Model</h2>
                <div class="best-model">
                    <p><strong>Model:</strong> {{ best_model.model_name }}</p>
                    <p><strong>Best Score:</strong> {{ "%.4f"|format(best_model.best_score) }} ({{ best_model.metric }})</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                {{ summary_table.to_html(classes='summary-table', escape=False)|safe }}
            </div>
            
            <div class="section">
                <h2>Performance Comparison</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{{ plots.performance_comparison }}" alt="Performance Comparison">
                </div>
            </div>
            
            {% if plots.radar_chart %}
            <div class="section">
                <h2>Performance Radar Chart</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{{ plots.radar_chart }}" alt="Radar Chart">
                </div>
            </div>
            {% endif %}
            
            {% if plots.training_time %}
            <div class="section">
                <h2>Training Time Comparison</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{{ plots.training_time }}" alt="Training Time">
                </div>
            </div>
            {% endif %}
            
            {% if plots.cv_distribution %}
            <div class="section">
                <h2>Cross-Validation Score Distribution</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{{ plots.cv_distribution }}" alt="CV Distribution">
                </div>
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(**report_data)
        
        output_path = self.output_dir / f"evaluation_report_{timestamp}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
        return str(output_path)
    
    def _generate_markdown_report(self, report_data: Dict[str, Any], timestamp: str) -> str:
        """生成Markdown报告"""
        md_content = f"""# Model Evaluation Report

Generated on: {timestamp}

## Dataset Information

"""
        
        for key, value in report_data['dataset_info'].items():
            md_content += f"- **{key}:** {value}\n"
        
        md_content += f"""
## Best Model

- **Model:** {report_data['best_model'].get('model_name', 'N/A')}
- **Best Score:** {report_data['best_model'].get('best_score', 0):.4f} ({report_data['best_model'].get('metric', 'N/A')})

## Performance Summary

{report_data['summary_table'].to_markdown(index=False)}

## Experiment Configuration

"""
        
        for key, value in report_data['experiment_config'].items():
            md_content += f"- **{key}:** {value}\n"
        
        output_path = self.output_dir / f"evaluation_report_{timestamp}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Markdown report saved to {output_path}")
        return str(output_path)
    
    def _create_dashboard_plots(self, evaluation_results: List[Any]) -> Dict[str, str]:
        """创建仪表板图表"""
        return self._generate_report_plots(evaluation_results)
    
    def _create_dashboard_html(self, plots: Dict[str, str], evaluation_results: List[Any]) -> str:
        """创建仪表板HTML"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .widget { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header { text-align: center; margin-bottom: 30px; background: white; padding: 20px; border-radius: 8px; }
                .plot img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Performance Dashboard</h1>
                <p>Real-time evaluation results</p>
            </div>
            
            <div class="dashboard">
                <div class="widget">
                    <h3>Performance Comparison</h3>
                    <div class="plot">
                        <img src="data:image/png;base64,{{ plots.performance_comparison }}" alt="Performance Comparison">
                    </div>
                </div>
                
                {% if plots.radar_chart %}
                <div class="widget">
                    <h3>Performance Radar</h3>
                    <div class="plot">
                        <img src="data:image/png;base64,{{ plots.radar_chart }}" alt="Radar Chart">
                    </div>
                </div>
                {% endif %}
                
                {% if plots.training_time %}
                <div class="widget">
                    <h3>Training Time</h3>
                    <div class="plot">
                        <img src="data:image/png;base64,{{ plots.training_time }}" alt="Training Time">
                    </div>
                </div>
                {% endif %}
                
                {% if plots.cv_distribution %}
                <div class="widget">
                    <h3>CV Score Distribution</h3>
                    <div class="plot">
                        <img src="data:image/png;base64,{{ plots.cv_distribution }}" alt="CV Distribution">
                    </div>
                </div>
                {% endif %}
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        return template.render(plots=plots, evaluation_results=evaluation_results)
    
    def _generate_comparison_html_report(self, report_data: Dict[str, Any], timestamp: str) -> str:
        """生成比较HTML报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 30px; }
                .plot { text-align: center; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Comparison Report</h1>
                <p>Generated on: {{ timestamp }}</p>
            </div>
            
            {% if plots.statistical_heatmap %}
            <div class="section">
                <h2>Statistical Significance</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{{ plots.statistical_heatmap }}" alt="Statistical Heatmap">
                </div>
            </div>
            {% endif %}
            
            {% if plots.ranking_plot %}
            <div class="section">
                <h2>Model Rankings</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{{ plots.ranking_plot }}" alt="Ranking Plot">
                </div>
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(**report_data)
        
        output_path = self.output_dir / f"comparison_report_{timestamp}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comparison report saved to {output_path}")
        return str(output_path)
    
    def _generate_comparison_markdown_report(self, report_data: Dict[str, Any], timestamp: str) -> str:
        """生成比较Markdown报告"""
        md_content = f"""# Model Comparison Report

Generated on: {timestamp}

## Statistical Tests Summary

"""
        
        if 'statistical_tests' in report_data['comparison_results']:
            tests = report_data['comparison_results']['statistical_tests']
            if 'overall_test' in tests:
                md_content += f"- **Overall Test:** {tests['overall_test']['test_name']}\n"
                md_content += f"- **P-value:** {tests['overall_test']['p_value']:.6f}\n"
                md_content += f"- **Significant:** {'Yes' if tests['overall_test']['significant'] else 'No'}\n\n"
        
        if 'rankings' in report_data['comparison_results']:
            md_content += "## Model Rankings\n\n"
            rankings = report_data['comparison_results']['rankings']
            if 'average_rank' in rankings:
                for model, rank in sorted(rankings['average_rank'].items(), key=lambda x: x[1]):
                    md_content += f"- **{model}:** {rank:.2f}\n"
        
        output_path = self.output_dir / f"comparison_report_{timestamp}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Comparison markdown report saved to {output_path}")
        return str(output_path)


def create_quick_report(evaluation_results: List[Any],
                       output_path: Optional[str] = None) -> str:
    """
    快速创建评估报告
    
    Args:
        evaluation_results: 评估结果列表
        output_path: 输出路径
        
    Returns:
        报告文件路径
    """
    generator = EvaluationReportGenerator()
    
    # 准备基本数据
    dataset_info = {'models_evaluated': len(evaluation_results)}
    experiment_config = {'quick_report': True}
    
    return generator.generate_comprehensive_report(
        evaluation_results=evaluation_results,
        dataset_info=dataset_info,
        experiment_config=experiment_config,
        output_format='html'
    )


def export_results_to_json(evaluation_results: List[Any],
                          output_path: str) -> None:
    """
    导出结果到JSON文件
    
    Args:
        evaluation_results: 评估结果列表
        output_path: 输出路径
    """
    results_data = []
    
    for result in evaluation_results:
        result_dict = {}
        
        # 提取基本信息
        if hasattr(result, 'model_name'):
            result_dict['model_name'] = result.model_name
        
        if hasattr(result, 'test_scores'):
            result_dict['test_scores'] = result.test_scores
        
        if hasattr(result, 'cv_scores'):
            result_dict['cv_scores'] = {
                metric: scores.tolist() if isinstance(scores, np.ndarray) else scores
                for metric, scores in result.cv_scores.items()
            }
        
        if hasattr(result, 'training_time'):
            result_dict['training_time'] = result.training_time
        
        if hasattr(result, 'prediction_time'):
            result_dict['prediction_time'] = result.prediction_time
        
        results_data.append(result_dict)
    
    # 保存到JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results exported to {output_path}")


def export_results_to_csv(evaluation_results: List[Any],
                         output_path: str) -> None:
    """
    导出结果到CSV文件
    
    Args:
        evaluation_results: 评估结果列表
        output_path: 输出路径
    """
    generator = EvaluationReportGenerator()
    summary_table = generator._create_summary_table(evaluation_results)
    summary_table.to_csv(output_path, index=False)
    
    logger.info(f"Results exported to {output_path}")