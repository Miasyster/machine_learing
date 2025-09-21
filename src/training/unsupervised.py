"""
无监督学习训练器

实现聚类、降维和异常检测任务的训练器
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Type
import logging
import time
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.cluster import MeanShift, AffinityPropagation, Birch, OPTICS
from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from .base import BaseTrainer, TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)


class UnsupervisedTrainer(BaseTrainer):
    """无监督学习训练器"""
    
    # 预定义的聚类算法
    CLUSTERING_ALGORITHMS = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'agglomerative': AgglomerativeClustering,
        'spectral': SpectralClustering,
        'meanshift': MeanShift,
        'affinity_propagation': AffinityPropagation,
        'birch': Birch,
        'optics': OPTICS
    }
    
    # 预定义的降维算法
    DIMENSIONALITY_REDUCTION_ALGORITHMS = {
        'pca': PCA,
        'ica': FastICA,
        'nmf': NMF,
        'factor_analysis': FactorAnalysis,
        'truncated_svd': TruncatedSVD,
        'tsne': TSNE,
        'isomap': Isomap,
        'lle': LocallyLinearEmbedding,
        'mds': MDS
    }
    
    # 预定义的异常检测算法
    ANOMALY_DETECTION_ALGORITHMS = {
        'isolation_forest': IsolationForest,
        'one_class_svm': OneClassSVM,
        'local_outlier_factor': LocalOutlierFactor,
        'elliptic_envelope': EllipticEnvelope
    }
    
    # 默认超参数网格
    DEFAULT_PARAM_GRIDS = {
        'kmeans': {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'init': ['k-means++', 'random'],
            'n_init': [10, 20],
            'max_iter': [300, 500]
        },
        'dbscan': {
            'eps': [0.1, 0.3, 0.5, 0.7, 1.0],
            'min_samples': [3, 5, 10, 15, 20],
            'metric': ['euclidean', 'manhattan', 'cosine']
        },
        'agglomerative': {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'linkage': ['ward', 'complete', 'average', 'single'],
            'affinity': ['euclidean', 'manhattan', 'cosine']
        },
        'pca': {
            'n_components': [2, 3, 5, 10, 20, 50],
            'whiten': [True, False]
        },
        'ica': {
            'n_components': [2, 3, 5, 10, 20],
            'algorithm': ['parallel', 'deflation'],
            'fun': ['logcosh', 'exp', 'cube']
        },
        'tsne': {
            'n_components': [2, 3],
            'perplexity': [5, 10, 30, 50],
            'learning_rate': [10, 50, 100, 200, 500],
            'n_iter': [1000, 2000, 3000]
        },
        'isolation_forest': {
            'n_estimators': [50, 100, 200],
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'max_features': [0.5, 0.7, 1.0]
        }
    }
    
    def __init__(self, 
                 task_type: str = 'clustering',
                 algorithms: Optional[Union[List[str], Dict[str, BaseEstimator]]] = None,
                 param_grids: Optional[Dict[str, Dict[str, List]]] = None,
                 config: Optional[TrainingConfig] = None,
                 **kwargs):
        """
        初始化无监督学习训练器
        
        Args:
            task_type: 任务类型 ('clustering', 'dimensionality_reduction', 'anomaly_detection')
            algorithms: 要使用的算法列表或字典
            param_grids: 超参数网格
            config: 训练配置
            **kwargs: 其他参数
        """
        super().__init__(config, **kwargs)
        
        self.task_type = task_type
        self.param_grids = param_grids or {}
        
        # 设置算法
        if algorithms is None:
            if task_type == 'clustering':
                self.algorithms = {name: cls() for name, cls in self.CLUSTERING_ALGORITHMS.items()}
            elif task_type == 'dimensionality_reduction':
                self.algorithms = {name: cls() for name, cls in self.DIMENSIONALITY_REDUCTION_ALGORITHMS.items()}
            elif task_type == 'anomaly_detection':
                self.algorithms = {name: cls() for name, cls in self.ANOMALY_DETECTION_ALGORITHMS.items()}
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        elif isinstance(algorithms, list):
            algorithm_dict = {}
            algorithm_dict.update(self.CLUSTERING_ALGORITHMS)
            algorithm_dict.update(self.DIMENSIONALITY_REDUCTION_ALGORITHMS)
            algorithm_dict.update(self.ANOMALY_DETECTION_ALGORITHMS)
            
            self.algorithms = {name: algorithm_dict[name]() for name in algorithms if name in algorithm_dict}
        else:
            self.algorithms = algorithms
        
        # 设置默认超参数网格
        for algorithm_name in self.algorithms.keys():
            if algorithm_name not in self.param_grids and algorithm_name in self.DEFAULT_PARAM_GRIDS:
                self.param_grids[algorithm_name] = self.DEFAULT_PARAM_GRIDS[algorithm_name]
        
        # 数据预处理
        self.scaler = StandardScaler()
        self.scale_data = kwargs.get('scale_data', True)
        
        if self.verbose:
            logger.info(f"Initialized UnsupervisedTrainer for {task_type} with algorithms: {list(self.algorithms.keys())}")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'UnsupervisedTrainer':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 真实标签（用于评估，可选）
            **kwargs: 其他参数
            
        Returns:
            训练后的训练器
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"Starting {self.task_type} training with {len(self.algorithms)} algorithms")
        
        # 数据预处理
        if self.scale_data:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        # 训练每个算法
        for algorithm_name, algorithm in self.algorithms.items():
            if self.verbose:
                logger.info(f"Training {algorithm_name}...")
            
            algorithm_start_time = time.time()
            
            try:
                # 超参数优化
                if self.config.hyperparameter_optimization and algorithm_name in self.param_grids:
                    best_algorithm = self._optimize_hyperparameters(
                        algorithm, X_scaled, y, algorithm_name
                    )
                else:
                    # 设置随机种子
                    if hasattr(algorithm, 'random_state'):
                        algorithm.set_params(random_state=self.config.random_state)
                    
                    # 训练算法
                    best_algorithm = algorithm.fit(X_scaled)
                
                # 获取结果
                if self.task_type == 'clustering':
                    labels = self._get_cluster_labels(best_algorithm, X_scaled)
                    scores = self._evaluate_clustering(X_scaled, labels, y)
                elif self.task_type == 'dimensionality_reduction':
                    transformed_data = self._get_transformed_data(best_algorithm, X_scaled)
                    scores = self._evaluate_dimensionality_reduction(X_scaled, transformed_data, best_algorithm)
                elif self.task_type == 'anomaly_detection':
                    anomaly_scores = self._get_anomaly_scores(best_algorithm, X_scaled)
                    scores = self._evaluate_anomaly_detection(X_scaled, anomaly_scores, y)
                else:
                    scores = {}
                
                # 创建训练结果
                algorithm_training_time = time.time() - algorithm_start_time
                result = TrainingResult(
                    model_name=algorithm_name,
                    task_type=self.task_type,
                    training_time=algorithm_training_time,
                    best_model=best_algorithm,
                    train_scores=scores,
                    best_params=best_algorithm.get_params() if hasattr(best_algorithm, 'get_params') else {},
                    config=self.config
                )
                
                # 添加任务特定的结果
                if self.task_type == 'clustering':
                    result.metadata['cluster_labels'] = labels
                    result.metadata['n_clusters'] = len(np.unique(labels)) if labels is not None else 0
                elif self.task_type == 'dimensionality_reduction':
                    result.metadata['transformed_data'] = transformed_data
                    result.metadata['explained_variance_ratio'] = getattr(best_algorithm, 'explained_variance_ratio_', None)
                elif self.task_type == 'anomaly_detection':
                    result.metadata['anomaly_scores'] = anomaly_scores
                    result.metadata['outliers'] = anomaly_scores < 0 if anomaly_scores is not None else None
                
                self.training_results_[algorithm_name] = result
                
                # 更新最佳模型
                primary_metric = self._get_primary_metric()
                current_score = scores.get(primary_metric, 0)
                
                if current_score > self.best_score_:
                    self.best_score_ = current_score
                    self.best_model_ = best_algorithm
                
                if self.verbose:
                    logger.info(f"Completed {algorithm_name} in {algorithm_training_time:.2f}s, "
                              f"score: {current_score:.4f}")
                
                # 调用回调
                self._call_callbacks('model_trained', 
                                   model_name=algorithm_name, 
                                   result=result)
                
            except Exception as e:
                logger.error(f"Failed to train {algorithm_name}: {e}")
                continue
        
        total_time = time.time() - start_time
        self.is_fitted_ = True
        
        if self.verbose:
            logger.info(f"Training completed in {total_time:.2f}s. "
                      f"Best algorithm: {self._get_best_algorithm_name()}")
        
        # 调用回调
        self._call_callbacks('training_completed', 
                           total_time=total_time,
                           best_score=self.best_score_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用最佳算法进行预测
        
        Args:
            X: 预测特征
            
        Returns:
            预测结果
        """
        if not self.is_fitted_:
            raise ValueError("Trainer must be fitted before prediction")
        
        if self.best_model_ is None:
            raise ValueError("No best model available")
        
        # 数据预处理
        if self.scale_data:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        if self.task_type == 'clustering':
            return self._get_cluster_labels(self.best_model_, X_scaled)
        elif self.task_type == 'dimensionality_reduction':
            return self._get_transformed_data(self.best_model_, X_scaled)
        elif self.task_type == 'anomaly_detection':
            return self._get_anomaly_scores(self.best_model_, X_scaled)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        数据变换（主要用于降维）
        
        Args:
            X: 输入特征
            
        Returns:
            变换后的数据
        """
        if self.task_type != 'dimensionality_reduction':
            raise ValueError("Transform is only available for dimensionality reduction")
        
        return self.predict(X)
    
    def _optimize_hyperparameters(self, 
                                algorithm: BaseEstimator, 
                                X: np.ndarray, 
                                y: Optional[np.ndarray],
                                algorithm_name: str) -> BaseEstimator:
        """优化超参数"""
        param_grid = self.param_grids[algorithm_name]
        
        best_score = -np.inf
        best_params = None
        best_algorithm = None
        
        # 简单的网格搜索（因为无监督学习的评估比较复杂）
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            try:
                # 创建算法实例
                algorithm_copy = algorithm.__class__(**params)
                if hasattr(algorithm_copy, 'random_state'):
                    algorithm_copy.set_params(random_state=self.config.random_state)
                
                # 训练算法
                algorithm_copy.fit(X)
                
                # 评估算法
                if self.task_type == 'clustering':
                    labels = self._get_cluster_labels(algorithm_copy, X)
                    scores = self._evaluate_clustering(X, labels, y)
                elif self.task_type == 'dimensionality_reduction':
                    transformed_data = self._get_transformed_data(algorithm_copy, X)
                    scores = self._evaluate_dimensionality_reduction(X, transformed_data, algorithm_copy)
                elif self.task_type == 'anomaly_detection':
                    anomaly_scores = self._get_anomaly_scores(algorithm_copy, X)
                    scores = self._evaluate_anomaly_detection(X, anomaly_scores, y)
                else:
                    scores = {}
                
                # 获取主要指标
                primary_metric = self._get_primary_metric()
                current_score = scores.get(primary_metric, -np.inf)
                
                if current_score > best_score:
                    best_score = current_score
                    best_params = params
                    best_algorithm = algorithm_copy
                
            except Exception as e:
                logger.warning(f"Failed to evaluate params {params}: {e}")
                continue
        
        if best_algorithm is None:
            logger.warning(f"Hyperparameter optimization failed for {algorithm_name}, using default")
            return algorithm
        
        if self.verbose:
            logger.info(f"Best parameters for {algorithm_name}: {best_params}")
            logger.info(f"Best score for {algorithm_name}: {best_score:.4f}")
        
        return best_algorithm
    
    def _get_cluster_labels(self, algorithm: BaseEstimator, X: np.ndarray) -> Optional[np.ndarray]:
        """获取聚类标签"""
        if hasattr(algorithm, 'labels_'):
            return algorithm.labels_
        elif hasattr(algorithm, 'predict'):
            return algorithm.predict(X)
        else:
            return None
    
    def _get_transformed_data(self, algorithm: BaseEstimator, X: np.ndarray) -> Optional[np.ndarray]:
        """获取变换后的数据"""
        if hasattr(algorithm, 'transform'):
            return algorithm.transform(X)
        elif hasattr(algorithm, 'fit_transform'):
            return algorithm.fit_transform(X)
        else:
            return None
    
    def _get_anomaly_scores(self, algorithm: BaseEstimator, X: np.ndarray) -> Optional[np.ndarray]:
        """获取异常分数"""
        if hasattr(algorithm, 'decision_function'):
            return algorithm.decision_function(X)
        elif hasattr(algorithm, 'score_samples'):
            return algorithm.score_samples(X)
        elif hasattr(algorithm, 'negative_outlier_factor_'):
            return algorithm.negative_outlier_factor_
        else:
            return None
    
    def _evaluate_clustering(self, X: np.ndarray, labels: Optional[np.ndarray], y: Optional[np.ndarray]) -> Dict[str, float]:
        """评估聚类结果"""
        scores = {}
        
        if labels is None:
            return scores
        
        try:
            # 内部评估指标
            scores['silhouette_score'] = silhouette_score(X, labels)
            scores['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            scores['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            
            # 外部评估指标（如果有真实标签）
            if y is not None:
                scores['adjusted_rand_score'] = adjusted_rand_score(y, labels)
                scores['normalized_mutual_info_score'] = normalized_mutual_info_score(y, labels)
            
        except Exception as e:
            logger.warning(f"Failed to compute clustering scores: {e}")
        
        return scores
    
    def _evaluate_dimensionality_reduction(self, X: np.ndarray, transformed_data: Optional[np.ndarray], algorithm: BaseEstimator) -> Dict[str, float]:
        """评估降维结果"""
        scores = {}
        
        if transformed_data is None:
            return scores
        
        try:
            # 解释方差比例
            if hasattr(algorithm, 'explained_variance_ratio_'):
                scores['explained_variance_ratio'] = np.sum(algorithm.explained_variance_ratio_)
            
            # 重构误差（如果可能）
            if hasattr(algorithm, 'inverse_transform'):
                try:
                    reconstructed = algorithm.inverse_transform(transformed_data)
                    scores['reconstruction_error'] = np.mean((X - reconstructed) ** 2)
                except:
                    pass
            
            # 降维比例
            scores['dimensionality_reduction_ratio'] = transformed_data.shape[1] / X.shape[1]
            
        except Exception as e:
            logger.warning(f"Failed to compute dimensionality reduction scores: {e}")
        
        return scores
    
    def _evaluate_anomaly_detection(self, X: np.ndarray, anomaly_scores: Optional[np.ndarray], y: Optional[np.ndarray]) -> Dict[str, float]:
        """评估异常检测结果"""
        scores = {}
        
        if anomaly_scores is None:
            return scores
        
        try:
            # 基本统计
            scores['mean_anomaly_score'] = np.mean(anomaly_scores)
            scores['std_anomaly_score'] = np.std(anomaly_scores)
            
            # 如果有真实标签，计算AUC等指标
            if y is not None:
                from sklearn.metrics import roc_auc_score, average_precision_score
                scores['roc_auc_score'] = roc_auc_score(y, anomaly_scores)
                scores['average_precision_score'] = average_precision_score(y, anomaly_scores)
            
        except Exception as e:
            logger.warning(f"Failed to compute anomaly detection scores: {e}")
        
        return scores
    
    def _get_primary_metric(self) -> str:
        """获取主要评估指标"""
        if self.task_type == 'clustering':
            return 'silhouette_score'
        elif self.task_type == 'dimensionality_reduction':
            return 'explained_variance_ratio'
        elif self.task_type == 'anomaly_detection':
            return 'roc_auc_score'
        else:
            return 'score'
    
    def _get_best_algorithm_name(self) -> str:
        """获取最佳算法名称"""
        for name, result in self.training_results_.items():
            if result.best_model is self.best_model_:
                return name
        return "Unknown"
    
    def get_algorithm_comparison(self) -> Dict[str, Dict[str, float]]:
        """获取算法比较结果"""
        comparison = {}
        
        for algorithm_name, result in self.training_results_.items():
            comparison[algorithm_name] = result.train_scores.copy()
            comparison[algorithm_name]['training_time'] = result.training_time
            
            # 添加任务特定的信息
            if self.task_type == 'clustering':
                comparison[algorithm_name]['n_clusters'] = result.metadata.get('n_clusters', 0)
            elif self.task_type == 'dimensionality_reduction':
                comparison[algorithm_name]['dimensionality_reduction_ratio'] = result.train_scores.get('dimensionality_reduction_ratio', 0)
        
        return comparison
    
    def add_algorithm(self, name: str, algorithm: BaseEstimator, param_grid: Optional[Dict] = None) -> None:
        """添加新算法"""
        self.algorithms[name] = algorithm
        if param_grid:
            self.param_grids[name] = param_grid
    
    def remove_algorithm(self, name: str) -> None:
        """移除算法"""
        if name in self.algorithms:
            del self.algorithms[name]
        if name in self.param_grids:
            del self.param_grids[name]
        if name in self.training_results_:
            del self.training_results_[name]


class ClusteringTrainer(UnsupervisedTrainer):
    """聚类任务训练器"""
    
    def __init__(self, **kwargs):
        kwargs['task_type'] = 'clustering'
        super().__init__(**kwargs)


class DimensionalityReductionTrainer(UnsupervisedTrainer):
    """降维任务训练器"""
    
    def __init__(self, **kwargs):
        kwargs['task_type'] = 'dimensionality_reduction'
        super().__init__(**kwargs)


class AnomalyDetectionTrainer(UnsupervisedTrainer):
    """异常检测任务训练器"""
    
    def __init__(self, **kwargs):
        kwargs['task_type'] = 'anomaly_detection'
        super().__init__(**kwargs)