"""
模型推理服务模块

提供模型推理、批量预测和实时服务功能
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import pickle

from .base import DeploymentError
from .serialization import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """预测请求"""
    request_id: str
    data: Any
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PredictionResponse:
    """预测响应"""
    request_id: str
    predictions: Any
    confidence: Optional[List[float]] = None
    processing_time: float = 0.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class ModelInferenceEngine:
    """模型推理引擎"""
    
    def __init__(self, 
                 model_path: str,
                 preprocessing_fn: Optional[Callable] = None,
                 postprocessing_fn: Optional[Callable] = None,
                 batch_size: int = 32,
                 max_workers: int = 4):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型文件路径
            preprocessing_fn: 预处理函数
            postprocessing_fn: 后处理函数
            batch_size: 批处理大小
            max_workers: 最大工作线程数
        """
        self.model_path = model_path
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # 加载模型
        self.model = None
        self.model_metadata = None
        self._load_model()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        # 缓存
        self.cache = {}
        self.cache_enabled = False
        self.cache_max_size = 1000
    
    def _load_model(self) -> None:
        """加载模型"""
        try:
            loader = ModelLoader()
            self.model, self.model_metadata = loader.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise DeploymentError(f"Model loading failed: {e}")
    
    def predict(self, data: Any, request_id: Optional[str] = None) -> PredictionResponse:
        """
        单次预测
        
        Args:
            data: 输入数据
            request_id: 请求ID
            
        Returns:
            预测响应
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        start_time = time.time()
        
        try:
            # 检查缓存
            if self.cache_enabled:
                cache_key = self._generate_cache_key(data)
                if cache_key in self.cache:
                    cached_response = self.cache[cache_key]
                    cached_response.request_id = request_id
                    cached_response.timestamp = datetime.now()
                    return cached_response
            
            # 预处理
            processed_data = self._preprocess(data)
            
            # 预测
            predictions = self.model.predict(processed_data)
            
            # 计算置信度（如果支持）
            confidence = self._get_confidence(processed_data)
            
            # 后处理
            final_predictions = self._postprocess(predictions)
            
            processing_time = time.time() - start_time
            
            response = PredictionResponse(
                request_id=request_id,
                predictions=final_predictions,
                confidence=confidence,
                processing_time=processing_time
            )
            
            # 更新缓存
            if self.cache_enabled:
                self._update_cache(cache_key, response)
            
            # 更新统计
            self._update_stats(processing_time, success=True)
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=False)
            
            logger.error(f"Prediction failed for request {request_id}: {e}")
            raise DeploymentError(f"Prediction failed: {e}")
    
    def predict_batch(self, 
                     data_list: List[Any], 
                     request_ids: Optional[List[str]] = None) -> List[PredictionResponse]:
        """
        批量预测
        
        Args:
            data_list: 输入数据列表
            request_ids: 请求ID列表
            
        Returns:
            预测响应列表
        """
        if request_ids is None:
            request_ids = [f"batch_req_{i}_{int(time.time() * 1000000)}" 
                          for i in range(len(data_list))]
        
        if len(data_list) != len(request_ids):
            raise ValueError("Data list and request IDs must have the same length")
        
        responses = []
        
        # 分批处理
        for i in range(0, len(data_list), self.batch_size):
            batch_data = data_list[i:i + self.batch_size]
            batch_ids = request_ids[i:i + self.batch_size]
            
            batch_responses = self._process_batch(batch_data, batch_ids)
            responses.extend(batch_responses)
        
        return responses
    
    def predict_async(self, data: Any, request_id: Optional[str] = None) -> 'Future':
        """
        异步预测
        
        Args:
            data: 输入数据
            request_id: 请求ID
            
        Returns:
            Future对象
        """
        return self.executor.submit(self.predict, data, request_id)
    
    def predict_batch_async(self, 
                           data_list: List[Any], 
                           request_ids: Optional[List[str]] = None) -> List['Future']:
        """
        异步批量预测
        
        Args:
            data_list: 输入数据列表
            request_ids: 请求ID列表
            
        Returns:
            Future对象列表
        """
        if request_ids is None:
            request_ids = [f"async_batch_req_{i}_{int(time.time() * 1000000)}" 
                          for i in range(len(data_list))]
        
        futures = []
        for data, req_id in zip(data_list, request_ids):
            future = self.executor.submit(self.predict, data, req_id)
            futures.append(future)
        
        return futures
    
    def _process_batch(self, batch_data: List[Any], batch_ids: List[str]) -> List[PredictionResponse]:
        """处理批量数据"""
        try:
            # 预处理整个批次
            processed_batch = [self._preprocess(data) for data in batch_data]
            
            # 合并为单个数组（如果可能）
            if all(isinstance(data, np.ndarray) for data in processed_batch):
                combined_data = np.vstack(processed_batch)
                batch_predictions = self.model.predict(combined_data)
                
                # 分割预测结果
                responses = []
                for i, (data, req_id) in enumerate(zip(batch_data, batch_ids)):
                    prediction = batch_predictions[i:i+1] if batch_predictions.ndim > 1 else batch_predictions[i]
                    
                    response = PredictionResponse(
                        request_id=req_id,
                        predictions=self._postprocess(prediction),
                        processing_time=0.0  # 批处理时间分摊
                    )
                    responses.append(response)
                
                return responses
            else:
                # 逐个处理
                return [self.predict(data, req_id) for data, req_id in zip(batch_data, batch_ids)]
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # 回退到逐个处理
            return [self.predict(data, req_id) for data, req_id in zip(batch_data, batch_ids)]
    
    def _preprocess(self, data: Any) -> Any:
        """预处理数据"""
        if self.preprocessing_fn:
            return self.preprocessing_fn(data)
        return data
    
    def _postprocess(self, predictions: Any) -> Any:
        """后处理预测结果"""
        if self.postprocessing_fn:
            return self.postprocessing_fn(predictions)
        return predictions
    
    def _get_confidence(self, data: Any) -> Optional[List[float]]:
        """获取预测置信度"""
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(data)
                if probabilities.ndim == 2:
                    return np.max(probabilities, axis=1).tolist()
                else:
                    return [float(np.max(probabilities))]
            elif hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(data)
                # 将决策函数分数转换为置信度
                return [float(abs(score)) for score in scores]
        except Exception as e:
            logger.warning(f"Failed to get confidence scores: {e}")
        
        return None
    
    def _generate_cache_key(self, data: Any) -> str:
        """生成缓存键"""
        try:
            if isinstance(data, np.ndarray):
                return hashlib.md5(data.tobytes()).hexdigest()
            elif isinstance(data, (list, tuple)):
                return hashlib.md5(str(data).encode()).hexdigest()
            else:
                return hashlib.md5(pickle.dumps(data)).hexdigest()
        except Exception:
            return str(hash(str(data)))
    
    def _update_cache(self, cache_key: str, response: PredictionResponse) -> None:
        """更新缓存"""
        if len(self.cache) >= self.cache_max_size:
            # 简单的LRU：删除最旧的条目
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = response
    
    def _update_stats(self, processing_time: float, success: bool) -> None:
        """更新统计信息"""
        self.stats['total_requests'] += 1
        self.stats['total_processing_time'] += processing_time
        
        if success:
            self.stats['successful_predictions'] += 1
        else:
            self.stats['failed_predictions'] += 1
        
        self.stats['average_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['total_requests']
        )
    
    def enable_cache(self, max_size: int = 1000) -> None:
        """启用缓存"""
        self.cache_enabled = True
        self.cache_max_size = max_size
        logger.info(f"Cache enabled with max size: {max_size}")
    
    def disable_cache(self) -> None:
        """禁用缓存"""
        self.cache_enabled = False
        self.cache.clear()
        logger.info("Cache disabled")
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
    
    def shutdown(self) -> None:
        """关闭推理引擎"""
        self.executor.shutdown(wait=True)
        logger.info("Inference engine shutdown")


class StreamingInferenceEngine:
    """流式推理引擎"""
    
    def __init__(self, 
                 model_path: str,
                 queue_size: int = 1000,
                 batch_size: int = 32,
                 batch_timeout: float = 0.1):
        """
        初始化流式推理引擎
        
        Args:
            model_path: 模型路径
            queue_size: 队列大小
            batch_size: 批处理大小
            batch_timeout: 批处理超时时间
        """
        self.model_path = model_path
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # 加载模型
        self.inference_engine = ModelInferenceEngine(model_path)
        
        # 请求队列
        self.request_queue = queue.Queue(maxsize=queue_size)
        self.response_callbacks = {}
        
        # 控制标志
        self.running = False
        self.worker_thread = None
    
    def start(self) -> None:
        """启动流式处理"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_stream)
        self.worker_thread.start()
        logger.info("Streaming inference engine started")
    
    def stop(self) -> None:
        """停止流式处理"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Streaming inference engine stopped")
    
    def submit_request(self, 
                      data: Any, 
                      callback: Callable[[PredictionResponse], None],
                      request_id: Optional[str] = None) -> str:
        """
        提交预测请求
        
        Args:
            data: 输入数据
            callback: 回调函数
            request_id: 请求ID
            
        Returns:
            请求ID
        """
        if request_id is None:
            request_id = f"stream_req_{int(time.time() * 1000000)}"
        
        request = PredictionRequest(
            request_id=request_id,
            data=data,
            timestamp=datetime.now()
        )
        
        self.response_callbacks[request_id] = callback
        
        try:
            self.request_queue.put(request, timeout=1.0)
            return request_id
        except queue.Full:
            del self.response_callbacks[request_id]
            raise DeploymentError("Request queue is full")
    
    def _process_stream(self) -> None:
        """处理流式请求"""
        while self.running:
            batch_requests = []
            start_time = time.time()
            
            # 收集批次请求
            while (len(batch_requests) < self.batch_size and 
                   (time.time() - start_time) < self.batch_timeout):
                try:
                    request = self.request_queue.get(timeout=0.01)
                    batch_requests.append(request)
                except queue.Empty:
                    continue
            
            if not batch_requests:
                continue
            
            # 处理批次
            try:
                data_list = [req.data for req in batch_requests]
                request_ids = [req.request_id for req in batch_requests]
                
                responses = self.inference_engine.predict_batch(data_list, request_ids)
                
                # 调用回调函数
                for response in responses:
                    callback = self.response_callbacks.pop(response.request_id, None)
                    if callback:
                        try:
                            callback(response)
                        except Exception as e:
                            logger.error(f"Callback failed for request {response.request_id}: {e}")
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                
                # 为失败的请求调用回调
                for request in batch_requests:
                    callback = self.response_callbacks.pop(request.request_id, None)
                    if callback:
                        error_response = PredictionResponse(
                            request_id=request.request_id,
                            predictions=None,
                            metadata={'error': str(e)}
                        )
                        try:
                            callback(error_response)
                        except Exception as cb_e:
                            logger.error(f"Error callback failed: {cb_e}")


class ModelEnsemble:
    """模型集成推理"""
    
    def __init__(self, 
                 model_paths: List[str],
                 weights: Optional[List[float]] = None,
                 voting_strategy: str = 'soft'):
        """
        初始化模型集成
        
        Args:
            model_paths: 模型路径列表
            weights: 模型权重
            voting_strategy: 投票策略 ('soft', 'hard', 'weighted')
        """
        self.model_paths = model_paths
        self.weights = weights or [1.0] * len(model_paths)
        self.voting_strategy = voting_strategy
        
        if len(self.weights) != len(model_paths):
            raise ValueError("Weights must match the number of models")
        
        # 加载所有模型
        self.engines = []
        for model_path in model_paths:
            engine = ModelInferenceEngine(model_path)
            self.engines.append(engine)
        
        logger.info(f"Ensemble initialized with {len(self.engines)} models")
    
    def predict(self, data: Any, request_id: Optional[str] = None) -> PredictionResponse:
        """
        集成预测
        
        Args:
            data: 输入数据
            request_id: 请求ID
            
        Returns:
            预测响应
        """
        if request_id is None:
            request_id = f"ensemble_req_{int(time.time() * 1000000)}"
        
        start_time = time.time()
        
        try:
            # 获取所有模型的预测
            predictions = []
            confidences = []
            
            for engine in self.engines:
                response = engine.predict(data, f"{request_id}_sub")
                predictions.append(response.predictions)
                if response.confidence:
                    confidences.append(response.confidence)
            
            # 集成预测结果
            ensemble_prediction = self._ensemble_predictions(predictions)
            ensemble_confidence = self._ensemble_confidence(confidences) if confidences else None
            
            processing_time = time.time() - start_time
            
            return PredictionResponse(
                request_id=request_id,
                predictions=ensemble_prediction,
                confidence=ensemble_confidence,
                processing_time=processing_time,
                metadata={'ensemble_size': len(self.engines)}
            )
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise DeploymentError(f"Ensemble prediction failed: {e}")
    
    def _ensemble_predictions(self, predictions: List[Any]) -> Any:
        """集成预测结果"""
        if self.voting_strategy == 'soft':
            return self._soft_voting(predictions)
        elif self.voting_strategy == 'hard':
            return self._hard_voting(predictions)
        elif self.voting_strategy == 'weighted':
            return self._weighted_voting(predictions)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def _soft_voting(self, predictions: List[Any]) -> Any:
        """软投票"""
        if all(isinstance(pred, np.ndarray) for pred in predictions):
            # 数值预测：平均值
            return np.mean(predictions, axis=0)
        else:
            # 分类预测：返回最常见的预测
            return self._hard_voting(predictions)
    
    def _hard_voting(self, predictions: List[Any]) -> Any:
        """硬投票"""
        # 简单多数投票
        from collections import Counter
        
        if isinstance(predictions[0], (list, np.ndarray)) and len(predictions[0]) > 1:
            # 多输出情况
            ensemble_pred = []
            for i in range(len(predictions[0])):
                votes = [pred[i] for pred in predictions]
                most_common = Counter(votes).most_common(1)[0][0]
                ensemble_pred.append(most_common)
            return ensemble_pred
        else:
            # 单输出情况
            votes = [pred if not isinstance(pred, (list, np.ndarray)) else pred[0] 
                    for pred in predictions]
            return Counter(votes).most_common(1)[0][0]
    
    def _weighted_voting(self, predictions: List[Any]) -> Any:
        """加权投票"""
        if all(isinstance(pred, np.ndarray) for pred in predictions):
            # 加权平均
            weighted_sum = np.zeros_like(predictions[0])
            total_weight = sum(self.weights)
            
            for pred, weight in zip(predictions, self.weights):
                weighted_sum += pred * weight
            
            return weighted_sum / total_weight
        else:
            # 加权硬投票（简化版本）
            return self._hard_voting(predictions)
    
    def _ensemble_confidence(self, confidences: List[List[float]]) -> List[float]:
        """集成置信度"""
        if not confidences:
            return None
        
        # 加权平均置信度
        ensemble_conf = []
        for i in range(len(confidences[0])):
            weighted_conf = sum(conf[i] * weight 
                              for conf, weight in zip(confidences, self.weights))
            ensemble_conf.append(weighted_conf / sum(self.weights))
        
        return ensemble_conf


def create_inference_engine(model_path: str, **kwargs) -> ModelInferenceEngine:
    """
    创建推理引擎
    
    Args:
        model_path: 模型路径
        **kwargs: 其他参数
        
    Returns:
        推理引擎实例
    """
    return ModelInferenceEngine(model_path, **kwargs)


def create_streaming_engine(model_path: str, **kwargs) -> StreamingInferenceEngine:
    """
    创建流式推理引擎
    
    Args:
        model_path: 模型路径
        **kwargs: 其他参数
        
    Returns:
        流式推理引擎实例
    """
    return StreamingInferenceEngine(model_path, **kwargs)


def create_ensemble(model_paths: List[str], **kwargs) -> ModelEnsemble:
    """
    创建模型集成
    
    Args:
        model_paths: 模型路径列表
        **kwargs: 其他参数
        
    Returns:
        模型集成实例
    """
    return ModelEnsemble(model_paths, **kwargs)