"""
部署系统基础类

定义部署配置、模型元数据和基础部署接口
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """部署配置"""
    
    # 基本配置
    model_name: str
    model_version: str
    deployment_name: str
    environment: str = "production"  # development, staging, production
    
    # 资源配置
    cpu_limit: Optional[str] = "1000m"  # CPU限制
    memory_limit: Optional[str] = "2Gi"  # 内存限制
    gpu_enabled: bool = False
    gpu_count: int = 0
    
    # 扩展配置
    min_replicas: int = 1
    max_replicas: int = 10
    auto_scaling: bool = True
    target_cpu_utilization: int = 70
    
    # 网络配置
    port: int = 8080
    health_check_path: str = "/health"
    metrics_path: str = "/metrics"
    
    # 安全配置
    enable_auth: bool = False
    api_key_required: bool = False
    rate_limit: Optional[int] = None  # 每分钟请求数
    
    # 监控配置
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = False
    
    # 部署策略
    deployment_strategy: str = "rolling"  # rolling, blue_green, canary
    rollback_enabled: bool = True
    health_check_timeout: int = 30
    
    # 存储配置
    model_storage_path: Optional[str] = None
    artifact_storage_path: Optional[str] = None
    
    # 其他配置
    tags: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    """模型元数据"""
    
    # 基本信息
    model_name: str
    model_version: str
    model_type: str  # classification, regression, clustering, etc.
    framework: str  # sklearn, tensorflow, pytorch, etc.
    
    # 模型信息
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    
    # 性能指标
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_score: Optional[float] = None
    validation_score: Optional[float] = None
    test_score: Optional[float] = None
    
    # 训练信息
    training_data_size: Optional[int] = None
    training_time: Optional[float] = None
    training_date: Optional[datetime] = None
    
    # 数据信息
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    
    # 部署信息
    deployment_requirements: Dict[str, str] = field(default_factory=dict)
    inference_time: Optional[float] = None
    memory_usage: Optional[float] = None
    model_size: Optional[float] = None
    
    # 版本信息
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    parent_version: Optional[str] = None
    
    # 标签和注释
    tags: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, (list, dict)):
                result[key] = value.copy()
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建"""
        # 处理日期时间字段
        datetime_fields = ['training_date', 'created_at', 'updated_at']
        for field_name in datetime_fields:
            if field_name in data and data[field_name] is not None:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)
    
    def update_performance(self, metrics: Dict[str, float]) -> None:
        """更新性能指标"""
        self.performance_metrics.update(metrics)
        self.updated_at = datetime.now()
    
    def add_tag(self, key: str, value: str) -> None:
        """添加标签"""
        self.tags[key] = value
        self.updated_at = datetime.now()
    
    def remove_tag(self, key: str) -> None:
        """移除标签"""
        if key in self.tags:
            del self.tags[key]
            self.updated_at = datetime.now()


class BaseDeployment(ABC):
    """基础部署接口"""
    
    def __init__(self, config: DeploymentConfig):
        """
        初始化部署
        
        Args:
            config: 部署配置
        """
        self.config = config
        self.deployment_id = None
        self.status = "not_deployed"
        self.created_at = None
        self.updated_at = None
        self.metadata = {}
        
        # 设置日志
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if config.enable_logging:
            self.logger.setLevel(getattr(logging, config.log_level.upper()))
    
    @abstractmethod
    def deploy(self, model_path: str, metadata: Optional[ModelMetadata] = None) -> str:
        """
        部署模型
        
        Args:
            model_path: 模型文件路径
            metadata: 模型元数据
            
        Returns:
            部署ID
        """
        pass
    
    @abstractmethod
    def undeploy(self) -> bool:
        """
        取消部署
        
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def update(self, model_path: str, metadata: Optional[ModelMetadata] = None) -> bool:
        """
        更新部署
        
        Args:
            model_path: 新模型文件路径
            metadata: 新模型元数据
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        获取部署状态
        
        Returns:
            状态信息
        """
        pass
    
    @abstractmethod
    def get_logs(self, lines: int = 100) -> List[str]:
        """
        获取部署日志
        
        Args:
            lines: 日志行数
            
        Returns:
            日志列表
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取监控指标
        
        Returns:
            监控指标
        """
        pass
    
    def predict(self, data: Any) -> Any:
        """
        预测接口
        
        Args:
            data: 输入数据
            
        Returns:
            预测结果
        """
        raise NotImplementedError("Predict method not implemented")
    
    def batch_predict(self, data_list: List[Any]) -> List[Any]:
        """
        批量预测接口
        
        Args:
            data_list: 输入数据列表
            
        Returns:
            预测结果列表
        """
        return [self.predict(data) for data in data_list]
    
    def scale(self, replicas: int) -> bool:
        """
        扩缩容
        
        Args:
            replicas: 副本数
            
        Returns:
            是否成功
        """
        self.logger.info(f"Scaling to {replicas} replicas")
        return True
    
    def rollback(self, version: Optional[str] = None) -> bool:
        """
        回滚部署
        
        Args:
            version: 目标版本
            
        Returns:
            是否成功
        """
        if not self.config.rollback_enabled:
            self.logger.error("Rollback is not enabled")
            return False
        
        self.logger.info(f"Rolling back to version {version}")
        return True
    
    def get_endpoint(self) -> Optional[str]:
        """
        获取服务端点
        
        Returns:
            服务端点URL
        """
        if self.status == "running":
            return f"http://localhost:{self.config.port}"
        return None
    
    def validate_config(self) -> List[str]:
        """
        验证配置
        
        Returns:
            错误列表
        """
        errors = []
        
        # 验证基本配置
        if not self.config.model_name:
            errors.append("Model name is required")
        
        if not self.config.model_version:
            errors.append("Model version is required")
        
        if not self.config.deployment_name:
            errors.append("Deployment name is required")
        
        # 验证资源配置
        if self.config.min_replicas < 1:
            errors.append("Min replicas must be at least 1")
        
        if self.config.max_replicas < self.config.min_replicas:
            errors.append("Max replicas must be >= min replicas")
        
        # 验证端口配置
        if not (1 <= self.config.port <= 65535):
            errors.append("Port must be between 1 and 65535")
        
        return errors
    
    def _update_status(self, status: str) -> None:
        """更新状态"""
        self.status = status
        self.updated_at = datetime.now()
        self.logger.info(f"Status updated to: {status}")
    
    def _log_deployment_event(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        """记录部署事件"""
        log_data = {
            'event': event,
            'deployment_id': self.deployment_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.deployment_name
        }
        
        if details:
            log_data.update(details)
        
        self.logger.info(f"Deployment event: {log_data}")


class DeploymentError(Exception):
    """部署错误"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ValidationError(DeploymentError):
    """验证错误"""
    pass


class ResourceError(DeploymentError):
    """资源错误"""
    pass


class NetworkError(DeploymentError):
    """网络错误"""
    pass


class AuthenticationError(DeploymentError):
    """认证错误"""
    pass