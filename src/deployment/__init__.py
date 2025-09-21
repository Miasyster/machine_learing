"""模型部署模块

提供模型部署、版本管理、推理服务等功能
"""

# 基础类
from .base import (
    DeploymentConfig,
    ModelMetadata,
    BaseDeployment,
    DeploymentError,
    ModelNotFoundError,
    DeploymentFailedError
)

# 模型序列化
from .serialization import (
    ModelSerializer,
    ModelLoader,
    SerializationManager,
    validate_model_compatibility,
    optimize_model_size
)

# 版本管理
from .versioning import (
    VersionInfo,
    ModelVersionManager,
    VersioningStrategy
)

# 推理服务
from .inference import (
    PredictionRequest,
    PredictionResponse,
    ModelInferenceEngine,
    StreamingInferenceEngine,
    ModelEnsemble,
    create_inference_engine,
    create_streaming_engine,
    create_ensemble
)

# 部署管理
from .deployment import (
    DeploymentStatus,
    LocalDeployment,
    DockerDeployment,
    DeploymentManager,
    create_deployment_manager
)

# 工具函数
from .utils import (
    validate_deployment_config,
    check_system_requirements,
    check_python_version,
    check_disk_space,
    check_memory,
    check_docker_available,
    check_kubectl_available,
    check_kubernetes_cluster,
    generate_deployment_id,
    create_deployment_package,
    extract_deployment_package,
    calculate_file_hash,
    create_requirements_file,
    create_deployment_script,
    monitor_deployment_health,
    get_deployment_logs,
    cleanup_old_deployments,
    export_deployment_config,
    import_deployment_config,
    benchmark_deployment_performance
)

__all__ = [
    # 基础类
    'DeploymentConfig',
    'ModelMetadata', 
    'BaseDeployment',
    'DeploymentError',
    'ModelNotFoundError',
    'DeploymentFailedError',
    
    # 模型序列化
    'ModelSerializer',
    'ModelLoader',
    'SerializationManager',
    'validate_model_compatibility',
    'optimize_model_size',
    
    # 版本管理
    'VersionInfo',
    'ModelVersionManager',
    'VersioningStrategy',
    
    # 推理服务
    'PredictionRequest',
    'PredictionResponse',
    'ModelInferenceEngine',
    'StreamingInferenceEngine',
    'ModelEnsemble',
    'create_inference_engine',
    'create_streaming_engine',
    'create_ensemble',
    
    # 部署管理
    'DeploymentStatus',
    'LocalDeployment',
    'DockerDeployment',
    'DeploymentManager',
    'create_deployment_manager',
    
    # 工具函数
    'validate_deployment_config',
    'check_system_requirements',
    'check_python_version',
    'check_disk_space',
    'check_memory',
    'check_docker_available',
    'check_kubectl_available',
    'check_kubernetes_cluster',
    'generate_deployment_id',
    'create_deployment_package',
    'extract_deployment_package',
    'calculate_file_hash',
    'create_requirements_file',
    'create_deployment_script',
    'monitor_deployment_health',
    'get_deployment_logs',
    'cleanup_old_deployments',
    'export_deployment_config',
    'import_deployment_config',
    'benchmark_deployment_performance'
]

__version__ = "1.0.0"