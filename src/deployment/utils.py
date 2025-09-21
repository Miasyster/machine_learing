"""
部署工具函数模块

提供部署相关的工具函数和辅助功能
"""

import os
import json
import yaml
import shutil
import hashlib
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime
import requests
import subprocess
import tempfile
import psutil
import platform

from .base import DeploymentConfig, ModelMetadata

logger = logging.getLogger(__name__)


def validate_deployment_config(config: DeploymentConfig) -> List[str]:
    """
    验证部署配置
    
    Args:
        config: 部署配置
        
    Returns:
        验证错误列表
    """
    errors = []
    
    # 检查必需字段
    if not config.model_name:
        errors.append("Model name is required")
    
    if not config.model_version:
        errors.append("Model version is required")
    
    if not config.deployment_type:
        errors.append("Deployment type is required")
    
    # 检查部署类型
    valid_types = ['local', 'docker', 'kubernetes', 'cloud']
    if config.deployment_type not in valid_types:
        errors.append(f"Invalid deployment type. Must be one of: {valid_types}")
    
    # 检查端口
    if config.port and (config.port < 1 or config.port > 65535):
        errors.append("Port must be between 1 and 65535")
    
    # 检查路径
    if config.model_storage_path and not Path(config.model_storage_path).exists():
        errors.append(f"Model storage path does not exist: {config.model_storage_path}")
    
    if config.deployment_path:
        try:
            Path(config.deployment_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create deployment path: {e}")
    
    # 检查资源限制
    if config.cpu_limit and config.cpu_limit <= 0:
        errors.append("CPU limit must be positive")
    
    if config.memory_limit and config.memory_limit <= 0:
        errors.append("Memory limit must be positive")
    
    return errors


def check_system_requirements(deployment_type: str) -> Dict[str, bool]:
    """
    检查系统要求
    
    Args:
        deployment_type: 部署类型
        
    Returns:
        要求检查结果
    """
    requirements = {}
    
    # 基本要求
    requirements['python'] = check_python_version()
    requirements['disk_space'] = check_disk_space()
    requirements['memory'] = check_memory()
    
    # 特定部署类型要求
    if deployment_type == 'docker':
        requirements['docker'] = check_docker_available()
    elif deployment_type == 'kubernetes':
        requirements['kubectl'] = check_kubectl_available()
        requirements['kubernetes_cluster'] = check_kubernetes_cluster()
    
    return requirements


def check_python_version(min_version: str = "3.7") -> bool:
    """
    检查Python版本
    
    Args:
        min_version: 最低版本要求
        
    Returns:
        是否满足要求
    """
    try:
        import sys
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        return current_version >= min_version
    except Exception:
        return False


def check_disk_space(min_space_gb: float = 1.0) -> bool:
    """
    检查磁盘空间
    
    Args:
        min_space_gb: 最小空间要求（GB）
        
    Returns:
        是否满足要求
    """
    try:
        disk_usage = psutil.disk_usage('.')
        free_space_gb = disk_usage.free / (1024 ** 3)
        return free_space_gb >= min_space_gb
    except Exception:
        return False


def check_memory(min_memory_gb: float = 1.0) -> bool:
    """
    检查内存
    
    Args:
        min_memory_gb: 最小内存要求（GB）
        
    Returns:
        是否满足要求
    """
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        return available_gb >= min_memory_gb
    except Exception:
        return False


def check_docker_available() -> bool:
    """
    检查Docker是否可用
    
    Returns:
        是否可用
    """
    try:
        result = subprocess.run(
            ["docker", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def check_kubectl_available() -> bool:
    """
    检查kubectl是否可用
    
    Returns:
        是否可用
    """
    try:
        result = subprocess.run(
            ["kubectl", "version", "--client"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def check_kubernetes_cluster() -> bool:
    """
    检查Kubernetes集群连接
    
    Returns:
        是否可连接
    """
    try:
        result = subprocess.run(
            ["kubectl", "cluster-info"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def generate_deployment_id(model_name: str, 
                          model_version: str, 
                          deployment_type: str) -> str:
    """
    生成部署ID
    
    Args:
        model_name: 模型名称
        model_version: 模型版本
        deployment_type: 部署类型
        
    Returns:
        部署ID
    """
    timestamp = int(datetime.now().timestamp())
    base_string = f"{model_name}_{model_version}_{deployment_type}_{timestamp}"
    
    # 生成短哈希
    hash_object = hashlib.md5(base_string.encode())
    short_hash = hash_object.hexdigest()[:8]
    
    return f"{model_name}-{model_version}-{short_hash}"


def create_deployment_package(model_path: str, 
                            output_path: str,
                            include_dependencies: bool = True,
                            compression: str = 'zip') -> str:
    """
    创建部署包
    
    Args:
        model_path: 模型文件路径
        output_path: 输出路径
        include_dependencies: 是否包含依赖
        compression: 压缩格式 ('zip', 'tar', 'tar.gz')
        
    Returns:
        部署包路径
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        package_dir = temp_path / "deployment_package"
        package_dir.mkdir()
        
        # 复制模型文件
        shutil.copy2(model_path, package_dir / "model")
        
        # 创建元数据文件
        metadata = {
            'model_name': model_path.stem,
            'created_at': datetime.now().isoformat(),
            'model_size': model_path.stat().st_size,
            'model_hash': calculate_file_hash(model_path)
        }
        
        with open(package_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 包含依赖（如果需要）
        if include_dependencies:
            create_requirements_file(package_dir / "requirements.txt")
        
        # 创建部署脚本
        create_deployment_script(package_dir / "deploy.py")
        
        # 压缩包
        if compression == 'zip':
            package_path = output_path.with_suffix('.zip')
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_dir)
                        zipf.write(file_path, arcname)
        
        elif compression in ['tar', 'tar.gz']:
            suffix = '.tar.gz' if compression == 'tar.gz' else '.tar'
            package_path = output_path.with_suffix(suffix)
            mode = 'w:gz' if compression == 'tar.gz' else 'w'
            
            with tarfile.open(package_path, mode) as tar:
                tar.add(package_dir, arcname='deployment_package')
        
        else:
            raise ValueError(f"Unsupported compression format: {compression}")
    
    logger.info(f"Deployment package created: {package_path}")
    return str(package_path)


def extract_deployment_package(package_path: str, 
                             extract_path: str) -> str:
    """
    解压部署包
    
    Args:
        package_path: 部署包路径
        extract_path: 解压路径
        
    Returns:
        解压后的目录路径
    """
    package_path = Path(package_path)
    extract_path = Path(extract_path)
    
    if not package_path.exists():
        raise FileNotFoundError(f"Package not found: {package_path}")
    
    extract_path.mkdir(parents=True, exist_ok=True)
    
    if package_path.suffix == '.zip':
        with zipfile.ZipFile(package_path, 'r') as zipf:
            zipf.extractall(extract_path)
    
    elif package_path.suffix in ['.tar', '.gz']:
        with tarfile.open(package_path, 'r:*') as tar:
            tar.extractall(extract_path)
    
    else:
        raise ValueError(f"Unsupported package format: {package_path.suffix}")
    
    # 查找解压后的目录
    extracted_dirs = [d for d in extract_path.iterdir() if d.is_dir()]
    if extracted_dirs:
        return str(extracted_dirs[0])
    else:
        return str(extract_path)


def calculate_file_hash(file_path: Union[str, Path], 
                       algorithm: str = 'md5') -> str:
    """
    计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法
        
    Returns:
        哈希值
    """
    file_path = Path(file_path)
    
    if algorithm == 'md5':
        hash_obj = hashlib.md5()
    elif algorithm == 'sha256':
        hash_obj = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def create_requirements_file(output_path: Union[str, Path]) -> None:
    """
    创建requirements.txt文件
    
    Args:
        output_path: 输出路径
    """
    requirements = [
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "joblib>=1.0.0",
        "flask>=2.0.0",
        "requests>=2.25.0"
    ]
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(requirements))


def create_deployment_script(output_path: Union[str, Path]) -> None:
    """
    创建部署脚本
    
    Args:
        output_path: 输出路径
    """
    script_content = '''#!/usr/bin/env python3
"""
自动部署脚本
"""

import os
import sys
import json
import argparse
from pathlib import Path

def load_metadata():
    """加载元数据"""
    with open("metadata.json", "r") as f:
        return json.load(f)

def deploy_local(port=5000):
    """本地部署"""
    print("Starting local deployment...")
    
    # 安装依赖
    if Path("requirements.txt").exists():
        os.system("pip install -r requirements.txt")
    
    # 启动服务
    from flask import Flask, request, jsonify
    import joblib
    import numpy as np
    
    app = Flask(__name__)
    
    # 加载模型
    model = joblib.load("model")
    
    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.json
            input_data = np.array(data["data"])
            predictions = model.predict(input_data)
            return jsonify({"predictions": predictions.tolist()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"})
    
    app.run(host="0.0.0.0", port=port)

def main():
    parser = argparse.ArgumentParser(description="Deploy ML model")
    parser.add_argument("--type", choices=["local", "docker"], default="local")
    parser.add_argument("--port", type=int, default=5000)
    
    args = parser.parse_args()
    
    metadata = load_metadata()
    print(f"Deploying model: {metadata['model_name']}")
    
    if args.type == "local":
        deploy_local(args.port)
    else:
        print(f"Deployment type {args.type} not implemented")

if __name__ == "__main__":
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # 设置执行权限
    if platform.system() != 'Windows':
        os.chmod(output_path, 0o755)


def monitor_deployment_health(endpoint_url: str, 
                            timeout: int = 5,
                            max_retries: int = 3) -> Dict[str, Any]:
    """
    监控部署健康状态
    
    Args:
        endpoint_url: 端点URL
        timeout: 超时时间
        max_retries: 最大重试次数
        
    Returns:
        健康状态信息
    """
    health_info = {
        'status': 'unknown',
        'response_time': None,
        'error': None,
        'timestamp': datetime.now().isoformat()
    }
    
    for attempt in range(max_retries):
        try:
            start_time = datetime.now()
            
            # 发送健康检查请求
            health_url = f"{endpoint_url.rstrip('/')}/health"
            response = requests.get(health_url, timeout=timeout)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            if response.status_code == 200:
                health_info.update({
                    'status': 'healthy',
                    'response_time': response_time,
                    'response_data': response.json() if response.content else None
                })
                break
            else:
                health_info.update({
                    'status': 'unhealthy',
                    'response_time': response_time,
                    'error': f"HTTP {response.status_code}"
                })
        
        except requests.exceptions.Timeout:
            health_info.update({
                'status': 'timeout',
                'error': f"Request timeout after {timeout}s"
            })
        
        except requests.exceptions.ConnectionError:
            health_info.update({
                'status': 'connection_error',
                'error': "Cannot connect to endpoint"
            })
        
        except Exception as e:
            health_info.update({
                'status': 'error',
                'error': str(e)
            })
        
        if attempt < max_retries - 1:
            import time
            time.sleep(1)  # 重试前等待
    
    return health_info


def get_deployment_logs(deployment_id: str, 
                       log_type: str = 'all',
                       lines: int = 100) -> List[str]:
    """
    获取部署日志
    
    Args:
        deployment_id: 部署ID
        log_type: 日志类型 ('all', 'error', 'access')
        lines: 行数
        
    Returns:
        日志行列表
    """
    logs = []
    
    try:
        # 这里应该根据实际的日志存储方式实现
        # 例如从文件、数据库或容器中读取日志
        
        log_file = Path(f"/var/log/ml_deployment/{deployment_id}.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                logs = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        # 过滤日志类型
        if log_type == 'error':
            logs = [line for line in logs if 'ERROR' in line or 'Exception' in line]
        elif log_type == 'access':
            logs = [line for line in logs if 'GET' in line or 'POST' in line]
    
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        logs = [f"Error reading logs: {e}"]
    
    return logs


def cleanup_old_deployments(deployment_dir: str, 
                          max_age_days: int = 7) -> int:
    """
    清理旧的部署
    
    Args:
        deployment_dir: 部署目录
        max_age_days: 最大保留天数
        
    Returns:
        清理的部署数量
    """
    deployment_dir = Path(deployment_dir)
    if not deployment_dir.exists():
        return 0
    
    cleaned_count = 0
    cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
    
    try:
        for item in deployment_dir.iterdir():
            if item.is_dir():
                # 检查目录的修改时间
                if item.stat().st_mtime < cutoff_time:
                    shutil.rmtree(item)
                    cleaned_count += 1
                    logger.info(f"Cleaned old deployment: {item.name}")
    
    except Exception as e:
        logger.error(f"Failed to cleanup deployments: {e}")
    
    return cleaned_count


def export_deployment_config(config: DeploymentConfig, 
                           output_path: str,
                           format: str = 'json') -> None:
    """
    导出部署配置
    
    Args:
        config: 部署配置
        output_path: 输出路径
        format: 格式 ('json', 'yaml')
    """
    config_dict = {
        'model_name': config.model_name,
        'model_version': config.model_version,
        'deployment_type': config.deployment_type,
        'host': config.host,
        'port': config.port,
        'cpu_limit': config.cpu_limit,
        'memory_limit': config.memory_limit,
        'environment_variables': config.environment_variables,
        'health_check_path': config.health_check_path,
        'model_storage_path': config.model_storage_path,
        'deployment_path': config.deployment_path
    }
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    elif format == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def import_deployment_config(config_path: str) -> DeploymentConfig:
    """
    导入部署配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        部署配置对象
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    
    elif config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return DeploymentConfig(**config_dict)


def benchmark_deployment_performance(endpoint_url: str,
                                   test_data: List[Any],
                                   concurrent_requests: int = 10,
                                   total_requests: int = 100) -> Dict[str, Any]:
    """
    基准测试部署性能
    
    Args:
        endpoint_url: 端点URL
        test_data: 测试数据
        concurrent_requests: 并发请求数
        total_requests: 总请求数
        
    Returns:
        性能指标
    """
    import concurrent.futures
    import statistics
    
    response_times = []
    success_count = 0
    error_count = 0
    
    def send_request(data):
        try:
            start_time = datetime.now()
            response = requests.post(
                f"{endpoint_url}/predict",
                json={"data": data},
                timeout=30
            )
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            
            if response.status_code == 200:
                return response_time, True, None
            else:
                return response_time, False, f"HTTP {response.status_code}"
        
        except Exception as e:
            return None, False, str(e)
    
    # 准备测试数据
    test_requests = []
    for i in range(total_requests):
        data_index = i % len(test_data)
        test_requests.append(test_data[data_index])
    
    # 执行并发测试
    start_time = datetime.now()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(send_request, data) for data in test_requests]
        
        for future in concurrent.futures.as_completed(futures):
            response_time, success, error = future.result()
            
            if success:
                success_count += 1
                response_times.append(response_time)
            else:
                error_count += 1
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # 计算性能指标
    if response_times:
        performance_metrics = {
            'total_requests': total_requests,
            'successful_requests': success_count,
            'failed_requests': error_count,
            'success_rate': success_count / total_requests,
            'total_time': total_time,
            'requests_per_second': total_requests / total_time,
            'average_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
            'p99_response_time': statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
        }
    else:
        performance_metrics = {
            'total_requests': total_requests,
            'successful_requests': 0,
            'failed_requests': total_requests,
            'success_rate': 0.0,
            'error': 'All requests failed'
        }
    
    return performance_metrics