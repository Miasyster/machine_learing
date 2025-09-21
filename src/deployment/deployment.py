"""
模型部署管理模块

提供模型部署、管理和监控功能
"""

import os
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import threading
import psutil
import requests
from concurrent.futures import ThreadPoolExecutor

from .base import BaseDeployment, DeploymentConfig, ModelMetadata, DeploymentError
from .serialization import SerializationManager
from .versioning import ModelVersionManager
from .inference import ModelInferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class DeploymentStatus:
    """部署状态"""
    deployment_id: str
    model_name: str
    model_version: str
    status: str  # 'deploying', 'running', 'stopped', 'failed'
    endpoint_url: Optional[str] = None
    health_status: str = 'unknown'  # 'healthy', 'unhealthy', 'unknown'
    created_at: datetime = None
    updated_at: datetime = None
    resource_usage: Dict[str, float] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.resource_usage is None:
            self.resource_usage = {}


class LocalDeployment(BaseDeployment):
    """本地部署"""
    
    def __init__(self, config: DeploymentConfig):
        """
        初始化本地部署
        
        Args:
            config: 部署配置
        """
        super().__init__(config)
        self.deployments = {}  # deployment_id -> DeploymentStatus
        self.processes = {}    # deployment_id -> process
        self.inference_engines = {}  # deployment_id -> ModelInferenceEngine
        
        # 部署目录
        self.deployment_dir = Path(config.deployment_path) / "local_deployments"
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # 监控线程
        self.monitoring_thread = None
        self.monitoring_enabled = False
    
    def deploy(self, 
              model_name: str, 
              model_version: str, 
              deployment_id: Optional[str] = None) -> str:
        """
        部署模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            deployment_id: 部署ID
            
        Returns:
            部署ID
        """
        if deployment_id is None:
            deployment_id = f"{model_name}_{model_version}_{int(time.time())}"
        
        try:
            # 创建部署状态
            status = DeploymentStatus(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=model_version,
                status='deploying'
            )
            self.deployments[deployment_id] = status
            
            # 创建部署目录
            deploy_dir = self.deployment_dir / deployment_id
            deploy_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制模型文件
            model_path = self._get_model_path(model_name, model_version)
            target_model_path = deploy_dir / "model"
            shutil.copy2(model_path, target_model_path)
            
            # 创建推理引擎
            inference_engine = ModelInferenceEngine(str(target_model_path))
            self.inference_engines[deployment_id] = inference_engine
            
            # 启动服务（如果配置了端口）
            if self.config.port:
                self._start_service(deployment_id, deploy_dir)
            
            # 更新状态
            status.status = 'running'
            status.updated_at = datetime.now()
            
            if self.config.port:
                status.endpoint_url = f"http://localhost:{self.config.port}/predict"
            
            logger.info(f"Model deployed successfully: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            if deployment_id in self.deployments:
                self.deployments[deployment_id].status = 'failed'
                self.deployments[deployment_id].error_message = str(e)
            
            logger.error(f"Deployment failed: {e}")
            raise DeploymentError(f"Deployment failed: {e}")
    
    def undeploy(self, deployment_id: str) -> bool:
        """
        取消部署
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            是否成功
        """
        try:
            if deployment_id not in self.deployments:
                logger.warning(f"Deployment {deployment_id} not found")
                return False
            
            # 停止服务
            if deployment_id in self.processes:
                process = self.processes[deployment_id]
                process.terminate()
                process.wait(timeout=10)
                del self.processes[deployment_id]
            
            # 清理推理引擎
            if deployment_id in self.inference_engines:
                self.inference_engines[deployment_id].shutdown()
                del self.inference_engines[deployment_id]
            
            # 删除部署目录
            deploy_dir = self.deployment_dir / deployment_id
            if deploy_dir.exists():
                shutil.rmtree(deploy_dir)
            
            # 更新状态
            self.deployments[deployment_id].status = 'stopped'
            self.deployments[deployment_id].updated_at = datetime.now()
            
            logger.info(f"Deployment {deployment_id} undeployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Undeploy failed: {e}")
            return False
    
    def update(self, 
              deployment_id: str, 
              model_name: str, 
              model_version: str) -> bool:
        """
        更新部署
        
        Args:
            deployment_id: 部署ID
            model_name: 新模型名称
            model_version: 新模型版本
            
        Returns:
            是否成功
        """
        try:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            # 停止当前服务
            self.undeploy(deployment_id)
            
            # 重新部署新版本
            self.deploy(model_name, model_version, deployment_id)
            
            logger.info(f"Deployment {deployment_id} updated to {model_name}:{model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False
    
    def get_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """
        获取部署状态
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            部署状态
        """
        return self.deployments.get(deployment_id)
    
    def list_deployments(self) -> List[DeploymentStatus]:
        """
        列出所有部署
        
        Returns:
            部署状态列表
        """
        return list(self.deployments.values())
    
    def health_check(self, deployment_id: str) -> bool:
        """
        健康检查
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            是否健康
        """
        try:
            status = self.deployments.get(deployment_id)
            if not status or status.status != 'running':
                return False
            
            # 检查推理引擎
            if deployment_id in self.inference_engines:
                engine = self.inference_engines[deployment_id]
                # 简单的健康检查：尝试预测
                import numpy as np
                test_data = np.array([[0, 0, 0, 0]])  # 简单测试数据
                try:
                    engine.predict(test_data)
                    status.health_status = 'healthy'
                    return True
                except Exception:
                    status.health_status = 'unhealthy'
                    return False
            
            # 检查HTTP服务
            if status.endpoint_url:
                try:
                    response = requests.get(f"{status.endpoint_url}/health", timeout=5)
                    healthy = response.status_code == 200
                    status.health_status = 'healthy' if healthy else 'unhealthy'
                    return healthy
                except Exception:
                    status.health_status = 'unhealthy'
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def predict(self, deployment_id: str, data: Any) -> Any:
        """
        使用部署的模型进行预测
        
        Args:
            deployment_id: 部署ID
            data: 输入数据
            
        Returns:
            预测结果
        """
        if deployment_id not in self.inference_engines:
            raise ValueError(f"Deployment {deployment_id} not found or not running")
        
        engine = self.inference_engines[deployment_id]
        response = engine.predict(data)
        return response.predictions
    
    def get_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """
        获取部署指标
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            指标字典
        """
        metrics = {}
        
        if deployment_id in self.inference_engines:
            engine = self.inference_engines[deployment_id]
            metrics.update(engine.get_stats())
        
        if deployment_id in self.processes:
            process = self.processes[deployment_id]
            try:
                proc = psutil.Process(process.pid)
                metrics.update({
                    'cpu_percent': proc.cpu_percent(),
                    'memory_mb': proc.memory_info().rss / 1024 / 1024,
                    'num_threads': proc.num_threads()
                })
            except Exception as e:
                logger.warning(f"Failed to get process metrics: {e}")
        
        return metrics
    
    def start_monitoring(self, interval: int = 60) -> None:
        """
        启动监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,)
        )
        self.monitoring_thread.start()
        logger.info("Monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Monitoring stopped")
    
    def _get_model_path(self, model_name: str, model_version: str) -> str:
        """获取模型文件路径"""
        # 这里应该与SerializationManager集成
        model_dir = Path(self.config.model_storage_path) / model_name / model_version
        model_files = list(model_dir.glob("model*"))
        
        if not model_files:
            raise FileNotFoundError(f"Model file not found: {model_name}:{model_version}")
        
        return str(model_files[0])
    
    def _start_service(self, deployment_id: str, deploy_dir: Path) -> None:
        """启动HTTP服务"""
        # 创建简单的Flask服务脚本
        service_script = deploy_dir / "service.py"
        self._create_service_script(service_script, deploy_dir / "model")
        
        # 启动服务
        cmd = [
            "python", str(service_script),
            "--port", str(self.config.port),
            "--host", self.config.host or "localhost"
        ]
        
        process = subprocess.Popen(cmd, cwd=deploy_dir)
        self.processes[deployment_id] = process
        
        # 等待服务启动
        time.sleep(2)
    
    def _create_service_script(self, script_path: Path, model_path: Path) -> None:
        """创建服务脚本"""
        script_content = f'''
import sys
import argparse
from flask import Flask, request, jsonify
import numpy as np
import joblib
import pickle

app = Flask(__name__)

# 加载模型
try:
    model = joblib.load("{model_path}")
except:
    try:
        with open("{model_path}", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Failed to load model: {{e}}")
        sys.exit(1)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if "data" not in data:
            return jsonify({{"error": "Missing data field"}}), 400
        
        input_data = np.array(data["data"])
        predictions = model.predict(input_data)
        
        return jsonify({{"predictions": predictions.tolist()}})
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({{"status": "healthy"}})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    
    app.run(host=args.host, port=args.port)
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
    
    def _monitoring_loop(self, interval: int) -> None:
        """监控循环"""
        while self.monitoring_enabled:
            try:
                for deployment_id in list(self.deployments.keys()):
                    # 健康检查
                    self.health_check(deployment_id)
                    
                    # 更新资源使用情况
                    metrics = self.get_metrics(deployment_id)
                    if deployment_id in self.deployments:
                        self.deployments[deployment_id].resource_usage = metrics
                        self.deployments[deployment_id].updated_at = datetime.now()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)


class DockerDeployment(BaseDeployment):
    """Docker部署"""
    
    def __init__(self, config: DeploymentConfig):
        """
        初始化Docker部署
        
        Args:
            config: 部署配置
        """
        super().__init__(config)
        self.containers = {}  # deployment_id -> container_id
        
        # 检查Docker是否可用
        self._check_docker()
    
    def deploy(self, 
              model_name: str, 
              model_version: str, 
              deployment_id: Optional[str] = None) -> str:
        """
        部署模型到Docker容器
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            deployment_id: 部署ID
            
        Returns:
            部署ID
        """
        if deployment_id is None:
            deployment_id = f"{model_name}_{model_version}_{int(time.time())}"
        
        try:
            # 构建Docker镜像
            image_name = f"ml-model-{model_name}-{model_version}".lower()
            self._build_docker_image(model_name, model_version, image_name)
            
            # 运行容器
            container_id = self._run_container(image_name, deployment_id)
            self.containers[deployment_id] = container_id
            
            logger.info(f"Model deployed to Docker: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            raise DeploymentError(f"Docker deployment failed: {e}")
    
    def undeploy(self, deployment_id: str) -> bool:
        """
        停止并删除Docker容器
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            是否成功
        """
        try:
            if deployment_id not in self.containers:
                return False
            
            container_id = self.containers[deployment_id]
            
            # 停止容器
            subprocess.run(["docker", "stop", container_id], check=True)
            
            # 删除容器
            subprocess.run(["docker", "rm", container_id], check=True)
            
            del self.containers[deployment_id]
            
            logger.info(f"Docker container {deployment_id} removed")
            return True
            
        except Exception as e:
            logger.error(f"Docker undeploy failed: {e}")
            return False
    
    def _check_docker(self) -> None:
        """检查Docker是否可用"""
        try:
            subprocess.run(["docker", "--version"], 
                         check=True, 
                         capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise DeploymentError("Docker is not available")
    
    def _build_docker_image(self, 
                           model_name: str, 
                           model_version: str, 
                           image_name: str) -> None:
        """构建Docker镜像"""
        # 创建临时构建目录
        build_dir = Path("/tmp") / f"docker_build_{int(time.time())}"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 复制模型文件
            model_path = self._get_model_path(model_name, model_version)
            shutil.copy2(model_path, build_dir / "model")
            
            # 创建Dockerfile
            dockerfile_content = '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model .
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
'''
            
            with open(build_dir / "Dockerfile", 'w') as f:
                f.write(dockerfile_content)
            
            # 创建requirements.txt
            requirements = [
                "flask",
                "numpy",
                "scikit-learn",
                "joblib"
            ]
            
            with open(build_dir / "requirements.txt", 'w') as f:
                f.write('\n'.join(requirements))
            
            # 创建Flask应用
            self._create_flask_app(build_dir / "app.py")
            
            # 构建镜像
            subprocess.run([
                "docker", "build", 
                "-t", image_name, 
                str(build_dir)
            ], check=True)
            
        finally:
            # 清理构建目录
            shutil.rmtree(build_dir, ignore_errors=True)
    
    def _create_flask_app(self, app_path: Path) -> None:
        """创建Flask应用"""
        app_content = '''
from flask import Flask, request, jsonify
import numpy as np
import joblib
import pickle

app = Flask(__name__)

# 加载模型
try:
    model = joblib.load("model")
except:
    try:
        with open("model", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
'''
        
        with open(app_path, 'w') as f:
            f.write(app_content)
    
    def _run_container(self, image_name: str, deployment_id: str) -> str:
        """运行Docker容器"""
        cmd = [
            "docker", "run", "-d",
            "--name", deployment_id,
            "-p", f"{self.config.port}:5000",
            image_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    
    def _get_model_path(self, model_name: str, model_version: str) -> str:
        """获取模型文件路径"""
        # 与LocalDeployment相同的实现
        model_dir = Path(self.config.model_storage_path) / model_name / model_version
        model_files = list(model_dir.glob("model*"))
        
        if not model_files:
            raise FileNotFoundError(f"Model file not found: {model_name}:{model_version}")
        
        return str(model_files[0])


class DeploymentManager:
    """部署管理器"""
    
    def __init__(self, config: DeploymentConfig):
        """
        初始化部署管理器
        
        Args:
            config: 部署配置
        """
        self.config = config
        self.deployments = {}  # deployment_type -> BaseDeployment
        
        # 初始化部署器
        if config.deployment_type == 'local':
            self.deployments['local'] = LocalDeployment(config)
        elif config.deployment_type == 'docker':
            self.deployments['docker'] = DockerDeployment(config)
        else:
            raise ValueError(f"Unsupported deployment type: {config.deployment_type}")
        
        self.current_deployment = self.deployments[config.deployment_type]
    
    def deploy_model(self, 
                    model_name: str, 
                    model_version: str,
                    deployment_type: Optional[str] = None) -> str:
        """
        部署模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            deployment_type: 部署类型
            
        Returns:
            部署ID
        """
        if deployment_type is None:
            deployment_type = self.config.deployment_type
        
        if deployment_type not in self.deployments:
            raise ValueError(f"Deployment type {deployment_type} not available")
        
        deployment = self.deployments[deployment_type]
        return deployment.deploy(model_name, model_version)
    
    def undeploy_model(self, deployment_id: str) -> bool:
        """
        取消部署
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            是否成功
        """
        return self.current_deployment.undeploy(deployment_id)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """
        获取部署状态
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            部署状态
        """
        return self.current_deployment.get_status(deployment_id)
    
    def list_all_deployments(self) -> List[DeploymentStatus]:
        """
        列出所有部署
        
        Returns:
            部署状态列表
        """
        return self.current_deployment.list_deployments()
    
    def health_check_all(self) -> Dict[str, bool]:
        """
        检查所有部署的健康状态
        
        Returns:
            部署ID -> 健康状态的字典
        """
        results = {}
        for deployment in self.current_deployment.list_deployments():
            results[deployment.deployment_id] = self.current_deployment.health_check(
                deployment.deployment_id
            )
        return results


def create_deployment_manager(config: DeploymentConfig) -> DeploymentManager:
    """
    创建部署管理器
    
    Args:
        config: 部署配置
        
    Returns:
        部署管理器实例
    """
    return DeploymentManager(config)