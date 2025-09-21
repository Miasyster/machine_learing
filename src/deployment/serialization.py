"""
模型序列化模块

提供模型序列化、反序列化和管理功能
"""

import pickle
import joblib
import json
import gzip
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime
import hashlib
import numpy as np
from sklearn.base import BaseEstimator

from .base import ModelMetadata, DeploymentError

logger = logging.getLogger(__name__)


class ModelSerializer:
    """模型序列化器"""
    
    SUPPORTED_FORMATS = ['pickle', 'joblib', 'json']
    
    def __init__(self, compression: bool = True, compression_level: int = 6):
        """
        初始化序列化器
        
        Args:
            compression: 是否启用压缩
            compression_level: 压缩级别 (1-9)
        """
        self.compression = compression
        self.compression_level = compression_level
    
    def serialize_model(self, 
                       model: Any, 
                       output_path: str, 
                       format: str = 'joblib',
                       metadata: Optional[ModelMetadata] = None) -> Dict[str, Any]:
        """
        序列化模型
        
        Args:
            model: 要序列化的模型
            output_path: 输出路径
            format: 序列化格式
            metadata: 模型元数据
            
        Returns:
            序列化信息
        """
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 序列化模型
        start_time = datetime.now()
        
        try:
            if format == 'pickle':
                self._serialize_pickle(model, output_path)
            elif format == 'joblib':
                self._serialize_joblib(model, output_path)
            elif format == 'json':
                self._serialize_json(model, output_path)
            
            end_time = datetime.now()
            serialization_time = (end_time - start_time).total_seconds()
            
            # 计算文件信息
            file_size = output_path.stat().st_size
            file_hash = self._calculate_file_hash(output_path)
            
            # 保存元数据
            if metadata:
                metadata.model_size = file_size / (1024 * 1024)  # MB
                metadata.updated_at = datetime.now()
                self._save_metadata(metadata, output_path.parent / f"{output_path.stem}_metadata.json")
            
            serialization_info = {
                'model_path': str(output_path),
                'format': format,
                'file_size': file_size,
                'file_hash': file_hash,
                'serialization_time': serialization_time,
                'compression': self.compression,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Model serialized successfully: {serialization_info}")
            return serialization_info
            
        except Exception as e:
            logger.error(f"Failed to serialize model: {e}")
            raise DeploymentError(f"Serialization failed: {e}")
    
    def _serialize_pickle(self, model: Any, output_path: Path) -> None:
        """使用pickle序列化"""
        if self.compression:
            with gzip.open(f"{output_path}.pkl.gz", 'wb', compresslevel=self.compression_level) as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(f"{output_path}.pkl", 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _serialize_joblib(self, model: Any, output_path: Path) -> None:
        """使用joblib序列化"""
        if self.compression:
            joblib.dump(model, f"{output_path}.joblib.gz", compress=self.compression_level)
        else:
            joblib.dump(model, f"{output_path}.joblib")
    
    def _serialize_json(self, model: Any, output_path: Path) -> None:
        """使用JSON序列化（仅支持简单模型）"""
        if hasattr(model, 'get_params') and hasattr(model, '__class__'):
            model_data = {
                'class_name': model.__class__.__name__,
                'module': model.__class__.__module__,
                'params': model.get_params(),
                'attributes': {}
            }
            
            # 保存重要属性
            for attr in ['coef_', 'intercept_', 'feature_importances_', 'classes_']:
                if hasattr(model, attr):
                    value = getattr(model, attr)
                    if isinstance(value, np.ndarray):
                        model_data['attributes'][attr] = value.tolist()
                    else:
                        model_data['attributes'][attr] = value
            
            with open(f"{output_path}.json", 'w') as f:
                json.dump(model_data, f, indent=2)
        else:
            raise ValueError("Model does not support JSON serialization")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _save_metadata(self, metadata: ModelMetadata, metadata_path: Path) -> None:
        """保存元数据"""
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)


class ModelLoader:
    """模型加载器"""
    
    def __init__(self):
        """初始化加载器"""
        pass
    
    def load_model(self, model_path: str, format: Optional[str] = None) -> Tuple[Any, Optional[ModelMetadata]]:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            format: 文件格式（自动检测如果为None）
            
        Returns:
            (模型对象, 元数据)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 自动检测格式
        if format is None:
            format = self._detect_format(model_path)
        
        try:
            # 加载模型
            if format == 'pickle':
                model = self._load_pickle(model_path)
            elif format == 'joblib':
                model = self._load_joblib(model_path)
            elif format == 'json':
                model = self._load_json(model_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # 加载元数据
            metadata = self._load_metadata(model_path)
            
            logger.info(f"Model loaded successfully from {model_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise DeploymentError(f"Model loading failed: {e}")
    
    def _detect_format(self, model_path: Path) -> str:
        """检测文件格式"""
        suffix = model_path.suffix.lower()
        
        if suffix in ['.pkl', '.pickle']:
            return 'pickle'
        elif suffix == '.gz':
            # 检查压缩文件的内部格式
            if '.pkl' in model_path.name or '.pickle' in model_path.name:
                return 'pickle'
            elif '.joblib' in model_path.name:
                return 'joblib'
        elif suffix == '.joblib':
            return 'joblib'
        elif suffix == '.json':
            return 'json'
        else:
            # 默认尝试joblib
            return 'joblib'
    
    def _load_pickle(self, model_path: Path) -> Any:
        """加载pickle文件"""
        if model_path.suffix == '.gz':
            with gzip.open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
    
    def _load_joblib(self, model_path: Path) -> Any:
        """加载joblib文件"""
        return joblib.load(model_path)
    
    def _load_json(self, model_path: Path) -> Any:
        """加载JSON文件"""
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # 重建模型（简化版本）
        class_name = model_data['class_name']
        module_name = model_data['module']
        
        # 动态导入模块
        import importlib
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        
        # 创建模型实例
        model = model_class(**model_data['params'])
        
        # 设置属性
        for attr, value in model_data['attributes'].items():
            if isinstance(value, list):
                setattr(model, attr, np.array(value))
            else:
                setattr(model, attr, value)
        
        return model
    
    def _load_metadata(self, model_path: Path) -> Optional[ModelMetadata]:
        """加载元数据"""
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                return ModelMetadata.from_dict(metadata_dict)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return None


class SerializationManager:
    """序列化管理器"""
    
    def __init__(self, storage_path: str = "models"):
        """
        初始化管理器
        
        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.serializer = ModelSerializer()
        self.loader = ModelLoader()
        
        # 模型注册表
        self.registry_path = self.storage_path / "registry.json"
        self.registry = self._load_registry()
    
    def save_model(self, 
                   model: Any, 
                   model_name: str, 
                   version: str,
                   metadata: Optional[ModelMetadata] = None,
                   format: str = 'joblib') -> str:
        """
        保存模型
        
        Args:
            model: 模型对象
            model_name: 模型名称
            version: 版本号
            metadata: 模型元数据
            format: 序列化格式
            
        Returns:
            模型路径
        """
        # 创建模型目录
        model_dir = self.storage_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 序列化模型
        model_path = model_dir / "model"
        serialization_info = self.serializer.serialize_model(
            model, model_path, format, metadata
        )
        
        # 更新注册表
        self._update_registry(model_name, version, serialization_info, metadata)
        
        logger.info(f"Model saved: {model_name}:{version}")
        return serialization_info['model_path']
    
    def load_model(self, model_name: str, version: str = "latest") -> Tuple[Any, Optional[ModelMetadata]]:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            version: 版本号
            
        Returns:
            (模型对象, 元数据)
        """
        # 获取实际版本
        if version == "latest":
            version = self._get_latest_version(model_name)
        
        # 获取模型路径
        model_info = self.registry.get(model_name, {}).get(version)
        if not model_info:
            raise ValueError(f"Model not found: {model_name}:{version}")
        
        model_path = model_info['model_path']
        return self.loader.load_model(model_path)
    
    def list_models(self) -> Dict[str, List[str]]:
        """
        列出所有模型
        
        Returns:
            模型名称和版本的字典
        """
        return {name: list(versions.keys()) for name, versions in self.registry.items()}
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """
        删除模型
        
        Args:
            model_name: 模型名称
            version: 版本号（None表示删除所有版本）
            
        Returns:
            是否成功
        """
        try:
            if version is None:
                # 删除所有版本
                model_dir = self.storage_path / model_name
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                
                if model_name in self.registry:
                    del self.registry[model_name]
            else:
                # 删除特定版本
                version_dir = self.storage_path / model_name / version
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                
                if model_name in self.registry and version in self.registry[model_name]:
                    del self.registry[model_name][version]
                    
                    # 如果没有版本了，删除模型条目
                    if not self.registry[model_name]:
                        del self.registry[model_name]
            
            self._save_registry()
            logger.info(f"Model deleted: {model_name}:{version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def get_model_info(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            version: 版本号
            
        Returns:
            模型信息
        """
        if version == "latest":
            version = self._get_latest_version(model_name)
        
        model_info = self.registry.get(model_name, {}).get(version)
        if not model_info:
            raise ValueError(f"Model not found: {model_name}:{version}")
        
        return model_info.copy()
    
    def copy_model(self, 
                   source_name: str, 
                   source_version: str,
                   target_name: str, 
                   target_version: str) -> bool:
        """
        复制模型
        
        Args:
            source_name: 源模型名称
            source_version: 源版本号
            target_name: 目标模型名称
            target_version: 目标版本号
            
        Returns:
            是否成功
        """
        try:
            # 加载源模型
            model, metadata = self.load_model(source_name, source_version)
            
            # 更新元数据
            if metadata:
                metadata.model_name = target_name
                metadata.model_version = target_version
                metadata.parent_version = f"{source_name}:{source_version}"
                metadata.created_at = datetime.now()
            
            # 保存为新模型
            source_info = self.get_model_info(source_name, source_version)
            format = source_info.get('format', 'joblib')
            
            self.save_model(model, target_name, target_version, metadata, format)
            
            logger.info(f"Model copied: {source_name}:{source_version} -> {target_name}:{target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy model: {e}")
            return False
    
    def _load_registry(self) -> Dict[str, Any]:
        """加载注册表"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
        
        return {}
    
    def _save_registry(self) -> None:
        """保存注册表"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _update_registry(self, 
                        model_name: str, 
                        version: str, 
                        serialization_info: Dict[str, Any],
                        metadata: Optional[ModelMetadata]) -> None:
        """更新注册表"""
        if model_name not in self.registry:
            self.registry[model_name] = {}
        
        registry_entry = serialization_info.copy()
        if metadata:
            registry_entry['metadata'] = metadata.to_dict()
        
        self.registry[model_name][version] = registry_entry
        self._save_registry()
    
    def _get_latest_version(self, model_name: str) -> str:
        """获取最新版本"""
        if model_name not in self.registry:
            raise ValueError(f"Model not found: {model_name}")
        
        versions = list(self.registry[model_name].keys())
        if not versions:
            raise ValueError(f"No versions found for model: {model_name}")
        
        # 简单的版本排序（可以改进为语义版本排序）
        versions.sort(reverse=True)
        return versions[0]


def validate_model_compatibility(model: Any) -> List[str]:
    """
    验证模型兼容性
    
    Args:
        model: 模型对象
        
    Returns:
        兼容性问题列表
    """
    issues = []
    
    # 检查是否有predict方法
    if not hasattr(model, 'predict'):
        issues.append("Model does not have predict method")
    
    # 检查是否是sklearn兼容的
    if isinstance(model, BaseEstimator):
        # 检查是否已训练
        if hasattr(model, 'fit') and not hasattr(model, 'classes_') and not hasattr(model, 'coef_'):
            issues.append("Model appears to be untrained")
    
    # 检查序列化兼容性
    try:
        pickle.dumps(model)
    except Exception as e:
        issues.append(f"Model is not pickle serializable: {e}")
    
    return issues


def optimize_model_size(model: Any) -> Any:
    """
    优化模型大小
    
    Args:
        model: 模型对象
        
    Returns:
        优化后的模型
    """
    # 这里可以实现模型压缩、量化等优化技术
    # 目前返回原模型
    logger.info("Model size optimization not implemented yet")
    return model