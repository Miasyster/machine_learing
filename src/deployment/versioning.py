"""
模型版本管理模块

提供模型版本控制、回滚和历史管理功能
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import hashlib
import git
from packaging import version

from .base import ModelMetadata, DeploymentError

logger = logging.getLogger(__name__)


@dataclass
class VersionInfo:
    """版本信息"""
    version: str
    created_at: datetime
    created_by: str
    description: str
    parent_version: Optional[str] = None
    tags: List[str] = None
    metrics: Dict[str, float] = None
    model_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionInfo':
        """从字典创建"""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self, repository_path: str = "model_repository"):
        """
        初始化版本管理器
        
        Args:
            repository_path: 仓库路径
        """
        self.repository_path = Path(repository_path)
        self.repository_path.mkdir(parents=True, exist_ok=True)
        
        # 版本信息文件
        self.versions_file = self.repository_path / "versions.json"
        self.versions = self._load_versions()
        
        # Git仓库（可选）
        self.git_repo = None
        self._init_git_repo()
    
    def create_version(self, 
                      model_name: str,
                      version_number: str,
                      model_path: str,
                      description: str = "",
                      created_by: str = "system",
                      parent_version: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      metrics: Optional[Dict[str, float]] = None) -> VersionInfo:
        """
        创建新版本
        
        Args:
            model_name: 模型名称
            version_number: 版本号
            model_path: 模型文件路径
            description: 版本描述
            created_by: 创建者
            parent_version: 父版本
            tags: 标签
            metrics: 性能指标
            
        Returns:
            版本信息
        """
        # 验证版本号
        self._validate_version(version_number)
        
        # 检查版本是否已存在
        if self._version_exists(model_name, version_number):
            raise ValueError(f"Version {version_number} already exists for model {model_name}")
        
        # 计算模型哈希
        model_hash = self._calculate_model_hash(model_path)
        
        # 创建版本目录
        version_dir = self._get_version_dir(model_name, version_number)
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件
        model_file_path = Path(model_path)
        target_path = version_dir / model_file_path.name
        shutil.copy2(model_path, target_path)
        
        # 创建版本信息
        version_info = VersionInfo(
            version=version_number,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            parent_version=parent_version,
            tags=tags or [],
            metrics=metrics or {},
            model_hash=model_hash
        )
        
        # 保存版本信息
        self._save_version_info(model_name, version_info)
        
        # Git提交（如果启用）
        if self.git_repo:
            self._git_commit(model_name, version_number, description)
        
        logger.info(f"Created version {version_number} for model {model_name}")
        return version_info
    
    def get_version(self, model_name: str, version_number: str) -> Optional[VersionInfo]:
        """
        获取版本信息
        
        Args:
            model_name: 模型名称
            version_number: 版本号
            
        Returns:
            版本信息
        """
        model_versions = self.versions.get(model_name, {})
        version_data = model_versions.get(version_number)
        
        if version_data:
            return VersionInfo.from_dict(version_data)
        return None
    
    def list_versions(self, model_name: str) -> List[VersionInfo]:
        """
        列出模型的所有版本
        
        Args:
            model_name: 模型名称
            
        Returns:
            版本信息列表
        """
        model_versions = self.versions.get(model_name, {})
        versions = []
        
        for version_data in model_versions.values():
            versions.append(VersionInfo.from_dict(version_data))
        
        # 按创建时间排序
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions
    
    def get_latest_version(self, model_name: str) -> Optional[VersionInfo]:
        """
        获取最新版本
        
        Args:
            model_name: 模型名称
            
        Returns:
            最新版本信息
        """
        versions = self.list_versions(model_name)
        return versions[0] if versions else None
    
    def delete_version(self, model_name: str, version_number: str) -> bool:
        """
        删除版本
        
        Args:
            model_name: 模型名称
            version_number: 版本号
            
        Returns:
            是否成功
        """
        try:
            # 检查版本是否存在
            if not self._version_exists(model_name, version_number):
                logger.warning(f"Version {version_number} does not exist for model {model_name}")
                return False
            
            # 删除版本目录
            version_dir = self._get_version_dir(model_name, version_number)
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            # 从版本记录中删除
            if model_name in self.versions and version_number in self.versions[model_name]:
                del self.versions[model_name][version_number]
                
                # 如果模型没有版本了，删除模型条目
                if not self.versions[model_name]:
                    del self.versions[model_name]
            
            self._save_versions()
            
            logger.info(f"Deleted version {version_number} for model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            return False
    
    def tag_version(self, model_name: str, version_number: str, tag: str) -> bool:
        """
        为版本添加标签
        
        Args:
            model_name: 模型名称
            version_number: 版本号
            tag: 标签
            
        Returns:
            是否成功
        """
        version_info = self.get_version(model_name, version_number)
        if not version_info:
            return False
        
        if tag not in version_info.tags:
            version_info.tags.append(tag)
            self._save_version_info(model_name, version_info)
            logger.info(f"Added tag '{tag}' to version {version_number} of model {model_name}")
        
        return True
    
    def remove_tag(self, model_name: str, version_number: str, tag: str) -> bool:
        """
        移除版本标签
        
        Args:
            model_name: 模型名称
            version_number: 版本号
            tag: 标签
            
        Returns:
            是否成功
        """
        version_info = self.get_version(model_name, version_number)
        if not version_info:
            return False
        
        if tag in version_info.tags:
            version_info.tags.remove(tag)
            self._save_version_info(model_name, version_info)
            logger.info(f"Removed tag '{tag}' from version {version_number} of model {model_name}")
        
        return True
    
    def find_versions_by_tag(self, tag: str) -> List[Tuple[str, VersionInfo]]:
        """
        根据标签查找版本
        
        Args:
            tag: 标签
            
        Returns:
            (模型名称, 版本信息) 列表
        """
        results = []
        
        for model_name in self.versions:
            for version_info in self.list_versions(model_name):
                if tag in version_info.tags:
                    results.append((model_name, version_info))
        
        return results
    
    def compare_versions(self, 
                        model_name: str, 
                        version1: str, 
                        version2: str) -> Dict[str, Any]:
        """
        比较两个版本
        
        Args:
            model_name: 模型名称
            version1: 版本1
            version2: 版本2
            
        Returns:
            比较结果
        """
        v1_info = self.get_version(model_name, version1)
        v2_info = self.get_version(model_name, version2)
        
        if not v1_info or not v2_info:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'model_name': model_name,
            'version1': {
                'version': v1_info.version,
                'created_at': v1_info.created_at,
                'description': v1_info.description,
                'metrics': v1_info.metrics,
                'tags': v1_info.tags
            },
            'version2': {
                'version': v2_info.version,
                'created_at': v2_info.created_at,
                'description': v2_info.description,
                'metrics': v2_info.metrics,
                'tags': v2_info.tags
            },
            'metrics_diff': {},
            'time_diff': (v2_info.created_at - v1_info.created_at).total_seconds()
        }
        
        # 计算指标差异
        all_metrics = set(v1_info.metrics.keys()) | set(v2_info.metrics.keys())
        for metric in all_metrics:
            val1 = v1_info.metrics.get(metric, 0)
            val2 = v2_info.metrics.get(metric, 0)
            comparison['metrics_diff'][metric] = val2 - val1
        
        return comparison
    
    def get_version_history(self, model_name: str) -> List[Dict[str, Any]]:
        """
        获取版本历史
        
        Args:
            model_name: 模型名称
            
        Returns:
            版本历史列表
        """
        versions = self.list_versions(model_name)
        history = []
        
        for version_info in versions:
            history.append({
                'version': version_info.version,
                'created_at': version_info.created_at,
                'created_by': version_info.created_by,
                'description': version_info.description,
                'parent_version': version_info.parent_version,
                'tags': version_info.tags,
                'metrics': version_info.metrics
            })
        
        return history
    
    def rollback_to_version(self, 
                           model_name: str, 
                           target_version: str,
                           new_version: Optional[str] = None) -> VersionInfo:
        """
        回滚到指定版本
        
        Args:
            model_name: 模型名称
            target_version: 目标版本
            new_version: 新版本号（如果不指定则自动生成）
            
        Returns:
            新版本信息
        """
        # 检查目标版本是否存在
        target_info = self.get_version(model_name, target_version)
        if not target_info:
            raise ValueError(f"Target version {target_version} not found")
        
        # 生成新版本号
        if new_version is None:
            latest_version = self.get_latest_version(model_name)
            if latest_version:
                new_version = self._increment_version(latest_version.version)
            else:
                new_version = "1.0.0"
        
        # 获取目标版本的模型文件
        target_dir = self._get_version_dir(model_name, target_version)
        model_files = list(target_dir.glob("*"))
        
        if not model_files:
            raise ValueError(f"No model files found in version {target_version}")
        
        # 创建回滚版本
        model_file = model_files[0]  # 假设只有一个模型文件
        
        rollback_info = self.create_version(
            model_name=model_name,
            version_number=new_version,
            model_path=str(model_file),
            description=f"Rollback to version {target_version}",
            parent_version=target_version,
            tags=["rollback"],
            metrics=target_info.metrics.copy()
        )
        
        logger.info(f"Rolled back model {model_name} to version {target_version} as {new_version}")
        return rollback_info
    
    def export_version(self, 
                      model_name: str, 
                      version_number: str, 
                      export_path: str) -> bool:
        """
        导出版本
        
        Args:
            model_name: 模型名称
            version_number: 版本号
            export_path: 导出路径
            
        Returns:
            是否成功
        """
        try:
            version_info = self.get_version(model_name, version_number)
            if not version_info:
                return False
            
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # 复制版本目录
            version_dir = self._get_version_dir(model_name, version_number)
            target_dir = export_path / f"{model_name}_{version_number}"
            
            shutil.copytree(version_dir, target_dir, dirs_exist_ok=True)
            
            # 保存版本信息
            version_info_file = target_dir / "version_info.json"
            with open(version_info_file, 'w') as f:
                json.dump(version_info.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Exported version {version_number} of model {model_name} to {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export version: {e}")
            return False
    
    def import_version(self, import_path: str) -> bool:
        """
        导入版本
        
        Args:
            import_path: 导入路径
            
        Returns:
            是否成功
        """
        try:
            import_path = Path(import_path)
            version_info_file = import_path / "version_info.json"
            
            if not version_info_file.exists():
                raise ValueError("version_info.json not found in import path")
            
            # 加载版本信息
            with open(version_info_file, 'r') as f:
                version_data = json.load(f)
            
            version_info = VersionInfo.from_dict(version_data)
            
            # 提取模型名称和版本号
            dir_name = import_path.name
            if '_' in dir_name:
                model_name, version_number = dir_name.rsplit('_', 1)
            else:
                raise ValueError("Invalid import directory name format")
            
            # 检查版本是否已存在
            if self._version_exists(model_name, version_number):
                raise ValueError(f"Version {version_number} already exists for model {model_name}")
            
            # 复制到版本目录
            target_dir = self._get_version_dir(model_name, version_number)
            shutil.copytree(import_path, target_dir, dirs_exist_ok=True)
            
            # 移除版本信息文件（已经在内存中）
            (target_dir / "version_info.json").unlink(missing_ok=True)
            
            # 保存版本信息
            self._save_version_info(model_name, version_info)
            
            logger.info(f"Imported version {version_number} of model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import version: {e}")
            return False
    
    def _validate_version(self, version_number: str) -> None:
        """验证版本号格式"""
        try:
            version.parse(version_number)
        except Exception:
            raise ValueError(f"Invalid version format: {version_number}")
    
    def _version_exists(self, model_name: str, version_number: str) -> bool:
        """检查版本是否存在"""
        return (model_name in self.versions and 
                version_number in self.versions[model_name])
    
    def _get_version_dir(self, model_name: str, version_number: str) -> Path:
        """获取版本目录路径"""
        return self.repository_path / model_name / version_number
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """计算模型文件哈希"""
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _load_versions(self) -> Dict[str, Any]:
        """加载版本信息"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load versions: {e}")
        
        return {}
    
    def _save_versions(self) -> None:
        """保存版本信息"""
        try:
            with open(self.versions_file, 'w') as f:
                json.dump(self.versions, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
    
    def _save_version_info(self, model_name: str, version_info: VersionInfo) -> None:
        """保存单个版本信息"""
        if model_name not in self.versions:
            self.versions[model_name] = {}
        
        self.versions[model_name][version_info.version] = version_info.to_dict()
        self._save_versions()
    
    def _increment_version(self, current_version: str) -> str:
        """递增版本号"""
        try:
            v = version.parse(current_version)
            # 简单递增补丁版本
            if hasattr(v, 'micro'):
                return f"{v.major}.{v.minor}.{v.micro + 1}"
            else:
                return f"{current_version}.1"
        except Exception:
            # 如果解析失败，简单添加后缀
            return f"{current_version}.1"
    
    def _init_git_repo(self) -> None:
        """初始化Git仓库"""
        try:
            git_dir = self.repository_path / ".git"
            if git_dir.exists():
                self.git_repo = git.Repo(self.repository_path)
            else:
                self.git_repo = git.Repo.init(self.repository_path)
                
            logger.info("Git repository initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Git repository: {e}")
            self.git_repo = None
    
    def _git_commit(self, model_name: str, version_number: str, description: str) -> None:
        """Git提交"""
        if not self.git_repo:
            return
        
        try:
            # 添加所有文件
            self.git_repo.git.add(A=True)
            
            # 提交
            commit_message = f"Add {model_name} version {version_number}: {description}"
            self.git_repo.index.commit(commit_message)
            
            # 创建标签
            tag_name = f"{model_name}-{version_number}"
            self.git_repo.create_tag(tag_name, message=description)
            
            logger.info(f"Git commit created: {commit_message}")
        except Exception as e:
            logger.warning(f"Failed to create Git commit: {e}")


class VersioningStrategy:
    """版本控制策略"""
    
    @staticmethod
    def semantic_versioning(current_version: str, change_type: str) -> str:
        """
        语义化版本控制
        
        Args:
            current_version: 当前版本
            change_type: 变更类型 (major, minor, patch)
            
        Returns:
            新版本号
        """
        try:
            v = version.parse(current_version)
            
            if change_type == "major":
                return f"{v.major + 1}.0.0"
            elif change_type == "minor":
                return f"{v.major}.{v.minor + 1}.0"
            elif change_type == "patch":
                return f"{v.major}.{v.minor}.{v.micro + 1}"
            else:
                raise ValueError(f"Invalid change type: {change_type}")
                
        except Exception as e:
            raise ValueError(f"Failed to increment version: {e}")
    
    @staticmethod
    def timestamp_versioning() -> str:
        """
        时间戳版本控制
        
        Returns:
            时间戳版本号
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def auto_versioning(current_version: str, metrics: Dict[str, float]) -> str:
        """
        自动版本控制（基于性能指标）
        
        Args:
            current_version: 当前版本
            metrics: 性能指标
            
        Returns:
            新版本号
        """
        # 简单的自动版本控制逻辑
        # 可以根据性能提升程度决定版本类型
        
        accuracy_improvement = metrics.get('accuracy_improvement', 0)
        
        if accuracy_improvement > 0.1:  # 10%以上提升
            return VersioningStrategy.semantic_versioning(current_version, "major")
        elif accuracy_improvement > 0.05:  # 5%以上提升
            return VersioningStrategy.semantic_versioning(current_version, "minor")
        else:
            return VersioningStrategy.semantic_versioning(current_version, "patch")


def create_version_manager(repository_path: str = "model_repository") -> ModelVersionManager:
    """
    创建版本管理器
    
    Args:
        repository_path: 仓库路径
        
    Returns:
        版本管理器实例
    """
    return ModelVersionManager(repository_path)