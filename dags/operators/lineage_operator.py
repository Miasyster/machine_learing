"""
数据血缘追踪和版本管理操作符
提供数据血缘关系追踪、版本管理和数据治理功能
"""

import os
import sys
import json
import logging
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import sqlite3
import pickle

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.models import Variable
from airflow.exceptions import AirflowException

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DataLineageTracker:
    """
    数据血缘追踪器
    
    负责记录和管理数据的血缘关系
    """
    
    def __init__(self, lineage_db_path: str):
        """
        初始化数据血缘追踪器
        
        Args:
            lineage_db_path: 血缘数据库路径
        """
        self.lineage_db_path = lineage_db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化血缘数据库"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.lineage_db_path), exist_ok=True)
            
            with sqlite3.connect(self.lineage_db_path) as conn:
                cursor = conn.cursor()
                
                # 创建数据资产表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_assets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        asset_id TEXT UNIQUE NOT NULL,
                        asset_name TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        asset_path TEXT,
                        schema_info TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 创建血缘关系表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS lineage_relationships (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_asset_id TEXT NOT NULL,
                        target_asset_id TEXT NOT NULL,
                        relationship_type TEXT NOT NULL,
                        transformation_info TEXT,
                        dag_id TEXT,
                        task_id TEXT,
                        execution_date TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_asset_id) REFERENCES data_assets (asset_id),
                        FOREIGN KEY (target_asset_id) REFERENCES data_assets (asset_id)
                    )
                """)
                
                # 创建数据版本表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        asset_id TEXT NOT NULL,
                        version TEXT NOT NULL,
                        version_hash TEXT NOT NULL,
                        file_path TEXT,
                        file_size INTEGER,
                        row_count INTEGER,
                        column_count INTEGER,
                        checksum TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (asset_id) REFERENCES data_assets (asset_id),
                        UNIQUE(asset_id, version)
                    )
                """)
                
                # 创建数据质量历史表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quality_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        asset_id TEXT NOT NULL,
                        version TEXT,
                        quality_score REAL,
                        quality_metrics TEXT,
                        issues TEXT,
                        checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (asset_id) REFERENCES data_assets (asset_id)
                    )
                """)
                
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_source ON lineage_relationships (source_asset_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_target ON lineage_relationships (target_asset_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_asset ON data_versions (asset_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_asset ON quality_history (asset_id)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"初始化血缘数据库失败: {str(e)}")
            raise
    
    def register_data_asset(self, asset_id: str, asset_name: str, asset_type: str, 
                          asset_path: str = None, schema_info: Dict = None, 
                          metadata: Dict = None) -> None:
        """
        注册数据资产
        
        Args:
            asset_id: 资产唯一标识
            asset_name: 资产名称
            asset_type: 资产类型（raw_data, processed_data, features, model等）
            asset_path: 资产文件路径
            schema_info: 数据模式信息
            metadata: 元数据
        """
        try:
            with sqlite3.connect(self.lineage_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO data_assets 
                    (asset_id, asset_name, asset_type, asset_path, schema_info, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    asset_id,
                    asset_name,
                    asset_type,
                    asset_path,
                    json.dumps(schema_info) if schema_info else None,
                    json.dumps(metadata) if metadata else None
                ))
                
                conn.commit()
                self.logger.info(f"数据资产已注册: {asset_id}")
                
        except Exception as e:
            self.logger.error(f"注册数据资产失败 {asset_id}: {str(e)}")
            raise
    
    def record_lineage(self, source_asset_id: str, target_asset_id: str, 
                      relationship_type: str, transformation_info: Dict = None,
                      dag_id: str = None, task_id: str = None, 
                      execution_date: datetime = None) -> None:
        """
        记录血缘关系
        
        Args:
            source_asset_id: 源数据资产ID
            target_asset_id: 目标数据资产ID
            relationship_type: 关系类型（transform, derive, aggregate等）
            transformation_info: 转换信息
            dag_id: DAG ID
            task_id: 任务ID
            execution_date: 执行日期
        """
        try:
            with sqlite3.connect(self.lineage_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO lineage_relationships 
                    (source_asset_id, target_asset_id, relationship_type, transformation_info, 
                     dag_id, task_id, execution_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    source_asset_id,
                    target_asset_id,
                    relationship_type,
                    json.dumps(transformation_info) if transformation_info else None,
                    dag_id,
                    task_id,
                    execution_date
                ))
                
                conn.commit()
                self.logger.info(f"血缘关系已记录: {source_asset_id} -> {target_asset_id}")
                
        except Exception as e:
            self.logger.error(f"记录血缘关系失败: {str(e)}")
            raise
    
    def create_data_version(self, asset_id: str, version: str, file_path: str,
                          metadata: Dict = None) -> str:
        """
        创建数据版本
        
        Args:
            asset_id: 数据资产ID
            version: 版本号
            file_path: 文件路径
            metadata: 版本元数据
            
        Returns:
            str: 版本哈希值
        """
        try:
            # 计算文件哈希
            version_hash = self._calculate_file_hash(file_path)
            
            # 获取文件信息
            file_stats = self._get_file_stats(file_path)
            
            with sqlite3.connect(self.lineage_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO data_versions 
                    (asset_id, version, version_hash, file_path, file_size, 
                     row_count, column_count, checksum, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    asset_id,
                    version,
                    version_hash,
                    file_path,
                    file_stats.get('file_size'),
                    file_stats.get('row_count'),
                    file_stats.get('column_count'),
                    file_stats.get('checksum'),
                    json.dumps(metadata) if metadata else None
                ))
                
                conn.commit()
                self.logger.info(f"数据版本已创建: {asset_id} v{version}")
                
            return version_hash
            
        except Exception as e:
            self.logger.error(f"创建数据版本失败: {str(e)}")
            raise
    
    def record_quality_metrics(self, asset_id: str, version: str, 
                             quality_score: float, quality_metrics: Dict,
                             issues: List[str] = None) -> None:
        """
        记录数据质量指标
        
        Args:
            asset_id: 数据资产ID
            version: 版本号
            quality_score: 质量分数
            quality_metrics: 质量指标详情
            issues: 质量问题列表
        """
        try:
            with sqlite3.connect(self.lineage_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO quality_history 
                    (asset_id, version, quality_score, quality_metrics, issues)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    asset_id,
                    version,
                    quality_score,
                    json.dumps(quality_metrics),
                    json.dumps(issues) if issues else None
                ))
                
                conn.commit()
                self.logger.info(f"质量指标已记录: {asset_id} v{version}")
                
        except Exception as e:
            self.logger.error(f"记录质量指标失败: {str(e)}")
            raise
    
    def get_lineage_graph(self, asset_id: str, direction: str = 'both', 
                         max_depth: int = 5) -> Dict[str, Any]:
        """
        获取数据血缘图
        
        Args:
            asset_id: 数据资产ID
            direction: 方向（upstream, downstream, both）
            max_depth: 最大深度
            
        Returns:
            Dict: 血缘图数据
        """
        try:
            with sqlite3.connect(self.lineage_db_path) as conn:
                cursor = conn.cursor()
                
                nodes = {}
                edges = []
                visited = set()
                
                def traverse(current_id: str, current_depth: int, traverse_direction: str):
                    if current_depth > max_depth or current_id in visited:
                        return
                    
                    visited.add(current_id)
                    
                    # 获取节点信息
                    cursor.execute("""
                        SELECT asset_name, asset_type, asset_path, metadata
                        FROM data_assets WHERE asset_id = ?
                    """, (current_id,))
                    
                    asset_info = cursor.fetchone()
                    if asset_info:
                        nodes[current_id] = {
                            'id': current_id,
                            'name': asset_info[0],
                            'type': asset_info[1],
                            'path': asset_info[2],
                            'metadata': json.loads(asset_info[3]) if asset_info[3] else {}
                        }
                    
                    # 获取关系
                    if traverse_direction in ['upstream', 'both']:
                        cursor.execute("""
                            SELECT source_asset_id, relationship_type, transformation_info
                            FROM lineage_relationships WHERE target_asset_id = ?
                        """, (current_id,))
                        
                        for source_id, rel_type, transform_info in cursor.fetchall():
                            edges.append({
                                'source': source_id,
                                'target': current_id,
                                'type': rel_type,
                                'transformation': json.loads(transform_info) if transform_info else {}
                            })
                            traverse(source_id, current_depth + 1, 'upstream')
                    
                    if traverse_direction in ['downstream', 'both']:
                        cursor.execute("""
                            SELECT target_asset_id, relationship_type, transformation_info
                            FROM lineage_relationships WHERE source_asset_id = ?
                        """, (current_id,))
                        
                        for target_id, rel_type, transform_info in cursor.fetchall():
                            edges.append({
                                'source': current_id,
                                'target': target_id,
                                'type': rel_type,
                                'transformation': json.loads(transform_info) if transform_info else {}
                            })
                            traverse(target_id, current_depth + 1, 'downstream')
                
                # 开始遍历
                traverse(asset_id, 0, direction)
                
                return {
                    'nodes': list(nodes.values()),
                    'edges': edges,
                    'root_asset': asset_id,
                    'direction': direction,
                    'max_depth': max_depth
                }
                
        except Exception as e:
            self.logger.error(f"获取血缘图失败: {str(e)}")
            raise
    
    def get_version_history(self, asset_id: str) -> List[Dict[str, Any]]:
        """
        获取版本历史
        
        Args:
            asset_id: 数据资产ID
            
        Returns:
            List: 版本历史列表
        """
        try:
            with sqlite3.connect(self.lineage_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT version, version_hash, file_path, file_size, 
                           row_count, column_count, checksum, metadata, created_at
                    FROM data_versions 
                    WHERE asset_id = ? 
                    ORDER BY created_at DESC
                """, (asset_id,))
                
                versions = []
                for row in cursor.fetchall():
                    versions.append({
                        'version': row[0],
                        'version_hash': row[1],
                        'file_path': row[2],
                        'file_size': row[3],
                        'row_count': row[4],
                        'column_count': row[5],
                        'checksum': row[6],
                        'metadata': json.loads(row[7]) if row[7] else {},
                        'created_at': row[8]
                    })
                
                return versions
                
        except Exception as e:
            self.logger.error(f"获取版本历史失败: {str(e)}")
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.warning(f"计算文件哈希失败 {file_path}: {str(e)}")
            return ""
    
    def _get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """获取文件统计信息"""
        stats = {}
        
        try:
            # 基本文件信息
            file_stat = os.stat(file_path)
            stats['file_size'] = file_stat.st_size
            
            # 如果是CSV文件，获取行列数
            if file_path.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path, nrows=1)  # 只读第一行获取列数
                    stats['column_count'] = len(df.columns)
                    
                    # 获取行数（更高效的方法）
                    with open(file_path, 'r') as f:
                        stats['row_count'] = sum(1 for line in f) - 1  # 减去标题行
                        
                except Exception as e:
                    self.logger.warning(f"获取CSV文件统计信息失败: {str(e)}")
            
            # 计算校验和
            stats['checksum'] = self._calculate_file_hash(file_path)
            
        except Exception as e:
            self.logger.warning(f"获取文件统计信息失败 {file_path}: {str(e)}")
        
        return stats


class DataLineageOperator(BaseOperator):
    """
    数据血缘追踪操作符
    
    记录数据处理过程中的血缘关系
    """
    
    template_fields = ['source_assets', 'target_assets', 'transformation_info']
    
    @apply_defaults
    def __init__(
        self,
        source_assets: List[Dict[str, str]],
        target_assets: List[Dict[str, str]],
        relationship_type: str = 'transform',
        transformation_info: Optional[Dict[str, Any]] = None,
        lineage_db_path: Optional[str] = None,
        *args,
        **kwargs
    ):
        """
        初始化数据血缘追踪操作符
        
        Args:
            source_assets: 源数据资产列表
            target_assets: 目标数据资产列表
            relationship_type: 关系类型
            transformation_info: 转换信息
            lineage_db_path: 血缘数据库路径
        """
        super().__init__(*args, **kwargs)
        self.source_assets = source_assets
        self.target_assets = target_assets
        self.relationship_type = relationship_type
        self.transformation_info = transformation_info or {}
        self.lineage_db_path = lineage_db_path or Variable.get(
            'ml_lineage_db_path', 
            '/opt/airflow/data/lineage/lineage.db'
        )
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行血缘追踪任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 血缘追踪结果
        """
        self.logger.info("开始记录数据血缘关系")
        
        try:
            # 初始化血缘追踪器
            tracker = DataLineageTracker(self.lineage_db_path)
            
            # 注册数据资产
            for asset in self.source_assets + self.target_assets:
                tracker.register_data_asset(
                    asset_id=asset['asset_id'],
                    asset_name=asset['asset_name'],
                    asset_type=asset['asset_type'],
                    asset_path=asset.get('asset_path'),
                    schema_info=asset.get('schema_info'),
                    metadata=asset.get('metadata')
                )
            
            # 记录血缘关系
            lineage_records = []
            for source_asset in self.source_assets:
                for target_asset in self.target_assets:
                    tracker.record_lineage(
                        source_asset_id=source_asset['asset_id'],
                        target_asset_id=target_asset['asset_id'],
                        relationship_type=self.relationship_type,
                        transformation_info=self.transformation_info,
                        dag_id=context['dag'].dag_id,
                        task_id=context['task'].task_id,
                        execution_date=context['execution_date']
                    )
                    
                    lineage_records.append({
                        'source': source_asset['asset_id'],
                        'target': target_asset['asset_id'],
                        'type': self.relationship_type
                    })
            
            result = {
                'lineage_records': lineage_records,
                'source_assets_count': len(self.source_assets),
                'target_assets_count': len(self.target_assets),
                'relationship_type': self.relationship_type,
                'execution_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"血缘关系记录完成，共记录 {len(lineage_records)} 条关系")
            
            # 将结果推送到XCom
            context['task_instance'].xcom_push(key='lineage_result', value=result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"记录数据血缘关系失败: {str(e)}")
            raise AirflowException(f"记录数据血缘关系失败: {str(e)}")


class DataVersionOperator(BaseOperator):
    """
    数据版本管理操作符
    
    管理数据的版本控制和历史记录
    """
    
    template_fields = ['asset_id', 'data_path', 'version_metadata']
    
    @apply_defaults
    def __init__(
        self,
        asset_id: str,
        data_path: str,
        version: Optional[str] = None,
        version_metadata: Optional[Dict[str, Any]] = None,
        auto_version: bool = True,
        lineage_db_path: Optional[str] = None,
        *args,
        **kwargs
    ):
        """
        初始化数据版本管理操作符
        
        Args:
            asset_id: 数据资产ID
            data_path: 数据文件路径
            version: 版本号（如果不提供则自动生成）
            version_metadata: 版本元数据
            auto_version: 是否自动生成版本号
            lineage_db_path: 血缘数据库路径
        """
        super().__init__(*args, **kwargs)
        self.asset_id = asset_id
        self.data_path = data_path
        self.version = version
        self.version_metadata = version_metadata or {}
        self.auto_version = auto_version
        self.lineage_db_path = lineage_db_path or Variable.get(
            'ml_lineage_db_path', 
            '/opt/airflow/data/lineage/lineage.db'
        )
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据版本管理任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 版本管理结果
        """
        self.logger.info(f"开始管理数据版本: {self.asset_id}")
        
        try:
            # 初始化血缘追踪器
            tracker = DataLineageTracker(self.lineage_db_path)
            
            # 生成版本号
            if not self.version:
                if self.auto_version:
                    self.version = self._generate_version_number(context)
                else:
                    raise AirflowException("未提供版本号且未启用自动版本生成")
            
            # 添加执行上下文到元数据
            enhanced_metadata = self.version_metadata.copy()
            enhanced_metadata.update({
                'dag_id': context['dag'].dag_id,
                'task_id': context['task'].task_id,
                'execution_date': context['execution_date'].isoformat(),
                'created_by': 'airflow',
                'creation_timestamp': datetime.now().isoformat()
            })
            
            # 创建数据版本
            version_hash = tracker.create_data_version(
                asset_id=self.asset_id,
                version=self.version,
                file_path=self.data_path,
                metadata=enhanced_metadata
            )
            
            # 获取版本历史
            version_history = tracker.get_version_history(self.asset_id)
            
            result = {
                'asset_id': self.asset_id,
                'version': self.version,
                'version_hash': version_hash,
                'data_path': self.data_path,
                'metadata': enhanced_metadata,
                'version_count': len(version_history),
                'creation_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"数据版本创建完成: {self.asset_id} v{self.version}")
            
            # 将结果推送到XCom
            context['task_instance'].xcom_push(key='version_result', value=result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据版本管理失败: {str(e)}")
            raise AirflowException(f"数据版本管理失败: {str(e)}")
    
    def _generate_version_number(self, context: Dict[str, Any]) -> str:
        """
        生成版本号
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            str: 版本号
        """
        execution_date = context['execution_date']
        
        # 使用执行日期生成版本号
        version = execution_date.strftime('%Y.%m.%d.%H%M')
        
        # 如果同一时间有多个版本，添加序号
        tracker = DataLineageTracker(self.lineage_db_path)
        existing_versions = tracker.get_version_history(self.asset_id)
        
        base_version = version
        counter = 1
        while any(v['version'] == version for v in existing_versions):
            version = f"{base_version}.{counter}"
            counter += 1
        
        return version


class LineageReportOperator(BaseOperator):
    """
    血缘报告生成操作符
    
    生成数据血缘关系报告
    """
    
    template_fields = ['asset_ids', 'report_config']
    
    @apply_defaults
    def __init__(
        self,
        asset_ids: Optional[List[str]] = None,
        report_type: str = 'full',
        report_format: str = 'json',
        report_config: Optional[Dict[str, Any]] = None,
        lineage_db_path: Optional[str] = None,
        *args,
        **kwargs
    ):
        """
        初始化血缘报告生成操作符
        
        Args:
            asset_ids: 要生成报告的资产ID列表
            report_type: 报告类型（full, summary, impact_analysis）
            report_format: 报告格式（json, html, graphviz）
            report_config: 报告配置
            lineage_db_path: 血缘数据库路径
        """
        super().__init__(*args, **kwargs)
        self.asset_ids = asset_ids or []
        self.report_type = report_type
        self.report_format = report_format
        self.report_config = report_config or {}
        self.lineage_db_path = lineage_db_path or Variable.get(
            'ml_lineage_db_path', 
            '/opt/airflow/data/lineage/lineage.db'
        )
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行血缘报告生成任务
        
        Args:
            context: Airflow任务上下文
            
        Returns:
            Dict: 报告生成结果
        """
        self.logger.info("开始生成数据血缘报告")
        
        try:
            # 初始化血缘追踪器
            tracker = DataLineageTracker(self.lineage_db_path)
            
            # 如果没有指定资产ID，获取所有资产
            if not self.asset_ids:
                self.asset_ids = self._get_all_asset_ids(tracker)
            
            # 生成报告
            if self.report_type == 'full':
                report_data = self._generate_full_report(tracker)
            elif self.report_type == 'summary':
                report_data = self._generate_summary_report(tracker)
            elif self.report_type == 'impact_analysis':
                report_data = self._generate_impact_analysis(tracker)
            else:
                raise AirflowException(f"不支持的报告类型: {self.report_type}")
            
            # 保存报告
            report_file = self._save_report(report_data, context)
            
            result = {
                'report_type': self.report_type,
                'report_format': self.report_format,
                'report_file': report_file,
                'asset_count': len(self.asset_ids),
                'generation_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"血缘报告生成完成: {report_file}")
            
            # 将结果推送到XCom
            context['task_instance'].xcom_push(key='report_result', value=result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"生成血缘报告失败: {str(e)}")
            raise AirflowException(f"生成血缘报告失败: {str(e)}")
    
    def _get_all_asset_ids(self, tracker: DataLineageTracker) -> List[str]:
        """获取所有资产ID"""
        try:
            with sqlite3.connect(tracker.lineage_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT asset_id FROM data_assets")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"获取资产ID列表失败: {str(e)}")
            return []
    
    def _generate_full_report(self, tracker: DataLineageTracker) -> Dict[str, Any]:
        """生成完整报告"""
        report = {
            'report_type': 'full',
            'generation_timestamp': datetime.now().isoformat(),
            'assets': {},
            'lineage_graphs': {},
            'version_histories': {},
            'quality_histories': {},
            'statistics': {}
        }
        
        # 收集每个资产的详细信息
        for asset_id in self.asset_ids:
            # 获取血缘图
            lineage_graph = tracker.get_lineage_graph(asset_id, direction='both')
            report['lineage_graphs'][asset_id] = lineage_graph
            
            # 获取版本历史
            version_history = tracker.get_version_history(asset_id)
            report['version_histories'][asset_id] = version_history
        
        # 生成统计信息
        report['statistics'] = self._generate_statistics(tracker)
        
        return report
    
    def _generate_summary_report(self, tracker: DataLineageTracker) -> Dict[str, Any]:
        """生成摘要报告"""
        report = {
            'report_type': 'summary',
            'generation_timestamp': datetime.now().isoformat(),
            'statistics': self._generate_statistics(tracker),
            'asset_summary': {},
            'relationship_summary': {}
        }
        
        # 生成资产摘要
        with sqlite3.connect(tracker.lineage_db_path) as conn:
            cursor = conn.cursor()
            
            # 按类型统计资产
            cursor.execute("""
                SELECT asset_type, COUNT(*) 
                FROM data_assets 
                GROUP BY asset_type
            """)
            
            asset_types = {}
            for asset_type, count in cursor.fetchall():
                asset_types[asset_type] = count
            
            report['asset_summary']['by_type'] = asset_types
            
            # 统计关系类型
            cursor.execute("""
                SELECT relationship_type, COUNT(*) 
                FROM lineage_relationships 
                GROUP BY relationship_type
            """)
            
            relationship_types = {}
            for rel_type, count in cursor.fetchall():
                relationship_types[rel_type] = count
            
            report['relationship_summary']['by_type'] = relationship_types
        
        return report
    
    def _generate_impact_analysis(self, tracker: DataLineageTracker) -> Dict[str, Any]:
        """生成影响分析报告"""
        report = {
            'report_type': 'impact_analysis',
            'generation_timestamp': datetime.now().isoformat(),
            'impact_analysis': {}
        }
        
        # 为每个资产分析其影响范围
        for asset_id in self.asset_ids:
            downstream_graph = tracker.get_lineage_graph(asset_id, direction='downstream')
            upstream_graph = tracker.get_lineage_graph(asset_id, direction='upstream')
            
            report['impact_analysis'][asset_id] = {
                'downstream_assets': len(downstream_graph['nodes']) - 1,  # 减去自身
                'upstream_assets': len(upstream_graph['nodes']) - 1,
                'total_dependencies': len(downstream_graph['nodes']) + len(upstream_graph['nodes']) - 2,
                'downstream_graph': downstream_graph,
                'upstream_graph': upstream_graph
            }
        
        return report
    
    def _generate_statistics(self, tracker: DataLineageTracker) -> Dict[str, Any]:
        """生成统计信息"""
        stats = {}
        
        try:
            with sqlite3.connect(tracker.lineage_db_path) as conn:
                cursor = conn.cursor()
                
                # 资产统计
                cursor.execute("SELECT COUNT(*) FROM data_assets")
                stats['total_assets'] = cursor.fetchone()[0]
                
                # 关系统计
                cursor.execute("SELECT COUNT(*) FROM lineage_relationships")
                stats['total_relationships'] = cursor.fetchone()[0]
                
                # 版本统计
                cursor.execute("SELECT COUNT(*) FROM data_versions")
                stats['total_versions'] = cursor.fetchone()[0]
                
                # 质量记录统计
                cursor.execute("SELECT COUNT(*) FROM quality_history")
                stats['total_quality_records'] = cursor.fetchone()[0]
                
        except Exception as e:
            self.logger.warning(f"生成统计信息失败: {str(e)}")
        
        return stats
    
    def _save_report(self, report_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """保存报告"""
        try:
            execution_date = context['execution_date']
            
            # 确定报告保存路径
            reports_path = Variable.get('ml_lineage_reports_path', '/opt/airflow/data/lineage/reports')
            report_file = os.path.join(
                reports_path,
                execution_date.strftime('%Y'),
                execution_date.strftime('%m'),
                f'lineage_{self.report_type}_{execution_date.strftime("%Y-%m-%d_%H-%M")}.{self.report_format}'
            )
            
            # 确保目录存在
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            # 保存报告
            if self.report_format == 'json':
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)
            elif self.report_format == 'html':
                html_content = self._generate_html_report(report_data)
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            self.logger.info(f"血缘报告已保存到: {report_file}")
            
            return report_file
            
        except Exception as e:
            self.logger.error(f"保存血缘报告失败: {str(e)}")
            raise
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """生成HTML格式报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据血缘报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .stats { display: flex; justify-content: space-around; margin: 20px 0; }
                .stat { text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>数据血缘报告</h1>
                <p>报告类型: {report_type}</p>
                <p>生成时间: {generation_time}</p>
            </div>
            
            <div class="section">
                <h2>统计概览</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>总资产数</h3>
                        <h2>{total_assets}</h2>
                    </div>
                    <div class="stat">
                        <h3>总关系数</h3>
                        <h2>{total_relationships}</h2>
                    </div>
                    <div class="stat">
                        <h3>总版本数</h3>
                        <h2>{total_versions}</h2>
                    </div>
                </div>
            </div>
            
            {content}
        </body>
        </html>
        """
        
        # 获取统计信息
        stats = report_data.get('statistics', {})
        
        # 生成内容
        content = ""
        if report_data['report_type'] == 'summary':
            content = self._generate_summary_html_content(report_data)
        elif report_data['report_type'] == 'impact_analysis':
            content = self._generate_impact_html_content(report_data)
        
        # 填充模板
        return html_template.format(
            report_type=report_data['report_type'],
            generation_time=report_data['generation_timestamp'],
            total_assets=stats.get('total_assets', 0),
            total_relationships=stats.get('total_relationships', 0),
            total_versions=stats.get('total_versions', 0),
            content=content
        )
    
    def _generate_summary_html_content(self, report_data: Dict[str, Any]) -> str:
        """生成摘要HTML内容"""
        content = "<div class='section'><h2>资产类型分布</h2><table><tr><th>类型</th><th>数量</th></tr>"
        
        asset_summary = report_data.get('asset_summary', {}).get('by_type', {})
        for asset_type, count in asset_summary.items():
            content += f"<tr><td>{asset_type}</td><td>{count}</td></tr>"
        
        content += "</table></div>"
        
        return content
    
    def _generate_impact_html_content(self, report_data: Dict[str, Any]) -> str:
        """生成影响分析HTML内容"""
        content = "<div class='section'><h2>影响分析</h2><table><tr><th>资产ID</th><th>上游依赖</th><th>下游影响</th><th>总依赖</th></tr>"
        
        impact_analysis = report_data.get('impact_analysis', {})
        for asset_id, analysis in impact_analysis.items():
            content += f"""
            <tr>
                <td>{asset_id}</td>
                <td>{analysis['upstream_assets']}</td>
                <td>{analysis['downstream_assets']}</td>
                <td>{analysis['total_dependencies']}</td>
            </tr>
            """
        
        content += "</table></div>"
        
        return content