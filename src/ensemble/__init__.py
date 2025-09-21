"""模型集成模块

提供多种集成学习方法，包括投票、平均、堆叠、混合、动态集成等策略
"""

# 基础类
from .base import BaseEnsemble, EnsembleResult

# 集成方法
from .voting import VotingEnsemble, WeightedVotingEnsemble
from .averaging import AveragingEnsemble, DynamicAveragingEnsemble, AdaptiveAveragingEnsemble
from .stacking import StackingEnsemble, MultiLevelStackingEnsemble, DynamicStackingEnsemble
from .blending import BlendingEnsemble, DynamicBlendingEnsemble, AdaptiveBlendingEnsemble

# 动态集成
from .dynamic import (
    DynamicModelSelectionEnsemble,
    OnlineEnsembleLearning,
    AdaptiveEnsembleStrategy
)

# 工具函数
from .utils import (
    evaluate_ensemble_performance,
    compare_ensembles,
    visualize_ensemble_performance,
    visualize_ensemble_comparison,
    analyze_model_diversity,
    save_ensemble_results,
    load_ensemble_results,
    generate_ensemble_report,
    suggest_ensemble_strategy
)

__all__ = [
    # 基础类
    'BaseEnsemble',
    'EnsembleResult',
    
    # 投票集成
    'VotingEnsemble',
    'WeightedVotingEnsemble',
    
    # 平均集成
    'AveragingEnsemble',
    'DynamicAveragingEnsemble',
    'AdaptiveAveragingEnsemble',
    
    # 堆叠集成
    'StackingEnsemble',
    'MultiLevelStackingEnsemble',
    'DynamicStackingEnsemble',
    
    # 混合集成
    'BlendingEnsemble',
    'DynamicBlendingEnsemble',
    'AdaptiveBlendingEnsemble',
    
    # 动态集成
    'DynamicModelSelectionEnsemble',
    'OnlineEnsembleLearning',
    'AdaptiveEnsembleStrategy',
    
    # 工具函数
    'evaluate_ensemble_performance',
    'compare_ensembles',
    'visualize_ensemble_performance',
    'visualize_ensemble_comparison',
    'analyze_model_diversity',
    'save_ensemble_results',
    'load_ensemble_results',
    'generate_ensemble_report',
    'suggest_ensemble_strategy'
]

__version__ = "1.0.0"