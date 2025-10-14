"""
Tensor Decision Trees for High-Resolution Imaging Data

A Python library for building decision trees and ensemble methods
that preserve spatial structure in high-dimensional tensor data.
"""

from .tensor_tree import TensorDecisionTree, Node
from .ensemble import (
    TensorRandomForest,
    TensorGradientBoosting,
    TensorLADBoost
)
from .utils import (
    custom_kmeans,
    L2_norm,
    leverage_score_sampling,
    relative_mse,
    print_tree_structure
)

__version__ = "0.1.0"

__all__ = [
    'TensorDecisionTree',
    'Node',
    'TensorRandomForest',
    'TensorGradientBoosting',
    'TensorLADBoost',
    'custom_kmeans',
    'L2_norm',
    'leverage_score_sampling',
    'relative_mse',
    'print_tree_structure',
]
