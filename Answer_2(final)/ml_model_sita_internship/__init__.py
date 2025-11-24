"""
ML Model Package
A simple package for classification and regression tasks
"""

from .question2 import (
    generate,
    learn,
    predict,
    statistics,
    target_statistics,
    features_statistics,
    correlation,
    get_metric
)

__version__ = "0.1.0"
__all__ = [
    "generate",
    "learn", 
    "predict",
    "statistics",
    "target_statistics",
    "features_statistics",
    "correlation",
    "get_metric"
]