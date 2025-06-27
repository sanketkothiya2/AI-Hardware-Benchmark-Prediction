"""
Phase 2: Data Optimization and Bias/Weight-Based Modeling Package

This package implements bias/weight-based modeling for AI benchmark data,
including manufacturer bias correction, architecture normalization,
and temporal bias adjustment.

Modules:
- bias_modeling: Core bias correction framework
- architecture_normalization: Cross-architecture scaling
- temporal_adjustment: Generation-based corrections
- validation_framework: Model validation system
"""

__version__ = "1.0.0"
__author__ = "AI Benchmark KPI Project"

from .bias_modeling import BiasModelingFramework
from .architecture_normalization import ArchitectureNormalizer
from .temporal_adjustment import TemporalBiasAdjuster

__all__ = [
    'BiasModelingFramework',
    'ArchitectureNormalizer', 
    'TemporalBiasAdjuster'
] 