"""
AI Benchmark KPI Prediction Models - Phase 3 Implementation

This package contains machine learning models for predicting AI hardware performance KPIs.

Modules:
    data_preprocessing: Feature engineering and data preparation pipeline
    performance_prediction: FLOPS and raw performance prediction models  
    efficiency_prediction: Power efficiency (TOPs/Watt) prediction models
    classification_models: Performance tier and category classification
    model_evaluation: Model performance evaluation and comparison framework
    prediction_pipeline: End-to-end prediction system for deployment

Model Storage:
    All trained models are saved as .pkl files in data/models/phase3_outputs/

Author: AI Benchmark Project Team
Version: 1.0
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "AI Benchmark Project Team"

# Core model types
MODEL_TYPES = {
    'regression': ['performance_prediction', 'efficiency_prediction'],
    'classification': ['performance_classification', 'tier_classification'],
    'ensemble': ['multi_output_prediction']
}

# Supported algorithms
ALGORITHMS = {
    'random_forest': 'Random Forest (Interpretable)',
    'xgboost': 'XGBoost (High Performance)',
    'lightgbm': 'LightGBM (Fast Training)'
}

# Target variables for prediction
TARGET_VARIABLES = {
    'performance': ['FP32_Final', 'Bias_Corrected_Performance'],
    'efficiency': ['TOPs_per_Watt', 'GFLOPS_per_Watt'],
    'classification': ['AI_Performance_Category', 'PerformanceTier']
} 