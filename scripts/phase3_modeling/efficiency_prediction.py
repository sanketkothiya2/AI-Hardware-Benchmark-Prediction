"""
Efficiency Prediction Models for AI Benchmark KPI

This module implements ML models for predicting GPU power efficiency.
Uses Random Forest (interpretable) and XGBoost (high-performance) algorithms.

Target Variables:
- TOPs_per_Watt: Power efficiency metric
- GFLOPS_per_Watt: Graphics performance efficiency

Author: AI Benchmark Project Team
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from data_preprocessing import AIBenchmarkPreprocessor
import warnings
warnings.filterwarnings('ignore')

class EfficiencyPredictionModels:
    """Efficiency prediction models for GPU power efficiency"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.feature_importance = {}
        self.model_path = '../../data/models/phase3_outputs/'
        os.makedirs(self.model_path, exist_ok=True)
        
        print("⚡ Efficiency Prediction Models Initialized")
        print("Target: GPU Power Efficiency (TOPs/Watt)")
        print("Algorithms: Random Forest + XGBoost")
    
    def initialize_models(self):
        """Initialize Random Forest and XGBoost models for efficiency prediction"""
        print("🔧 Initializing Efficiency Models...")
        
        # Random Forest - Interpretable
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=120,           # Slightly more trees for efficiency prediction
            max_depth=12,               # Adjusted depth for efficiency patterns
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost - High performance
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=120,
            max_depth=5,                # Slightly shallower for efficiency
            learning_rate=0.08,         # Lower learning rate for stability
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print("✅ Efficiency models initialized:")
        print("   - Random Forest: Optimized for interpretability")
        print("   - XGBoost: Tuned for efficiency prediction")
    
    def train_models(self, X_train, y_train, target_name):
        """Train both Random Forest and XGBoost models for efficiency"""
        print(f"⚡ Training efficiency models for {target_name}...")
        
        # Handle target selection
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            if target_name in y_train.columns:
                y_target = y_train[target_name]
            else:
                y_target = y_train.iloc[:, 0]
                target_name = y_train.columns[0]
        else:
            y_target = y_train.ravel() if len(y_train.shape) > 1 else y_train
        
        # Remove any infinite or NaN values
        finite_mask = np.isfinite(y_target)
        X_train_clean = X_train[finite_mask]
        y_target_clean = y_target[finite_mask]
        
        print(f"   Training on {len(y_target_clean)} clean samples")
        
        trained_models = {}
        
        # Train Random Forest
        print("🌲 Training Random Forest for efficiency...")
        rf_model = self.models['random_forest']
        rf_model.fit(X_train_clean, y_target_clean)
        trained_models['random_forest'] = rf_model
        self.feature_importance[f'random_forest_{target_name}'] = rf_model.feature_importances_
        print("✅ Random Forest efficiency model complete")
        
        # Train XGBoost
        print("⚡ Training XGBoost for efficiency...")
        xgb_model = self.models['xgboost']
        xgb_model.fit(X_train_clean, y_target_clean)
        trained_models['xgboost'] = xgb_model
        self.feature_importance[f'xgboost_{target_name}'] = xgb_model.feature_importances_
        print("✅ XGBoost efficiency model complete")
        
        return trained_models
    
    def evaluate_models(self, models, X_val, y_val, X_test, y_test, target_name):
        """Evaluate trained efficiency models"""
        print(f"📊 Evaluating efficiency models for {target_name}...")
        
        # Handle target selection
        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            if target_name in y_val.columns:
                y_val_target = y_val[target_name]
                y_test_target = y_test[target_name]
            else:
                y_val_target = y_val.iloc[:, 0]
                y_test_target = y_test.iloc[:, 0]
        else:
            y_val_target = y_val.ravel() if len(y_val.shape) > 1 else y_val
            y_test_target = y_test.ravel() if len(y_test.shape) > 1 else y_test
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            print(f"🔍 Evaluating {model_name.replace('_', ' ').title()}...")
            
            # Predictions
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            # Handle infinite predictions
            y_val_pred = np.clip(y_val_pred, 0, np.percentile(y_val_target, 99))
            y_test_pred = np.clip(y_test_pred, 0, np.percentile(y_test_target, 99))
            
            # Metrics
            val_r2 = r2_score(y_val_target, y_val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val_target, y_val_pred))
            val_mae = mean_absolute_error(y_val_target, y_val_pred)
            
            test_r2 = r2_score(y_test_target, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test_target, y_test_pred))
            test_mae = mean_absolute_error(y_test_target, y_test_pred)
            
            # MAPE calculation with protection against division by zero
            def calculate_mape(y_true, y_pred):
                mask = (y_true != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
                if mask.sum() == 0:
                    return float('inf')
                return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            
            val_mape = calculate_mape(y_val_target, y_val_pred)
            test_mape = calculate_mape(y_test_target, y_test_pred)
            
            # Efficiency-specific metric: Median Absolute Percentage Error
            def calculate_median_ape(y_true, y_pred):
                mask = (y_true != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
                if mask.sum() == 0:
                    return float('inf')
                return np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            
            val_median_ape = calculate_median_ape(y_val_target, y_val_pred)
            test_median_ape = calculate_median_ape(y_test_target, y_test_pred)
            
            # Store results
            evaluation_results[model_name] = {
                'validation': {
                    'r2_score': val_r2, 'rmse': val_rmse, 'mae': val_mae, 
                    'mape': val_mape, 'median_ape': val_median_ape
                },
                'test': {
                    'r2_score': test_r2, 'rmse': test_rmse, 'mae': test_mae,
                    'mape': test_mape, 'median_ape': test_median_ape
                }
            }
            
            # Print results
            print(f"   Validation - R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
            print(f"   Test - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%")
        
        self.model_scores[target_name] = evaluation_results
        return evaluation_results
    
    def save_models(self, models, target_name):
        """Save trained efficiency models as .pkl files"""
        print(f"💾 Saving efficiency models for {target_name}...")
        
        for model_name, model in models.items():
            filename = f'efficiency_{model_name}_{target_name}_model.pkl'
            filepath = os.path.join(self.model_path, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"   ✅ {model_name} saved: {filename}")
        
        # Save feature importance
        importance_filename = f'efficiency_feature_importance_{target_name}.pkl'
        importance_filepath = os.path.join(self.model_path, importance_filename)
        
        target_importance = {k: v for k, v in self.feature_importance.items() if target_name in k}
        
        with open(importance_filepath, 'wb') as f:
            pickle.dump(target_importance, f)
        
        print(f"   ✅ Feature importance saved: {importance_filename}")
    
    def create_efficiency_report(self, target_name, feature_names):
        """Create comprehensive efficiency prediction report"""
        print(f"📈 Creating efficiency report for {target_name}...")
        
        if target_name not in self.model_scores:
            print("❌ No evaluation results found")
            return
        
        results = self.model_scores[target_name]
        
        # Create report
        report = []
        report.append("=" * 80)
        report.append(f"EFFICIENCY PREDICTION REPORT - {target_name.upper()}")
        report.append("=" * 80)
        
        # Model comparison
        report.append("\n⚡ MODEL PERFORMANCE (Power Efficiency):")
        for model_name, metrics in results.items():
            report.append(f"\n🔸 {model_name.replace('_', ' ').title()}:")
            report.append(f"   Test R² Score:      {metrics['test']['r2_score']:.4f}")
            report.append(f"   Test RMSE:          {metrics['test']['rmse']:.4f}")
            report.append(f"   Test MAPE:          {metrics['test']['mape']:.2f}%")
            report.append(f"   Test Median APE:    {metrics['test']['median_ape']:.2f}%")
        
        # Best model for efficiency
        best_model = max(results.keys(), key=lambda x: results[x]['test']['r2_score'])
        best_r2 = results[best_model]['test']['r2_score']
        
        report.append(f"\n🏆 BEST EFFICIENCY MODEL: {best_model.replace('_', ' ').title()}")
        report.append(f"   R² Score: {best_r2:.4f}")
        
        # Top efficiency-related features
        if f'random_forest_{target_name}' in self.feature_importance:
            importance = self.feature_importance[f'random_forest_{target_name}']
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(10)
            
            report.append("\n🔍 TOP 10 EFFICIENCY FEATURES:")
            for _, row in feature_df.iterrows():
                report.append(f"   {row['feature']:<30} {row['importance']:.4f}")
        
        # Efficiency insights
        report.append("\n💡 EFFICIENCY INSIGHTS:")
        report.append("   - Higher R² indicates better power efficiency prediction")
        report.append("   - Lower MAPE shows more accurate efficiency forecasting")
        report.append("   - Median APE provides robust efficiency error measurement")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_filepath = os.path.join(self.model_path, f'efficiency_report_{target_name}.txt')
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Efficiency report saved: efficiency_report_{target_name}.txt")
        print("\n".join(report))
    
    def train_and_evaluate_all_targets(self):
        """Complete pipeline for all efficiency targets"""
        print("⚡ Starting Efficiency Prediction Pipeline...")
        print("=" * 80)
        
        # Initialize models
        self.initialize_models()
        
        # Get preprocessed data
        preprocessor = AIBenchmarkPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names = \
            preprocessor.full_preprocessing_pipeline(task_type='efficiency')
        
        print(f"\n⚡ Training for {len(target_names)} efficiency targets: {target_names}")
        
        # Process each target
        for target_name in target_names:
            print(f"\n{'='*60}")
            print(f"PROCESSING EFFICIENCY: {target_name}")
            print(f"{'='*60}")
            
            # Train models
            trained_models = self.train_models(X_train, y_train, target_name)
            
            # Evaluate models
            self.evaluate_models(trained_models, X_val, y_val, X_test, y_test, target_name)
            
            # Save models
            self.save_models(trained_models, target_name)
            
            # Create report
            self.create_efficiency_report(target_name, feature_names)
        
        print(f"\n{'='*80}")
        print("🎉 EFFICIENCY PREDICTION COMPLETE!")
        print(f"✅ Efficiency models saved to: {self.model_path}")
        print(f"{'='*80}")

if __name__ == "__main__":
    print("AI Benchmark Efficiency Prediction Models")
    print("Phase 3 Implementation - Priority 2")
    print("=" * 50)
    
    efficiency_models = EfficiencyPredictionModels()
    efficiency_models.train_and_evaluate_all_targets() 