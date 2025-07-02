"""
Classification Models for AI Benchmark KPI

This module implements ML models for classifying GPU performance categories.
Uses Random Forest and XGBoost for performance tier classification.

Author: AI Benchmark Project Team
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from data_preprocessing import AIBenchmarkPreprocessor
import warnings
warnings.filterwarnings('ignore')

class ClassificationModels:
    """Classification models for GPU performance categories"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.feature_importance = {}
        self.label_encoders = {}
        self.model_path = '../../data/models/phase3_outputs/'
        os.makedirs(self.model_path, exist_ok=True)
        
        print("🎯 Classification Models Initialized")
        print("Target: GPU Performance Categories & Tiers")
        print("Algorithms: Random Forest + XGBoost")
    
    def initialize_models(self):
        """Initialize Random Forest and XGBoost classifiers"""
        print("🔧 Initializing Classification Models...")
        
        # Random Forest - Interpretable classifier
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost - High performance classifier
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        print("✅ Classification models initialized")
    
    def train_models(self, X_train, y_train, target_name):
        """Train both Random Forest and XGBoost classifiers"""
        print(f"🎯 Training classification models for {target_name}...")
        
        # Handle target selection
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            if target_name in y_train.columns:
                y_target = y_train[target_name]
            else:
                y_target = y_train.iloc[:, 0]
                target_name = y_train.columns[0]
        else:
            y_target = y_train.ravel() if len(y_train.shape) > 1 else y_train
        
        # Clean data
        y_target = y_target.astype(str)
        valid_mask = y_target != 'nan'
        X_train_clean = X_train[valid_mask]
        y_target_clean = y_target[valid_mask]
        
        print(f"   Training on {len(y_target_clean)} samples")
        print(f"   Classes: {sorted(y_target_clean.unique())}")
        
        # Create label encoder for XGBoost
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_target_encoded = label_encoder.fit_transform(y_target_clean)
        
        # Store label encoder for later use
        self.label_encoders[target_name] = label_encoder
        
        trained_models = {}
        
        # Train Random Forest (can handle string labels)
        print("🌲 Training Random Forest classifier...")
        rf_model = self.models['random_forest']
        rf_model.fit(X_train_clean, y_target_clean)
        trained_models['random_forest'] = rf_model
        self.feature_importance[f'random_forest_{target_name}'] = rf_model.feature_importances_
        print("✅ Random Forest complete")
        
        # Train XGBoost (needs numeric labels)
        print("⚡ Training XGBoost classifier...")
        xgb_model = self.models['xgboost']
        xgb_model.fit(X_train_clean, y_target_encoded)
        trained_models['xgboost'] = xgb_model
        self.feature_importance[f'xgboost_{target_name}'] = xgb_model.feature_importances_
        print("✅ XGBoost complete")
        
        return trained_models
    
    def evaluate_models(self, models, X_val, y_val, X_test, y_test, target_name):
        """Evaluate trained classification models"""
        print(f"📊 Evaluating classification models for {target_name}...")
        
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
        
        # Clean data
        y_val_target = y_val_target.astype(str)
        y_test_target = y_test_target.astype(str)
        
        val_valid_mask = y_val_target != 'nan'
        test_valid_mask = y_test_target != 'nan'
        
        X_val_clean = X_val[val_valid_mask]
        y_val_clean = y_val_target[val_valid_mask]
        X_test_clean = X_test[test_valid_mask]
        y_test_clean = y_test_target[test_valid_mask]
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            print(f"🔍 Evaluating {model_name.replace('_', ' ').title()}...")
            
            # Predictions
            y_val_pred = model.predict(X_val_clean)
            y_test_pred = model.predict(X_test_clean)
            
            # Convert XGBoost numeric predictions back to string labels
            if model_name == 'xgboost' and target_name in self.label_encoders:
                y_val_pred = self.label_encoders[target_name].inverse_transform(y_val_pred)
                y_test_pred = self.label_encoders[target_name].inverse_transform(y_test_pred)
            
            # Metrics
            val_accuracy = accuracy_score(y_val_clean, y_val_pred)
            val_f1 = f1_score(y_val_clean, y_val_pred, average='weighted', zero_division=0)
            
            test_accuracy = accuracy_score(y_test_clean, y_test_pred)
            test_precision = precision_score(y_test_clean, y_test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test_clean, y_test_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test_clean, y_test_pred, average='weighted', zero_division=0)
            
            # Store results
            evaluation_results[model_name] = {
                'validation': {'accuracy': val_accuracy, 'f1_score': val_f1},
                'test': {
                    'accuracy': test_accuracy, 'precision': test_precision,
                    'recall': test_recall, 'f1_score': test_f1
                }
            }
            
            # Print results
            print(f"   Validation - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
            print(f"   Test - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        
        self.model_scores[target_name] = evaluation_results
        return evaluation_results
    
    def save_models(self, models, target_name):
        """Save trained classification models as .pkl files"""
        print(f"💾 Saving classification models for {target_name}...")
        
        for model_name, model in models.items():
            filename = f'classification_{model_name}_{target_name}_model.pkl'
            filepath = os.path.join(self.model_path, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"   ✅ {model_name} saved: {filename}")
        
        # Save feature importance
        importance_filename = f'classification_feature_importance_{target_name}.pkl'
        importance_filepath = os.path.join(self.model_path, importance_filename)
        
        target_importance = {k: v for k, v in self.feature_importance.items() if target_name in k}
        
        with open(importance_filepath, 'wb') as f:
            pickle.dump(target_importance, f)
        
        print(f"   ✅ Feature importance saved: {importance_filename}")
        
        # Save label encoder if exists
        if target_name in self.label_encoders:
            encoder_filename = f'classification_label_encoder_{target_name}.pkl'
            encoder_filepath = os.path.join(self.model_path, encoder_filename)
            
            with open(encoder_filepath, 'wb') as f:
                pickle.dump(self.label_encoders[target_name], f)
            
            print(f"   ✅ Label encoder saved: {encoder_filename}")
    
    def create_classification_report(self, target_name, feature_names):
        """Create comprehensive classification report"""
        print(f"📈 Creating classification report for {target_name}...")
        
        if target_name not in self.model_scores:
            print("❌ No evaluation results found")
            return
        
        results = self.model_scores[target_name]
        
        # Create report
        report = []
        report.append("=" * 80)
        report.append(f"CLASSIFICATION REPORT - {target_name.upper()}")
        report.append("=" * 80)
        
        # Model comparison
        report.append("\n🎯 MODEL PERFORMANCE:")
        for model_name, metrics in results.items():
            report.append(f"\n🔸 {model_name.replace('_', ' ').title()}:")
            report.append(f"   Test Accuracy:    {metrics['test']['accuracy']:.4f}")
            report.append(f"   Test Precision:   {metrics['test']['precision']:.4f}")
            report.append(f"   Test Recall:      {metrics['test']['recall']:.4f}")
            report.append(f"   Test F1-Score:    {metrics['test']['f1_score']:.4f}")
        
        # Best model
        best_model = max(results.keys(), key=lambda x: results[x]['test']['f1_score'])
        best_f1 = results[best_model]['test']['f1_score']
        
        report.append(f"\n🏆 BEST MODEL: {best_model.replace('_', ' ').title()}")
        report.append(f"   F1-Score: {best_f1:.4f}")
        
        # Top features
        if f'random_forest_{target_name}' in self.feature_importance:
            importance = self.feature_importance[f'random_forest_{target_name}']
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(10)
            
            report.append("\n🔍 TOP 10 FEATURES:")
            for _, row in feature_df.iterrows():
                report.append(f"   {row['feature']:<30} {row['importance']:.4f}")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_filepath = os.path.join(self.model_path, f'classification_report_{target_name}.txt')
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Report saved: classification_report_{target_name}.txt")
        print("\n".join(report))
    
    def train_and_evaluate_all_targets(self):
        """Complete pipeline for all classification targets"""
        print("🎯 Starting Classification Pipeline...")
        print("=" * 80)
        
        # Initialize models
        self.initialize_models()
        
        # Get preprocessed data
        preprocessor = AIBenchmarkPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names = \
            preprocessor.full_preprocessing_pipeline(task_type='classification')
        
        print(f"\n🎯 Training for {len(target_names)} targets: {target_names}")
        
        # Process each target
        for target_name in target_names:
            print(f"\n{'='*60}")
            print(f"PROCESSING: {target_name}")
            print(f"{'='*60}")
            
            # Train models
            trained_models = self.train_models(X_train, y_train, target_name)
            
            # Evaluate models
            self.evaluate_models(trained_models, X_val, y_val, X_test, y_test, target_name)
            
            # Save models
            self.save_models(trained_models, target_name)
            
            # Create report
            self.create_classification_report(target_name, feature_names)
        
        print(f"\n{'='*80}")
        print("🎉 CLASSIFICATION COMPLETE!")
        print(f"✅ Models saved to: {self.model_path}")
        print(f"{'='*80}")

if __name__ == "__main__":
    print("AI Benchmark Classification Models")
    print("Phase 3 Implementation - Priority 3")
    print("=" * 50)
    
    classification_models = ClassificationModels()
    classification_models.train_and_evaluate_all_targets() 