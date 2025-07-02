"""
Data Preprocessing Pipeline for AI Benchmark KPI Prediction

This module handles feature engineering, data cleaning, and preprocessing for ML models.
Prepares data for performance prediction, efficiency prediction, and classification tasks.

Author: AI Benchmark Project Team
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class AIBenchmarkPreprocessor:
    """Comprehensive preprocessing pipeline for AI benchmark data"""
    
    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.target_columns = {}
        self.model_path = '../../data/models/phase3_outputs/'
    
    def load_data(self, filepath='../../data/phase2_outputs/phase2_final_enhanced_dataset.csv'):
        """Load the Phase 2 enhanced dataset"""
        print("📊 Loading Phase 2 Enhanced Dataset...")
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def identify_feature_types(self, df):
        """Identify categorical and numerical features"""
        print("🔍 Identifying Feature Types...")
        
        # Define target variables
        self.target_columns = {
            'performance': ['FP32_Final', 'Bias_Corrected_Performance'],
            'efficiency': ['TOPs_per_Watt', 'GFLOPS_per_Watt'],
            'classification': ['AI_Performance_Category', 'PerformanceTier']
        }
        
        # Exclude problematic columns
        exclude_columns = [
            'gpuName', 'testDate', 'price', 'gpuValue',
            'FP16 (half precision) performance (FLOP/s)',
            'INT8 performance (OP/s)',
            'Memory size per board (Byte)',
            'Memory_GB', 'Memory bandwidth (byte/s)'
        ]
        
        # Get all targets
        all_targets = []
        for targets in self.target_columns.values():
            all_targets.extend(targets)
        
        # Feature columns
        self.feature_columns = [col for col in df.columns 
                               if col not in all_targets and col not in exclude_columns]
        
        # Separate categorical and numerical
        self.categorical_columns = df[self.feature_columns].select_dtypes(include=['object', 'bool']).columns.tolist()
        self.numerical_columns = df[self.feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"✅ Features: {len(self.feature_columns)} total")
        print(f"   - Categorical: {len(self.categorical_columns)}")
        print(f"   - Numerical: {len(self.numerical_columns)}")
    
    def handle_missing_values(self, df):
        """Handle missing values in dataset"""
        print("🔧 Handling Missing Values...")
        df_processed = df.copy()
        
        # Categorical missing values
        for col in self.categorical_columns:
            if col in df_processed.columns:
                missing_count = df_processed[col].isnull().sum()
                if missing_count > 0:
                    df_processed[col] = df_processed[col].fillna('Unknown')
                    print(f"   - {col}: {missing_count} filled with 'Unknown'")
        
        # Numerical missing values
        numerical_features = [col for col in self.numerical_columns if col in df_processed.columns]
        if numerical_features:
            df_processed[numerical_features] = self.imputer.fit_transform(df_processed[numerical_features])
        
        print("✅ Missing values handled")
        return df_processed
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features using Label Encoding"""
        print("🔤 Encoding Categorical Features...")
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        print(f"✅ {len(self.categorical_columns)} categorical features encoded")
        return df_encoded
    
    def scale_numerical_features(self, X, fit=True):
        """Scale numerical features using StandardScaler"""
        print("📏 Scaling Features...")
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        print("✅ Features scaled")
        return X_scaled
    
    def prepare_features_and_targets(self, df, task_type='performance'):
        """Prepare feature matrix and target variables"""
        print(f"🎯 Preparing {task_type} prediction data...")
        
        X = df[self.feature_columns].copy()
        
        if task_type in self.target_columns:
            target_cols = self.target_columns[task_type]
            available_targets = [col for col in target_cols if col in df.columns]
            
            if not available_targets:
                raise ValueError(f"No targets found for {task_type}")
            
            y = df[available_targets].copy()
            print(f"✅ {X.shape[1]} features, {len(available_targets)} targets prepared")
            return X, y, self.feature_columns, available_targets
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        print(f"✂️ Splitting data...")
        
        # Split train and temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Split validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state
        )
        
        print(f"✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, filename='preprocessor.pkl'):
        """Save preprocessor to disk"""
        filepath = os.path.join(self.model_path, filename)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'target_columns': self.target_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        print(f"💾 Preprocessor saved: {filepath}")
    
    def full_preprocessing_pipeline(self, task_type='performance'):
        """Complete preprocessing pipeline"""
        print("🚀 Starting Full Preprocessing Pipeline...")
        print("=" * 60)
        
        # Load and process data
        df = self.load_data()
        self.identify_feature_types(df)
        df_clean = self.handle_missing_values(df)
        df_encoded = self.encode_categorical_features(df_clean, fit=True)
        
        # Prepare features and targets
        X, y, feature_names, target_names = self.prepare_features_and_targets(df_encoded, task_type)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled = self.scale_numerical_features(X_train, fit=True)
        X_val_scaled = self.scale_numerical_features(X_val, fit=False)
        X_test_scaled = self.scale_numerical_features(X_test, fit=False)
        
        # Save preprocessor
        self.save_preprocessor(f'preprocessor_{task_type}.pkl')
        
        print("=" * 60)
        print("🎉 Preprocessing Complete!")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names, target_names

if __name__ == "__main__":
    preprocessor = AIBenchmarkPreprocessor()
    results = preprocessor.full_preprocessing_pipeline(task_type='performance')
    print("Preprocessing pipeline test completed successfully!") 