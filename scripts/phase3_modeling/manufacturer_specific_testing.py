#!/usr/bin/env python3
"""
Manufacturer-Specific Model Testing
Test Phase 3 models on Intel, NVIDIA, and AMD data separately
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ManufacturerSpecificTesting:
    """Test models on manufacturer-specific data"""
    
    def __init__(self):
        self.model_path = '../../data/models/phase3_outputs/'
        self.data_path = '../../data/phase2_outputs/phase2_final_enhanced_dataset.csv'
        self.results = {}
        self.test_data = {}
        
    def load_test_data(self):
        """Load and prepare test data with manufacturer information"""
        print("📊 Loading test dataset...")
        
        # Load full dataset
        df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded: {df.shape[0]} records")
        
        # Check manufacturers
        manufacturers = df['Manufacturer'].value_counts()
        print(f"📈 Manufacturer distribution:")
        for mfg, count in manufacturers.items():
            print(f"   - {mfg}: {count} records ({count/len(df)*100:.1f}%)")
        
        return df
    
    def prepare_manufacturer_data(self, df, manufacturer):
        """Prepare data for specific manufacturer"""
        print(f"🔧 Preparing {manufacturer} data...")
        
        # Filter by manufacturer
        mfg_data = df[df['Manufacturer'] == manufacturer].copy()
        print(f"✅ {manufacturer}: {len(mfg_data)} records")
        
        # Load the preprocessor to match training data format
        preprocessor_files = [
            'preprocessor_performance.pkl',
            'preprocessor_efficiency.pkl', 
            'preprocessor_classification.pkl'
        ]
        
        prepared_data = {}
        
        for prep_file in preprocessor_files:
            prep_path = os.path.join(self.model_path, prep_file)
            if os.path.exists(prep_path):
                with open(prep_path, 'rb') as f:
                    preprocessor = pickle.load(f)
                
                task_type = prep_file.replace('preprocessor_', '').replace('.pkl', '')
                
                # Apply same preprocessing as training
                mfg_processed = self.apply_preprocessing(mfg_data, preprocessor, task_type)
                prepared_data[task_type] = mfg_processed
        
        return prepared_data
    
    def apply_preprocessing(self, data, preprocessor, task_type):
        """Apply the same preprocessing as used in training"""
        
        # Get feature columns and targets
        feature_columns = preprocessor['feature_columns']
        target_columns = preprocessor['target_columns'][task_type]
        
        # Handle missing values for categorical columns
        categorical_columns = preprocessor['categorical_columns']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].fillna('Unknown')
        
        # Handle missing values for numerical columns  
        numerical_columns = preprocessor['numerical_columns']
        numerical_features = [col for col in numerical_columns if col in data.columns]
        if numerical_features:
            imputer = preprocessor['imputer']
            data[numerical_features] = imputer.transform(data[numerical_features])
        
        # Encode categorical features
        label_encoders = preprocessor['label_encoders']
        for col in categorical_columns:
            if col in data.columns and col in label_encoders:
                try:
                    data[col] = label_encoders[col].transform(data[col].astype(str))
                except ValueError as e:
                    # Handle unseen categories
                    print(f"⚠️ Warning: Unseen categories in {col}, using default encoding")
                    data[col] = 0
        
        # Prepare features and targets
        X = data[feature_columns].copy()
        
        # Scale features
        scaler = preprocessor['scaler']
        X_scaled = scaler.transform(X)
        
        # Prepare targets
        available_targets = [col for col in target_columns if col in data.columns]
        y = data[available_targets].copy() if available_targets else None
        
        return {
            'X': X_scaled,
            'y': y,
            'feature_names': feature_columns,
            'target_names': available_targets,
            'raw_data': data
        }
    
    def test_performance_models(self, manufacturer_data, manufacturer):
        """Test performance prediction models"""
        print(f"🏆 Testing Performance Models - {manufacturer}")
        
        if 'performance' not in manufacturer_data:
            print("❌ No performance data available")
            return {}
        
        data = manufacturer_data['performance']
        X, y = data['X'], data['y']
        
        if y is None or len(y) == 0:
            print("❌ No target data available")
            return {}
        
        results = {}
        
        # Test each target
        for target in data['target_names']:
            print(f"🎯 Testing {target}...")
            
            y_target = y[target] if target in y.columns else y.iloc[:, 0]
            
            target_results = {}
            
            # Test Random Forest
            rf_path = os.path.join(self.model_path, f'random_forest_{target}_model.pkl')
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    rf_model = pickle.load(f)
                
                y_pred = rf_model.predict(X)
                
                r2 = r2_score(y_target, y_pred)
                rmse = np.sqrt(mean_squared_error(y_target, y_pred))
                mae = mean_absolute_error(y_target, y_pred)
                
                # MAPE calculation
                mask = y_target != 0
                mape = np.mean(np.abs((y_target[mask] - y_pred[mask]) / y_target[mask])) * 100 if mask.sum() > 0 else 0
                
                target_results['random_forest'] = {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                }
                
                print(f"   🌲 Random Forest: R² = {r2:.4f}, MAPE = {mape:.2f}%")
            
            # Test XGBoost
            xgb_path = os.path.join(self.model_path, f'xgboost_{target}_model.pkl')
            if os.path.exists(xgb_path):
                with open(xgb_path, 'rb') as f:
                    xgb_model = pickle.load(f)
                
                y_pred = xgb_model.predict(X)
                
                r2 = r2_score(y_target, y_pred)
                rmse = np.sqrt(mean_squared_error(y_target, y_pred))
                mae = mean_absolute_error(y_target, y_pred)
                
                # MAPE calculation
                mask = y_target != 0
                mape = np.mean(np.abs((y_target[mask] - y_pred[mask]) / y_target[mask])) * 100 if mask.sum() > 0 else 0
                
                target_results['xgboost'] = {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                }
                
                print(f"   ⚡ XGBoost: R² = {r2:.4f}, MAPE = {mape:.2f}%")
            
            results[target] = target_results
        
        return results
    
    def test_efficiency_models(self, manufacturer_data, manufacturer):
        """Test efficiency prediction models"""
        print(f"⚡ Testing Efficiency Models - {manufacturer}")
        
        if 'efficiency' not in manufacturer_data:
            print("❌ No efficiency data available")
            return {}
        
        data = manufacturer_data['efficiency']
        X, y = data['X'], data['y']
        
        if y is None or len(y) == 0:
            print("❌ No target data available")
            return {}
        
        results = {}
        
        # Test each target
        for target in data['target_names']:
            print(f"🎯 Testing {target}...")
            
            y_target = y[target] if target in y.columns else y.iloc[:, 0]
            
            target_results = {}
            
            # Test Random Forest
            rf_path = os.path.join(self.model_path, f'efficiency_random_forest_{target}_model.pkl')
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    rf_model = pickle.load(f)
                
                y_pred = rf_model.predict(X)
                
                r2 = r2_score(y_target, y_pred)
                rmse = np.sqrt(mean_squared_error(y_target, y_pred))
                mae = mean_absolute_error(y_target, y_pred)
                
                # MAPE calculation
                mask = y_target != 0
                mape = np.mean(np.abs((y_target[mask] - y_pred[mask]) / y_target[mask])) * 100 if mask.sum() > 0 else 0
                
                target_results['random_forest'] = {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                }
                
                print(f"   🌲 Random Forest: R² = {r2:.4f}, MAPE = {mape:.2f}%")
            
            # Test XGBoost
            xgb_path = os.path.join(self.model_path, f'efficiency_xgboost_{target}_model.pkl')
            if os.path.exists(xgb_path):
                with open(xgb_path, 'rb') as f:
                    xgb_model = pickle.load(f)
                
                y_pred = xgb_model.predict(X)
                
                r2 = r2_score(y_target, y_pred)
                rmse = np.sqrt(mean_squared_error(y_target, y_pred))
                mae = mean_absolute_error(y_target, y_pred)
                
                # MAPE calculation  
                mask = y_target != 0
                mape = np.mean(np.abs((y_target[mask] - y_pred[mask]) / y_target[mask])) * 100 if mask.sum() > 0 else 0
                
                target_results['xgboost'] = {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                }
                
                print(f"   ⚡ XGBoost: R² = {r2:.4f}, MAPE = {mape:.2f}%")
            
            results[target] = target_results
        
        return results
    
    def test_classification_models(self, manufacturer_data, manufacturer):
        """Test classification models"""
        print(f"🎯 Testing Classification Models - {manufacturer}")
        
        if 'classification' not in manufacturer_data:
            print("❌ No classification data available")
            return {}
        
        data = manufacturer_data['classification']
        X, y = data['X'], data['y']
        
        if y is None or len(y) == 0:
            print("❌ No target data available")
            return {}
        
        results = {}
        
        # Test each target
        for target in data['target_names']:
            print(f"🎯 Testing {target}...")
            
            y_target = y[target] if target in y.columns else y.iloc[:, 0]
            
            target_results = {}
            
            # Load label encoder
            encoder_path = os.path.join(self.model_path, f'classification_label_encoder_{target}.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                
                # Encode targets
                try:
                    y_encoded = label_encoder.transform(y_target.astype(str))
                except ValueError:
                    print(f"⚠️ Warning: Unseen categories in {target}")
                    continue
                
                # Test Random Forest
                rf_path = os.path.join(self.model_path, f'classification_random_forest_{target}_model.pkl')
                if os.path.exists(rf_path):
                    with open(rf_path, 'rb') as f:
                        rf_model = pickle.load(f)
                    
                    y_pred = rf_model.predict(X)
                    
                    accuracy = accuracy_score(y_encoded, y_pred)
                    precision = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)
                    
                    target_results['random_forest'] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
                    
                    print(f"   🌲 Random Forest: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
                
                # Test XGBoost
                xgb_path = os.path.join(self.model_path, f'classification_xgboost_{target}_model.pkl')
                if os.path.exists(xgb_path):
                    with open(xgb_path, 'rb') as f:
                        xgb_model = pickle.load(f)
                    
                    y_pred = xgb_model.predict(X)
                    
                    accuracy = accuracy_score(y_encoded, y_pred)
                    precision = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)
                    
                    target_results['xgboost'] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
                    
                    print(f"   ⚡ XGBoost: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
            
            results[target] = target_results
        
        return results
    
    def create_manufacturer_report(self, manufacturer, results):
        """Create detailed report for manufacturer"""
        
        report = []
        report.append("=" * 80)
        report.append(f"MANUFACTURER-SPECIFIC TESTING REPORT - {manufacturer.upper()}")
        report.append("=" * 80)
        
        # Performance Models
        if 'performance' in results and results['performance']:
            report.append("\n🏆 PERFORMANCE PREDICTION RESULTS:")
            for target, models in results['performance'].items():
                report.append(f"\n📊 {target}:")
                for model_name, metrics in models.items():
                    report.append(f"   {model_name.replace('_', ' ').title()}:")
                    report.append(f"      R² Score: {metrics['r2_score']:.4f}")
                    report.append(f"      RMSE:     {metrics['rmse']:.2e}")
                    report.append(f"      MAPE:     {metrics['mape']:.2f}%")
        
        # Efficiency Models
        if 'efficiency' in results and results['efficiency']:
            report.append("\n⚡ EFFICIENCY PREDICTION RESULTS:")
            for target, models in results['efficiency'].items():
                report.append(f"\n📊 {target}:")
                for model_name, metrics in models.items():
                    report.append(f"   {model_name.replace('_', ' ').title()}:")
                    report.append(f"      R² Score: {metrics['r2_score']:.4f}")
                    report.append(f"      RMSE:     {metrics['rmse']:.4f}")
                    report.append(f"      MAPE:     {metrics['mape']:.2f}%")
        
        # Classification Models
        if 'classification' in results and results['classification']:
            report.append("\n🎯 CLASSIFICATION RESULTS:")
            for target, models in results['classification'].items():
                report.append(f"\n📊 {target}:")
                for model_name, metrics in models.items():
                    report.append(f"   {model_name.replace('_', ' ').title()}:")
                    report.append(f"      Accuracy:  {metrics['accuracy']:.4f}")
                    report.append(f"      Precision: {metrics['precision']:.4f}")
                    report.append(f"      Recall:    {metrics['recall']:.4f}")
                    report.append(f"      F1-Score:  {metrics['f1_score']:.4f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def run_manufacturer_testing(self):
        """Run complete manufacturer-specific testing"""
        print("🚀 Starting Manufacturer-Specific Model Testing...")
        print("=" * 80)
        
        # Load data
        df = self.load_test_data()
        
        # Get unique manufacturers
        manufacturers = ['NVIDIA', 'AMD', 'Intel']
        
        all_results = {}
        
        for manufacturer in manufacturers:
            if manufacturer in df['Manufacturer'].values:
                print(f"\n{'='*60}")
                print(f"TESTING: {manufacturer}")
                print(f"{'='*60}")
                
                # Prepare manufacturer data
                manufacturer_data = self.prepare_manufacturer_data(df, manufacturer)
                
                # Test all model types
                results = {}
                results['performance'] = self.test_performance_models(manufacturer_data, manufacturer)
                results['efficiency'] = self.test_efficiency_models(manufacturer_data, manufacturer)
                results['classification'] = self.test_classification_models(manufacturer_data, manufacturer)
                
                # Create and save report
                report = self.create_manufacturer_report(manufacturer, results)
                
                report_path = f'manufacturer_testing_report_{manufacturer.lower()}.txt'
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                print(f"\n📁 Report saved: {report_path}")
                print(report)
                
                all_results[manufacturer] = results
            else:
                print(f"⚠️ No data found for {manufacturer}")
        
        print(f"\n{'='*80}")
        print("🎉 MANUFACTURER-SPECIFIC TESTING COMPLETE!")
        print(f"{'='*80}")
        
        return all_results

if __name__ == "__main__":
    print("Manufacturer-Specific Model Testing")
    print("Testing Intel, NVIDIA, and AMD performance separately")
    print("=" * 60)
    
    tester = ManufacturerSpecificTesting()
    results = tester.run_manufacturer_testing() 