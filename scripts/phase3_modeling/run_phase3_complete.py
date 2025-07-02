"""
Complete Phase 3 Machine Learning Pipeline

This script executes the complete Phase 3 machine learning pipeline for AI benchmark prediction.
Runs all three model types sequentially: Performance, Efficiency, and Classification.

Features:
- Performance Prediction (FLOPS) - Random Forest & XGBoost
- Efficiency Prediction (TOPs/Watt) - Random Forest & XGBoost  
- Classification (Performance Tiers) - Random Forest & XGBoost
- Comprehensive model evaluation and reporting
- All models saved as .pkl files for deployment

Author: AI Benchmark Project Team
"""

import sys
import os
import time
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from performance_prediction import PerformancePredictionModels
from efficiency_prediction import EfficiencyPredictionModels
from classification_models import ClassificationModels

class Phase3CompleteExecution:
    """
    Complete Phase 3 execution manager
    
    Coordinates the execution of all machine learning models and provides
    comprehensive reporting and error handling.
    """
    
    def __init__(self):
        self.start_time = None
        self.model_results = {}
        self.execution_log = []
        
        print("🚀 PHASE 3 COMPLETE MACHINE LEARNING PIPELINE")
        print("=" * 80)
        print("AI Benchmark KPI Prediction Models")
        print("Implementation: Random Forest + XGBoost")
        print("Targets: Performance + Efficiency + Classification")
        print("=" * 80)
    
    def log_execution(self, message, status="INFO"):
        """Log execution messages with timestamps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {status}: {message}"
        self.execution_log.append(log_entry)
        print(log_entry)
    
    def run_performance_prediction(self):
        """Execute performance prediction models"""
        try:
            self.log_execution("Starting Performance Prediction Models...", "INFO")
            
            performance_models = PerformancePredictionModels()
            performance_models.train_and_evaluate_all_targets()
            
            self.model_results['performance'] = {
                'status': 'SUCCESS',
                'models_trained': ['random_forest', 'xgboost'],
                'targets': ['FP32_Final', 'Bias_Corrected_Performance']
            }
            
            self.log_execution("Performance Prediction Models completed successfully", "SUCCESS")
            
        except Exception as e:
            error_msg = f"Performance Prediction failed: {str(e)}"
            self.log_execution(error_msg, "ERROR")
            self.model_results['performance'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            raise
    
    def run_efficiency_prediction(self):
        """Execute efficiency prediction models"""
        try:
            self.log_execution("Starting Efficiency Prediction Models...", "INFO")
            
            efficiency_models = EfficiencyPredictionModels()
            efficiency_models.train_and_evaluate_all_targets()
            
            self.model_results['efficiency'] = {
                'status': 'SUCCESS',
                'models_trained': ['random_forest', 'xgboost'],
                'targets': ['TOPs_per_Watt', 'GFLOPS_per_Watt']
            }
            
            self.log_execution("Efficiency Prediction Models completed successfully", "SUCCESS")
            
        except Exception as e:
            error_msg = f"Efficiency Prediction failed: {str(e)}"
            self.log_execution(error_msg, "ERROR")
            self.model_results['efficiency'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            raise
    
    def run_classification_models(self):
        """Execute classification models"""
        try:
            self.log_execution("Starting Classification Models...", "INFO")
            
            classification_models = ClassificationModels()
            classification_models.train_and_evaluate_all_targets()
            
            self.model_results['classification'] = {
                'status': 'SUCCESS',
                'models_trained': ['random_forest', 'xgboost'],
                'targets': ['AI_Performance_Category', 'PerformanceTier']
            }
            
            self.log_execution("Classification Models completed successfully", "SUCCESS")
            
        except Exception as e:
            error_msg = f"Classification Models failed: {str(e)}"
            self.log_execution(error_msg, "ERROR")
            self.model_results['classification'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            raise
    
    def generate_execution_summary(self):
        """Generate comprehensive execution summary"""
        end_time = time.time()
        execution_duration = end_time - self.start_time
        
        summary = []
        summary.append("=" * 80)
        summary.append("PHASE 3 EXECUTION SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Total Duration: {execution_duration:.2f} seconds ({execution_duration/60:.2f} minutes)")
        summary.append("")
        
        # Model Results Summary
        summary.append("📊 MODEL EXECUTION RESULTS:")
        summary.append("-" * 40)
        
        successful_models = 0
        total_models = 0
        
        for model_type, result in self.model_results.items():
            summary.append(f"\n🔸 {model_type.upper()} PREDICTION:")
            if result['status'] == 'SUCCESS':
                summary.append(f"   Status: ✅ SUCCESS")
                summary.append(f"   Models: {', '.join(result['models_trained'])}")
                summary.append(f"   Targets: {', '.join(result['targets'])}")
                successful_models += len(result['models_trained'])
                total_models += len(result['models_trained'])
            else:
                summary.append(f"   Status: ❌ FAILED")
                summary.append(f"   Error: {result['error']}")
                total_models += 2  # Assume 2 models per type
        
        # Overall Statistics
        summary.append(f"\n📈 OVERALL STATISTICS:")
        summary.append(f"   Successful Models: {successful_models}/{total_models}")
        summary.append(f"   Success Rate: {(successful_models/total_models)*100:.1f}%")
        
        # Files Generated
        summary.append(f"\n📁 FILES GENERATED:")
        summary.append(f"   Model Files (.pkl): Saved to data/models/phase3_outputs/")
        summary.append(f"   Performance Reports: Available for each model type")
        summary.append(f"   Feature Importance: Saved for interpretability")
        
        # Next Steps
        summary.append(f"\n🎯 NEXT STEPS:")
        summary.append(f"   1. Review model performance reports")
        summary.append(f"   2. Validate model predictions on new data")
        summary.append(f"   3. Deploy best-performing models")
        summary.append(f"   4. Implement prediction pipeline for production")
        
        summary.append("\n" + "=" * 80)
        
        # Save summary to file
        summary_filepath = '../../data/models/phase3_outputs/phase3_execution_summary.txt'
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))
        
        # Save execution log
        log_filepath = '../../data/models/phase3_outputs/phase3_execution_log.txt'
        with open(log_filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.execution_log))
        
        print('\n'.join(summary))
        
        self.log_execution(f"Execution summary saved: {summary_filepath}", "INFO")
        self.log_execution(f"Execution log saved: {log_filepath}", "INFO")
    
    def execute_complete_pipeline(self):
        """Execute the complete Phase 3 machine learning pipeline"""
        self.start_time = time.time()
        self.log_execution("Starting Phase 3 Complete Pipeline Execution", "START")
        
        try:
            # Step 1: Performance Prediction Models
            print("\n" + "="*60)
            print("STEP 1: PERFORMANCE PREDICTION MODELS")
            print("="*60)
            self.run_performance_prediction()
            
            # Step 2: Efficiency Prediction Models  
            print("\n" + "="*60)
            print("STEP 2: EFFICIENCY PREDICTION MODELS")
            print("="*60)
            self.run_efficiency_prediction()
            
            # Step 3: Classification Models
            print("\n" + "="*60)
            print("STEP 3: CLASSIFICATION MODELS")
            print("="*60)
            self.run_classification_models()
            
            # Generate Summary
            print("\n" + "="*60)
            print("EXECUTION COMPLETE - GENERATING SUMMARY")
            print("="*60)
            self.generate_execution_summary()
            
            self.log_execution("Phase 3 Complete Pipeline executed successfully!", "COMPLETE")
            
            return True
            
        except Exception as e:
            self.log_execution(f"Pipeline execution failed: {str(e)}", "CRITICAL")
            self.generate_execution_summary()  # Generate summary even on failure
            return False

def main():
    """
    Main execution function for Phase 3 complete pipeline
    """
    print("AI Benchmark Machine Learning Pipeline - Phase 3")
    print("Complete Implementation: Performance + Efficiency + Classification")
    print("=" * 80)
    
    # Create and execute the complete pipeline
    executor = Phase3CompleteExecution()
    success = executor.execute_complete_pipeline()
    
    if success:
        print("\n🎉 SUCCESS: Phase 3 pipeline completed successfully!")
        print("📁 All models and reports saved to: data/models/phase3_outputs/")
        print("🚀 Ready for model deployment and prediction!")
    else:
        print("\n❌ FAILURE: Phase 3 pipeline encountered errors.")
        print("📝 Check execution log for details: data/models/phase3_outputs/phase3_execution_log.txt")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 