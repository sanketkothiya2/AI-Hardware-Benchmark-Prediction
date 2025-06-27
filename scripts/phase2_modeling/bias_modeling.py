"""
Bias/Weight-Based Modeling Framework for AI Benchmark Data

This module implements bias correction for manufacturer-specific optimizations,
cross-architecture performance normalization, and temporal bias adjustments.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import json
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class BiasModelingFramework:
    """
    Core framework for bias/weight-based modeling of AI benchmark data.
    Handles manufacturer bias, architecture normalization, and temporal adjustments.
    """
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the bias modeling framework.
        
        Args:
            dataset_path: Path to the AI benchmark dataset
        """
        self.dataset_path = dataset_path
        self.data = None
        self.bias_factors = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Initialize manufacturer bias factors based on domain knowledge
        self.manufacturer_bias_factors = {
            'NVIDIA': {
                'cuda_boost': 1.15,
                'ai_optimization': 1.25,
                'fp16_efficiency': 1.3,
                'tensor_cores': 1.4,
                'base_adjustment': 1.0
            },
            'AMD': {
                'opencl_boost': 1.1,
                'memory_efficiency': 1.2,
                'compute_optimization': 1.15,
                'rdna_advantage': 1.1,
                'base_adjustment': 0.95
            },
            'Intel': {
                'power_efficiency': 1.4,
                'integrated_optimization': 1.1,
                'general_compute': 0.9,
                'efficiency_focus': 1.2,
                'base_adjustment': 0.85
            }
        }
        
        # Generation-based temporal bias weights
        self.generation_weights = {
            'Current Gen (2022+)': 1.0,      # Latest baseline
            'Recent Gen (2020-2021)': 0.95,  # Slight penalty
            'Previous Gen (2018-2019)': 0.85, # Moderate penalty
            'Older Gen (2016-2017)': 0.75,   # Higher penalty
            'Legacy Gen (2014-2015)': 0.6,   # Significant penalty
            'Very Legacy (<2014)': 0.5       # Major penalty
        }
        
        # Performance category weights for different use cases
        self.category_weights = {
            'AI_Flagship': {'accuracy': 0.9, 'efficiency': 0.1, 'cost': 0.0},
            'AI_High_End': {'accuracy': 0.7, 'efficiency': 0.25, 'cost': 0.05},
            'AI_Mid_Range': {'accuracy': 0.5, 'efficiency': 0.35, 'cost': 0.15},
            'AI_Entry': {'accuracy': 0.3, 'efficiency': 0.5, 'cost': 0.2},
            'AI_Basic': {'accuracy': 0.2, 'efficiency': 0.6, 'cost': 0.2}
        }
    
    def load_data(self, dataset_path: str = None) -> pd.DataFrame:
        """
        Load and prepare the AI benchmark dataset.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Loaded and prepared DataFrame
        """
        if dataset_path:
            self.dataset_path = dataset_path
        
        if not self.dataset_path:
            raise ValueError("Dataset path must be provided")
        
        print(f"📊 Loading dataset from: {self.dataset_path}")
        self.data = pd.read_csv(self.dataset_path)
        
        print(f"   Dataset shape: {self.data.shape}")
        print(f"   Manufacturers: {self.data['Manufacturer'].unique()}")
        print(f"   Architectures: {len(self.data['Architecture'].unique())}")
        print(f"   Performance categories: {self.data['AI_Performance_Category'].unique()}")
        
        return self.data
    
    def analyze_manufacturer_bias(self) -> Dict:
        """
        Analyze performance patterns by manufacturer to identify bias factors.
        
        Returns:
            Dictionary containing bias analysis results
        """
        if self.data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        print("🔍 Analyzing manufacturer bias patterns...")
        
        analysis_results = {}
        
        # Performance metrics to analyze
        performance_metrics = ['FP32_Final', 'TOPs_per_Watt', 'G3Dmark', 
                             'powerPerformance', 'GFLOPS_per_Watt']
        
        for manufacturer in self.data['Manufacturer'].unique():
            if pd.isna(manufacturer):
                continue
                
            manufacturer_data = self.data[self.data['Manufacturer'] == manufacturer]
            analysis_results[manufacturer] = {
                'count': len(manufacturer_data),
                'architectures': manufacturer_data['Architecture'].unique().tolist(),
                'performance_stats': {}
            }
            
            # Calculate performance statistics
            for metric in performance_metrics:
                if metric in manufacturer_data.columns:
                    valid_data = manufacturer_data[manufacturer_data[metric].notna()][metric]
                    if len(valid_data) > 0:
                        analysis_results[manufacturer]['performance_stats'][metric] = {
                            'mean': float(valid_data.mean()),
                            'median': float(valid_data.median()),
                            'std': float(valid_data.std()),
                            'count': len(valid_data)
                        }
        
        # Calculate relative performance factors
        print("   Calculating relative performance factors...")
        overall_performance = {}
        for metric in performance_metrics:
            if metric in self.data.columns:
                valid_data = self.data[self.data[metric].notna()][metric]
                if len(valid_data) > 0:
                    overall_performance[metric] = float(valid_data.mean())
        
        # Calculate bias factors relative to overall mean
        for manufacturer in analysis_results:
            bias_factors = {}
            for metric in performance_metrics:
                if (metric in analysis_results[manufacturer]['performance_stats'] and 
                    metric in overall_performance):
                    
                    manufacturer_mean = analysis_results[manufacturer]['performance_stats'][metric]['mean']
                    overall_mean = overall_performance[metric]
                    
                    if overall_mean > 0:
                        bias_factor = manufacturer_mean / overall_mean
                        bias_factors[metric] = bias_factor
            
            analysis_results[manufacturer]['bias_factors'] = bias_factors
        
        self.bias_factors = analysis_results
        
        # Display summary
        print("\n   📈 Manufacturer Performance Summary:")
        for manufacturer, data in analysis_results.items():
            print(f"   {manufacturer}: {data['count']} GPUs, {len(data['architectures'])} architectures")
            if 'bias_factors' in data and 'FP32_Final' in data['bias_factors']:
                bias = data['bias_factors']['FP32_Final']
                print(f"      Performance bias: {bias:.3f}x relative to average")
        
        return analysis_results
    
    def calculate_bias_corrected_performance(self, gpu_data: pd.Series) -> Dict:
        """
        Calculate bias-corrected performance for a single GPU record.
        
        Args:
            gpu_data: Series containing GPU specifications
            
        Returns:
            Dictionary with corrected performance metrics
        """
        if 'FP32_Final' not in gpu_data or pd.isna(gpu_data['FP32_Final']):
            return {'error': 'Missing FP32_Final performance data'}
        
        base_performance = gpu_data['FP32_Final']
        manufacturer = gpu_data.get('Manufacturer', 'Unknown')
        generation = gpu_data.get('GenerationCategory', 'Unknown')
        category = gpu_data.get('AI_Performance_Category', 'AI_Basic')
        architecture = gpu_data.get('Architecture', 'Unknown')
        
        # Get manufacturer bias factor
        manufacturer_factor = 1.0
        if manufacturer in self.manufacturer_bias_factors:
            # Use base adjustment + architecture-specific bonuses
            manufacturer_factor = self.manufacturer_bias_factors[manufacturer]['base_adjustment']
            
            # Add architecture-specific optimizations
            if manufacturer == 'NVIDIA' and 'Ampere' in str(architecture):
                manufacturer_factor *= self.manufacturer_bias_factors[manufacturer]['ai_optimization']
            elif manufacturer == 'AMD' and 'RDNA' in str(architecture):
                manufacturer_factor *= self.manufacturer_bias_factors[manufacturer]['rdna_advantage']
        
        # Get generation weight
        generation_factor = self.generation_weights.get(generation, 0.7)
        
        # Get category weight (use accuracy weight as primary factor)
        category_factor = 1.0
        if category in self.category_weights:
            category_factor = self.category_weights[category]['accuracy']
        
        # Calculate corrected performance
        corrected_performance = (
            base_performance * 
            manufacturer_factor * 
            generation_factor * 
            (0.5 + category_factor * 0.5)  # Blend with baseline
        )
        
        return {
            'original_performance': base_performance,
            'corrected_performance': corrected_performance,
            'manufacturer_factor': manufacturer_factor,
            'generation_factor': generation_factor,
            'category_factor': category_factor,
            'bias_correction_applied': corrected_performance / base_performance if base_performance > 0 else 1.0
        }
    
    def apply_bias_correction_to_dataset(self) -> pd.DataFrame:
        """
        Apply bias correction to the entire dataset.
        
        Returns:
            DataFrame with bias-corrected performance metrics
        """
        if self.data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        print("🔧 Applying bias correction to dataset...")
        
        # Create a copy for modifications
        corrected_data = self.data.copy()
        
        # Initialize new columns
        corrected_data['Bias_Corrected_Performance'] = np.nan
        corrected_data['Manufacturer_Bias_Factor'] = np.nan
        corrected_data['Generation_Bias_Factor'] = np.nan
        corrected_data['Category_Bias_Factor'] = np.nan
        corrected_data['Total_Bias_Correction'] = np.nan
        
        successful_corrections = 0
        
        for idx, row in corrected_data.iterrows():
            try:
                correction_result = self.calculate_bias_corrected_performance(row)
                
                if 'error' not in correction_result:
                    corrected_data.loc[idx, 'Bias_Corrected_Performance'] = correction_result['corrected_performance']
                    corrected_data.loc[idx, 'Manufacturer_Bias_Factor'] = correction_result['manufacturer_factor']
                    corrected_data.loc[idx, 'Generation_Bias_Factor'] = correction_result['generation_factor']
                    corrected_data.loc[idx, 'Category_Bias_Factor'] = correction_result['category_factor']
                    corrected_data.loc[idx, 'Total_Bias_Correction'] = correction_result['bias_correction_applied']
                    successful_corrections += 1
                    
            except Exception as e:
                print(f"   ⚠️  Error processing row {idx}: {str(e)}")
                continue
        
        print(f"   ✅ Successfully applied bias correction to {successful_corrections} records")
        print(f"   📊 Bias correction statistics:")
        
        if successful_corrections > 0:
            bias_stats = corrected_data['Total_Bias_Correction'].describe()
            print(f"      Mean bias factor: {bias_stats['mean']:.3f}")
            print(f"      Median bias factor: {bias_stats['50%']:.3f}")
            print(f"      Min/Max bias factor: {bias_stats['min']:.3f} / {bias_stats['max']:.3f}")
        
        return corrected_data
    
    def validate_bias_correction(self, corrected_data: pd.DataFrame) -> Dict:
        """
        Validate the effectiveness of bias correction.
        
        Args:
            corrected_data: DataFrame with bias corrections applied
            
        Returns:
            Dictionary containing validation metrics
        """
        print("🧪 Validating bias correction effectiveness...")
        
        validation_results = {}
        
        # Compare manufacturer performance variance before and after correction
        manufacturers = corrected_data['Manufacturer'].unique()
        manufacturers = [m for m in manufacturers if not pd.isna(m)]
        
        # Original performance variance
        original_variance = {}
        corrected_variance = {}
        
        for manufacturer in manufacturers:
            manufacturer_data = corrected_data[corrected_data['Manufacturer'] == manufacturer]
            
            # Original performance
            original_perf = manufacturer_data['FP32_Final'].dropna()
            if len(original_perf) > 1:
                original_variance[manufacturer] = float(original_perf.var())
            
            # Corrected performance
            corrected_perf = manufacturer_data['Bias_Corrected_Performance'].dropna()
            if len(corrected_perf) > 1:
                corrected_variance[manufacturer] = float(corrected_perf.var())
        
        validation_results['manufacturer_variance'] = {
            'original': original_variance,
            'corrected': corrected_variance
        }
        
        # Calculate bias reduction
        total_original_var = sum(original_variance.values()) if original_variance else 0
        total_corrected_var = sum(corrected_variance.values()) if corrected_variance else 0
        
        if total_original_var > 0:
            bias_reduction = (total_original_var - total_corrected_var) / total_original_var
            validation_results['bias_reduction_percentage'] = bias_reduction * 100
        else:
            validation_results['bias_reduction_percentage'] = 0
        
        # Performance correlation analysis
        if ('FP32_Final' in corrected_data.columns and 
            'Bias_Corrected_Performance' in corrected_data.columns):
            
            valid_pairs = corrected_data[['FP32_Final', 'Bias_Corrected_Performance']].dropna()
            if len(valid_pairs) > 1:
                correlation = valid_pairs.corr().iloc[0, 1]
                validation_results['performance_correlation'] = float(correlation)
        
        print(f"   📈 Validation Results:")
        print(f"      Bias reduction: {validation_results.get('bias_reduction_percentage', 0):.2f}%")
        print(f"      Performance correlation: {validation_results.get('performance_correlation', 0):.3f}")
        
        return validation_results
    
    def save_bias_corrected_dataset(self, corrected_data: pd.DataFrame, 
                                   output_path: str = "data/phase2_outputs/bias_corrected_dataset.csv"):
        """
        Save the bias-corrected dataset to file.
        
        Args:
            corrected_data: DataFrame with bias corrections
            output_path: Path to save the corrected dataset
        """
        print(f"💾 Saving bias-corrected dataset to: {output_path}")
        
        # Ensure output directory exists
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        corrected_data.to_csv(output_path, index=False)
        print(f"   ✅ Dataset saved successfully")
        
        # Save bias factors and metadata
        metadata_path = output_path.replace('.csv', '_metadata.json')
        metadata = {
            'manufacturer_bias_factors': self.manufacturer_bias_factors,
            'generation_weights': self.generation_weights,
            'category_weights': self.category_weights,
            'bias_analysis': self.bias_factors,
            'dataset_info': {
                'total_records': len(corrected_data),
                'manufacturers': corrected_data['Manufacturer'].unique().tolist(),
                'architectures': corrected_data['Architecture'].unique().tolist()
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"   📋 Metadata saved to: {metadata_path}")
    
    def generate_bias_analysis_report(self) -> str:
        """
        Generate a comprehensive bias analysis report.
        
        Returns:
            Formatted report string
        """
        if not self.bias_factors:
            return "No bias analysis available. Run analyze_manufacturer_bias() first."
        
        report = []
        report.append("# Bias Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        report.append("## Manufacturer Performance Analysis")
        report.append("")
        
        for manufacturer, data in self.bias_factors.items():
            report.append(f"### {manufacturer}")
            report.append(f"- **Total GPUs**: {data['count']}")
            report.append(f"- **Architectures**: {', '.join(data['architectures'])}")
            
            if 'bias_factors' in data:
                report.append("- **Performance Bias Factors**:")
                for metric, factor in data['bias_factors'].items():
                    report.append(f"  - {metric}: {factor:.3f}x")
            report.append("")
        
        report.append("## Bias Correction Strategy")
        report.append("")
        report.append("### Manufacturer Adjustments")
        for manufacturer, factors in self.manufacturer_bias_factors.items():
            report.append(f"- **{manufacturer}**: Base adjustment {factors['base_adjustment']}")
        
        report.append("")
        report.append("### Generation Weights")
        for generation, weight in self.generation_weights.items():
            report.append(f"- **{generation}**: {weight}")
        
        return "\n".join(report)


def run_bias_modeling_analysis(dataset_path: str = "data/final/Ai-Benchmark-Final-enhanced-fixed.csv"):
    """
    Main function to run complete bias modeling analysis.
    
    Args:
        dataset_path: Path to the AI benchmark dataset
    """
    print("🚀 Starting Phase 2: Bias/Weight-Based Modeling Analysis")
    print("=" * 60)
    
    # Initialize framework
    framework = BiasModelingFramework(dataset_path)
    
    try:
        # Load and analyze data
        data = framework.load_data()
        
        # Analyze manufacturer bias
        bias_analysis = framework.analyze_manufacturer_bias()
        
        # Apply bias correction
        corrected_data = framework.apply_bias_correction_to_dataset()
        
        # Validate corrections
        validation_results = framework.validate_bias_correction(corrected_data)
        
        # Save results
        framework.save_bias_corrected_dataset(corrected_data)
        
        # Generate and save report
        report = framework.generate_bias_analysis_report()
        
        report_path = "documentation/bias_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📋 Bias analysis report saved to: {report_path}")
        print(f"✅ Phase 2.1 (Bias Modeling Framework) completed successfully!")
        
        return framework, corrected_data, validation_results
        
    except Exception as e:
        print(f"❌ Error in bias modeling analysis: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the analysis
    framework, corrected_data, validation = run_bias_modeling_analysis() 