"""
Architecture Normalization Module for AI Benchmark Data

This module implements cross-architecture performance normalization to enable
fair comparisons between different GPU architectures and generations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')


class ArchitectureNormalizer:
    """
    Architecture normalization framework for cross-architecture performance comparison.
    Implements scaling factors and normalization techniques for different GPU architectures.
    """
    
    def __init__(self):
        """Initialize the architecture normalizer."""
        self.data = None
        self.architecture_factors = {}
        self.normalization_baseline = {}
        self.scaler = StandardScaler()
        
        # Architecture-specific performance scaling factors
        # Based on architectural efficiency and optimization levels
        self.base_architecture_factors = {
            # NVIDIA Architectures
            'Ampere': {
                'compute_efficiency': 1.0,      # Current baseline
                'ai_acceleration': 1.25,        # Tensor cores
                'memory_efficiency': 1.1,       # Improved memory subsystem
                'power_efficiency': 1.15,       # 8nm/7nm process
                'generation_factor': 1.0
            },
            'Ada Lovelace': {
                'compute_efficiency': 1.15,     # Next-gen improvements
                'ai_acceleration': 1.4,         # Enhanced Tensor cores
                'memory_efficiency': 1.2,       # Better memory bandwidth
                'power_efficiency': 1.25,       # 5nm process
                'generation_factor': 1.1
            },
            'Turing': {
                'compute_efficiency': 0.85,     # Previous generation
                'ai_acceleration': 1.0,         # First-gen RT/Tensor cores
                'memory_efficiency': 0.95,      # Older memory subsystem
                'power_efficiency': 0.9,        # 12nm process
                'generation_factor': 0.85
            },
            'Pascal': {
                'compute_efficiency': 0.75,     # Older architecture
                'ai_acceleration': 0.6,         # No dedicated AI units
                'memory_efficiency': 0.8,       # Basic memory subsystem
                'power_efficiency': 0.8,        # 16nm process
                'generation_factor': 0.75
            },
            'Volta': {
                'compute_efficiency': 0.9,      # Datacenter focus
                'ai_acceleration': 1.1,         # Early Tensor cores
                'memory_efficiency': 1.0,       # HBM2 memory
                'power_efficiency': 0.85,       # 12nm process
                'generation_factor': 0.9
            },
            'Maxwell': {
                'compute_efficiency': 0.65,     # Legacy architecture
                'ai_acceleration': 0.4,         # No AI acceleration
                'memory_efficiency': 0.7,       # Basic memory
                'power_efficiency': 0.75,       # 28nm process
                'generation_factor': 0.65
            },
            'Kepler': {
                'compute_efficiency': 0.5,      # Very legacy
                'ai_acceleration': 0.3,         # No AI support
                'memory_efficiency': 0.6,       # Limited memory bandwidth
                'power_efficiency': 0.6,        # 28nm process
                'generation_factor': 0.5
            },
            
            # AMD Architectures
            'RDNA 3': {
                'compute_efficiency': 1.0,      # Latest AMD baseline
                'ai_acceleration': 0.8,         # Limited AI acceleration
                'memory_efficiency': 1.05,      # Good memory efficiency
                'power_efficiency': 1.1,        # 5nm process
                'generation_factor': 1.0
            },
            'RDNA 2': {
                'compute_efficiency': 0.9,      # Current AMD
                'ai_acceleration': 0.7,         # Basic AI features
                'memory_efficiency': 1.0,       # Improved memory
                'power_efficiency': 1.0,        # 7nm process
                'generation_factor': 0.9
            },
            'RDNA': {
                'compute_efficiency': 0.8,      # First RDNA
                'ai_acceleration': 0.6,         # Limited AI
                'memory_efficiency': 0.9,       # Better than GCN
                'power_efficiency': 0.95,       # 7nm process
                'generation_factor': 0.8
            },
            'GCN (Vega)': {
                'compute_efficiency': 0.7,      # Legacy compute
                'ai_acceleration': 0.5,         # Minimal AI support
                'memory_efficiency': 0.85,      # HBM in some models
                'power_efficiency': 0.75,       # Power hungry
                'generation_factor': 0.7
            },
            'GCN': {
                'compute_efficiency': 0.6,      # Older GCN
                'ai_acceleration': 0.4,         # No real AI support
                'memory_efficiency': 0.75,      # Basic memory
                'power_efficiency': 0.7,        # Less efficient
                'generation_factor': 0.6
            },
            
            # Intel Architectures
            'Xe': {
                'compute_efficiency': 0.6,      # New but limited
                'ai_acceleration': 0.5,         # Basic AI features
                'memory_efficiency': 0.8,       # Integrated memory
                'power_efficiency': 1.2,        # Power efficient
                'generation_factor': 0.6
            },
            'Gen9/Gen11': {
                'compute_efficiency': 0.4,      # Integrated graphics
                'ai_acceleration': 0.3,         # Minimal AI
                'memory_efficiency': 0.6,       # Shared memory
                'power_efficiency': 1.1,        # Power efficient
                'generation_factor': 0.4
            },
            
            # Unknown/Generic
            'Unknown': {
                'compute_efficiency': 0.7,      # Conservative estimate
                'ai_acceleration': 0.6,         # Conservative estimate
                'memory_efficiency': 0.8,       # Conservative estimate
                'power_efficiency': 0.8,        # Conservative estimate
                'generation_factor': 0.7
            }
        }
    
    def load_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Load and prepare data for architecture normalization.
        
        Args:
            data: DataFrame containing GPU benchmark data
            
        Returns:
            Loaded and prepared DataFrame
        """
        self.data = data.copy()
        print(f"🏗️  Architecture Normalizer loaded {len(self.data)} GPU records")
        print(f"   Architectures found: {sorted(self.data['Architecture'].unique())}")
        return self.data
    
    def analyze_architecture_patterns(self) -> Dict:
        """
        Analyze performance patterns across different architectures.
        
        Returns:
            Dictionary containing architecture analysis results
        """
        if self.data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        print("🔍 Analyzing architecture performance patterns...")
        
        analysis_results = {}
        performance_metrics = ['FP32_Final', 'TOPs_per_Watt', 'G3Dmark', 
                             'powerPerformance', 'GFLOPS_per_Watt']
        
        # Calculate overall baselines
        overall_baselines = {}
        for metric in performance_metrics:
            if metric in self.data.columns:
                valid_data = self.data[self.data[metric].notna()][metric]
                if len(valid_data) > 0:
                    overall_baselines[metric] = {
                        'mean': float(valid_data.mean()),
                        'median': float(valid_data.median()),
                        'std': float(valid_data.std())
                    }
        
        self.normalization_baseline = overall_baselines
        
        # Analyze each architecture
        for architecture in self.data['Architecture'].unique():
            if pd.isna(architecture):
                continue
            
            arch_data = self.data[self.data['Architecture'] == architecture]
            
            analysis_results[architecture] = {
                'count': len(arch_data),
                'manufacturers': arch_data['Manufacturer'].unique().tolist(),
                'performance_stats': {},
                'relative_performance': {}
            }
            
            # Calculate architecture-specific statistics
            for metric in performance_metrics:
                if metric in arch_data.columns:
                    valid_data = arch_data[arch_data[metric].notna()][metric]
                    if len(valid_data) > 0:
                        analysis_results[architecture]['performance_stats'][metric] = {
                            'mean': float(valid_data.mean()),
                            'median': float(valid_data.median()),
                            'std': float(valid_data.std()),
                            'count': len(valid_data)
                        }
                        
                        # Calculate relative performance vs overall baseline
                        if (metric in overall_baselines and 
                            overall_baselines[metric]['mean'] > 0):
                            
                            relative_perf = valid_data.mean() / overall_baselines[metric]['mean']
                            analysis_results[architecture]['relative_performance'][metric] = float(relative_perf)
        
        # Calculate normalized architecture factors
        print("   Calculating normalized architecture factors...")
        self.architecture_factors = {}
        
        for architecture, data in analysis_results.items():
            # Get base factors or calculate from data
            if architecture in self.base_architecture_factors:
                base_factors = self.base_architecture_factors[architecture].copy()
            else:
                # Calculate factors for unknown architectures
                base_factors = self.base_architecture_factors['Unknown'].copy()
            
            # Adjust factors based on actual performance data
            if 'relative_performance' in data and data['relative_performance']:
                # Use FP32_Final as primary performance indicator
                if 'FP32_Final' in data['relative_performance']:
                    performance_ratio = data['relative_performance']['FP32_Final']
                    # Adjust generation factor based on actual performance
                    base_factors['generation_factor'] *= min(performance_ratio, 1.5)
            
            self.architecture_factors[architecture] = base_factors
        
        print(f"   📊 Architecture analysis completed for {len(analysis_results)} architectures")
        return analysis_results
    
    def normalize_performance_by_architecture(self, metric_column: str = 'FP32_Final') -> pd.DataFrame:
        """
        Normalize performance metrics by architecture.
        
        Args:
            metric_column: Performance metric to normalize
            
        Returns:
            DataFrame with normalized performance metrics
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        if not self.architecture_factors:
            raise ValueError("Architecture analysis must be run first")
        
        print(f"🔧 Normalizing {metric_column} by architecture...")
        
        normalized_data = self.data.copy()
        normalized_column = f'{metric_column}_Architecture_Normalized'
        architecture_factor_column = f'{metric_column}_Architecture_Factor'
        
        # Initialize new columns
        normalized_data[normalized_column] = np.nan
        normalized_data[architecture_factor_column] = np.nan
        
        successful_normalizations = 0
        
        for idx, row in normalized_data.iterrows():
            try:
                if pd.isna(row[metric_column]):
                    continue
                
                architecture = row.get('Architecture', 'Unknown')
                original_value = row[metric_column]
                
                # Get architecture factor
                if architecture in self.architecture_factors:
                    arch_factors = self.architecture_factors[architecture]
                    # Combine multiple factors for comprehensive normalization
                    combined_factor = (
                        arch_factors['compute_efficiency'] * 0.4 +
                        arch_factors['ai_acceleration'] * 0.3 +
                        arch_factors['memory_efficiency'] * 0.2 +
                        arch_factors['generation_factor'] * 0.1
                    )
                else:
                    combined_factor = 0.7  # Conservative default
                
                # Apply normalization
                normalized_value = original_value / combined_factor if combined_factor > 0 else original_value
                
                normalized_data.loc[idx, normalized_column] = normalized_value
                normalized_data.loc[idx, architecture_factor_column] = combined_factor
                successful_normalizations += 1
                
            except Exception as e:
                print(f"   ⚠️  Error normalizing row {idx}: {str(e)}")
                continue
        
        print(f"   ✅ Successfully normalized {successful_normalizations} records")
        
        # Calculate normalization statistics
        if successful_normalizations > 0:
            original_stats = normalized_data[metric_column].describe()
            normalized_stats = normalized_data[normalized_column].describe()
            
            print(f"   📊 Normalization impact:")
            print(f"      Original mean: {original_stats['mean']:.2e}")
            print(f"      Normalized mean: {normalized_stats['mean']:.2e}")
            print(f"      Variance reduction: {(1 - normalized_stats['std'] / original_stats['std']) * 100:.2f}%")
        
        return normalized_data
    
    def create_architecture_performance_matrix(self) -> pd.DataFrame:
        """
        Create a comprehensive architecture performance comparison matrix.
        
        Returns:
            DataFrame containing architecture performance matrix
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        print("📋 Creating architecture performance matrix...")
        
        # Performance metrics to include
        metrics = ['FP32_Final', 'TOPs_per_Watt', 'G3Dmark', 'powerPerformance', 
                  'GFLOPS_per_Watt', 'Memory_GB', 'TDP']
        
        architecture_matrix = []
        
        for architecture in sorted(self.data['Architecture'].unique()):
            if pd.isna(architecture):
                continue
            
            arch_data = self.data[self.data['Architecture'] == architecture]
            
            row = {
                'Architecture': architecture,
                'Total_GPUs': len(arch_data),
                'Manufacturers': ', '.join(arch_data['Manufacturer'].unique()),
                'Performance_Categories': ', '.join(arch_data['AI_Performance_Category'].unique())
            }
            
            # Calculate statistics for each metric
            for metric in metrics:
                if metric in arch_data.columns:
                    valid_data = arch_data[arch_data[metric].notna()][metric]
                    if len(valid_data) > 0:
                        row[f'{metric}_Mean'] = float(valid_data.mean())
                        row[f'{metric}_Median'] = float(valid_data.median())
                        row[f'{metric}_Count'] = len(valid_data)
                    else:
                        row[f'{metric}_Mean'] = np.nan
                        row[f'{metric}_Median'] = np.nan
                        row[f'{metric}_Count'] = 0
            
            # Add architecture factors
            if architecture in self.architecture_factors:
                factors = self.architecture_factors[architecture]
                row['Compute_Efficiency_Factor'] = factors['compute_efficiency']
                row['AI_Acceleration_Factor'] = factors['ai_acceleration']
                row['Memory_Efficiency_Factor'] = factors['memory_efficiency']
                row['Power_Efficiency_Factor'] = factors['power_efficiency']
                row['Generation_Factor'] = factors['generation_factor']
            
            architecture_matrix.append(row)
        
        matrix_df = pd.DataFrame(architecture_matrix)
        
        print(f"   ✅ Architecture matrix created with {len(matrix_df)} architectures")
        return matrix_df
    
    def validate_normalization_effectiveness(self, normalized_data: pd.DataFrame, 
                                           original_column: str = 'FP32_Final',
                                           normalized_column: str = None) -> Dict:
        """
        Validate the effectiveness of architecture normalization.
        
        Args:
            normalized_data: DataFrame with normalized metrics
            original_column: Original performance metric column
            normalized_column: Normalized performance metric column
            
        Returns:
            Dictionary containing validation results
        """
        if normalized_column is None:
            normalized_column = f'{original_column}_Architecture_Normalized'
        
        print("🧪 Validating architecture normalization effectiveness...")
        
        validation_results = {}
        
        # Calculate variance reduction
        original_variance = normalized_data[original_column].var()
        normalized_variance = normalized_data[normalized_column].var()
        
        if original_variance > 0:
            variance_reduction = (original_variance - normalized_variance) / original_variance
            validation_results['variance_reduction_percentage'] = variance_reduction * 100
        else:
            validation_results['variance_reduction_percentage'] = 0
        
        # Architecture-wise performance consistency
        architecture_consistency = {}
        for architecture in normalized_data['Architecture'].unique():
            if pd.isna(architecture):
                continue
            
            arch_data = normalized_data[normalized_data['Architecture'] == architecture]
            
            original_cv = arch_data[original_column].std() / arch_data[original_column].mean() if arch_data[original_column].mean() > 0 else 0
            normalized_cv = arch_data[normalized_column].std() / arch_data[normalized_column].mean() if arch_data[normalized_column].mean() > 0 else 0
            
            architecture_consistency[architecture] = {
                'original_cv': float(original_cv),
                'normalized_cv': float(normalized_cv),
                'consistency_improvement': float(original_cv - normalized_cv) if original_cv > 0 else 0
            }
        
        validation_results['architecture_consistency'] = architecture_consistency
        
        # Cross-architecture comparison fairness
        architectures = [arch for arch in normalized_data['Architecture'].unique() if not pd.isna(arch)]
        cross_arch_correlations = {}
        
        for i, arch1 in enumerate(architectures):
            for arch2 in architectures[i+1:]:
                arch1_data = normalized_data[normalized_data['Architecture'] == arch1][normalized_column].dropna()
                arch2_data = normalized_data[normalized_data['Architecture'] == arch2][normalized_column].dropna()
                
                if len(arch1_data) > 1 and len(arch2_data) > 1:
                    # Calculate overlap in performance ranges
                    arch1_range = (arch1_data.min(), arch1_data.max())
                    arch2_range = (arch2_data.min(), arch2_data.max())
                    
                    overlap_start = max(arch1_range[0], arch2_range[0])
                    overlap_end = min(arch1_range[1], arch2_range[1])
                    overlap_ratio = max(0, overlap_end - overlap_start) / (max(arch1_range[1], arch2_range[1]) - min(arch1_range[0], arch2_range[0]))
                    
                    cross_arch_correlations[f'{arch1}_vs_{arch2}'] = {
                        'performance_overlap_ratio': float(overlap_ratio),
                        'mean_difference_ratio': float(abs(arch1_data.mean() - arch2_data.mean()) / max(arch1_data.mean(), arch2_data.mean()))
                    }
        
        validation_results['cross_architecture_fairness'] = cross_arch_correlations
        
        # Overall normalization score
        avg_variance_reduction = validation_results['variance_reduction_percentage']
        avg_consistency_improvement = np.mean([data['consistency_improvement'] for data in architecture_consistency.values()])
        
        normalization_score = (avg_variance_reduction * 0.6 + avg_consistency_improvement * 100 * 0.4)
        validation_results['overall_normalization_score'] = float(normalization_score)
        
        print(f"   📈 Validation Results:")
        print(f"      Variance reduction: {validation_results['variance_reduction_percentage']:.2f}%")
        print(f"      Overall normalization score: {validation_results['overall_normalization_score']:.2f}")
        
        return validation_results
    
    def save_architecture_analysis(self, output_dir: str = "data/phase2_outputs/"):
        """
        Save architecture analysis results and normalization data.
        
        Args:
            output_dir: Directory to save analysis results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving architecture analysis to: {output_dir}")
        
        # Save architecture factors
        factors_path = os.path.join(output_dir, "architecture_factors.json")
        with open(factors_path, 'w') as f:
            json.dump(self.architecture_factors, f, indent=2, default=str)
        
        # Save normalization baseline
        baseline_path = os.path.join(output_dir, "normalization_baseline.json")
        with open(baseline_path, 'w') as f:
            json.dump(self.normalization_baseline, f, indent=2, default=str)
        
        print(f"   ✅ Architecture analysis saved successfully")


def run_architecture_normalization(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to run architecture normalization analysis.
    
    Args:
        data: Input DataFrame with GPU benchmark data
        
    Returns:
        Tuple of (normalized_data, validation_results)
    """
    print("\n🏗️  Starting Architecture Normalization Analysis")
    print("=" * 50)
    
    # Initialize normalizer
    normalizer = ArchitectureNormalizer()
    
    try:
        # Load and analyze data
        normalizer.load_data(data)
        
        # Analyze architecture patterns
        architecture_analysis = normalizer.analyze_architecture_patterns()
        
        # Normalize performance metrics
        normalized_data = normalizer.normalize_performance_by_architecture('FP32_Final')
        
        # Create architecture matrix
        arch_matrix = normalizer.create_architecture_performance_matrix()
        
        # Validate normalization
        validation_results = normalizer.validate_normalization_effectiveness(normalized_data)
        
        # Save results
        normalizer.save_architecture_analysis()
        
        # Save architecture matrix
        matrix_path = "data/phase2_outputs/architecture_performance_matrix.csv"
        arch_matrix.to_csv(matrix_path, index=False)
        
        print(f"\n✅ Architecture normalization completed successfully!")
        print(f"📊 Variance reduction: {validation_results.get('variance_reduction_percentage', 0):.2f}%")
        print(f"🎯 Normalization score: {validation_results.get('overall_normalization_score', 0):.2f}")
        
        return normalized_data, validation_results
        
    except Exception as e:
        print(f"❌ Error in architecture normalization: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    print("Architecture Normalization Module")
    print("This module should be imported and used with bias_modeling.py") 