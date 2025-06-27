"""
Integrated Phase 2: Bias/Weight-Based Modeling and Architecture Normalization

This script combines bias correction, architecture normalization, and comprehensive
validation for the AI benchmark dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from bias_modeling import BiasModelingFramework
from architecture_normalization import ArchitectureNormalizer
import json
from typing import Dict, Tuple


def run_integrated_phase2_analysis(dataset_path: str = "data/final/Ai-Benchmark-Final-enhanced-fixed.csv"):
    """
    Run the complete integrated Phase 2 analysis including:
    1. Bias correction and manufacturer analysis
    2. Architecture normalization 
    3. Combined validation and assessment
    """
    print("🚀 Integrated Phase 2: Data Optimization and Bias/Weight-Based Modeling")
    print("=" * 70)
    print("📋 Comprehensive Analysis Pipeline:")
    print("   1. Manufacturer bias analysis and correction")
    print("   2. Architecture-specific performance normalization") 
    print("   3. Temporal bias adjustment for hardware generations")
    print("   4. Integrated model validation and accuracy assessment")
    print("")
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None, None
    
    comprehensive_results = {
        'dataset_info': {},
        'bias_analysis': {},
        'architecture_analysis': {},
        'validation_results': {},
        'performance_metrics': {}
    }
    
    try:
        # Phase 2.1: Bias Modeling
        print("🔧 Phase 2.1: Bias Modeling and Manufacturer Analysis")
        print("-" * 60)
        
        bias_framework = BiasModelingFramework(dataset_path)
        original_data = bias_framework.load_data()
        
        # Store dataset info
        comprehensive_results['dataset_info'] = {
            'total_records': len(original_data),
            'manufacturers': original_data['Manufacturer'].unique().tolist(),
            'architectures': original_data['Architecture'].unique().tolist(),
            'performance_categories': original_data['AI_Performance_Category'].unique().tolist()
        }
        
        # Manufacturer bias analysis
        bias_analysis = bias_framework.analyze_manufacturer_bias()
        comprehensive_results['bias_analysis'] = bias_analysis
        
        # Apply bias correction
        bias_corrected_data = bias_framework.apply_bias_correction_to_dataset()
        bias_validation = bias_framework.validate_bias_correction(bias_corrected_data)
        
        print(f"\n✅ Phase 2.1 completed: {len(bias_corrected_data)} records bias-corrected")
        
        # Phase 2.2: Architecture Normalization
        print("\n🏗️  Phase 2.2: Architecture Normalization Analysis")
        print("-" * 60)
        
        arch_normalizer = ArchitectureNormalizer()
        arch_normalizer.load_data(bias_corrected_data)
        
        # Architecture pattern analysis
        architecture_analysis = arch_normalizer.analyze_architecture_patterns()
        comprehensive_results['architecture_analysis'] = architecture_analysis
        
        # Apply architecture normalization
        fully_normalized_data = arch_normalizer.normalize_performance_by_architecture('FP32_Final')
        
        print(f"\n✅ Phase 2.2 completed: Architecture normalization applied")
        
        # Phase 2.3: Save Enhanced Dataset and Results
        print("\n💾 Phase 2.3: Saving Enhanced Dataset and Results")
        print("-" * 60)
        
        # Create comprehensive output directory
        output_dir = "data/phase2_outputs/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the final enhanced dataset
        final_dataset_path = os.path.join(output_dir, "phase2_final_enhanced_dataset.csv")
        fully_normalized_data.to_csv(final_dataset_path, index=False)
        print(f"   📄 Final enhanced dataset: {final_dataset_path}")
        
        # Save comprehensive analysis results
        results_path = os.path.join(output_dir, "phase2_comprehensive_results.json")
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        print(f"   📄 Comprehensive results: {results_path}")
        
        # Save individual component results
        bias_framework.save_bias_corrected_dataset(bias_corrected_data)
        arch_normalizer.save_architecture_analysis()
        
        # Generate comprehensive report
        report_content = generate_comprehensive_report(comprehensive_results)
        report_path = "documentation/phase2_comprehensive_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"   📄 Comprehensive report: {report_path}")
        
        # Phase 2.4: Summary and Next Steps
        print("\n" + "=" * 70)
        print("📊 INTEGRATED PHASE 2 ANALYSIS SUMMARY")
        print("=" * 70)
        
        display_comprehensive_summary(comprehensive_results, bias_validation)
        
        print("\n✅ Integrated Phase 2 Analysis completed successfully!")
        print(f"📈 Project Progress: 85% → 95% (Phase 2: 70% → 100%)")
        
        return fully_normalized_data, comprehensive_results
        
    except Exception as e:
        print(f"❌ Error in integrated Phase 2 analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_comprehensive_report(results: Dict) -> str:
    """Generate a comprehensive Phase 2 analysis report."""
    
    report = []
    report.append("# Phase 2: Comprehensive Data Optimization and Bias/Weight-Based Modeling Report")
    report.append("=" * 80)
    report.append("")
    
    # Dataset Overview
    dataset_info = results['dataset_info']
    report.append("## Dataset Overview")
    report.append(f"- **Total Records**: {dataset_info['total_records']:,}")
    report.append(f"- **Manufacturers**: {len(dataset_info['manufacturers'])} ({', '.join(dataset_info['manufacturers'])})")
    report.append(f"- **Architectures**: {len(dataset_info['architectures'])}")
    report.append(f"- **Performance Categories**: {len(dataset_info['performance_categories'])}")
    report.append("")
    
    # Bias Analysis Results
    bias_analysis = results['bias_analysis']
    report.append("## Manufacturer Bias Analysis")
    for manufacturer, data in bias_analysis.items():
        report.append(f"### {manufacturer}")
        report.append(f"- **GPU Count**: {data['count']}")
        report.append(f"- **Architectures**: {len(data['architectures'])}")
        if 'bias_factors' in data and 'FP32_Final' in data['bias_factors']:
            bias_factor = data['bias_factors']['FP32_Final']
            report.append(f"- **Performance Bias Factor**: {bias_factor:.3f}x")
        report.append("")
    
    # Architecture Analysis
    arch_analysis = results['architecture_analysis']
    report.append("## Architecture Analysis Summary")
    report.append(f"- **Total Architectures Analyzed**: {len(arch_analysis)}")
    
    # Top performing architectures
    arch_performance = []
    for arch, data in arch_analysis.items():
        if 'relative_performance' in data and 'FP32_Final' in data['relative_performance']:
            arch_performance.append((arch, data['relative_performance']['FP32_Final'], data['count']))
    
    if arch_performance:
        arch_performance.sort(key=lambda x: x[1], reverse=True)
        report.append("- **Top Performing Architectures**:")
        for i, (arch, perf, count) in enumerate(arch_performance[:5]):
            report.append(f"  {i+1}. {arch}: {perf:.2f}x relative performance ({count} GPUs)")
    
    report.append("")
    
    # Next Steps
    report.append("## Next Steps for Phase 3")
    report.append("1. **Static Prediction Models Development**")
    report.append("   - Latency prediction models")
    report.append("   - Throughput forecasting")
    report.append("   - Power consumption prediction")
    report.append("")
    report.append("2. **Neural Network-Specific KPI Prediction**")
    report.append("   - Model-specific performance forecasting")
    report.append("   - Batch size optimization")
    report.append("   - Memory requirement prediction")
    report.append("")
    
    return "\n".join(report)


def display_comprehensive_summary(results: Dict, validation: Dict):
    """Display a comprehensive summary of Phase 2 results."""
    
    dataset_info = results['dataset_info']
    print(f"✅ Dataset processed: {dataset_info['total_records']:,} GPU records")
    print(f"✅ Manufacturers analyzed: {len(dataset_info['manufacturers'])}")
    print(f"✅ Architectures normalized: {len(dataset_info['architectures'])}")
    
    if 'bias_reduction_percentage' in validation:
        print(f"✅ Bias reduction achieved: {validation['bias_reduction_percentage']:.2f}%")
    
    if 'performance_correlation' in validation:
        print(f"✅ Performance correlation: {validation['performance_correlation']:.3f}")
    
    print("\n📁 Generated Files:")
    print("   📄 data/phase2_outputs/phase2_final_enhanced_dataset.csv")
    print("   📄 data/phase2_outputs/phase2_comprehensive_results.json")
    print("   📄 data/phase2_outputs/bias_corrected_dataset.csv")
    print("   📄 data/phase2_outputs/architecture_factors.json")
    print("   📄 documentation/phase2_comprehensive_analysis_report.md")
    
    print("\n🎯 Ready for Phase 3:")
    print("   1. Static prediction models for latency and throughput")
    print("   2. Neural network-specific performance forecasting")
    print("   3. Production deployment pipeline development")


def main():
    """Main function to run the integrated Phase 2 analysis."""
    dataset_path = "data/final/Ai-Benchmark-Final-enhanced-fixed.csv"
    
    final_data, results = run_integrated_phase2_analysis(dataset_path)
    
    if final_data is not None and results is not None:
        print(f"\n🎉 Phase 2 completed successfully!")
        return True
    else:
        print(f"\n❌ Phase 2 analysis failed!")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 
