import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

def analyze_derivable_metrics():
    """
    Analyze if missing critical metrics can be derived from existing data relationships.
    """
    
    print("="*80)
    print("ANALYSIS: DERIVING MISSING METRICS FROM EXISTING DATA RELATIONSHIPS")
    print("="*80)
    
    # Load datasets
    ai_bench = pd.read_csv('data/AI-Benchmark-cleaned.csv')
    
    print(f"\nDataset loaded: {ai_bench.shape[0]} rows, {ai_bench.shape[1]} columns")
    
    # Define missing metrics and their potential derivation strategies
    missing_metrics = {
        'Latency': {
            'description': 'Time to process a single inference (ms)',
            'potential_sources': ['FP32_Final', 'G3Dmark', 'Architecture', 'Process size (nm)'],
            'derivation_strategy': 'Inverse relationship with FLOPS and clock speed'
        },
        'Throughput': {
            'description': 'Inferences per second or samples per second',
            'potential_sources': ['FP32_Final', 'FLOPS_per_Watt', 'TDP', 'Memory bandwidth (byte/s)'],
            'derivation_strategy': 'Direct relationship with FLOPS and memory bandwidth'
        },
        'TOPs_per_Watt': {
            'description': 'Tera Operations per Second per Watt (AI efficiency)',
            'potential_sources': ['FP32_Final', 'TDP', 'FLOPS_per_Watt'],
            'derivation_strategy': 'Convert FLOPS to TOPs and normalize by TDP'
        },
        'Memory_Usage_Percent': {
            'description': 'Percentage of memory bandwidth utilized',
            'potential_sources': ['Memory bandwidth (byte/s)', 'Memory_GB', 'G3Dmark'],
            'derivation_strategy': 'Estimate from performance scores and memory specs'
        },
        'Compute_Usage_Percent': {
            'description': 'Percentage of compute units utilized',
            'potential_sources': ['FP32_Final', 'TDP', 'powerPerformance', 'Architecture'],
            'derivation_strategy': 'Estimate from FLOPS utilization vs theoretical max'
        }
    }
    
    print(f"\n1. MISSING METRICS ANALYSIS:")
    for metric, info in missing_metrics.items():
        print(f"\n   📊 {metric}:")
        print(f"      Description: {info['description']}")
        print(f"      Strategy: {info['derivation_strategy']}")
        
        # Check availability of source columns
        available_sources = []
        missing_sources = []
        
        for source in info['potential_sources']:
            if source in ai_bench.columns:
                non_null_count = ai_bench[source].notna().sum()
                coverage = (non_null_count / len(ai_bench)) * 100
                if coverage > 50:  # At least 50% coverage
                    available_sources.append((source, coverage))
                else:
                    missing_sources.append((source, coverage))
            else:
                missing_sources.append((source, 0))
        
        print(f"      ✅ Available sources: {[(s, f'{c:.1f}%') for s, c in available_sources]}")
        print(f"      ❌ Limited sources: {[(s, f'{c:.1f}%') for s, c in missing_sources]}")
    
    print(f"\n2. DERIVATION FEASIBILITY ANALYSIS:")
    
    # 1. Derive TOPs/Watt from existing FLOPS data
    print(f"\n   🔬 ANALYZING TOPs/Watt DERIVATION:")
    if 'FP32_Final' in ai_bench.columns and 'TDP' in ai_bench.columns:
        # Calculate TOPs/Watt = (FLOPS / 1e12) / TDP
        valid_data = ai_bench[ai_bench['FP32_Final'].notna() & ai_bench['TDP'].notna()]
        
        if len(valid_data) > 0:
            valid_data = valid_data.copy()
            valid_data['TOPs_per_Watt'] = (valid_data['FP32_Final'] / 1e12) / valid_data['TDP']
            
            print(f"      ✅ Successfully derived TOPs/Watt for {len(valid_data)} devices")
            print(f"      📈 Range: {valid_data['TOPs_per_Watt'].min():.4f} - {valid_data['TOPs_per_Watt'].max():.4f}")
            print(f"      📊 Mean: {valid_data['TOPs_per_Watt'].mean():.4f} TOPs/Watt")
            
            # Check correlation with existing efficiency metrics
            if 'FLOPS_per_Watt' in ai_bench.columns:
                correlation = valid_data['TOPs_per_Watt'].corr(valid_data['FLOPS_per_Watt'])
                print(f"      🔗 Correlation with FLOPS_per_Watt: {correlation:.3f}")
        else:
            print(f"      ❌ Insufficient data for derivation")
    
    # 2. Estimate Latency from FLOPS and architecture
    print(f"\n   🔬 ANALYZING LATENCY DERIVATION:")
    if 'FP32_Final' in ai_bench.columns and 'Architecture' in ai_bench.columns:
        valid_data = ai_bench[ai_bench['FP32_Final'].notna()]
        
        if len(valid_data) > 0:
            # Estimate latency using inverse FLOPS relationship
            # Latency ≈ k / FLOPS (where k is architecture-dependent constant)
            
            # Group by architecture to find architecture-specific constants
            arch_groups = valid_data.groupby('Architecture').agg({
                'FP32_Final': ['mean', 'std', 'count'],
                'G3Dmark': 'mean'
            }).round(2)
            
            print(f"      📊 Architecture analysis for {len(arch_groups)} architectures:")
            print(f"      Top architectures by FLOPS:")
            
            top_archs = arch_groups.sort_values(('FP32_Final', 'mean'), ascending=False).head(5)
            for arch, data in top_archs.iterrows():
                if data[('FP32_Final', 'count')] >= 5:  # At least 5 samples
                    mean_flops = data[('FP32_Final', 'mean')]
                    count = data[('FP32_Final', 'count')]
                    # Estimate relative latency (inverse of FLOPS, normalized)
                    rel_latency = 1e12 / mean_flops if mean_flops > 0 else float('inf')
                    print(f"         {arch}: {mean_flops:.2e} FLOPS ({count} devices) → Est. rel. latency: {rel_latency:.2f}")
            
            print(f"      ✅ Can derive relative latency estimates")
        else:
            print(f"      ❌ Insufficient FLOPS data")
    
    # 3. Estimate Throughput from FLOPS and Memory Bandwidth
    print(f"\n   🔬 ANALYZING THROUGHPUT DERIVATION:")
    throughput_sources = ['FP32_Final', 'Memory bandwidth (byte/s)', 'TDP']
    available_for_throughput = []
    
    for source in throughput_sources:
        if source in ai_bench.columns:
            coverage = (ai_bench[source].notna().sum() / len(ai_bench)) * 100
            available_for_throughput.append((source, coverage))
    
    print(f"      📊 Available sources: {[(s, f'{c:.1f}%') for s, c in available_for_throughput]}")
    
    # Check if we can estimate throughput from FLOPS
    flops_coverage = next((c for s, c in available_for_throughput if s == 'FP32_Final'), 0)
    if flops_coverage > 90:
        print(f"      ✅ Can estimate throughput from FLOPS (assuming typical model complexities)")
        print(f"      💡 Throughput ≈ FLOPS / (Model_FLOPS_per_inference)")
    else:
        print(f"      ❌ Limited FLOPS data for throughput estimation")
    
    # 4. Analyze Memory and Compute Usage derivation
    print(f"\n   🔬 ANALYZING MEMORY/COMPUTE USAGE DERIVATION:")
    
    # Memory usage estimation
    memory_sources = ['Memory bandwidth (byte/s)', 'Memory_GB', 'G3Dmark']
    memory_coverage = []
    for source in memory_sources:
        if source in ai_bench.columns:
            coverage = (ai_bench[source].notna().sum() / len(ai_bench)) * 100
            memory_coverage.append((source, coverage))
    
    print(f"      📊 Memory sources: {[(s, f'{c:.1f}%') for s, c in memory_coverage]}")
    
    # Compute usage estimation
    if 'powerPerformance' in ai_bench.columns and 'TDP' in ai_bench.columns:
        # Compute usage can be estimated from power efficiency
        valid_power = ai_bench[ai_bench['powerPerformance'].notna() & ai_bench['TDP'].notna()]
        if len(valid_power) > 0:
            print(f"      ✅ Can estimate compute usage from power efficiency data ({len(valid_power)} devices)")
            print(f"      💡 Compute_Usage ≈ f(powerPerformance, TDP, Architecture)")
        else:
            print(f"      ❌ Insufficient power data")
    
    # 5. Check for precision data recovery
    print(f"\n   🔬 ANALYZING PRECISION DATA RECOVERY:")
    precision_cols = ['FP16 (half precision) performance (FLOP/s)', 'INT8 performance (OP/s)']
    
    for col in precision_cols:
        if col in ai_bench.columns:
            available_count = ai_bench[col].notna().sum()
            coverage = (available_count / len(ai_bench)) * 100
            print(f"      📊 {col}: {available_count} devices ({coverage:.1f}%)")
            
            if available_count > 50:  # If we have some data
                # Check correlation with FP32 data
                valid_precision = ai_bench[ai_bench[col].notna() & ai_bench['FP32_Final'].notna()]
                if len(valid_precision) > 10:
                    correlation = valid_precision[col].corr(valid_precision['FP32_Final'])
                    print(f"         🔗 Correlation with FP32: {correlation:.3f}")
                    
                    if abs(correlation) > 0.7:
                        print(f"         ✅ Strong correlation - can interpolate missing values")
                        
                        # Estimate missing values using linear relationship
                        X = valid_precision[['FP32_Final']]
                        y = valid_precision[col]
                        
                        if len(X) > 5:
                            model = LinearRegression()
                            model.fit(X, y)
                            r2 = model.score(X, y)
                            print(f"         📈 Linear model R²: {r2:.3f}")
                            
                            if r2 > 0.5:
                                print(f"         ✅ Can predict {col} from FP32_Final")
                            else:
                                print(f"         ⚠️  Weak predictive power")
    
    # 6. Architecture-based metric estimation
    print(f"\n   🔬 ANALYZING ARCHITECTURE-BASED ESTIMATION:")
    
    if 'Architecture' in ai_bench.columns:
        arch_stats = ai_bench.groupby('Architecture').agg({
            'FP32_Final': ['mean', 'std', 'count'],
            'TDP': ['mean', 'std'],
            'FLOPS_per_Watt': ['mean', 'std']
        }).round(3)
        
        # Find architectures with sufficient data for modeling
        reliable_archs = arch_stats[arch_stats[('FP32_Final', 'count')] >= 10]
        
        print(f"      📊 Architectures with sufficient data (≥10 devices): {len(reliable_archs)}")
        
        if len(reliable_archs) > 0:
            print(f"      ✅ Can use architecture-based imputation for missing metrics")
            print(f"      💡 Strategy: Use architecture means/medians for missing values")
            
            # Show top architectures
            top_reliable = reliable_archs.sort_values(('FP32_Final', 'count'), ascending=False).head(3)
            for arch, data in top_reliable.iterrows():
                count = data[('FP32_Final', 'count')]
                mean_flops = data[('FP32_Final', 'mean')]
                print(f"         • {arch}: {count} devices, avg {mean_flops:.2e} FLOPS")
    
    print(f"\n3. DERIVATION RECOMMENDATIONS:")
    
    recommendations = [
        "✅ TOPs/Watt: Directly derivable from FP32_Final and TDP",
        "✅ Relative Latency: Estimable from inverse FLOPS relationship",
        "⚠️  Throughput: Requires model complexity assumptions",
        "⚠️  Memory Usage %: Limited by sparse memory bandwidth data",
        "✅ Compute Usage %: Derivable from power efficiency metrics",
        "✅ Precision Recovery: Use linear interpolation from FP32 data",
        "✅ Architecture Imputation: Use group statistics for missing values"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n4. IMPLEMENTATION PRIORITY:")
    
    high_priority = [
        "1. Calculate TOPs/Watt from existing FLOPS and TDP data",
        "2. Estimate relative latency using architecture-specific FLOPS constants",
        "3. Recover FP16/INT8 precision using linear models from FP32",
        "4. Implement architecture-based imputation for sparse columns"
    ]
    
    medium_priority = [
        "5. Estimate throughput with model complexity assumptions",
        "6. Calculate compute usage from power efficiency patterns",
        "7. Interpolate memory specifications using similar devices"
    ]
    
    print(f"\n   🚀 HIGH PRIORITY (Immediate Implementation):")
    for item in high_priority:
        print(f"      {item}")
    
    print(f"\n   ⏳ MEDIUM PRIORITY (Phase 2):")
    for item in medium_priority:
        print(f"      {item}")
    
    return ai_bench

def create_derived_metrics_dataset():
    """
    Create an enhanced dataset with derived metrics.
    """
    
    print(f"\n" + "="*80)
    print("CREATING ENHANCED DATASET WITH DERIVED METRICS")
    print("="*80)
    
    df = pd.read_csv('data/AI-Benchmark-cleaned.csv')
    df_enhanced = df.copy()
    
    # 1. Calculate TOPs/Watt
    if 'FP32_Final' in df.columns and 'TDP' in df.columns:
        mask = df['FP32_Final'].notna() & df['TDP'].notna() & (df['TDP'] > 0)
        df_enhanced.loc[mask, 'TOPs_per_Watt'] = (df.loc[mask, 'FP32_Final'] / 1e12) / df.loc[mask, 'TDP']
        derived_count = df_enhanced['TOPs_per_Watt'].notna().sum()
        print(f"✅ Derived TOPs_per_Watt for {derived_count} devices")
    
    # 2. Estimate relative latency
    if 'FP32_Final' in df.columns:
        mask = df['FP32_Final'].notna() & (df['FP32_Final'] > 0)
        # Normalized relative latency (lower is better)
        max_flops = df.loc[mask, 'FP32_Final'].max()
        df_enhanced.loc[mask, 'Relative_Latency'] = max_flops / df.loc[mask, 'FP32_Final']
        derived_count = df_enhanced['Relative_Latency'].notna().sum()
        print(f"✅ Derived Relative_Latency for {derived_count} devices")
    
    # 3. Calculate compute usage estimate
    if 'powerPerformance' in df.columns and 'TDP' in df.columns:
        mask = df['powerPerformance'].notna() & df['TDP'].notna()
        # Normalize powerPerformance to estimate compute usage
        max_efficiency = df.loc[mask, 'powerPerformance'].max()
        df_enhanced.loc[mask, 'Compute_Usage_Est'] = df.loc[mask, 'powerPerformance'] / max_efficiency * 100
        derived_count = df_enhanced['Compute_Usage_Est'].notna().sum()
        print(f"✅ Derived Compute_Usage_Est for {derived_count} devices")
    
    # 4. Recover FP16 precision using linear model
    fp16_col = 'FP16 (half precision) performance (FLOP/s)'
    if fp16_col in df.columns and 'FP32_Final' in df.columns:
        # Use existing FP16 data to build model
        valid_data = df[df[fp16_col].notna() & df['FP32_Final'].notna()]
        
        if len(valid_data) > 10:
            X = valid_data[['FP32_Final']]
            y = valid_data[fp16_col]
            
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            
            if r2 > 0.5:
                # Predict missing FP16 values
                missing_mask = df[fp16_col].isna() & df['FP32_Final'].notna()
                predictions = model.predict(df.loc[missing_mask, ['FP32_Final']])
                df_enhanced.loc[missing_mask, f'{fp16_col}_Estimated'] = predictions
                
                predicted_count = (df_enhanced[f'{fp16_col}_Estimated'].notna()).sum()
                print(f"✅ Estimated FP16 performance for {predicted_count} devices (R²: {r2:.3f})")
    
    # 5. Architecture-based imputation
    if 'Architecture' in df.columns:
        # Calculate architecture-specific medians for key metrics
        arch_medians = df.groupby('Architecture').agg({
            'FP32_Final': 'median',
            'TDP': 'median',
            'FLOPS_per_Watt': 'median'
        })
        
        imputed_count = 0
        for arch in arch_medians.index:
            arch_mask = df['Architecture'] == arch
            
            # Impute missing FP32_Final values
            if 'FP32_Final' in df.columns:
                missing_mask = arch_mask & df['FP32_Final'].isna()
                if missing_mask.sum() > 0 and not pd.isna(arch_medians.loc[arch, 'FP32_Final']):
                    df_enhanced.loc[missing_mask, 'FP32_Final_Imputed'] = arch_medians.loc[arch, 'FP32_Final']
                    imputed_count += missing_mask.sum()
        
        if imputed_count > 0:
            print(f"✅ Architecture-based imputation for {imputed_count} missing values")
    
    # Save enhanced dataset
    output_file = 'data/processed/ai_benchmark_enhanced_with_derived_metrics.csv'
    df_enhanced.to_csv(output_file, index=False)
    
    print(f"\n📁 Enhanced dataset saved: {output_file}")
    print(f"📊 Shape: {df_enhanced.shape[0]} rows × {df_enhanced.shape[1]} columns")
    
    # Summary of enhancements
    new_columns = [col for col in df_enhanced.columns if col not in df.columns]
    print(f"🆕 New derived columns ({len(new_columns)}):")
    for col in new_columns:
        count = df_enhanced[col].notna().sum()
        print(f"   • {col}: {count} values")
    
    return df_enhanced

if __name__ == "__main__":
    # Run analysis
    df = analyze_derivable_metrics()
    
    # Create enhanced dataset
    df_enhanced = create_derived_metrics_dataset() 