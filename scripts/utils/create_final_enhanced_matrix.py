import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os

def create_final_enhanced_ai_benchmark_matrix():
    """
    Create the final enhanced AI benchmark matrix with all derived metrics.
    Apply comprehensive derivation techniques and save with standard naming.
    """
    
    print("="*80)
    print("CREATING FINAL ENHANCED AI BENCHMARK MATRIX")
    print("="*80)
    
    # Ensure output directory exists
    output_dir = 'data/final'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original dataset
    print(f"\n1. Loading AI-Benchmark-cleaned.csv...")
    df = pd.read_csv('data/AI-Benchmark-cleaned.csv')
    df_enhanced = df.copy()
    
    print(f"   Original dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    derived_metrics_count = 0
    
    # 2. DERIVE TOPs/Watt (AI Efficiency Metric)
    print(f"\n2. Deriving TOPs/Watt (AI Efficiency)...")
    if 'FP32_Final' in df.columns and 'TDP' in df.columns:
        mask = df['FP32_Final'].notna() & df['TDP'].notna() & (df['TDP'] > 0)
        df_enhanced.loc[mask, 'TOPs_per_Watt'] = (df.loc[mask, 'FP32_Final'] / 1e12) / df.loc[mask, 'TDP']
        
        derived_count = df_enhanced['TOPs_per_Watt'].notna().sum()
        print(f"   ✅ Derived TOPs_per_Watt for {derived_count} devices")
        print(f"   📊 Range: {df_enhanced['TOPs_per_Watt'].min():.6f} - {df_enhanced['TOPs_per_Watt'].max():.6f}")
        derived_metrics_count += 1
    
    # 3. DERIVE Relative Latency
    print(f"\n3. Deriving Relative Latency...")
    if 'FP32_Final' in df.columns:
        mask = df['FP32_Final'].notna() & (df['FP32_Final'] > 0)
        max_flops = df.loc[mask, 'FP32_Final'].max()
        df_enhanced.loc[mask, 'Relative_Latency_Index'] = max_flops / df.loc[mask, 'FP32_Final']
        
        derived_count = df_enhanced['Relative_Latency_Index'].notna().sum()
        print(f"   ✅ Derived Relative_Latency_Index for {derived_count} devices")
        print(f"   📊 Range: {df_enhanced['Relative_Latency_Index'].min():.2f} - {df_enhanced['Relative_Latency_Index'].max():.2f}")
        derived_metrics_count += 1
    
    # 4. DERIVE Compute Usage Estimation
    print(f"\n4. Deriving Compute Usage Estimation...")
    if 'powerPerformance' in df.columns and 'TDP' in df.columns:
        mask = df['powerPerformance'].notna() & df['TDP'].notna()
        max_efficiency = df.loc[mask, 'powerPerformance'].max()
        df_enhanced.loc[mask, 'Compute_Usage_Percent'] = (df.loc[mask, 'powerPerformance'] / max_efficiency) * 100
        
        derived_count = df_enhanced['Compute_Usage_Percent'].notna().sum()
        print(f"   ✅ Derived Compute_Usage_Percent for {derived_count} devices")
        print(f"   📊 Range: {df_enhanced['Compute_Usage_Percent'].min():.2f}% - {df_enhanced['Compute_Usage_Percent'].max():.2f}%")
        derived_metrics_count += 1
    
    # 5. DERIVE Throughput Estimation (with standard model assumptions)
    print(f"\n5. Deriving Throughput Estimation...")
    if 'FP32_Final' in df.columns:
        # Standard model complexities (FLOPS per inference)
        standard_models = {
            'ResNet50_ImageNet': 4.1e9,      # 4.1 GFLOPS
            'BERT_Base': 22.5e9,             # 22.5 GFLOPS  
            'GPT2_Small': 1.5e9,             # 1.5 GFLOPS
            'MobileNetV2': 0.3e9,            # 0.3 GFLOPS
            'EfficientNet_B0': 0.39e9        # 0.39 GFLOPS
        }
        
        mask = df['FP32_Final'].notna() & (df['FP32_Final'] > 0)
        
        # Calculate throughput for each standard model
        for model_name, model_flops in standard_models.items():
            col_name = f'Throughput_{model_name}_fps'
            df_enhanced.loc[mask, col_name] = df.loc[mask, 'FP32_Final'] / model_flops
        
        # Average throughput across models
        throughput_cols = [f'Throughput_{model}_fps' for model in standard_models.keys()]
        df_enhanced['Avg_Throughput_fps'] = df_enhanced[throughput_cols].mean(axis=1)
        
        derived_count = df_enhanced['Avg_Throughput_fps'].notna().sum()
        print(f"   ✅ Derived throughput for {len(standard_models)} standard models + average")
        print(f"   📊 Average throughput range: {df_enhanced['Avg_Throughput_fps'].min():.2f} - {df_enhanced['Avg_Throughput_fps'].max():.2f} fps")
        derived_metrics_count += len(standard_models) + 1
    
    # 6. DERIVE Enhanced Precision Metrics
    print(f"\n6. Deriving Enhanced Precision Metrics...")
    
    # FP16 Performance Prediction
    fp16_col = 'FP16 (half precision) performance (FLOP/s)'
    if fp16_col in df.columns and 'FP32_Final' in df.columns:
        valid_data = df[df[fp16_col].notna() & df['FP32_Final'].notna()]
        
        if len(valid_data) >= 10:
            # Build linear regression model
            X = valid_data[['FP32_Final']]
            y = valid_data[fp16_col]
            
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            
            if r2 > 0.3:  # Reasonable correlation
                # Predict missing FP16 values
                missing_mask = df[fp16_col].isna() & df['FP32_Final'].notna()
                predictions = model.predict(df.loc[missing_mask, ['FP32_Final']])
                df_enhanced.loc[missing_mask, 'FP16_Performance_Predicted'] = predictions
                
                predicted_count = (df_enhanced['FP16_Performance_Predicted'].notna()).sum()
                print(f"   ✅ Predicted FP16 performance for {predicted_count} devices (R²: {r2:.3f})")
                derived_metrics_count += 1
            else:
                print(f"   ⚠️  FP16 correlation too weak (R²: {r2:.3f})")
        else:
            print(f"   ⚠️  Insufficient FP16 training data ({len(valid_data)} samples)")
    
    # Architecture-based INT8 estimation
    if 'Architecture' in df.columns and 'FP32_Final' in df.columns:
        # Typical INT8 speedup factors by architecture
        int8_speedup_factors = {
            'Turing': 4.0,        # Tensor cores
            'Ampere': 5.0,        # Improved tensor cores
            'Ada Lovelace': 6.0,  # Latest tensor cores
            'RDNA 2': 2.0,        # AMD INT8 support
            'RDNA': 1.8,          # Limited INT8
            'GCN': 1.5,           # Basic INT8
            'Pascal': 2.0,        # DP4A instruction
            'Maxwell': 1.2,       # Limited INT8
            'Unknown': 2.0        # Conservative estimate
        }
        
        mask = df['FP32_Final'].notna()
        df_enhanced['INT8_Performance_Estimated'] = np.nan
        
        for arch, speedup in int8_speedup_factors.items():
            arch_mask = mask & (df['Architecture'] == arch)
            if arch_mask.sum() > 0:
                df_enhanced.loc[arch_mask, 'INT8_Performance_Estimated'] = (
                    df.loc[arch_mask, 'FP32_Final'] * speedup
                )
        
        derived_count = df_enhanced['INT8_Performance_Estimated'].notna().sum()
        print(f"   ✅ Estimated INT8 performance for {derived_count} devices")
        derived_metrics_count += 1
    
    # 7. DERIVE Memory Usage Estimation
    print(f"\n7. Deriving Memory Usage Estimation...")
    if 'G3Dmark' in df.columns and 'Memory_GB' in df.columns:
        # Estimate memory usage based on performance and available memory
        mask = df['G3Dmark'].notna() & df['Memory_GB'].notna() & (df['Memory_GB'] > 0)
        
        if mask.sum() > 100:  # Sufficient data
            # Normalize by memory size - higher performance uses more memory percentage
            max_score = df.loc[mask, 'G3Dmark'].max()
            df_enhanced.loc[mask, 'Memory_Usage_Estimated_Percent'] = (
                (df.loc[mask, 'G3Dmark'] / max_score) * 
                (8 / df.loc[mask, 'Memory_GB']).clip(0.2, 1.0) * 100
            ).clip(10, 95)  # Reasonable usage range
            
            derived_count = df_enhanced['Memory_Usage_Estimated_Percent'].notna().sum()
            print(f"   ✅ Estimated memory usage for {derived_count} devices")
            derived_metrics_count += 1
        else:
            print(f"   ⚠️  Insufficient memory specification data")
    
    # 8. DERIVE Power Efficiency Metrics
    print(f"\n8. Deriving Additional Power Efficiency Metrics...")
    if 'FP32_Final' in df.columns and 'TDP' in df.columns:
        mask = df['FP32_Final'].notna() & df['TDP'].notna() & (df['TDP'] > 0)
        
        # GFLOPS per Watt
        df_enhanced.loc[mask, 'GFLOPS_per_Watt'] = (df.loc[mask, 'FP32_Final'] / 1e9) / df.loc[mask, 'TDP']
        
        # Performance per Dollar per Watt (where price available)
        if 'price' in df.columns:
            price_mask = mask & df['price'].notna() & (df['price'] > 0)
            df_enhanced.loc[price_mask, 'Performance_per_Dollar_per_Watt'] = (
                (df.loc[price_mask, 'FP32_Final'] / 1e9) / 
                (df.loc[price_mask, 'price'] * df.loc[price_mask, 'TDP'])
            )
            
            derived_count = df_enhanced['Performance_per_Dollar_per_Watt'].notna().sum()
            print(f"   ✅ Derived Performance_per_Dollar_per_Watt for {derived_count} devices")
        
        derived_count = df_enhanced['GFLOPS_per_Watt'].notna().sum()
        print(f"   ✅ Derived GFLOPS_per_Watt for {derived_count} devices")
        derived_metrics_count += 2
    
    # 9. ARCHITECTURE-BASED IMPUTATION
    print(f"\n9. Performing Architecture-based Imputation...")
    if 'Architecture' in df.columns:
        # Calculate architecture medians for key metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        arch_medians = df.groupby('Architecture')[numeric_cols].median()
        
        imputed_values = 0
        for arch in arch_medians.index:
            arch_mask = df['Architecture'] == arch
            
            # Impute critical missing values
            for col in ['TDP', 'FP32_Final', 'FLOPS_per_Watt']:
                if col in df.columns:
                    missing_mask = arch_mask & df[col].isna()
                    if missing_mask.sum() > 0 and not pd.isna(arch_medians.loc[arch, col]):
                        df_enhanced.loc[missing_mask, f'{col}_Imputed'] = arch_medians.loc[arch, col]
                        imputed_values += missing_mask.sum()
        
        print(f"   ✅ Architecture-based imputation: {imputed_values} values")
        derived_metrics_count += 3  # For the imputed columns
    
    # 10. CREATE PERFORMANCE TIERS AND CATEGORIES
    print(f"\n10. Creating Enhanced Performance Categories...")
    if 'TOPs_per_Watt' in df_enhanced.columns:
        # AI Efficiency Tiers
        df_enhanced['AI_Efficiency_Tier'] = pd.cut(
            df_enhanced['TOPs_per_Watt'],
            bins=[0, 0.01, 0.03, 0.06, 0.1, float('inf')],
            labels=['Entry', 'Mid-Range', 'High-End', 'Premium', 'Ultra']
        )
        
        # Performance Category based on multiple metrics
        if 'FP32_Final' in df.columns:
            df_enhanced['AI_Performance_Category'] = 'Unknown'
            
            # Define categories based on FLOPS and efficiency
            conditions = [
                (df_enhanced['FP32_Final'] >= 20e12) & (df_enhanced['TOPs_per_Watt'] >= 0.05),
                (df_enhanced['FP32_Final'] >= 15e12) & (df_enhanced['TOPs_per_Watt'] >= 0.03),
                (df_enhanced['FP32_Final'] >= 10e12) & (df_enhanced['TOPs_per_Watt'] >= 0.02),
                (df_enhanced['FP32_Final'] >= 5e12) & (df_enhanced['TOPs_per_Watt'] >= 0.01),
                df_enhanced['FP32_Final'] >= 1e12
            ]
            
            choices = ['AI_Flagship', 'AI_High_End', 'AI_Mid_Range', 'AI_Entry', 'AI_Basic']
            
            df_enhanced['AI_Performance_Category'] = np.select(conditions, choices, default='AI_Legacy')
        
        print(f"   ✅ Created AI performance tiers and categories")
        derived_metrics_count += 2
    
    # 11. SAVE THE ENHANCED MATRIX
    print(f"\n11. Saving Final Enhanced Matrix...")
    
    # Standard naming convention
    output_filename = 'ai_benchmark_enhanced_comprehensive_matrix.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    # Save with proper formatting
    df_enhanced.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"   📁 Saved: {output_path}")
    print(f"   📊 Final dataset: {df_enhanced.shape[0]} rows × {df_enhanced.shape[1]} columns")
    print(f"   🆕 Added {df_enhanced.shape[1] - df.shape[1]} new columns")
    
    # 12. GENERATE SUMMARY REPORT
    print(f"\n12. Generating Enhancement Summary...")
    
    new_columns = [col for col in df_enhanced.columns if col not in df.columns]
    
    print(f"\n📊 ENHANCEMENT SUMMARY:")
    print(f"   Original columns: {df.shape[1]}")
    print(f"   Enhanced columns: {df_enhanced.shape[1]}")
    print(f"   New derived metrics: {len(new_columns)}")
    
    print(f"\n🆕 NEW DERIVED COLUMNS:")
    for i, col in enumerate(new_columns, 1):
        non_null_count = df_enhanced[col].notna().sum()
        coverage = (non_null_count / len(df_enhanced)) * 100
        print(f"   {i:2d}. {col}: {non_null_count} values ({coverage:.1f}%)")
    
    # Data quality metrics
    print(f"\n📈 DATA QUALITY IMPROVEMENT:")
    
    original_complete_cols = sum(1 for col in df.columns if df[col].notna().sum() / len(df) > 0.9)
    enhanced_complete_cols = sum(1 for col in df_enhanced.columns if df_enhanced[col].notna().sum() / len(df_enhanced) > 0.9)
    
    print(f"   Columns with >90% data:")
    print(f"     Original: {original_complete_cols}")
    print(f"     Enhanced: {enhanced_complete_cols}")
    print(f"     Improvement: +{enhanced_complete_cols - original_complete_cols}")
    
    # Key metrics coverage
    key_metrics = [
        'TOPs_per_Watt', 'Relative_Latency_Index', 'Compute_Usage_Percent',
        'Avg_Throughput_fps', 'GFLOPS_per_Watt', 'AI_Efficiency_Tier'
    ]
    
    print(f"\n🎯 KEY AI METRICS COVERAGE:")
    for metric in key_metrics:
        if metric in df_enhanced.columns:
            coverage = (df_enhanced[metric].notna().sum() / len(df_enhanced)) * 100
            print(f"   • {metric}: {coverage:.1f}% coverage")
    
    print(f"\n✅ FINAL ENHANCED MATRIX READY FOR AI MODELING!")
    print(f"   Location: {output_path}")
    print(f"   Ready for: Bias/weight-based modeling, Performance prediction, KPI analysis")
    
    return df_enhanced, output_path

if __name__ == "__main__":
    enhanced_df, file_path = create_final_enhanced_ai_benchmark_matrix()
    print(f"\n🎉 Enhancement complete! Final matrix saved at: {file_path}") 