import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_final_dataset():
    """Analyze the final enhanced dataset structure for documentation."""
    
    print("="*80)
    print("FINAL ENHANCED AI BENCHMARK MATRIX - COLUMN ANALYSIS")
    print("="*80)
    
    # Load the dataset
    df = pd.read_csv('data/final/ai_benchmark_enhanced_comprehensive_matrix.csv')
    
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn analysis:")
    
    # Original columns (first 31)
    original_cols = df.columns[:31]
    derived_cols = df.columns[31:]
    
    print(f"\n📊 ORIGINAL COLUMNS (1-31):")
    for i, col in enumerate(original_cols, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = ((len(df) - non_null) / len(df)) * 100
        
        # Sample values
        sample_vals = df[col].dropna().head(3).tolist()
        if dtype == 'object':
            unique_count = df[col].nunique()
            sample_str = f"Examples: {sample_vals} (Total unique: {unique_count})"
        else:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            sample_str = f"Range: {min_val:.2f} - {max_val:.2f}, Mean: {mean_val:.2f}"
        
        print(f"{i:2d}. {col:<35} | {dtype:<10} | {non_null:4d}/{len(df)} ({null_pct:5.1f}% missing)")
        print(f"    {sample_str}")
    
    print(f"\n🆕 DERIVED COLUMNS (32-46):")
    for i, col in enumerate(derived_cols, 32):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = ((len(df) - non_null) / len(df)) * 100
        
        # Sample values
        sample_vals = df[col].dropna().head(3).tolist()
        if dtype == 'object':
            unique_count = df[col].nunique()
            sample_str = f"Examples: {sample_vals} (Total unique: {unique_count})"
        else:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            sample_str = f"Range: {min_val:.6f} - {max_val:.6f}, Mean: {mean_val:.6f}"
        
        print(f"{i:2d}. {col:<35} | {dtype:<10} | {non_null:4d}/{len(df)} ({null_pct:5.1f}% missing)")
        print(f"    {sample_str}")
    
    # Key statistics
    print(f"\n📈 KEY STATISTICS:")
    print(f"Total devices: {len(df)}")
    print(f"Manufacturers: {df['Manufacturer'].nunique()} unique")
    print(f"Architectures: {df['Architecture'].nunique()} unique")
    print(f"Complete records (no missing values): {df.dropna().shape[0]}")
    print(f"Records with >90% data: {sum(1 for col in df.columns if df[col].notna().sum() / len(df) > 0.9)}")

if __name__ == "__main__":
    analyze_final_dataset()

# Load the final dataset
print("Loading the final dataset...")
df = pd.read_csv('data/processed/ai_hardware_performance_matrix.csv')

# Basic dataset information
print(f"Dataset shape: {df.shape}")
print(f"Number of columns: {len(df.columns)}")

# Count non-null values for key columns
print("\nNumber of non-null values for key metrics:")
key_metrics = [
    'FP32_TFLOPS', 'FP16_TFLOPS', 'INT8_TFLOPS', 
    'Memory_bandwidth_GB_s', 'TDP_numeric', 'perf_per_watt',
    'memory_bandwidth_compute_ratio'
]
print(df[key_metrics].count())

# Check for hardware types distribution
if 'Type' in df.columns:
    print("\nHardware type distribution:")
    print(df['Type'].value_counts())

# Check for hardware manufacturers distribution
if 'Manufacturer' in df.columns:
    print("\nTop 10 manufacturers:")
    print(df['Manufacturer'].value_counts().head(10))

# Check for release year distribution
if 'Release_year' in df.columns:
    print("\nRelease year distribution:")
    print(df['Release_year'].value_counts().sort_index())

# Check for correlations between key performance metrics
print("\nCorrelation between key performance metrics:")
corr_metrics = key_metrics + ['FP16_to_FP32_scaling', 'INT8_to_FP32_scaling']
corr_matrix = df[corr_metrics].corr()
print(corr_matrix)

# Save correlation matrix to CSV for reference
corr_matrix.to_csv('data/processed/performance_metrics_correlation.csv')

# Check for MLPerf benchmarks availability
mlperf_columns = [col for col in df.columns if 'Offline' in col or 'Server' in col]
print(f"\nNumber of MLPerf benchmark columns: {len(mlperf_columns)}")
print("MLPerf benchmark columns:", mlperf_columns)

# Check for normalized metrics
normalized_columns = [col for col in df.columns if 'normalized' in col]
print(f"\nNumber of normalized metric columns: {len(normalized_columns)}")
print("Normalized metric columns:", normalized_columns)

# Check for missing values percentage
print("\nMissing values percentage for key columns:")
missing_pct = df[key_metrics].isna().mean() * 100
print(missing_pct)

# Print a sample of rows with the most complete data
print("\nSample of rows with most complete data:")
# Calculate completeness score for each row (percentage of non-null values in key metrics)
completeness = df[key_metrics].notna().mean(axis=1)
# Get indices of the 5 most complete rows
most_complete_indices = completeness.nlargest(5).index
# Print these rows
print(df.loc[most_complete_indices, ['Hardware name', 'Type', 'Release_year', 'TDP_numeric'] + key_metrics])

# Print summary statistics for key performance metrics
print("\nSummary statistics for key performance metrics:")
print(df[key_metrics].describe())

print("\nAnalysis complete!") 