import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory for visualizations
os.makedirs('analysis/visualizations', exist_ok=True)

# Load the final dataset
print("Loading the final dataset...")
df = pd.read_csv('data/processed/ai_hardware_performance_matrix.csv')

# Set the style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Performance Distribution by Hardware Type
print("Generating performance distribution by hardware type...")
if 'Type' in df.columns and 'FP32_TFLOPS' in df.columns:
    plt.figure(figsize=(14, 8))
    # Filter to include only rows with both Type and FP32_TFLOPS
    filtered_df = df[df['Type'].notna() & df['FP32_TFLOPS'].notna()]
    if len(filtered_df) > 0:
        sns.boxplot(x='Type', y='FP32_TFLOPS', data=filtered_df)
        plt.title('FP32 Performance Distribution by Hardware Type')
        plt.ylabel('TFLOPS (FP32)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analysis/visualizations/perf_by_hardware_type.png')
        plt.close()

# 2. Performance vs. Power Efficiency
print("Generating performance vs. power efficiency plot...")
if 'FP32_TFLOPS' in df.columns and 'TDP_numeric' in df.columns:
    plt.figure(figsize=(12, 8))
    # Filter to include only rows with both metrics
    filtered_df = df[df['FP32_TFLOPS'].notna() & df['TDP_numeric'].notna()]
    if len(filtered_df) > 0:
        # Add Type for color if available
        if 'Type' in df.columns:
            types = filtered_df['Type'].fillna('Unknown')
            sns.scatterplot(x='TDP_numeric', y='FP32_TFLOPS', hue=types, data=filtered_df, alpha=0.7)
        else:
            sns.scatterplot(x='TDP_numeric', y='FP32_TFLOPS', data=filtered_df, alpha=0.7)
        
        plt.title('Performance vs. Power Consumption')
        plt.xlabel('TDP (Watts)')
        plt.ylabel('TFLOPS (FP32)')
        plt.tight_layout()
        plt.savefig('analysis/visualizations/perf_vs_power.png')
        plt.close()

# 3. Memory Bandwidth vs. Compute Performance
print("Generating memory bandwidth vs. compute performance plot...")
if 'Memory_bandwidth_GB_s' in df.columns and 'FP32_TFLOPS' in df.columns:
    plt.figure(figsize=(12, 8))
    # Filter to include only rows with both metrics
    filtered_df = df[df['Memory_bandwidth_GB_s'].notna() & df['FP32_TFLOPS'].notna()]
    if len(filtered_df) > 0:
        # Add Type for color if available
        if 'Type' in df.columns:
            types = filtered_df['Type'].fillna('Unknown')
            sns.scatterplot(x='FP32_TFLOPS', y='Memory_bandwidth_GB_s', hue=types, data=filtered_df, alpha=0.7)
        else:
            sns.scatterplot(x='FP32_TFLOPS', y='Memory_bandwidth_GB_s', data=filtered_df, alpha=0.7)
        
        plt.title('Memory Bandwidth vs. Compute Performance')
        plt.xlabel('TFLOPS (FP32)')
        plt.ylabel('Memory Bandwidth (GB/s)')
        plt.tight_layout()
        plt.savefig('analysis/visualizations/memory_vs_compute.png')
        plt.close()

# 4. Performance Evolution Over Time
print("Generating performance evolution over time...")
if 'Release_year' in df.columns and 'FP32_TFLOPS' in df.columns:
    plt.figure(figsize=(14, 8))
    # Filter to include only rows with both metrics
    filtered_df = df[df['Release_year'].notna() & df['FP32_TFLOPS'].notna()]
    if len(filtered_df) > 0:
        # Group by year and calculate mean, min, max
        yearly_stats = filtered_df.groupby('Release_year')['FP32_TFLOPS'].agg(['mean', 'min', 'max']).reset_index()
        
        # Plot the evolution
        plt.plot(yearly_stats['Release_year'], yearly_stats['mean'], marker='o', linewidth=2, label='Mean')
        plt.fill_between(yearly_stats['Release_year'], yearly_stats['min'], yearly_stats['max'], alpha=0.2, label='Range')
        
        plt.title('FP32 Performance Evolution Over Time')
        plt.xlabel('Release Year')
        plt.ylabel('TFLOPS (FP32)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis/visualizations/perf_evolution.png')
        plt.close()

# 5. Correlation Heatmap for Key Metrics
print("Generating correlation heatmap...")
key_metrics = [
    'FP32_TFLOPS', 'FP16_TFLOPS', 'INT8_TFLOPS', 
    'Memory_bandwidth_GB_s', 'TDP_numeric', 'perf_per_watt',
    'memory_bandwidth_compute_ratio', 'FP16_to_FP32_scaling', 'INT8_to_FP32_scaling'
]

# Filter to include only columns that exist in the dataset
available_metrics = [col for col in key_metrics if col in df.columns]
if len(available_metrics) > 1:
    # Calculate correlation matrix
    corr_matrix = df[available_metrics].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Matrix of Key Performance Metrics')
    plt.tight_layout()
    plt.savefig('analysis/visualizations/correlation_heatmap.png')
    plt.close()

# 6. Performance Across Precision Levels
print("Generating performance across precision levels...")
precision_metrics = ['FP32_TFLOPS', 'FP16_TFLOPS', 'INT8_TFLOPS']
available_precision = [col for col in precision_metrics if col in df.columns]

if len(available_precision) > 1:
    # Filter to include only rows with values for all precision levels
    filtered_df = df.dropna(subset=available_precision)
    
    if len(filtered_df) > 0:
        # Prepare data for plotting
        data_to_plot = []
        
        for _, row in filtered_df.iterrows():
            for metric in available_precision:
                precision = metric.split('_')[0]
                data_to_plot.append({
                    'Hardware': row.get('Hardware name', 'Unknown'),
                    'Precision': precision,
                    'TFLOPS': row[metric]
                })
        
        plot_df = pd.DataFrame(data_to_plot)
        
        # Plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Hardware', y='TFLOPS', hue='Precision', data=plot_df)
        plt.title('Performance Across Precision Levels')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('analysis/visualizations/precision_comparison.png')
        plt.close()

# 7. Performance Per Watt Distribution
print("Generating performance per watt distribution...")
if 'perf_per_watt' in df.columns:
    plt.figure(figsize=(12, 8))
    # Filter to include only rows with perf_per_watt
    filtered_df = df[df['perf_per_watt'].notna()]
    
    if len(filtered_df) > 0:
        # Plot histogram
        sns.histplot(filtered_df['perf_per_watt'], bins=30, kde=True)
        plt.title('Distribution of Performance per Watt')
        plt.xlabel('TFLOPS per Watt')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('analysis/visualizations/perf_per_watt_dist.png')
        plt.close()

# 8. MLPerf Results Comparison
print("Generating MLPerf results comparison...")
mlperf_columns = [col for col in df.columns if ('Offline' in col or 'Server' in col) and 'normalized' not in col]

if len(mlperf_columns) > 0:
    # Filter to include only rows with MLPerf results
    filtered_df = df.dropna(subset=mlperf_columns, how='all')
    
    if len(filtered_df) > 0 and 'Hardware name' in filtered_df.columns:
        # Select top 10 hardware by first MLPerf metric
        top_hw = filtered_df.sort_values(by=mlperf_columns[0], ascending=False).head(10)
        
        # Plot
        plt.figure(figsize=(14, 10))
        
        for col in mlperf_columns:
            if col in top_hw.columns:
                plt.barh(top_hw['Hardware name'], top_hw[col], label=col)
        
        plt.title('Top 10 Hardware by MLPerf Results')
        plt.xlabel('Result Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig('analysis/visualizations/mlperf_comparison.png')
        plt.close()

# 9. Memory Bandwidth to Compute Ratio
print("Generating memory bandwidth to compute ratio plot...")
if 'memory_bandwidth_compute_ratio' in df.columns:
    plt.figure(figsize=(12, 8))
    # Filter to include only rows with the ratio
    filtered_df = df[df['memory_bandwidth_compute_ratio'].notna()]
    
    if len(filtered_df) > 0 and 'Hardware name' in filtered_df.columns:
        # Sort by ratio and take top 15
        top_hw = filtered_df.sort_values(by='memory_bandwidth_compute_ratio', ascending=False).head(15)
        
        # Plot
        plt.figure(figsize=(14, 10))
        sns.barplot(x='memory_bandwidth_compute_ratio', y='Hardware name', data=top_hw)
        plt.title('Top 15 Hardware by Memory Bandwidth to Compute Ratio')
        plt.xlabel('Memory Bandwidth (GB/s) / Compute (TFLOPS)')
        plt.tight_layout()
        plt.savefig('analysis/visualizations/memory_compute_ratio.png')
        plt.close()

print("Visualizations generated successfully in the 'analysis/visualizations' directory!") 