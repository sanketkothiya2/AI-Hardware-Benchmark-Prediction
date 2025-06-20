#!/usr/bin/env python3
"""
Data Quality Analysis for Focused AI Hardware Dataset
Analyzes completeness, missing values, and data coverage
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_data_quality():
    """Comprehensive data quality analysis"""
    
    # Load the dataset
    df = pd.read_csv('data/processed/focused_ai_hardware_metrics.csv')
    
    print('=== DATA QUALITY ANALYSIS ===')
    print(f'Total rows: {len(df):,}')
    print(f'Total columns: {len(df.columns)}')
    
    print('\n=== MISSING VALUES ANALYSIS ===')
    missing_stats = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        non_missing = len(df) - missing_count
        missing_stats.append({
            'column': col,
            'missing_count': missing_count, 
            'missing_percent': missing_pct,
            'non_missing': non_missing,
            'data_coverage': 100 - missing_pct
        })
    
    missing_df = pd.DataFrame(missing_stats)
    missing_df = missing_df.sort_values('missing_percent', ascending=True)
    
    print('Columns with significant missing data (>50% missing):')
    high_missing = missing_df[missing_df['missing_percent'] > 50]
    for _, row in high_missing.iterrows():
        print(f'  {row["column"]}: {row["missing_count"]:,} missing ({row["missing_percent"]:.1f}%), only {row["non_missing"]} valid')
    
    print('\nColumns with good data coverage (<50% missing):')
    good_coverage = missing_df[missing_df['missing_percent'] <= 50]
    for _, row in good_coverage.iterrows():
        if row['missing_count'] > 0:
            print(f'  {row["column"]}: {row["non_missing"]:,} valid ({row["data_coverage"]:.1f}% coverage)')
        else:
            print(f'  {row["column"]}: {row["non_missing"]:,} valid (100% coverage)')
    
    print('\n=== KEY METRICS COVERAGE ===')
    key_metrics = [
        ('fp32_tflops', 'FP32 FLOPS'),
        ('fp16_tflops', 'FP16 FLOPS'),
        ('int8_tops', 'INT8 Performance'),
        ('memory_bandwidth_gbps', 'Memory Bandwidth'),
        ('memory_size_gb', 'Memory Size'),
        ('energy_consumption_watts', 'Power Consumption'),
        ('latency_ms', 'Latency'),
        ('throughput_ops_sec', 'Throughput'),
        ('fps_indicator', 'FPS Indicator'),
        ('cuda_score', 'CUDA Score'),
        ('price_usd', 'Price Data')
    ]
    
    for metric, label in key_metrics:
        if metric in df.columns:
            valid_count = df[metric].notna().sum()
            coverage = (valid_count / len(df)) * 100
            print(f'  {label}: {valid_count:,}/{len(df):,} ({coverage:.1f}% coverage)')
    
    print('\n=== DATA SOURCES DISTRIBUTION ===')
    source_dist = df['data_source'].value_counts()
    for source, count in source_dist.items():
        pct = (count / len(df)) * 100
        print(f'  {source}: {count:,} entries ({pct:.1f}%)')
    
    print('\n=== DATA COMPLETENESS CATEGORIES ===')
    
    # Basic hardware info completeness
    basic_cols = ['hardware_name', 'manufacturer', 'type']
    complete_basic = df[basic_cols].notna().all(axis=1).sum()
    print(f'Hardware with basic info (name, manufacturer, type): {complete_basic:,}/{len(df):,} ({complete_basic/len(df)*100:.1f}%)')
    
    # Power consumption data
    power_complete = df['energy_consumption_watts'].notna().sum()
    print(f'Hardware with power data: {power_complete:,}/{len(df):,} ({power_complete/len(df)*100:.1f}%)')
    
    # Compute performance data
    compute_cols = ['fp32_tflops', 'fp16_tflops', 'int8_tops'] 
    has_any_compute = df[compute_cols].notna().any(axis=1).sum()
    has_all_compute = df[compute_cols].notna().all(axis=1).sum()
    print(f'Hardware with ANY compute performance: {has_any_compute:,}/{len(df):,} ({has_any_compute/len(df)*100:.1f}%)')
    print(f'Hardware with ALL compute performance: {has_all_compute:,}/{len(df):,} ({has_all_compute/len(df)*100:.1f}%)')
    
    # Memory data
    memory_cols = ['memory_bandwidth_gbps', 'memory_size_gb']
    has_any_memory = df[memory_cols].notna().any(axis=1).sum()
    has_all_memory = df[memory_cols].notna().all(axis=1).sum()
    print(f'Hardware with ANY memory data: {has_any_memory:,}/{len(df):,} ({has_any_memory/len(df)*100:.1f}%)')
    print(f'Hardware with ALL memory data: {has_all_memory:,}/{len(df):,} ({has_all_memory/len(df)*100:.1f}%)')
    
    # Graphics performance
    graphics_cols = ['cuda_score', 'opencl_score', 'fps_indicator']
    has_graphics = df[graphics_cols].notna().any(axis=1).sum()
    print(f'Hardware with graphics performance: {has_graphics:,}/{len(df):,} ({has_graphics/len(df)*100:.1f}%)')
    
    # Price data
    price_complete = df['price_usd'].notna().sum()
    print(f'Hardware with price data: {price_complete:,}/{len(df):,} ({price_complete/len(df)*100:.1f}%)')
    
    print('\n=== MOST COMPLETE RECORDS ===')
    # Calculate completeness score
    df['completeness_score'] = df.notna().sum(axis=1)
    top_complete = df.nlargest(10, 'completeness_score')[['hardware_name', 'manufacturer', 'data_source', 'completeness_score']]
    
    print('Top 10 most complete hardware entries:')
    total_cols = len(df.columns) - 1  # Exclude completeness_score column
    for _, row in top_complete.iterrows():
        completeness_pct = (row['completeness_score'] / total_cols) * 100
        print(f'  {row["hardware_name"]} ({row["manufacturer"]}, {row["data_source"]}): {row["completeness_score"]}/{total_cols} fields ({completeness_pct:.1f}%)')
    
    print('\n=== RECOMMENDATIONS FOR CLEAN DATASET ===')
    
    # Identify rows with minimal useful data
    essential_cols = ['hardware_name', 'manufacturer', 'energy_consumption_watts']
    performance_cols = ['fp32_tflops', 'fp16_tflops', 'int8_tops', 'cuda_score', 'memory_bandwidth_gbps']
    
    # Rows with essential info AND some performance data
    has_essential = df[essential_cols].notna().all(axis=1)
    has_performance = df[performance_cols].notna().any(axis=1)
    useful_rows = has_essential & has_performance
    useful_count = useful_rows.sum()
    
    print(f'Rows with essential info + performance data: {useful_count:,}/{len(df):,} ({useful_count/len(df)*100:.1f}%)')
    print(f'Recommended to keep: {useful_count:,} rows')
    print(f'Recommended to remove: {len(df) - useful_count:,} rows with insufficient data')
    
    # Show data sources of useful rows
    print('\nData source distribution of useful rows:')
    useful_df = df[useful_rows]
    useful_sources = useful_df['data_source'].value_counts()
    for source, count in useful_sources.items():
        pct = (count / useful_count) * 100
        print(f'  {source}: {count:,} entries ({pct:.1f}%)')
    
    return df, useful_rows

def create_clean_dataset(df, useful_rows):
    """Create a cleaned dataset with only useful rows"""
    
    print('\n=== CREATING CLEAN DATASET ===')
    
    # Filter to useful rows
    clean_df = df[useful_rows].copy()
    
    # Remove completeness_score column if it exists
    if 'completeness_score' in clean_df.columns:
        clean_df = clean_df.drop('completeness_score', axis=1)
    
    # Focus on most important columns
    priority_cols = [
        'hardware_name', 'manufacturer', 'type', 'data_source',
        'fp32_tflops', 'fp16_tflops', 'int8_tops',
        'memory_bandwidth_gbps', 'memory_size_gb', 'energy_consumption_watts',
        'cuda_score', 'opencl_score', 'fps_indicator',
        'g3d_mark_score', 'price_usd', 'perf_per_dollar',
        'throughput_ops_sec', 'latency_ms',
        'tops_per_watt_fp32', 'tops_per_watt_fp16', 'tops_per_watt_int8',
        'memory_compute_ratio', 'fp16_to_fp32_scaling', 'int8_to_fp32_scaling'
    ]
    
    # Keep only columns that exist in the dataframe
    available_cols = [col for col in priority_cols if col in clean_df.columns]
    clean_df = clean_df[available_cols]
    
    # Save clean dataset
    output_path = 'data/processed/clean_ai_hardware_matrix.csv'
    clean_df.to_csv(output_path, index=False)
    
    print(f'Clean dataset saved to: {output_path}')
    print(f'Clean dataset shape: {clean_df.shape[0]:,} rows × {clean_df.shape[1]} columns')
    
    # Show final coverage statistics
    print('\nFinal coverage in clean dataset:')
    key_metrics = [
        ('fp32_tflops', 'FP32 FLOPS'),
        ('fp16_tflops', 'FP16 FLOPS'), 
        ('int8_tops', 'INT8 Performance'),
        ('memory_bandwidth_gbps', 'Memory Bandwidth'),
        ('energy_consumption_watts', 'Power Consumption'),
        ('cuda_score', 'CUDA Score'),
        ('price_usd', 'Price Data')
    ]
    
    for metric, label in key_metrics:
        if metric in clean_df.columns:
            valid_count = clean_df[metric].notna().sum()
            coverage = (valid_count / len(clean_df)) * 100
            print(f'  {label}: {valid_count:,}/{len(clean_df):,} ({coverage:.1f}% coverage)')
    
    return clean_df

if __name__ == "__main__":
    df, useful_rows = analyze_data_quality()
    clean_df = create_clean_dataset(df, useful_rows) 