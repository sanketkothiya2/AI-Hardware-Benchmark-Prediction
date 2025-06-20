#!/usr/bin/env python3
"""
AI Hardware Performance Matrix Builder
Creates a comprehensive matrix with complete data coverage
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_ai_hardware_matrix():
    """Build comprehensive AI hardware performance matrix"""
    
    # Load the clean dataset
    df = pd.read_csv('data/processed/clean_ai_hardware_matrix.csv')
    
    print('=== AI HARDWARE PERFORMANCE MATRIX BUILDER ===')
    print(f'Input dataset: {len(df):,} rows × {len(df.columns)} columns')
    
    # Define matrix structure based on available data
    matrix_metrics = {
        'COMPUTE_PERFORMANCE': ['fp32_tflops', 'fp16_tflops', 'int8_tops'],
        'MEMORY_SPECS': ['memory_bandwidth_gbps', 'memory_size_gb'],
        'POWER_EFFICIENCY': ['energy_consumption_watts', 'tops_per_watt_fp32', 'tops_per_watt_fp16', 'tops_per_watt_int8'],
        'GRAPHICS_PERFORMANCE': ['cuda_score', 'opencl_score', 'fps_indicator'],
        'MARKET_DATA': ['price_usd', 'perf_per_dollar'],
        'REAL_WORLD_PERFORMANCE': ['throughput_ops_sec', 'latency_ms'],
        'DERIVED_METRICS': ['memory_compute_ratio', 'fp16_to_fp32_scaling', 'int8_to_fp32_scaling']
    }
    
    # 1. COMPLETE PERFORMANCE MATRIX - Hardware with full compute specs
    print('\n=== 1. COMPLETE PERFORMANCE MATRIX ===')
    compute_complete = df[['fp32_tflops', 'fp16_tflops', 'int8_tops']].notna().all(axis=1)
    memory_complete = df[['memory_bandwidth_gbps', 'memory_size_gb']].notna().all(axis=1)
    power_complete = df['energy_consumption_watts'].notna()
    
    complete_matrix = df[compute_complete & memory_complete & power_complete].copy()
    print(f'Hardware with complete compute+memory+power data: {len(complete_matrix):,} entries')
    
    if len(complete_matrix) > 0:
        complete_matrix = complete_matrix.sort_values('fp32_tflops', ascending=False)
        complete_matrix.to_csv('data/processed/complete_performance_matrix.csv', index=False)
        print('Saved: data/processed/complete_performance_matrix.csv')
        
        # Show top performers
        print('\nTop 10 Complete Performance Systems:')
        for _, row in complete_matrix.head(10).iterrows():
            print(f'  {row["hardware_name"]} ({row["manufacturer"]}): '
                  f'{row["fp32_tflops"]:.1f} TFLOPS FP32, '
                  f'{row["memory_bandwidth_gbps"]:.0f} GB/s, '
                  f'{row["energy_consumption_watts"]:.0f}W')
    
    # 2. COMPUTE PERFORMANCE MATRIX - Hardware with any compute data
    print('\n=== 2. COMPUTE PERFORMANCE MATRIX ===')
    has_compute = df[['fp32_tflops', 'fp16_tflops', 'int8_tops']].notna().any(axis=1)
    has_power = df['energy_consumption_watts'].notna()
    
    compute_matrix = df[has_compute & has_power].copy()
    
    # Fill missing compute values using available data
    for _, row in compute_matrix.iterrows():
        idx = row.name
        
        # Estimate FP16 from FP32 (typically 2x performance)
        if pd.isna(row['fp16_tflops']) and pd.notna(row['fp32_tflops']):
            compute_matrix.loc[idx, 'fp16_tflops'] = row['fp32_tflops'] * 2.0
            compute_matrix.loc[idx, 'fp16_to_fp32_scaling'] = 2.0
        
        # Estimate INT8 from FP32 (typically 4x performance) 
        if pd.isna(row['int8_tops']) and pd.notna(row['fp32_tflops']):
            compute_matrix.loc[idx, 'int8_tops'] = row['fp32_tflops'] * 4.0
            compute_matrix.loc[idx, 'int8_to_fp32_scaling'] = 4.0
        
        # Estimate FP32 from FP16 (divide by 2)
        if pd.isna(row['fp32_tflops']) and pd.notna(row['fp16_tflops']):
            compute_matrix.loc[idx, 'fp32_tflops'] = row['fp16_tflops'] / 2.0
            compute_matrix.loc[idx, 'fp16_to_fp32_scaling'] = 2.0
        
        # Calculate efficiency metrics
        if pd.notna(row['fp32_tflops']) and pd.notna(row['energy_consumption_watts']):
            compute_matrix.loc[idx, 'tops_per_watt_fp32'] = row['fp32_tflops'] / row['energy_consumption_watts']
        
        if pd.notna(compute_matrix.loc[idx, 'fp16_tflops']) and pd.notna(row['energy_consumption_watts']):
            compute_matrix.loc[idx, 'tops_per_watt_fp16'] = compute_matrix.loc[idx, 'fp16_tflops'] / row['energy_consumption_watts']
        
        if pd.notna(compute_matrix.loc[idx, 'int8_tops']) and pd.notna(row['energy_consumption_watts']):
            compute_matrix.loc[idx, 'tops_per_watt_int8'] = compute_matrix.loc[idx, 'int8_tops'] / row['energy_consumption_watts']
    
    print(f'Hardware with compute performance data: {len(compute_matrix):,} entries')
    compute_matrix = compute_matrix.sort_values('fp32_tflops', ascending=False, na_position='last')
    compute_matrix.to_csv('data/processed/compute_performance_matrix.csv', index=False)
    print('Saved: data/processed/compute_performance_matrix.csv')
    
    # 3. MEMORY BANDWIDTH MATRIX - Memory-intensive workloads
    print('\n=== 3. MEMORY BANDWIDTH MATRIX ===')
    has_memory = df['memory_bandwidth_gbps'].notna()
    has_power = df['energy_consumption_watts'].notna()
    
    memory_matrix = df[has_memory & has_power].copy()
    
    # Calculate memory efficiency
    for _, row in memory_matrix.iterrows():
        idx = row.name
        if pd.notna(row['memory_bandwidth_gbps']) and pd.notna(row['energy_consumption_watts']):
            memory_matrix.loc[idx, 'memory_bandwidth_per_watt'] = row['memory_bandwidth_gbps'] / row['energy_consumption_watts']
    
    print(f'Hardware with memory bandwidth data: {len(memory_matrix):,} entries')
    memory_matrix = memory_matrix.sort_values('memory_bandwidth_gbps', ascending=False)
    memory_matrix.to_csv('data/processed/memory_bandwidth_matrix.csv', index=False)
    print('Saved: data/processed/memory_bandwidth_matrix.csv')
    
    # Show top memory performers
    print('\nTop 10 Memory Bandwidth Systems:')
    for _, row in memory_matrix.head(10).iterrows():
        efficiency = row.get('memory_bandwidth_per_watt', 0)
        print(f'  {row["hardware_name"]} ({row["manufacturer"]}): '
              f'{row["memory_bandwidth_gbps"]:.0f} GB/s, '
              f'{efficiency:.2f} GB/s/W')
    
    # 4. GRAPHICS PERFORMANCE MATRIX - GPU/graphics workloads
    print('\n=== 4. GRAPHICS PERFORMANCE MATRIX ===')
    graphics_cols = ['cuda_score', 'opencl_score', 'fps_indicator']
    has_graphics = df[graphics_cols].notna().any(axis=1)
    has_power = df['energy_consumption_watts'].notna()
    
    graphics_matrix = df[has_graphics & has_power].copy()
    
    # Normalize graphics scores and create composite score
    for col in graphics_cols:
        if col in graphics_matrix.columns:
            max_val = graphics_matrix[col].max()
            if pd.notna(max_val) and max_val > 0:
                graphics_matrix[f'{col}_normalized'] = graphics_matrix[col] / max_val
    
    # Create composite graphics score
    norm_cols = [f'{col}_normalized' for col in graphics_cols if f'{col}_normalized' in graphics_matrix.columns]
    if norm_cols:
        graphics_matrix['composite_graphics_score'] = graphics_matrix[norm_cols].mean(axis=1, skipna=True)
    
    print(f'Hardware with graphics performance data: {len(graphics_matrix):,} entries')
    graphics_matrix = graphics_matrix.sort_values('cuda_score', ascending=False, na_position='last')
    graphics_matrix.to_csv('data/processed/graphics_performance_matrix.csv', index=False)
    print('Saved: data/processed/graphics_performance_matrix.csv')
    
    # 5. PRICE PERFORMANCE MATRIX - Cost-effectiveness analysis
    print('\n=== 5. PRICE PERFORMANCE MATRIX ===')
    has_price = df['price_usd'].notna()
    has_performance = df[['fp32_tflops', 'cuda_score', 'fps_indicator']].notna().any(axis=1)
    
    price_matrix = df[has_price & has_performance].copy()
    
    # Calculate various price-performance metrics
    for _, row in price_matrix.iterrows():
        idx = row.name
        price = row['price_usd']
        
        if pd.notna(row['fp32_tflops']) and price > 0:
            price_matrix.loc[idx, 'tflops_per_dollar'] = row['fp32_tflops'] / price
        
        if pd.notna(row['cuda_score']) and price > 0:
            price_matrix.loc[idx, 'cuda_score_per_dollar'] = row['cuda_score'] / price
        
        if pd.notna(row['fps_indicator']) and price > 0:
            price_matrix.loc[idx, 'fps_per_dollar'] = row['fps_indicator'] / price
    
    print(f'Hardware with price data: {len(price_matrix):,} entries')
    price_matrix = price_matrix.sort_values('perf_per_dollar', ascending=False, na_position='last')
    price_matrix.to_csv('data/processed/price_performance_matrix.csv', index=False)
    print('Saved: data/processed/price_performance_matrix.csv')
    
    # Show best value systems
    print('\nTop 10 Price-Performance Systems:')
    for _, row in price_matrix.head(10).iterrows():
        print(f'  {row["hardware_name"]} ({row["manufacturer"]}): '
              f'${row["price_usd"]:.0f}, '
              f'{row["perf_per_dollar"]:.2f} perf/dollar')
    
    # 6. UNIFIED AI PERFORMANCE MATRIX - Comprehensive view
    print('\n=== 6. UNIFIED AI PERFORMANCE MATRIX ===')
    
    # Start with base data (name, manufacturer, type, power)
    essential_cols = ['hardware_name', 'manufacturer', 'type', 'data_source', 'energy_consumption_watts']
    has_essentials = df[essential_cols].notna().all(axis=1)
    
    unified_matrix = df[has_essentials].copy()
    
    # Fill in missing values using industry averages and scaling factors
    print('Filling missing values with estimates...')
    
    for _, row in unified_matrix.iterrows():
        idx = row.name
        
        # 1. Compute Performance Estimation
        if pd.isna(row['fp32_tflops']):
            # Estimate from CUDA score if available
            if pd.notna(row['cuda_score']):
                # Rough scaling: high-end cards ~200k CUDA = ~80 TFLOPS
                unified_matrix.loc[idx, 'fp32_tflops'] = row['cuda_score'] / 2500
            # Estimate from power consumption (rough scaling)
            elif pd.notna(row['energy_consumption_watts']):
                if row['energy_consumption_watts'] > 300:  # High-end
                    unified_matrix.loc[idx, 'fp32_tflops'] = row['energy_consumption_watts'] * 0.2
                elif row['energy_consumption_watts'] > 150:  # Mid-range
                    unified_matrix.loc[idx, 'fp32_tflops'] = row['energy_consumption_watts'] * 0.15
                else:  # Low-end
                    unified_matrix.loc[idx, 'fp32_tflops'] = row['energy_consumption_watts'] * 0.1
        
        # 2. Multi-precision scaling
        fp32_perf = unified_matrix.loc[idx, 'fp32_tflops']
        if pd.notna(fp32_perf):
            if pd.isna(row['fp16_tflops']):
                unified_matrix.loc[idx, 'fp16_tflops'] = fp32_perf * 2.0  # Standard 2x scaling
                unified_matrix.loc[idx, 'fp16_to_fp32_scaling'] = 2.0
            
            if pd.isna(row['int8_tops']):
                unified_matrix.loc[idx, 'int8_tops'] = fp32_perf * 4.0  # Standard 4x scaling
                unified_matrix.loc[idx, 'int8_to_fp32_scaling'] = 4.0
        
        # 3. Memory bandwidth estimation
        if pd.isna(row['memory_bandwidth_gbps']) and pd.notna(fp32_perf):
            # Scale based on compute performance
            if fp32_perf > 50:  # High-end
                unified_matrix.loc[idx, 'memory_bandwidth_gbps'] = fp32_perf * 40
            elif fp32_perf > 20:  # Mid-range
                unified_matrix.loc[idx, 'memory_bandwidth_gbps'] = fp32_perf * 30
            else:  # Low-end
                unified_matrix.loc[idx, 'memory_bandwidth_gbps'] = fp32_perf * 20
        
        # 4. Memory size estimation
        if pd.isna(row['memory_size_gb']) and pd.notna(row['energy_consumption_watts']):
            power = row['energy_consumption_watts']
            if power > 400:  # High-end
                unified_matrix.loc[idx, 'memory_size_gb'] = 80.0
            elif power > 200:  # Mid-range
                unified_matrix.loc[idx, 'memory_size_gb'] = 24.0
            else:  # Low-end
                unified_matrix.loc[idx, 'memory_size_gb'] = 8.0
        
        # 5. Calculate all efficiency metrics
        power = row['energy_consumption_watts']
        if pd.notna(power) and power > 0:
            fp32_perf = unified_matrix.loc[idx, 'fp32_tflops']
            fp16_perf = unified_matrix.loc[idx, 'fp16_tflops']
            int8_perf = unified_matrix.loc[idx, 'int8_tops']
            
            if pd.notna(fp32_perf):
                unified_matrix.loc[idx, 'tops_per_watt_fp32'] = fp32_perf / power
            if pd.notna(fp16_perf):
                unified_matrix.loc[idx, 'tops_per_watt_fp16'] = fp16_perf / power
            if pd.notna(int8_perf):
                unified_matrix.loc[idx, 'tops_per_watt_int8'] = int8_perf / power
        
        # 6. Memory-compute ratio
        mem_bw = unified_matrix.loc[idx, 'memory_bandwidth_gbps']
        if pd.notna(mem_bw) and pd.notna(fp32_perf) and fp32_perf > 0:
            unified_matrix.loc[idx, 'memory_compute_ratio'] = mem_bw / fp32_perf
    
    # Performance categorization
    unified_matrix['performance_tier'] = 'Unknown'
    fp32_values = unified_matrix['fp32_tflops'].dropna()
    
    if len(fp32_values) > 0:
        high_threshold = fp32_values.quantile(0.8)
        mid_threshold = fp32_values.quantile(0.4)
        
        unified_matrix.loc[unified_matrix['fp32_tflops'] >= high_threshold, 'performance_tier'] = 'High-End'
        unified_matrix.loc[(unified_matrix['fp32_tflops'] >= mid_threshold) & 
                          (unified_matrix['fp32_tflops'] < high_threshold), 'performance_tier'] = 'Mid-Range'
        unified_matrix.loc[unified_matrix['fp32_tflops'] < mid_threshold, 'performance_tier'] = 'Entry-Level'
    
    # Sort by performance
    unified_matrix = unified_matrix.sort_values(['fp32_tflops'], ascending=False, na_position='last')
    
    print(f'Unified AI Performance Matrix: {len(unified_matrix):,} entries')
    unified_matrix.to_csv('data/processed/unified_ai_performance_matrix.csv', index=False)
    print('Saved: data/processed/unified_ai_performance_matrix.csv')
    
    # Final summary
    print('\n=== MATRIX SUMMARY ===')
    print(f'Complete Performance Matrix: {len(complete_matrix):,} entries (100% data coverage)')
    print(f'Compute Performance Matrix: {len(compute_matrix):,} entries')
    print(f'Memory Bandwidth Matrix: {len(memory_matrix):,} entries')
    print(f'Graphics Performance Matrix: {len(graphics_matrix):,} entries')
    print(f'Price Performance Matrix: {len(price_matrix):,} entries')
    print(f'Unified AI Performance Matrix: {len(unified_matrix):,} entries (estimated missing values)')
    
    print('\nPerformance tier distribution in unified matrix:')
    tier_dist = unified_matrix['performance_tier'].value_counts()
    for tier, count in tier_dist.items():
        pct = (count / len(unified_matrix)) * 100
        print(f'  {tier}: {count:,} systems ({pct:.1f}%)')
    
    return unified_matrix

if __name__ == "__main__":
    unified_matrix = create_ai_hardware_matrix() 