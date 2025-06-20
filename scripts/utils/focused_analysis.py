#!/usr/bin/env python3
"""
Focused AI Hardware Metrics Analysis
Demonstrates extraction of specific key metrics for AI performance prediction
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def clean_hardware_name(name):
    """Clean hardware names for better matching"""
    if pd.isna(name): 
        return ''
    name = str(name).strip()
    # Remove common suffixes
    name = re.sub(r'\s*(PCIe|SXM\d*|OAM|NVL\d*)(\s+\d+GB)?$', '', name, flags=re.IGNORECASE)
    return name

def extract_numeric(val):
    """Extract numeric values from various formats"""
    if pd.isna(val): 
        return None
    if isinstance(val, (int, float)): 
        return float(val) if not pd.isna(val) else None
    try: 
        clean_val = re.sub(r'[^\d.-]', '', str(val))
        return float(clean_val) if clean_val else None
    except: 
        return None

def main():
    print('=== FOCUSED AI HARDWARE METRICS ANALYSIS ===\n')
    
    # Define data paths
    base_path = Path('data/raw')
    
    # Load datasets
    try:
        print('Loading datasets...')
        ml_hw = pd.read_csv(base_path / 'ml_hardware.csv')
        gpu_bench = pd.read_csv(base_path / 'GPU_benchmarks_v7.csv')
        gpu_graphics = pd.read_csv(base_path / 'GPU_scores_graphicsAPIs.csv')
        mlperf = pd.read_csv(base_path / 'mlperf.csv')
        
        print(f'✓ ML Hardware: {len(ml_hw)} records')
        print(f'✓ GPU Benchmarks: {len(gpu_bench)} records')
        print(f'✓ GPU Graphics APIs: {len(gpu_graphics)} records')
        print(f'✓ MLPerf: {len(mlperf)} records')
        
    except Exception as e:
        print(f'Error loading datasets: {e}')
        return
    
    print('\n=== EXTRACTING KEY METRICS ===')
    
    # Initialize metrics collection
    key_metrics = []
    
    # 1. Extract FLOPS and compute metrics from ML Hardware
    print('1. Processing ML Hardware data for FLOPS metrics...')
    
    for _, row in ml_hw.iterrows():
        hw_name = clean_hardware_name(row.get('Hardware name', ''))
        if not hw_name:
            continue
        
        # Extract FLOPS for different precisions
        fp32_flops = extract_numeric(row.get('FP32 (single precision) performance (FLOP/s)', 0))
        fp16_flops = extract_numeric(row.get('FP16 (half precision) performance (FLOP/s)', 0))
        int8_ops = extract_numeric(row.get('INT8 performance (OP/s)', 0))
        
        # Memory metrics
        mem_bandwidth = extract_numeric(row.get('Memory bandwidth (byte/s)', 0))
        mem_size = extract_numeric(row.get('Memory size per board (Byte)', 0))
        
        # Power consumption
        tdp = extract_numeric(row.get('TDP (W)', 0))
        
        metrics = {
            'hardware_name': hw_name,
            'manufacturer': row.get('Manufacturer', ''),
            'type': row.get('Type', ''),
            
            # Key Metrics (converted to standard units)
            'fp32_tflops': fp32_flops / 1e12 if fp32_flops else None,
            'fp16_tflops': fp16_flops / 1e12 if fp16_flops else None,
            'int8_tops': int8_ops / 1e12 if int8_ops else None,
            'memory_bandwidth_gbps': mem_bandwidth / 1e9 if mem_bandwidth else None,
            'memory_size_gb': mem_size / 1e9 if mem_size else None,
            'energy_consumption_watts': tdp,
            
            'data_source': 'ml_hardware'
        }
        
        key_metrics.append(metrics)
    
    df = pd.DataFrame(key_metrics)
    print(f'   ✓ Extracted {len(df)} hardware entries with FLOPS data')
    
    # 2. Add Graphics Performance (FPS indicators)
    print('2. Adding graphics performance metrics...')
    
    graphics_added = 0
    for _, row in gpu_graphics.iterrows():
        hw_name = clean_hardware_name(row.get('Device', ''))
        if not hw_name:
            continue
        
        # Find existing entry or create new
        existing_idx = df[df['hardware_name'] == hw_name].index
        
        cuda_score = extract_numeric(row.get('CUDA'))
        opencl_score = extract_numeric(row.get('OpenCL'))
        vulkan_score = extract_numeric(row.get('Vulkan'))
        
        if len(existing_idx) > 0:
            # Update existing entry
            idx = existing_idx[0]
            df.at[idx, 'cuda_score'] = cuda_score
            df.at[idx, 'opencl_score'] = opencl_score
            df.at[idx, 'vulkan_score'] = vulkan_score
            
            # FPS indicator from CUDA score (approximation)
            if cuda_score:
                df.at[idx, 'fps_indicator'] = cuda_score / 1000
            
            graphics_added += 1
        else:
            # Create new entry for graphics-only hardware
            new_entry = {
                'hardware_name': hw_name,
                'manufacturer': row.get('Manufacturer', ''),
                'cuda_score': cuda_score,
                'opencl_score': opencl_score,
                'vulkan_score': vulkan_score,
                'fps_indicator': cuda_score / 1000 if cuda_score else None,
                'data_source': 'graphics'
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            graphics_added += 1
    
    print(f'   ✓ Added graphics data for {graphics_added} entries')
    
    # 3. Add benchmark price/performance data
    print('3. Adding benchmark and price data...')
    
    benchmark_added = 0
    for _, row in gpu_bench.iterrows():
        hw_name = clean_hardware_name(row.get('gpuName', ''))
        if not hw_name:
            continue
        
        existing_idx = df[df['hardware_name'] == hw_name].index
        
        g3d_score = extract_numeric(row.get('G3Dmark'))
        price = extract_numeric(row.get('price'))
        tdp_bench = extract_numeric(row.get('TDP'))
        
        if len(existing_idx) > 0:
            idx = existing_idx[0]
            df.at[idx, 'g3d_mark_score'] = g3d_score
            df.at[idx, 'price_usd'] = price
            
            # Fill TDP if missing
            if pd.isna(df.at[idx, 'energy_consumption_watts']) and tdp_bench:
                df.at[idx, 'energy_consumption_watts'] = tdp_bench
            
            # Calculate performance per dollar
            if g3d_score and price:
                df.at[idx, 'perf_per_dollar'] = g3d_score / price
            
            benchmark_added += 1
    
    print(f'   ✓ Added benchmark data for {benchmark_added} entries')
    
    # 4. Add MLPerf latency and throughput
    print('4. Processing MLPerf for latency/throughput...')
    
    # Aggregate MLPerf results by hardware
    mlperf_data = {}
    
    for _, row in mlperf.iterrows():
        hw_name = clean_hardware_name(row.get('Accelerator', ''))
        scenario = row.get('Scenario', '')
        result = extract_numeric(row.get('Avg. Result at System Name', 0))
        
        if not hw_name or not result:
            continue
        
        if hw_name not in mlperf_data:
            mlperf_data[hw_name] = {'throughput': [], 'latency': []}
        
        # All scenarios provide throughput
        mlperf_data[hw_name]['throughput'].append(result)
        
        # Server scenario provides latency-related data
        if scenario == 'Server' and result > 0:
            # Approximate latency from throughput (ms per query)
            latency_ms = 1000 / result
            mlperf_data[hw_name]['latency'].append(latency_ms)
    
    # Add aggregated MLPerf metrics
    mlperf_added = 0
    for hw_name, perf_data in mlperf_data.items():
        existing_idx = df[df['hardware_name'] == hw_name].index
        
        avg_throughput = np.mean(perf_data['throughput']) if perf_data['throughput'] else None
        avg_latency = np.mean(perf_data['latency']) if perf_data['latency'] else None
        max_throughput = np.max(perf_data['throughput']) if perf_data['throughput'] else None
        min_latency = np.min(perf_data['latency']) if perf_data['latency'] else None
        
        if len(existing_idx) > 0:
            idx = existing_idx[0]
            df.at[idx, 'throughput_ops_sec'] = avg_throughput
            df.at[idx, 'latency_ms'] = avg_latency
            df.at[idx, 'max_throughput_ops_sec'] = max_throughput
            df.at[idx, 'min_latency_ms'] = min_latency
            mlperf_added += 1
    
    print(f'   ✓ Added MLPerf data for {mlperf_added} entries')
    
    # 5. Calculate derived efficiency metrics
    print('5. Calculating efficiency metrics...')
    
    # TOPs per Watt
    df['tops_per_watt_fp32'] = (df['fp32_tflops'] / df['energy_consumption_watts']).where(df['energy_consumption_watts'] > 0)
    df['tops_per_watt_fp16'] = (df['fp16_tflops'] / df['energy_consumption_watts']).where(df['energy_consumption_watts'] > 0)
    df['tops_per_watt_int8'] = (df['int8_tops'] / df['energy_consumption_watts']).where(df['energy_consumption_watts'] > 0)
    
    # Memory efficiency
    df['memory_compute_ratio'] = (df['memory_bandwidth_gbps'] / df['fp32_tflops']).where(df['fp32_tflops'] > 0)
    
    # Precision scaling
    df['fp16_to_fp32_scaling'] = (df['fp16_tflops'] / df['fp32_tflops']).where(df['fp32_tflops'] > 0)
    df['int8_to_fp32_scaling'] = (df['int8_tops'] / df['fp32_tflops']).where(df['fp32_tflops'] > 0)
    
    # Compute density
    df['compute_density_tflops_per_gb'] = (df['fp32_tflops'] / df['memory_size_gb']).where(df['memory_size_gb'] > 0)
    
    # Memory usage estimation
    df['memory_usage_percent_est'] = np.clip(
        (df['fp32_tflops'] * 4) / (df['memory_bandwidth_gbps'] * 1000) * 100, 
        0, 100
    )
    
    print(f'   ✓ Calculated efficiency metrics')
    
    # Clean up final dataset
    df = df.dropna(subset=['hardware_name'])
    df = df[df['hardware_name'].str.len() > 0]
    
    print(f'\n=== FINAL RESULTS ===')
    print(f'Total hardware entries: {len(df)}')
    
    # Show coverage of key metrics
    print(f'\nKEY METRICS COVERAGE:')
    key_metrics_cols = [
        ('fp32_tflops', 'FP32 FLOPS'),
        ('fp16_tflops', 'FP16 FLOPS'), 
        ('int8_tops', 'INT8 Performance'),
        ('memory_bandwidth_gbps', 'Memory Bandwidth'),
        ('energy_consumption_watts', 'Energy Consumption'),
        ('latency_ms', 'Latency'),
        ('throughput_ops_sec', 'Throughput'),
        ('fps_indicator', 'FPS Indicator'),
        ('tops_per_watt_fp32', 'TOPS/Watt'),
        ('price_usd', 'Price')
    ]
    
    for col, label in key_metrics_cols:
        if col in df.columns:
            count = df[col].notna().sum()
            pct = count / len(df) * 100
            print(f'- {label}: {count}/{len(df)} ({pct:.1f}%)')
    
    # Show data sources distribution
    print(f'\nDATA SOURCES:')
    if 'data_source' in df.columns:
        source_counts = df['data_source'].value_counts()
        for source, count in source_counts.items():
            print(f'- {source}: {count} entries')
    
    # Save focused dataset
    output_path = Path('data/processed/focused_ai_hardware_metrics.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'\n✓ Saved focused dataset to: {output_path}')
    
    # Show top performers
    print(f'\n=== TOP PERFORMERS ===')
    
    # Top compute performance
    if 'fp32_tflops' in df.columns:
        top_compute = df.nlargest(5, 'fp32_tflops')[['hardware_name', 'fp32_tflops', 'manufacturer']]
        print(f'\nTop 5 Compute Performance (FP32):')
        for _, row in top_compute.iterrows():
            if pd.notna(row['fp32_tflops']):
                print(f'  {row["hardware_name"]}: {row["fp32_tflops"]:.1f} TFLOPS ({row["manufacturer"]})')
    
    # Top energy efficiency
    if 'tops_per_watt_fp32' in df.columns:
        top_efficiency = df.nlargest(5, 'tops_per_watt_fp32')[['hardware_name', 'tops_per_watt_fp32', 'energy_consumption_watts']]
        print(f'\nTop 5 Energy Efficiency (TOPS/Watt):')
        for _, row in top_efficiency.iterrows():
            if pd.notna(row['tops_per_watt_fp32']):
                print(f'  {row["hardware_name"]}: {row["tops_per_watt_fp32"]:.3f} TOPS/W')
    
    # Best latency performers
    if 'min_latency_ms' in df.columns:
        best_latency = df.nsmallest(5, 'min_latency_ms')[['hardware_name', 'min_latency_ms', 'max_throughput_ops_sec']]
        print(f'\nBest Latency Performance:')
        for _, row in best_latency.iterrows():
            if pd.notna(row['min_latency_ms']):
                throughput = f", {row['max_throughput_ops_sec']:.0f} ops/sec" if pd.notna(row['max_throughput_ops_sec']) else ""
                print(f'  {row["hardware_name"]}: {row["min_latency_ms"]:.2f} ms{throughput}')
    
    print(f'\n=== MATRIX RELATIONS THAT CAN BE BUILT ===')
    
    matrix_possibilities = [
        ('Latency vs Hardware', 'Hardware → Model latency performance'),
        ('Throughput vs Precision', 'Hardware → FP32/FP16/INT8 throughput'),
        ('FLOPS Capability', 'Hardware → Multi-precision FLOPS'),
        ('Memory Bandwidth vs Performance', 'Hardware → Memory efficiency'),
        ('Power Efficiency', 'Hardware → TOPS/Watt across precisions'),
        ('Price-Performance', 'Hardware → Performance per dollar'),
        ('Graphics Performance', 'Hardware → FPS and API scores'),
        ('Scaling Relationships', 'Precision scaling factors')
    ]
    
    for matrix_name, description in matrix_possibilities:
        print(f'- {matrix_name}: {description}')
    
    print(f'\nThis focused approach extracts {len(key_metrics_cols)} key metrics')
    print(f'with much higher data quality and relevance for AI performance prediction!')
    print(f'\n=== ANALYSIS COMPLETED ===')

if __name__ == "__main__":
    main() 