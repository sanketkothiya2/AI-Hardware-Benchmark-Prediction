import pandas as pd
import numpy as np
import re
from datetime import datetime

# Function to convert string values with units to numeric values
def convert_to_numeric(value):
    if pd.isna(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return value
    
    # Convert string to string (in case it's not)
    value = str(value).strip()
    
    # Return NaN for empty strings
    if value == '':
        return np.nan
    
    # Try direct conversion first
    try:
        return float(value)
    except ValueError:
        pass
    
    # Handle scientific notation
    try:
        return float(value)
    except:
        # Handle values with units
        multipliers = {
            'K': 1e3,
            'M': 1e6,
            'B': 1e9,
            'T': 1e12,
            'GB': 1e9,
            'MB': 1e6,
            'KB': 1e3,
            'TB': 1e12,
            'G': 1e9
        }
        
        for unit, multiplier in multipliers.items():
            if unit in value:
                try:
                    numeric_part = value.replace(unit, '').strip()
                    return float(numeric_part) * multiplier
                except:
                    pass
        
        # If we get here, we couldn't parse the value
        return np.nan

# Function to standardize GPU names
def standardize_gpu_name(name):
    if pd.isna(name):
        return name
    
    # Convert to string and lowercase for standardization
    name = str(name).lower()
    
    # Replace common variations
    name = name.replace('geforce ', '')
    name = name.replace('radeon ', '')
    name = name.replace('nvidia ', '')
    name = name.replace('amd ', '')
    name = name.replace('intel ', '')
    
    # Standardize RTX/GTX naming
    name = re.sub(r'rtx\s*', 'rtx ', name)
    name = re.sub(r'gtx\s*', 'gtx ', name)
    
    # Standardize series naming
    name = re.sub(r'(\d+)\s*ti', r'\1 ti', name)
    name = re.sub(r'(\d+)\s*super', r'\1 super', name)
    
    return name.strip()

# Function to extract year from date string
def extract_year(date_str):
    if pd.isna(date_str):
        return np.nan
    
    try:
        # Try to parse as year only
        if len(str(date_str)) == 4 and str(date_str).isdigit():
            return int(date_str)
        
        # Try to parse as full date
        date_obj = pd.to_datetime(date_str)
        return date_obj.year
    except:
        return np.nan

# Function to normalize performance metrics
def normalize_performance(df, column):
    if column not in df.columns or df[column].isna().all():
        return df
    
    # Get the max value for normalization
    max_val = df[column].max()
    
    if max_val > 0:
        df[f'{column}_normalized'] = df[column] / max_val
    
    return df

# Load datasets
print("Loading datasets...")
gpu_benchmarks = pd.read_csv('data/raw/GPU_benchmarks_v7.csv')
gpu_api_scores = pd.read_csv('data/raw/GPU_scores_graphicsAPIs.csv')
ml_hardware = pd.read_csv('data/raw/ml_hardware.csv')
mlperf = pd.read_csv('data/raw/mlperf.csv')

print("Cleaning GPU benchmarks data...")
# Clean GPU benchmarks data
gpu_benchmarks['gpuName_std'] = gpu_benchmarks['gpuName'].apply(standardize_gpu_name)
gpu_benchmarks['release_year'] = gpu_benchmarks['testDate'].apply(extract_year)
gpu_benchmarks['price_numeric'] = gpu_benchmarks['price'].apply(convert_to_numeric)
gpu_benchmarks['TDP_numeric'] = gpu_benchmarks['TDP'].apply(convert_to_numeric)

# Calculate performance metrics
gpu_benchmarks['perf_per_watt'] = gpu_benchmarks['G3Dmark'] / gpu_benchmarks['TDP_numeric']
gpu_benchmarks['perf_per_dollar'] = gpu_benchmarks['G3Dmark'] / gpu_benchmarks['price_numeric']

print("Cleaning GPU API scores data...")
# Clean GPU API scores data
gpu_api_scores['Device_std'] = gpu_api_scores['Device'].apply(standardize_gpu_name)

# Convert API scores to numeric
for api in ['CUDA', 'Metal', 'OpenCL', 'Vulkan']:
    gpu_api_scores[f'{api}_numeric'] = gpu_api_scores[api].apply(convert_to_numeric)

# Calculate API performance ratios
apis = ['CUDA', 'Metal', 'OpenCL', 'Vulkan']
for i, api1 in enumerate(apis):
    for api2 in apis[i+1:]:
        col_name = f'{api1}_to_{api2}_ratio'
        gpu_api_scores[col_name] = gpu_api_scores[f'{api1}_numeric'] / gpu_api_scores[f'{api2}_numeric']

print("Cleaning ML hardware data...")
# Clean ML hardware data
ml_hardware['Hardware_std'] = ml_hardware['Hardware name'].apply(standardize_gpu_name)
ml_hardware['Release_year'] = ml_hardware['Release date'].apply(extract_year)
ml_hardware['TDP_numeric'] = ml_hardware['TDP (W)'].apply(convert_to_numeric)
ml_hardware['Memory_size_GB'] = ml_hardware['Memory size per board (Byte)'].apply(convert_to_numeric) / 1e9
ml_hardware['Memory_bandwidth_GB_s'] = ml_hardware['Memory bandwidth (byte/s)'].apply(convert_to_numeric) / 1e9

# Convert FLOPS metrics to numeric
flops_columns = [
    'FP64 (double precision) performance (FLOP/s)',
    'FP32 (single precision) performance (FLOP/s)',
    'FP16 (half precision) performance (FLOP/s)',
    'TF32 (TensorFloat-32) performance (FLOP/s)',
    'Tensor-FP16/BF16 performance (FLOP/s)',
    'INT8 performance (OP/s)',
    'INT4 performance (OP/s)'
]

for col in flops_columns:
    short_name = col.split(' ')[0]
    ml_hardware[f'{short_name}_TFLOPS'] = ml_hardware[col].apply(convert_to_numeric) / 1e12

# Calculate efficiency metrics
ml_hardware['TFLOPS_per_watt_FP32'] = ml_hardware['FP32_TFLOPS'] / ml_hardware['TDP_numeric']
ml_hardware['TFLOPS_per_watt_FP16'] = ml_hardware['FP16_TFLOPS'] / ml_hardware['TDP_numeric']
ml_hardware['TFLOPS_per_watt_INT8'] = ml_hardware['INT8_TFLOPS'] / ml_hardware['TDP_numeric']
ml_hardware['Memory_bandwidth_per_TFLOPS'] = ml_hardware['Memory_bandwidth_GB_s'] / ml_hardware['FP32_TFLOPS']

print("Cleaning MLPerf data...")
# Clean MLPerf data
mlperf['Accelerator_std'] = mlperf['Accelerator'].apply(standardize_gpu_name)
mlperf['Result_numeric'] = mlperf['Avg. Result at System Name'].apply(convert_to_numeric)

# Group MLPerf results by hardware and benchmark
mlperf_grouped = mlperf.groupby(['Accelerator', 'Benchmark', 'Scenario']).agg({
    'Result_numeric': 'mean',
    '# of Accelerators': 'first',
    'Host Processor Core Count': 'first'
}).reset_index()

# Create pivot table for model performance by hardware
mlperf_pivot = mlperf.pivot_table(
    index='Accelerator_std',
    columns=['Benchmark', 'Scenario'],
    values='Result_numeric',
    aggfunc='mean'
).reset_index()

# Flatten column names
mlperf_pivot.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in mlperf_pivot.columns]

print("Merging datasets...")
# Merge GPU benchmarks with API scores
merged_gpu = pd.merge(
    gpu_benchmarks,
    gpu_api_scores,
    left_on='gpuName_std',
    right_on='Device_std',
    how='outer'
)

# Try to match ML hardware with GPU data
# This is challenging due to naming differences, so we'll use a fuzzy approach
merged_all = pd.DataFrame()

# Create a mapping dictionary for common hardware
hardware_mapping = {}

# For each ML hardware entry, try to find a matching GPU
for _, hw_row in ml_hardware.iterrows():
    hw_name = hw_row['Hardware_std']
    if pd.isna(hw_name):
        continue
    
    # Look for exact matches first
    matches = merged_gpu[merged_gpu['gpuName_std'] == hw_name]
    
    if len(matches) > 0:
        hardware_mapping[hw_name] = hw_name
    else:
        # Look for partial matches
        for gpu_name in merged_gpu['gpuName_std'].dropna().unique():
            if pd.isna(gpu_name):
                continue
                
            # Check if the hardware name contains the GPU name or vice versa
            if hw_name in gpu_name or gpu_name in hw_name:
                hardware_mapping[hw_name] = gpu_name
                break

# Create a new dataframe with ML hardware and matched GPU data
merged_all = pd.DataFrame()

# For each ML hardware entry, add the matched GPU data
for _, hw_row in ml_hardware.iterrows():
    hw_name = hw_row['Hardware_std']
    
    # Create a new row with ML hardware data
    new_row = hw_row.to_dict()
    
    # If we have a match, add the GPU data
    if hw_name in hardware_mapping:
        gpu_name = hardware_mapping[hw_name]
        gpu_matches = merged_gpu[merged_gpu['gpuName_std'] == gpu_name]
        
        if len(gpu_matches) > 0:
            # Use the first match
            gpu_data = gpu_matches.iloc[0].to_dict()
            
            # Add GPU data to the new row
            for key, value in gpu_data.items():
                if key not in new_row:
                    new_row[key] = value
    
    # Add MLPerf data if available
    mlperf_matches = mlperf[mlperf['Accelerator_std'] == hw_name]
    
    if len(mlperf_matches) > 0:
        # Group MLPerf data by benchmark and scenario
        mlperf_data = mlperf_matches.groupby(['Benchmark', 'Scenario']).agg({
            'Result_numeric': 'mean'
        }).reset_index()
        
        # Add MLPerf data to the new row
        for _, perf_row in mlperf_data.iterrows():
            benchmark = perf_row['Benchmark']
            scenario = perf_row['Scenario']
            result = perf_row['Result_numeric']
            
            new_row[f'{benchmark}_{scenario}'] = result
    
    # Add the new row to the merged dataframe
    merged_all = pd.concat([merged_all, pd.DataFrame([new_row])], ignore_index=True)

# For GPUs that don't have ML hardware data, add them with available information
for _, gpu_row in merged_gpu.iterrows():
    gpu_name = gpu_row['gpuName_std']
    
    # Skip if this GPU is already in the merged dataframe
    if gpu_name in merged_all['Hardware_std'].values:
        continue
    
    # Create a new row with GPU data
    new_row = gpu_row.to_dict()
    
    # Add MLPerf data if available
    mlperf_matches = mlperf[mlperf['Accelerator_std'] == gpu_name]
    
    if len(mlperf_matches) > 0:
        # Group MLPerf data by benchmark and scenario
        mlperf_data = mlperf_matches.groupby(['Benchmark', 'Scenario']).agg({
            'Result_numeric': 'mean'
        }).reset_index()
        
        # Add MLPerf data to the new row
        for _, perf_row in mlperf_data.iterrows():
            benchmark = perf_row['Benchmark']
            scenario = perf_row['Scenario']
            result = perf_row['Result_numeric']
            
            new_row[f'{benchmark}_{scenario}'] = result
    
    # Add the new row to the merged dataframe
    merged_all = pd.concat([merged_all, pd.DataFrame([new_row])], ignore_index=True)

print("Calculating derived metrics...")
# Calculate additional derived metrics
# Performance scaling across precision levels
merged_all['FP16_to_FP32_scaling'] = merged_all['FP16_TFLOPS'] / merged_all['FP32_TFLOPS']
merged_all['INT8_to_FP32_scaling'] = merged_all['INT8_TFLOPS'] / merged_all['FP32_TFLOPS']

# Memory-compute balance metrics
merged_all['memory_bandwidth_compute_ratio'] = merged_all['Memory_bandwidth_GB_s'] / merged_all['FP32_TFLOPS']

# Efficiency metrics
merged_all['perf_per_watt'] = merged_all['FP32_TFLOPS'] / merged_all['TDP_numeric']
merged_all['memory_efficiency'] = merged_all['Memory_bandwidth_GB_s'] / merged_all['Memory_size_GB']

# Normalize key performance metrics
perf_columns = [col for col in merged_all.columns if 'TFLOPS' in col or 'Offline' in col or 'Server' in col]
for col in perf_columns:
    if col in merged_all.columns and not merged_all[col].isna().all():
        max_val = merged_all[col].max()
        if max_val > 0:
            merged_all[f'{col}_normalized'] = merged_all[col] / max_val

# Add generation indicators
merged_all['is_ampere'] = merged_all['Hardware_std'].apply(lambda x: 1 if isinstance(x, str) and ('a100' in x or 'a40' in x or 'a30' in x or 'a10' in x) else 0)
merged_all['is_hopper'] = merged_all['Hardware_std'].apply(lambda x: 1 if isinstance(x, str) and ('h100' in x or 'h200' in x) else 0)
merged_all['is_ada_lovelace'] = merged_all['Hardware_std'].apply(lambda x: 1 if isinstance(x, str) and ('rtx 40' in x or 'rtx a60' in x) else 0)

print("Cleaning final dataset...")
# Clean up the final dataset
# Remove duplicate columns
merged_all = merged_all.loc[:, ~merged_all.columns.duplicated()]

# Fill missing values for key metrics with column means
key_metrics = ['FP32_TFLOPS', 'Memory_bandwidth_GB_s', 'TDP_numeric', 'perf_per_watt']
for col in key_metrics:
    if col in merged_all.columns:
        merged_all[col] = merged_all[col].fillna(merged_all[col].mean())

# Save the final dataset
print("Saving final dataset...")
merged_all.to_csv('data/processed/ai_hardware_performance_matrix.csv', index=False)

print("ETL process completed successfully!")

# Print summary statistics
print("\nFinal dataset shape:", merged_all.shape)
print("Number of columns:", len(merged_all.columns))
print("Key metrics available:", [col for col in merged_all.columns if 'TFLOPS' in col or 'bandwidth' in col or 'Offline' in col or 'Server' in col]) 