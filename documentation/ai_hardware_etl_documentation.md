# AI Hardware Performance Matrix ETL Documentation

## Overview

This document explains the Extract, Transform, Load (ETL) process used to create a comprehensive AI hardware performance matrix from four source datasets:

1. **GPU_benchmarks_v7.csv**: Contains detailed GPU specifications including name, G3Dmark/G2Dmark scores, price, TDP, memory, and release dates for over 2,300 GPUs.
2. **GPU_scores_graphicsAPIs.csv**: Shows GPU performance across different graphics APIs (CUDA, Metal, OpenCL, Vulkan) for approximately 1,200 GPUs.
3. **ml_hardware.csv**: Provides specifications for ML hardware including TPUs, GPUs, and AI accelerators with metrics like FLOPS performance, memory bandwidth, power consumption, and process technology.
4. **mlperf.csv**: Contains MLPerf benchmark results showing inference performance for various models across different hardware configurations.

The resulting dataset (`ai_hardware_performance_matrix.csv`) integrates these sources to create a comprehensive matrix for AI hardware performance prediction, focusing on key metrics like latency, throughput, FLOPS, memory bandwidth, power efficiency, and more.

## ETL Process

### 1. Data Extraction

The process begins by loading the four source CSV files using pandas:

```python
gpu_benchmarks = pd.read_csv('GPU_benchmarks_v7.csv')
gpu_api_scores = pd.read_csv('GPU_scores_graphicsAPIs.csv')
ml_hardware = pd.read_csv('ml_hardware.csv')
mlperf = pd.read_csv('mlperf.csv')
```

### 2. Data Cleaning and Transformation

#### 2.1 Standardizing Hardware Names

One of the main challenges in merging these datasets is the inconsistent naming of hardware across files. We implemented a standardization function to normalize GPU and accelerator names:

```python
def standardize_gpu_name(name):
    # Convert to lowercase
    name = str(name).lower()
    
    # Remove common prefixes
    name = name.replace('geforce ', '')
    name = name.replace('radeon ', '')
    name = name.replace('nvidia ', '')
    name = name.replace('amd ', '')
    name = name.replace('intel ', '')
    
    # Standardize naming conventions
    name = re.sub(r'rtx\s*', 'rtx ', name)
    name = re.sub(r'gtx\s*', 'gtx ', name)
    name = re.sub(r'(\d+)\s*ti', r'\1 ti', name)
    name = re.sub(r'(\d+)\s*super', r'\1 super', name)
    
    return name.strip()
```

#### 2.2 Converting String Values to Numeric

Many performance metrics in the datasets are stored as strings with units (e.g., "10.5 TFLOPS"). We implemented a function to convert these to numeric values:

```python
def convert_to_numeric(value):
    # Handle various formats including scientific notation and units
    multipliers = {
        'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12,
        'GB': 1e9, 'MB': 1e6, 'KB': 1e3, 'TB': 1e12, 'G': 1e9
    }
    
    # Try to extract numeric value and apply appropriate multiplier
    for unit, multiplier in multipliers.items():
        if unit in value:
            numeric_part = value.replace(unit, '').strip()
            return float(numeric_part) * multiplier
```

#### 2.3 Extracting Release Years

To enable temporal analysis, we extracted release years from date strings:

```python
def extract_year(date_str):
    try:
        # Handle year-only format
        if len(str(date_str)) == 4 and str(date_str).isdigit():
            return int(date_str)
        
        # Parse full date
        date_obj = pd.to_datetime(date_str)
        return date_obj.year
    except:
        return np.nan
```

#### 2.4 Dataset-Specific Transformations

##### GPU Benchmarks Data
- Standardized GPU names
- Converted price and TDP to numeric values
- Calculated performance-per-watt and performance-per-dollar metrics

##### GPU API Scores Data
- Standardized device names
- Converted API scores to numeric values
- Calculated performance ratios between different APIs (e.g., CUDA to OpenCL ratio)

##### ML Hardware Data
- Standardized hardware names
- Converted FLOPS metrics to TFLOPS (teraflops) for easier comparison
- Calculated efficiency metrics (TFLOPS per watt, memory bandwidth per TFLOPS)

##### MLPerf Data
- Standardized accelerator names
- Converted benchmark results to numeric values
- Created aggregated metrics by hardware, benchmark, and scenario

### 3. Data Integration

#### 3.1 Merging Strategy

The integration process follows these steps:

1. First, GPU benchmarks and API scores are merged based on standardized device names
2. ML hardware is then matched with the merged GPU data using both exact and fuzzy matching
3. MLPerf benchmark results are added to each hardware entry where available
4. GPUs without ML hardware data are also included with their available metrics

#### 3.2 Hardware Name Matching

To handle the challenge of matching hardware across datasets with different naming conventions:

```python
# Create a mapping dictionary for common hardware
hardware_mapping = {}

# For each ML hardware entry, try to find a matching GPU
for _, hw_row in ml_hardware.iterrows():
    hw_name = hw_row['Hardware_std']
    
    # Look for exact matches first
    matches = merged_gpu[merged_gpu['gpuName_std'] == hw_name]
    
    if len(matches) > 0:
        hardware_mapping[hw_name] = hw_name
    else:
        # Look for partial matches
        for gpu_name in merged_gpu['gpuName_std'].dropna().unique():
            # Check if the hardware name contains the GPU name or vice versa
            if hw_name in gpu_name or gpu_name in hw_name:
                hardware_mapping[hw_name] = gpu_name
                break
```

### 4. Derived Metrics Calculation

To enrich the dataset for KPI prediction, we calculated several derived metrics:

#### 4.1 Performance Scaling Metrics
- `FP16_to_FP32_scaling`: Ratio of FP16 to FP32 performance (shows mixed precision speedup)
- `INT8_to_FP32_scaling`: Ratio of INT8 to FP32 performance (shows quantization speedup)

#### 4.2 Efficiency Metrics
- `perf_per_watt`: TFLOPS per watt (energy efficiency)
- `memory_efficiency`: Memory bandwidth per GB of memory (memory system efficiency)
- `memory_bandwidth_compute_ratio`: Memory bandwidth to compute ratio (balance metric)

#### 4.3 Normalized Metrics
All key performance metrics are normalized (divided by their maximum value) to create scale-independent features for modeling:

```python
perf_columns = [col for col in merged_all.columns if 'TFLOPS' in col or 'Offline' in col or 'Server' in col]
for col in perf_columns:
    if col in merged_all.columns and not merged_all[col].isna().all():
        max_val = df[col].max()
        if max_val > 0:
            merged_all[f'{col}_normalized'] = merged_all[col] / max_val
```

#### 4.4 Architecture Generation Indicators
Binary indicators for major GPU architectures:
- `is_ampere`: NVIDIA Ampere architecture (A100, A40, A30, A10)
- `is_hopper`: NVIDIA Hopper architecture (H100, H200)
- `is_ada_lovelace`: NVIDIA Ada Lovelace architecture (RTX 40xx, RTX A60xx)

### 5. Data Cleaning and Final Processing

#### 5.1 Duplicate Column Removal
```python
merged_all = merged_all.loc[:, ~merged_all.columns.duplicated()]
```

#### 5.2 Missing Value Handling
For key metrics, missing values are filled with column means:
```python
key_metrics = ['FP32_TFLOPS', 'Memory_bandwidth_GB_s', 'TDP_numeric', 'perf_per_watt']
for col in key_metrics:
    if col in merged_all.columns:
        merged_all[col] = merged_all[col].fillna(merged_all[col].mean())
```

## Resulting Dataset Structure

The final dataset (`ai_hardware_performance_matrix.csv`) includes the following categories of metrics:

### Core Hardware Specifications
- Hardware name and manufacturer
- Release year and price
- Process technology (nm)
- TDP (Thermal Design Power)
- Memory size and bandwidth
- Clock speeds (base, boost, memory)

### Performance Metrics
- FLOPS at various precision levels (FP64, FP32, FP16, INT8, INT4)
- G3Dmark and G2Dmark scores (for GPUs)
- Graphics API performance (CUDA, Metal, OpenCL, Vulkan)
- MLPerf benchmark results for various models and scenarios

### Derived Efficiency Metrics
- Performance per watt
- Performance per dollar
- Memory bandwidth to compute ratio
- Performance scaling across precision levels
- Normalized performance metrics

### Architecture Indicators
- Binary flags for major GPU architectures

## Key Metrics for KPI Prediction

The dataset is specifically designed to support prediction of these key AI hardware performance metrics:

1. **Latency**: Available from MLPerf server scenario results
2. **Throughput**: Available from MLPerf offline scenario results
3. **FLOPS**: Available across multiple precision levels (FP64, FP32, FP16, INT8, INT4)
4. **Memory Bandwidth**: Available in GB/s
5. **Energy Consumption**: Derivable from TDP and performance metrics
6. **Precision**: Multiple precision levels with scaling factors
7. **TOPs/Watt**: Available as derived metrics
8. **Memory Usage**: Derivable from model size and hardware specifications
9. **Compute Usage**: Available from MLPerf results and hardware specifications

## Usage for Model Training

This integrated dataset provides a rich foundation for training machine learning models to predict AI hardware performance. Potential modeling approaches include:

1. **Regression models** for predicting specific performance metrics
2. **Classification models** for categorizing hardware into performance tiers
3. **Time series models** for predicting performance trends across hardware generations
4. **Multi-task models** for simultaneously predicting multiple related performance metrics

The normalized metrics and architecture indicators are particularly valuable for transfer learning approaches, where models trained on one performance domain can be adapted to others.

## Limitations and Considerations

1. **Missing Data**: Despite our best efforts, some hardware entries have missing values for certain metrics
2. **Name Matching**: The fuzzy matching approach for hardware names may introduce some errors
3. **Temporal Aspects**: Performance metrics may be affected by driver versions and software optimizations not captured in the dataset
4. **Benchmark Variability**: MLPerf results can vary based on implementation details and optimization levels

## Conclusion

The AI Hardware Performance Matrix provides a comprehensive, integrated view of AI accelerator performance across multiple dimensions. By combining general GPU benchmarks, graphics API scores, ML hardware specifications, and MLPerf results, it enables sophisticated modeling of hardware performance for AI workloads. 