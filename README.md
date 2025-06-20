# AI Hardware Performance Matrix ETL Pipeline

This project provides a comprehensive ETL (Extract, Transform, Load) pipeline for processing and analyzing AI hardware performance data from multiple sources.

## Project Structure

```
AI_Benchmark_kpi/
├── data/
│   ├── raw/                          # Original source datasets
│   │   ├── GPU_benchmarks_v7.csv     # GPU specifications and benchmark scores
│   │   ├── GPU_scores_graphicsAPIs.csv  # GPU performance across graphics APIs
│   │   ├── ml_hardware.csv           # ML hardware specifications and FLOPS
│   │   └── mlperf.csv               # MLPerf benchmark results
│   └── processed/                    # Processed and integrated datasets
│       ├── ai_hardware_performance_matrix.csv  # Main integrated dataset
│       └── performance_metrics_correlation.csv # Correlation analysis
│
├── scripts/
│   ├── etl/                         # ETL pipeline scripts
│   │   └── ai_hardware_etl.py       # Main ETL processing script
│   └── utils/                       # Utility and analysis scripts
│       ├── inspect_data.py          # Data inspection utility
│       ├── analyze_final_dataset.py # Dataset analysis script
│       └── visualize_dataset.py     # Data visualization script
│
├── analysis/
│   ├── reports/                     # Generated analysis reports
│   └── visualizations/              # Generated charts and plots
│       └── (various .png files)     # Performance charts and correlations
│
├── documentation/                   # Project documentation
│   ├── ai_hardware_etl_documentation.md  # Detailed ETL process documentation
│   └── project proposal.pdf        # Original project proposal
│
└── output/                         # Final output files (empty for now)
```

## Dataset Sources

1. **GPU_benchmarks_v7.csv** (2,317 rows): Comprehensive GPU specifications including G3Dmark/G2Dmark scores, pricing, TDP, and release dates
2. **GPU_scores_graphicsAPIs.csv** (1,213 rows): GPU performance across CUDA, Metal, OpenCL, and Vulkan APIs
3. **ml_hardware.csv** (161 rows): ML hardware specifications with FLOPS performance across different precision levels
4. **mlperf.csv** (861 rows): MLPerf benchmark results for inference performance across various models

## Key Features

- **Data Integration**: Merges four disparate datasets using intelligent name matching
- **Data Cleaning**: Handles missing values, standardizes units, and normalizes hardware names
- **Feature Engineering**: Creates derived metrics like performance-per-watt, scaling factors, and efficiency ratios
- **Rich Metrics**: Supports prediction of latency, throughput, FLOPS, memory bandwidth, energy consumption, and more

## Usage

### 1. Data Inspection
```bash
python scripts/utils/inspect_data.py
```
Provides basic statistics and structure information for all source datasets.

### 2. Run ETL Pipeline
```bash
python scripts/etl/ai_hardware_etl.py
```
Processes all source datasets and generates the integrated performance matrix.

### 3. Analyze Results
```bash
python scripts/utils/analyze_final_dataset.py
```
Generates comprehensive statistics and analysis of the final dataset.

### 4. Generate Visualizations
```bash
python scripts/utils/visualize_dataset.py
```
Creates various charts and plots showing performance relationships and trends.

## Output Dataset

The main output `ai_hardware_performance_matrix.csv` contains **3,008 rows** and **99 columns** including:

### Core Hardware Specifications
- Hardware name, manufacturer, and type
- Release year and pricing information
- Process technology and TDP
- Memory specifications and bandwidth
- Clock speeds (base, boost, memory)

### Performance Metrics
- FLOPS across precision levels (FP64, FP32, FP16, INT8, INT4)
- Graphics benchmark scores (G3Dmark, G2Dmark)
- API performance (CUDA, Metal, OpenCL, Vulkan)
- MLPerf benchmark results

### Derived Efficiency Metrics
- Performance per watt ratios
- Memory bandwidth to compute ratios
- Performance scaling across precision levels
- Normalized performance metrics for modeling

### Architecture Indicators
- Binary flags for major GPU architectures (Ampere, Hopper, Ada Lovelace)

## Key Metrics for AI Performance Prediction

The dataset enables prediction of these critical KPIs:
- **Latency**: Server scenario response times
- **Throughput**: Offline scenario processing rates
- **FLOPS**: Multi-precision floating point performance
- **Memory Bandwidth**: Data transfer capabilities
- **Energy Efficiency**: Performance per watt metrics
- **Precision Scaling**: Mixed precision speedup factors

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (for preprocessing utilities)

## Documentation

Detailed documentation about the ETL process, data transformations, and methodology can be found in:
- `documentation/ai_hardware_etl_documentation.md`

## Contributing

When adding new data sources or modifying the ETL pipeline:
1. Update the corresponding scripts in `scripts/etl/`
2. Document changes in the documentation folder
3. Run the analysis scripts to validate results
4. Update this README if the structure changes 