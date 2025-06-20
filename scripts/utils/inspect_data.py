import pandas as pd

# Load each CSV file
gpu_benchmarks = pd.read_csv('data/raw/GPU_benchmarks_v7.csv')
gpu_api_scores = pd.read_csv('data/raw/GPU_scores_graphicsAPIs.csv')
ml_hardware = pd.read_csv('data/raw/ml_hardware.csv')
mlperf = pd.read_csv('data/raw/mlperf.csv')

# Print shape and first few rows of each dataframe
print("GPU_benchmarks_v7.csv shape:", gpu_benchmarks.shape)
print("GPU_scores_graphicsAPIs.csv shape:", gpu_api_scores.shape)
print("ml_hardware.csv shape:", ml_hardware.shape)
print("mlperf.csv shape:", mlperf.shape)

# Print column names for each dataframe
print("\nGPU_benchmarks_v7.csv columns:", gpu_benchmarks.columns.tolist())
print("\nGPU_scores_graphicsAPIs.csv columns:", gpu_api_scores.columns.tolist())
print("\nml_hardware.csv columns:", ml_hardware.columns.tolist())
print("\nmlperf.csv columns:", mlperf.columns.tolist())

# Check for missing values in each dataframe
print("\nMissing values in GPU_benchmarks_v7.csv:")
print(gpu_benchmarks.isnull().sum())

print("\nMissing values in GPU_scores_graphicsAPIs.csv:")
print(gpu_api_scores.isnull().sum())

print("\nMissing values in ml_hardware.csv:")
print(ml_hardware.isnull().sum())

print("\nMissing values in mlperf.csv:")
print(mlperf.isnull().sum()) 