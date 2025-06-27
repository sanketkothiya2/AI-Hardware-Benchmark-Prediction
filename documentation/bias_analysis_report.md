# Bias Analysis Report
==================================================

## Manufacturer Performance Analysis

### NVIDIA
- **Total GPUs**: 733
- **Architectures**: Ampere, Turing, Pascal, Volta, Ada Lovelace, Maxwell, Kepler, Fermi, Unknown, Tesla (Legacy)
- **Performance Bias Factors**:
  - FP32_Final: 1.670x
  - TOPs_per_Watt: 1.590x
  - G3Dmark: 1.470x
  - powerPerformance: 1.435x
  - GFLOPS_per_Watt: 1.590x

### AMD
- **Total GPUs**: 1226
- **Architectures**: RDNA 2, RDNA, GCN (Vega), GCN, RDNA 3, VLIW5, Unknown
- **Performance Bias Factors**:
  - FP32_Final: 0.695x
  - TOPs_per_Watt: 0.736x
  - G3Dmark: 0.808x
  - powerPerformance: 0.824x
  - GFLOPS_per_Watt: 0.736x

### Intel
- **Total GPUs**: 149
- **Architectures**: Gen9/Gen11, Xe
- **Performance Bias Factors**:
  - FP32_Final: 0.216x
  - TOPs_per_Watt: 0.265x
  - G3Dmark: 0.268x
  - powerPerformance: 0.310x
  - GFLOPS_per_Watt: 0.265x

## Bias Correction Strategy

### Manufacturer Adjustments
- **NVIDIA**: Base adjustment 1.0
- **AMD**: Base adjustment 0.95
- **Intel**: Base adjustment 0.85

### Generation Weights
- **Current Gen (2022+)**: 1.0
- **Recent Gen (2020-2021)**: 0.95
- **Previous Gen (2018-2019)**: 0.85
- **Older Gen (2016-2017)**: 0.75
- **Legacy Gen (2014-2015)**: 0.6
- **Very Legacy (<2014)**: 0.5