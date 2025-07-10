# AI Benchmark KPI Project: Derived Column Documentation
<!-- Phase 2 Final Enhanced Dataset -->

<p align="center">
  <img src="https://img.shields.io/badge/Version-2.1.0-blue.svg" alt="Version 2.1.0">
  <img src="https://img.shields.io/badge/Status-Production-green.svg" alt="Status: Production">
  <img src="https://img.shields.io/badge/Records-2,108-orange.svg" alt="Records: 2,108">
  <img src="https://img.shields.io/badge/Columns-49-yellow.svg" alt="Columns: 49">
</p>

## Table of Contents

- [Overview](#overview)
- [Efficiency Metrics](#efficiency-metrics)
- [Bias Correction Metrics](#bias-correction-metrics)
- [Performance Prediction Metrics](#performance-prediction-metrics)
- [Throughput Metrics](#throughput-metrics)
- [Latency Metrics](#latency-metrics)
- [Classification Metrics](#classification-metrics)
- [Implementation Notes](#implementation-notes)

## Overview

This document details the mathematical derivations and calculations for the key derived columns in the `phase2_final_enhanced_dataset.csv`. These engineered features serve as the foundation for the AI Benchmark KPI project's machine learning models.

---

## Efficiency Metrics

### GFLOPS_per_Watt

```
GFLOPS_per_Watt = (FP32_Final / 1e9) / TDP
```

**Description:** Normalizes raw computational throughput by power consumption, providing a standardized efficiency metric. Higher values indicate more energy-efficient hardware.

**Units:** Giga Floating-Point Operations per Second per Watt

**Application:** Critical for energy efficiency analysis and data center deployment planning.

### TOPs_per_Watt

```
TOPs_per_Watt = (FP32_Final / 1e12) / TDP
```

**Description:** Tera Operations per Second per Watt - the AI industry standard for measuring computational efficiency. TOPs_per_Watt is particularly relevant for neural network inference efficiency.

**Units:** Tera Operations per Second per Watt

**Application:** Data center planning, operational cost estimation, and green AI initiatives.

### Performance_per_Dollar_per_Watt

```
Performance_per_Dollar_per_Watt = (FP32_Final / 1e9) / (price * TDP)
```

**Description:** Economic efficiency metric combining performance, acquisition cost, and operational power consumption into a unified value metric.

**Units:** GFLOPS/(USD × Watt)

**Application:** Total cost of ownership (TCO) analysis for hardware procurement decisions.

---

## Bias Correction Metrics

### Manufacturer_Bias_Factor

```
Manufacturer_Bias_Factor = manufacturer_bias_factors[Manufacturer][base_adjustment]
```

**Description:** Applies vendor-specific performance adjustments to normalize hardware across manufacturers:
- NVIDIA: 1.25 (CUDA optimization, Tensor cores)
- AMD: 1.045 (OpenCL optimization)
- Intel: 0.85 (Integrated graphics focus)

**Application:** Enables fair cross-vendor hardware comparisons by accounting for software ecosystem advantages.

### Generation_Bias_Factor

```
Generation_Bias_Factor = generation_weights[GenerationCategory]
```

**Description:** Applies temporal normalization to account for architectural advancements across hardware generations:

| Generation Category | Factor |
|---------------------|--------|
| Current Gen (2022+) | 1.0    |
| Recent Gen (2020-2021) | 0.95 |
| Previous Gen (2018-2019) | 0.85 |
| Older Gen (2016-2017) | 0.75 |
| Legacy Gen (2014-2015) | 0.6 |
| Very Legacy (<2014) | 0.5 |

**Application:** Cross-generational performance normalization for future-proof predictions.

### Category_Bias_Factor

```
Category_Bias_Factor = category_weights[AI_Performance_Category][accuracy]
```

**Description:** Applies workload-specific adjustment factors based on the GPU's AI performance category:

| AI Performance Category | Factor |
|-------------------------|--------|
| AI_Flagship            | 0.9    |
| AI_High_End            | 0.7    |
| AI_Mid_Range           | 0.5    |
| AI_Entry               | 0.3    |
| AI_Basic               | 0.2    |

**Application:** Optimizes predictions for the appropriate use case of each hardware class.

### FP32_Final_Architecture_Factor

```
FP32_Final_Architecture_Factor = architecture_factors[Architecture][generation_factor]
```

**Description:** Architecture-specific performance scaling factor derived from computational modeling of each GPU architecture's strengths:

| Architecture    | Factor |
|-----------------|--------|
| Ada Lovelace    | 1.65   |
| Ampere          | 1.50   |
| Volta           | 1.35   |
| RDNA 2          | 1.35   |
| Turing          | 1.275  |
| RDNA            | 1.20   |
| Pascal          | 1.125  |
| Other architectures | 0.43-0.98 |

**Application:** Precise normalization across different GPU design philosophies and compute approaches.

---

## Performance Prediction Metrics

### Bias_Corrected_Performance

```
Bias_Corrected_Performance = FP32_Final * Manufacturer_Bias_Factor * 
                             Generation_Bias_Factor * Category_Bias_Factor
```

**Description:** Core derived metric applying all bias correction factors to the raw FP32 performance to obtain a normalized performance value.

**Units:** FLOPS (Floating-Point Operations per Second)

**Application:** Foundation for performance prediction models that operate across vendors, generations, and use cases.

### FP32_Final_Architecture_Normalized

```
FP32_Final_Architecture_Normalized = FP32_Final / FP32_Final_Architecture_Factor
```

**Description:** Architecture-normalized raw performance that accounts for fundamental design differences between GPU architectures.

**Units:** FLOPS (Floating-Point Operations per Second)

**Application:** Cross-architecture performance comparisons and transfer learning.

### Total_Bias_Correction

```
Total_Bias_Correction = Manufacturer_Bias_Factor * Generation_Bias_Factor * Category_Bias_Factor
```

**Description:** Combined correction factor encompassing all bias adjustments applied to the raw performance metrics.

**Application:** Understanding the magnitude of adjustments applied to each hardware configuration.

---

## Throughput Metrics

### Throughput Model Estimations

```
Throughput_Model_fps = FP32_Final / Model_FLOPS_per_inference
```

**Models implemented:**

| Model | FLOPS per inference | Formula |
|-------|---------------------|---------|
| Throughput_ResNet50_ImageNet_fps | 4.1 GFLOPS | FP32_Final / 4.1e9 |
| Throughput_BERT_Base_fps | 22.5 GFLOPS | FP32_Final / 22.5e9 |
| Throughput_GPT2_Small_fps | 1.5 GFLOPS | FP32_Final / 1.5e9 |
| Throughput_MobileNetV2_fps | 0.3 GFLOPS | FP32_Final / 0.3e9 |
| Throughput_EfficientNet_B0_fps | 0.39 GFLOPS | FP32_Final / 0.39e9 |

**Description:** Estimated frames/inferences per second for standard AI models based on their computational complexity.

**Units:** Frames per Second (FPS) or Inferences per Second

**Application:** Real-world performance estimation for specific AI workloads.

### Avg_Throughput_fps

```
Avg_Throughput_fps = mean(All_Model_Throughputs)
```

**Description:** Average throughput across all standard model benchmarks, providing a generalized performance indicator.

**Units:** Average Frames per Second (FPS)

**Application:** General-purpose AI performance estimation.

---

## Latency Metrics

### Relative_Latency_Index

```
Relative_Latency_Index = max(FP32_Final) / FP32_Final
```

**Description:** Normalized latency estimation using an inverse relationship with computational throughput. Value of 1.0 represents the fastest device, with higher values indicating proportionally higher latency.

**Units:** Dimensionless ratio

**Application:** Inference response time estimation, particularly for real-time applications.

### Compute_Usage_Percent

```
Compute_Usage_Percent = (powerPerformance / max(powerPerformance)) × 100
```

**Description:** Estimated compute utilization percentage based on power efficiency normalization.

**Units:** Percentage (%)

**Application:** Resource utilization optimization and efficiency analysis.

---

## Classification Metrics

### AI_Performance_Category

**Categories:**
- AI_Flagship
- AI_High_End
- AI_Mid_Range  
- AI_Entry
- AI_Basic

**Description:** AI-specific performance tier classification based on FP32 performance and AI throughput.

**Application:** Hardware recommendation systems and workload matching.

### AI_Efficiency_Tier

**Categories:**
- Ultra
- Premium
- High-End
- Mid-Range
- Entry

**Description:** AI-specific efficiency tier classification based on TOPs_per_Watt thresholds.

**Application:** Quick AI efficiency comparison and energy-focused deployments.

---

## Implementation Notes

1. The bias correction methodology is based on empirical analysis of over 2,100 GPU configurations across 17 architectures.

2. Architecture factors were derived using both theoretical analysis and actual benchmark results, captured in the `architecture_factors.json` configuration file.

3. All derived columns maintain full traceability to source measurements, allowing for validation and adjustment as new hardware generations emerge.

4. These derived metrics form the foundation for the Phase 3 machine learning models, enabling accurate cross-vendor and cross-generation predictions.

---

<p align="center">
  <b>AI Benchmark KPI Project</b><br>
  © 2023 AI Performance Research Group<br>
  <small>Version 2.1.0 | Last Updated: November 2023</small>
</p> 