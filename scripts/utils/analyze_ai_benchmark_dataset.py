import pandas as pd
import numpy as np

def analyze_ai_benchmark_dataset():
    """
    Analyze the AI-Benchmark-cleaned.csv dataset against project requirements.
    """
    
    # Load the dataset
    print("Loading AI-Benchmark-cleaned.csv...")
    df = pd.read_csv('data/AI-Benchmark-cleaned.csv')
    
    print("="*80)
    print("AI BENCHMARK DATASET ANALYSIS FOR PROJECT REQUIREMENTS")
    print("="*80)
    
    # Basic dataset info
    print(f"\n1. DATASET OVERVIEW:")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Manufacturers: {df['Manufacturer'].value_counts().to_dict()}")
    
    # Required key metrics from project
    required_metrics = [
        'Latency', 'Throughput', 'FLOPS', 'Memory Bandwidth', 
        'Model Size', 'Energy Consumption', 'Precision', 
        'Network Density', 'FPS', 'TOPs/Watt', 
        'Memory Usage %', 'Compute Usage %'
    ]
    
    print(f"\n2. COLUMN ANALYSIS:")
    print(f"   Available columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\n3. KEY METRICS ALIGNMENT:")
    
    # Check which required metrics are available
    available_metrics = []
    missing_metrics = []
    
    for metric in required_metrics:
        # Look for columns that might contain this metric
        metric_clean = metric.lower().replace(' ', '').replace('/', '').replace('%', '')
        found_cols = []
        
        for col in df.columns:
            col_clean = col.lower().replace(' ', '').replace('/', '').replace('%', '').replace('(', '').replace(')', '')
            if metric_clean in col_clean or any(word in col_clean for word in metric_clean.split()):
                found_cols.append(col)
        
        if found_cols:
            available_metrics.append((metric, found_cols))
        else:
            missing_metrics.append(metric)
    
    print(f"\n   ✅ AVAILABLE METRICS ({len(available_metrics)}/{len(required_metrics)}):")
    for metric, cols in available_metrics:
        print(f"      • {metric}: {cols}")
    
    print(f"\n   ❌ MISSING METRICS ({len(missing_metrics)}/{len(required_metrics)}):")
    for metric in missing_metrics:
        print(f"      • {metric}")
    
    # Data completeness analysis
    print(f"\n4. DATA COMPLETENESS ANALYSIS:")
    high_missing = []
    good_data = []
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        
        if null_pct > 50:
            high_missing.append((col, null_pct))
        elif null_pct < 25:
            good_data.append((col, null_pct))
    
    print(f"\n   📊 COLUMNS WITH GOOD DATA COVERAGE (< 25% missing):")
    for col, pct in sorted(good_data, key=lambda x: x[1]):
        print(f"      • {col}: {100-pct:.1f}% populated")
    
    print(f"\n   ⚠️  COLUMNS WITH HIGH MISSING DATA (> 50% missing):")
    for col, pct in sorted(high_missing, key=lambda x: x[1], reverse=True):
        print(f"      • {col}: {pct:.1f}% missing")
    
    # Performance metrics analysis
    performance_cols = [
        'FP32_Final', 'FP16 (half precision) performance (FLOP/s)', 
        'INT8 performance (OP/s)', 'FLOPS_per_Watt', 
        'Memory bandwidth (byte/s)', 'TDP', 'powerPerformance',
        'G3Dmark', 'G2Dmark', 'Memory_GB', 'Process size (nm)'
    ]
    
    print(f"\n5. CORE PERFORMANCE METRICS STATUS:")
    for col in performance_cols:
        if col in df.columns:
            populated = df[col].notna().sum()
            pct = (populated / len(df)) * 100
            print(f"   • {col}: {populated}/{len(df)} ({pct:.1f}%) populated")
    
    # Architecture and generation analysis
    print(f"\n6. HARDWARE DIVERSITY:")
    print(f"   • Architectures: {len(df['Architecture'].unique())} unique")
    print(f"     Top architectures: {dict(df['Architecture'].value_counts().head())}")
    print(f"   • Categories: {dict(df['Category'].value_counts())}")
    print(f"   • Generations: {dict(df['Generation'].value_counts())}")
    
    # Price and value analysis
    print(f"\n7. PRICING DATA:")
    price_available = df['price'].notna().sum()
    print(f"   • Price data: {price_available}/{len(df)} ({(price_available/len(df)*100):.1f}%) available")
    if price_available > 0:
        print(f"   • Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"   • Median price: ${df['price'].median():.2f}")
    
    # Recommendations
    print(f"\n8. DATASET ASSESSMENT FOR PROJECT GOALS:")
    
    score = 0
    max_score = 6
    
    # Scoring criteria
    if len(available_metrics) >= len(required_metrics) * 0.6:  # 60% of required metrics
        score += 1
        print(f"   ✅ Metric Coverage: Good ({len(available_metrics)}/{len(required_metrics)} required metrics)")
    else:
        print(f"   ❌ Metric Coverage: Limited ({len(available_metrics)}/{len(required_metrics)} required metrics)")
    
    if len(good_data) >= 15:  # At least 15 columns with good data
        score += 1
        print(f"   ✅ Data Quality: Good ({len(good_data)} columns with <25% missing)")
    else:
        print(f"   ⚠️  Data Quality: Moderate ({len(good_data)} columns with <25% missing)")
    
    if df.shape[0] >= 2000:  # Sufficient sample size
        score += 1
        print(f"   ✅ Sample Size: Adequate ({df.shape[0]} devices)")
    else:
        print(f"   ⚠️  Sample Size: Limited ({df.shape[0]} devices)")
    
    if len(df['Manufacturer'].unique()) >= 3:  # Multiple manufacturers
        score += 1
        print(f"   ✅ Manufacturer Diversity: Good ({len(df['Manufacturer'].unique())} manufacturers)")
    else:
        print(f"   ⚠️  Manufacturer Diversity: Limited ({len(df['Manufacturer'].unique())} manufacturers)")
    
    if 'FP32_Final' in df.columns and df['FP32_Final'].notna().sum() > len(df) * 0.5:
        score += 1
        print(f"   ✅ Core Performance Data: Available (FP32 FLOPS)")
    else:
        print(f"   ❌ Core Performance Data: Limited")
    
    if price_available > len(df) * 0.2:  # At least 20% have price data
        score += 1
        print(f"   ✅ Economic Data: Available ({(price_available/len(df)*100):.1f}% have prices)")
    else:
        print(f"   ❌ Economic Data: Limited ({(price_available/len(df)*100):.1f}% have prices)")
    
    print(f"\n9. OVERALL ASSESSMENT:")
    print(f"   Score: {score}/{max_score}")
    
    if score >= 5:
        assessment = "EXCELLENT - Ready for advanced modeling"
        recommendations = [
            "✅ Dataset is well-suited for your AI benchmarking project",
            "✅ Proceed with comprehensive data modeling and prediction",
            "✅ Focus on bias/weight-based modeling with available metrics",
            "✅ Develop static prediction models for performance KPIs"
        ]
    elif score >= 3:
        assessment = "GOOD - Suitable with some enhancements"
        recommendations = [
            "✅ Dataset provides solid foundation for project",
            "⚠️  Consider data augmentation for missing metrics",
            "✅ Focus on available performance metrics for initial models",
            "⚠️  May need additional data sources for complete coverage"
        ]
    else:
        assessment = "LIMITED - Needs significant enhancement"
        recommendations = [
            "❌ Dataset has limitations for comprehensive AI benchmarking",
            "❌ Significant missing metrics need to be addressed",
            "❌ Consider integrating additional data sources",
            "❌ Focus on subset of available metrics for initial phase"
        ]
    
    print(f"   Status: {assessment}")
    print(f"\n10. RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"    {i}. {rec}")
    
    # Specific suggestions for missing metrics
    print(f"\n11. SUGGESTIONS FOR MISSING METRICS:")
    suggestions = {
        'Latency': 'Could be derived from throughput and batch size data',
        'Throughput': 'Might be estimated from FLOPS and clock speeds',
        'Model Size': 'Not directly available - would need separate neural network model database',
        'Network Density': 'Requires neural network architecture data',
        'Memory Usage %': 'Could be calculated from memory bandwidth and available memory',
        'Compute Usage %': 'Would need additional utilization metrics'
    }
    
    for metric in missing_metrics:
        if metric in suggestions:
            print(f"   • {metric}: {suggestions[metric]}")
        else:
            print(f"   • {metric}: Requires external data source")

if __name__ == "__main__":
    analyze_ai_benchmark_dataset() 