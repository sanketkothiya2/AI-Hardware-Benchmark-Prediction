import pandas as pd
import numpy as np
import os

def filter_major_manufacturers():
    """
    Filter the comprehensive AI hardware matrix to include only Intel, AMD, and NVIDIA manufacturers.
    """
    
    # Input and output file paths
    input_file = 'data/processed/comprehensive_ai_hardware_matrix.csv'
    output_file = 'data/processed/major_manufacturers_ai_hardware_matrix.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    try:
        # Read the comprehensive dataset
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Total records: {len(df)}")
        
        # Display unique manufacturers before filtering
        print("\nUnique manufacturers in the dataset:")
        unique_manufacturers = df['unified_manufacturer'].value_counts()
        print(unique_manufacturers)
        
        # Filter for Intel, AMD, and NVIDIA only
        major_manufacturers = ['Intel', 'AMD', 'NVIDIA']
        
        # Create a case-insensitive filter
        filtered_df = df[df['unified_manufacturer'].str.upper().isin([m.upper() for m in major_manufacturers])].copy()
        
        print(f"\nFiltered dataset shape: {filtered_df.shape}")
        print(f"Filtered records: {len(filtered_df)}")
        
        # Display manufacturer distribution after filtering
        print("\nManufacturer distribution after filtering:")
        manufacturer_counts = filtered_df['unified_manufacturer'].value_counts()
        print(manufacturer_counts)
        
        # Display some statistics about the filtered data
        print("\n=== FILTERED DATASET STATISTICS ===")
        print(f"Total Intel devices: {len(filtered_df[filtered_df['unified_manufacturer'].str.upper() == 'INTEL'])}")
        print(f"Total AMD devices: {len(filtered_df[filtered_df['unified_manufacturer'].str.upper() == 'AMD'])}")
        print(f"Total NVIDIA devices: {len(filtered_df[filtered_df['unified_manufacturer'].str.upper() == 'NVIDIA'])}")
        
        # Show performance tier distribution
        if 'performance_tier' in filtered_df.columns:
            print("\nPerformance tier distribution:")
            tier_counts = filtered_df['performance_tier'].value_counts()
            print(tier_counts)
        
        # Show device type distribution
        if 'type' in filtered_df.columns:
            print("\nDevice type distribution:")
            type_counts = filtered_df['type'].value_counts()
            print(type_counts)
        
        # Show release year distribution (if release_date exists)
        if 'release_date' in filtered_df.columns:
            filtered_df['release_year'] = pd.to_datetime(filtered_df['release_date'], errors='coerce').dt.year
            print("\nRelease year distribution:")
            year_counts = filtered_df['release_year'].value_counts().sort_index()
            print(year_counts.tail(10))  # Show last 10 years
        
        # Display some key performance metrics summary
        performance_columns = ['fp32_tflops', 'fp16_tflops', 'int8_tops', 'memory_bandwidth_gbps', 'unified_tdp']
        available_perf_cols = [col for col in performance_columns if col in filtered_df.columns]
        
        if available_perf_cols:
            print(f"\n=== PERFORMANCE METRICS SUMMARY ===")
            for col in available_perf_cols:
                non_zero_data = filtered_df[filtered_df[col] > 0][col]
                if len(non_zero_data) > 0:
                    print(f"\n{col.upper()}:")
                    print(f"  Min: {non_zero_data.min():.2f}")
                    print(f"  Max: {non_zero_data.max():.2f}")
                    print(f"  Mean: {non_zero_data.mean():.2f}")
                    print(f"  Records with data: {len(non_zero_data)}/{len(filtered_df)}")
        
        # Save the filtered dataset
        print(f"\nSaving filtered data to {output_file}...")
        filtered_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Successfully created filtered dataset with {len(filtered_df)} records!")
        print(f"📁 Saved to: {output_file}")
        
        # Show sample of filtered data
        print("\n=== SAMPLE OF FILTERED DATA ===")
        sample_cols = ['hardware_name', 'unified_manufacturer', 'type', 'performance_tier', 'release_date']
        available_sample_cols = [col for col in sample_cols if col in filtered_df.columns]
        
        if available_sample_cols:
            print(filtered_df[available_sample_cols].head(10).to_string(index=False))
        
        return filtered_df
        
    except Exception as e:
        print(f"Error processing the data: {str(e)}")
        return None

if __name__ == "__main__":
    print("🔧 Filtering AI Hardware Matrix for Major Manufacturers (Intel, AMD, NVIDIA)")
    print("=" * 80)
    
    result = filter_major_manufacturers()
    
    if result is not None:
        print("\n" + "=" * 80)
        print("✅ FILTERING COMPLETED SUCCESSFULLY!")
        print("The filtered dataset contains only Intel, AMD, and NVIDIA hardware.")
    else:
        print("\n" + "=" * 80)
        print("❌ FILTERING FAILED!") 