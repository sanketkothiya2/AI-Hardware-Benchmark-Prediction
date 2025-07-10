import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from typing import Dict, List, Any

def show_hardware_comparison():
    """Simplified Hardware Comparison Dashboard"""
    st.markdown("# 🔍 Hardware Comparison")
    st.markdown("**Compare graphics cards from our database or create custom configurations**")
    
    if st.session_state.data is None:
        st.error("❌ Dataset not loaded. Please check the data loading.")
        return
    
    df = st.session_state.data
    models = getattr(st.session_state, 'models', {})
    
    # Only show database comparison
    show_simple_database_comparison(df)

def show_simple_database_comparison(df):
    """Simplified GPU database comparison"""
    st.markdown("### 📊 Compare Existing GPUs")
    st.markdown("Select up to 3 GPUs from our database to compare their specifications and performance")
    
    # Simple filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            manufacturers = ['All'] + sorted([m for m in df['Manufacturer'].unique() if pd.notna(m) and str(m).strip() != ''])
            if not manufacturers or len(manufacturers) == 1:  # Only 'All'
                manufacturers = ['All', 'NVIDIA', 'AMD', 'Intel']  # Fallback
            selected_mfg = st.selectbox("🏭 Manufacturer", manufacturers)
        except Exception:
            selected_mfg = 'All'
    
    with col2:
        try:
            if selected_mfg != 'All':
                vendor_df = df[df['Manufacturer'] == selected_mfg]
            else:
                vendor_df = df
            
            archs = ['All'] + sorted([a for a in vendor_df['Architecture'].unique() if pd.notna(a) and str(a).strip() != '' and a != 'Unknown'])
            if not archs or len(archs) == 1:  # Only 'All'
                archs = ['All', 'Ada Lovelace', 'Ampere', 'RDNA 3', 'RDNA 2']  # Fallback
            selected_arch = st.selectbox("🏗️ Architecture", archs)
        except Exception:
            selected_arch = 'All'
    
    with col3:
        try:
            # Filter by performance tier if available
            if 'PerformanceTier' in df.columns:
                tiers = ['All'] + sorted([t for t in df['PerformanceTier'].unique() if pd.notna(t) and str(t).strip() != ''])
                if not tiers or len(tiers) == 1:  # Only 'All'
                    tiers = ['All', 'High-End', 'Mid-Range', 'Entry-Level']  # Fallback
                selected_tier = st.selectbox("⭐ Performance Tier", tiers)
            else:
                selected_tier = 'All'
        except Exception:
            selected_tier = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    if selected_mfg != 'All':
        filtered_df = filtered_df[filtered_df['Manufacturer'] == selected_mfg]
    if selected_arch != 'All':
        filtered_df = filtered_df[filtered_df['Architecture'] == selected_arch]
    if selected_tier != 'All' and 'PerformanceTier' in df.columns:
        filtered_df = filtered_df[filtered_df['PerformanceTier'] == selected_tier]
    
    st.info(f"📈 Found {len(filtered_df):,} GPUs matching your filters")
    
    # Check if we have any data after filtering
    if filtered_df.empty:
        st.warning("⚠️ No GPUs found matching your current filters. Please try different filter combinations.")
        st.info("💡 Try selecting 'All' for one or more filters to see more results.")
        return
    
    # GPU selection with search
    if not filtered_df.empty:
        # Create searchable GPU list
        gpu_options = []
        for idx, row in filtered_df.iterrows():
            try:
                # Safe data extraction with defaults
                gpu_name = row.get('gpuName', f'GPU_{idx}')
                if pd.isna(gpu_name) or gpu_name == '':
                    gpu_name = f'GPU_{idx}'
                
                perf = row.get('FP32_Final', 0)
                if pd.isna(perf) or perf <= 0:
                    perf = 0
                else:
                    perf = perf / 1e12
                
                eff = row.get('GFLOPS_per_Watt', 0)
                if pd.isna(eff):
                    eff = 0
                
                # Create display name
                display_name = f"{gpu_name} ({perf:.1f} TFLOPS, {eff:.1f} GFLOPS/W)"
                gpu_options.append((display_name, gpu_name))
            except Exception as e:
                # Skip problematic rows
                continue
        
        # Sort by performance (with error handling)
        def safe_sort_key(x):
            try:
                return float(x[0].split('(')[1].split(' ')[0])
            except (ValueError, IndexError):
                return 0.0
        
        gpu_options.sort(key=safe_sort_key, reverse=True)
        
        # Check if we have GPU options
        if not gpu_options:
            st.warning("⚠️ No GPUs found matching your current filters. Try adjusting the filters above.")
            return
        
        # GPU selection
        selected_gpus = st.multiselect(
            "🎯 Select GPUs to compare (choose 2-3 for best results):",
            options=[opt[0] for opt in gpu_options[:50]],  # Limit to top 50 for performance
            default=[],
            max_selections=3,
            help="Start typing to search for specific GPU models"
        )
        
        if len(selected_gpus) >= 2:
            # Get selected GPU data
            selected_gpu_names = []
            for sel in selected_gpus:
                gpu_name = next((opt[1] for opt in gpu_options if opt[0] == sel), None)
                if gpu_name:
                    selected_gpu_names.append(gpu_name)
            
            # Show comparison
            show_gpu_comparison_results(df, selected_gpu_names)
        elif len(selected_gpus) == 1:
            st.info("👆 Select at least one more GPU to start comparison")
        else:
            st.info("👆 Select 2-3 GPUs from the list above to compare them")

def show_gpu_comparison_results(df, gpu_names):
    """Show clean comparison results"""
    st.markdown("---")
    st.markdown("### 🏆 Comparison Results")
    
    # Get GPU data
    gpu_data_list = []
    for gpu_name in gpu_names:
        gpu_data = df[df['gpuName'] == gpu_name]
        if not gpu_data.empty:
            gpu_data_list.append(gpu_data.iloc[0])
    
    if not gpu_data_list:
        st.error("❌ Could not find data for selected GPUs")
        return
    
    # Summary cards
    cols = st.columns(len(gpu_data_list))
    
    for i, gpu_data in enumerate(gpu_data_list):
        with cols[i]:
            # Clean GPU name display
            gpu_name = gpu_data['gpuName']
            if len(gpu_name) > 25:
                gpu_name = gpu_name[:22] + "..."
            
            st.markdown(f"#### 🖥️ {gpu_name}")
            
            # Key metrics with better formatting
            perf = gpu_data.get('FP32_Final', 0) / 1e12 if pd.notna(gpu_data.get('FP32_Final', 0)) and gpu_data.get('FP32_Final', 0) > 0 else 0
            eff = gpu_data.get('GFLOPS_per_Watt', 0) if pd.notna(gpu_data.get('GFLOPS_per_Watt', 0)) else 0
            tdp = gpu_data.get('TDP', 0) if pd.notna(gpu_data.get('TDP', 0)) else 0
            memory = gpu_data.get('Memory_GB', 0) if pd.notna(gpu_data.get('Memory_GB', 0)) else 0
            
            st.metric("🚀 Performance", f"{perf:.1f} TFLOPS")
            st.metric("⚡ Efficiency", f"{eff:.1f} GFLOPS/W")
            st.metric("🔥 Power", f"{tdp:.0f}W")
            st.metric("🧠 Memory", f"{memory:.0f}GB")
            
            # Category if available
            if 'AI_Performance_Category' in gpu_data.index and pd.notna(gpu_data['AI_Performance_Category']):
                category = gpu_data['AI_Performance_Category'].replace('AI_', '')
                st.markdown(f"**Category:** {category}")
    
    # Performance comparison chart
    st.markdown("### 📊 Performance Comparison")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Performance bar chart
        gpu_names_short = [data['gpuName'][:15] + "..." if len(data['gpuName']) > 15 else data['gpuName'] for data in gpu_data_list]
        performances = [data.get('FP32_Final', 0) / 1e12 if pd.notna(data.get('FP32_Final', 0)) and data.get('FP32_Final', 0) > 0 else 0 for data in gpu_data_list]
        
        fig = px.bar(
            x=gpu_names_short,
            y=performances,
            title="🚀 Performance (TFLOPS)",
            color=performances,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, showlegend=False, xaxis_title="", yaxis_title="TFLOPS")
        fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Efficiency comparison
        efficiencies = [data.get('GFLOPS_per_Watt', 0) if pd.notna(data.get('GFLOPS_per_Watt', 0)) else 0 for data in gpu_data_list]
        
        fig = px.bar(
            x=gpu_names_short,
            y=efficiencies,
            title="⚡ Efficiency (GFLOPS/W)",
            color=efficiencies,
            color_continuous_scale='plasma'
        )
        fig.update_layout(height=400, showlegend=False, xaxis_title="", yaxis_title="GFLOPS/W")
        fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed specifications table
    show_detailed_specs_table(gpu_data_list)

def show_detailed_specs_table(gpu_data_list):
    """Show detailed specifications in a clean table"""
    st.markdown("### 📋 Detailed Specifications")
    
    # Key specifications to show
    specs = [
        ('Manufacturer', 'Manufacturer'),
        ('Architecture', 'Architecture'),
        ('Performance (TFLOPS)', 'FP32_Final'),
        ('Efficiency (GFLOPS/W)', 'GFLOPS_per_Watt'),
        ('Power (W)', 'TDP'),
        ('Memory (GB)', 'Memory_GB'),
        ('AI Category', 'AI_Performance_Category'),
        ('Performance Tier', 'PerformanceTier')
    ]
    
    comparison_data = []
    
    for display_name, column_name in specs:
        if column_name in gpu_data_list[0].index:
            row = {'Specification': display_name}
            
            for i, gpu_data in enumerate(gpu_data_list):
                value = gpu_data.get(column_name, 'N/A')
                
                # Format values nicely
                if pd.isna(value) or value == 'N/A':
                    formatted_value = 'N/A'
                elif column_name == 'FP32_Final':
                    formatted_value = f"{value / 1e12:.1f}"
                elif column_name == 'GFLOPS_per_Watt':
                    formatted_value = f"{value:.1f}"
                elif column_name in ['TDP', 'Memory_GB']:
                    formatted_value = f"{value:.0f}"
                elif column_name == 'AI_Performance_Category':
                    formatted_value = str(value).replace('AI_', '') if 'AI_' in str(value) else str(value)
                else:
                    formatted_value = str(value)
                
                row[f'{gpu_data["gpuName"][:20]}...'] = formatted_value
            
            comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, height=350)



 