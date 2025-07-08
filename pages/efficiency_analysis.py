import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show_efficiency_analysis():
    """Display efficiency analysis page"""
    st.markdown("## ⚡ Efficiency Analysis")
    
    if st.session_state.data is None:
        st.error("Dataset not loaded")
        return
    
    df = st.session_state.data
    
    # Check for required columns
    required_cols = ['GFLOPS_per_Watt', 'gpuName', 'Manufacturer', 'TDP']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.info("Available columns: " + ", ".join(df.columns.tolist()))
        return
    
    # Efficiency leaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Efficiency Leaders")
        try:
            # Clean data first
            efficiency_data = df[df['GFLOPS_per_Watt'].notna() & (df['GFLOPS_per_Watt'] > 0)]
            
            if len(efficiency_data) > 0:
                efficiency_leaders = efficiency_data.nlargest(10, 'GFLOPS_per_Watt')[
                    ['gpuName', 'Manufacturer', 'GFLOPS_per_Watt', 'TDP']
                ]
                st.dataframe(efficiency_leaders, use_container_width=True)
            else:
                st.warning("No valid efficiency data available")
        except Exception as e:
            st.error(f"Error displaying efficiency leaders: {e}")
    
    with col2:
        st.markdown("### 📊 Efficiency Distribution")
        try:
            # Clean data for visualization
            plot_data = df[df['GFLOPS_per_Watt'].notna() & (df['GFLOPS_per_Watt'] > 0)]
            
            if len(plot_data) > 0:
                fig = px.histogram(
                    plot_data, 
                    x='GFLOPS_per_Watt', 
                    color='Manufacturer',
                    title="GPU Efficiency Distribution",
                    labels={'GFLOPS_per_Watt': 'GFLOPS per Watt', 'count': 'Number of GPUs'},
                    nbins=30
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid data for efficiency distribution")
        except Exception as e:
            st.error(f"Error creating efficiency chart: {e}")
    
    # Additional efficiency insights
    st.markdown("### 📈 Efficiency Insights")
    
    try:
        if 'GFLOPS_per_Watt' in df.columns and df['GFLOPS_per_Watt'].notna().sum() > 0:
            col1, col2, col3 = st.columns(3)
            
            valid_data = df[df['GFLOPS_per_Watt'].notna() & (df['GFLOPS_per_Watt'] > 0)]
            
            with col1:
                avg_efficiency = valid_data['GFLOPS_per_Watt'].mean()
                st.metric("Average Efficiency", f"{avg_efficiency:.2f} GFLOPS/W")
            
            with col2:
                max_efficiency = valid_data['GFLOPS_per_Watt'].max()
                st.metric("Peak Efficiency", f"{max_efficiency:.2f} GFLOPS/W")
            
            with col3:
                efficient_count = len(valid_data[valid_data['GFLOPS_per_Watt'] > avg_efficiency])
                st.metric("Above Average", f"{efficient_count} GPUs")
            
            # Efficiency by manufacturer
            st.markdown("### 🏭 Efficiency by Manufacturer")
            
            mfg_efficiency = valid_data.groupby('Manufacturer')['GFLOPS_per_Watt'].agg(['mean', 'count']).round(2)
            mfg_efficiency.columns = ['Average Efficiency', 'GPU Count']
            st.dataframe(mfg_efficiency, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in efficiency insights: {e}") 