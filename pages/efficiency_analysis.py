import streamlit as st
import pandas as pd
import plotly.express as px

def show_efficiency_analysis():
    """Display efficiency analysis page"""
    st.markdown("## ⚡ Efficiency Analysis")
    
    if st.session_state.data is None:
        st.error("Dataset not loaded")
        return
    
    df = st.session_state.data
    
    # Efficiency leaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Efficiency Leaders")
        efficiency_leaders = df.nlargest(10, 'GFLOPS_per_Watt')[
            ['gpuName', 'Manufacturer', 'GFLOPS_per_Watt', 'TDP']
        ]
        st.dataframe(efficiency_leaders)
    
    with col2:
        st.markdown("### 📊 Efficiency Distribution")
        fig = px.histogram(df, x='GFLOPS_per_Watt', color='Manufacturer')
        st.plotly_chart(fig, use_container_width=True) 