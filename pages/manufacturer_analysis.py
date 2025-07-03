import streamlit as st
import pandas as pd
import plotly.express as px

def show_manufacturer_analysis():
    """Display manufacturer analysis page"""
    st.markdown("## 🏭 Manufacturer Analysis")
    
    if st.session_state.data is None:
        st.error("Dataset not loaded")
        return
    
    df = st.session_state.data
    
    # Market share
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Market Share")
        market_share = df['Manufacturer'].value_counts()
        fig = px.pie(values=market_share.values, names=market_share.index)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Performance by Manufacturer")
        fig = px.box(df, x='Manufacturer', y='FP32_Final')
        st.plotly_chart(fig, use_container_width=True) 