import streamlit as st
import pandas as pd
import plotly.express as px

def show_model_performance():
    """Display model performance analysis"""
    st.markdown("## 🎯 Model Performance Analysis")
    
    # Model accuracy metrics
    models_data = {
        'Model': ['Random Forest (Performance)', 'XGBoost (Performance)', 'Random Forest (Efficiency)'],
        'Accuracy': [0.925, 0.912, 0.889],
        'RMSE': [1247.3, 1389.2, 0.156]
    }
    
    models_df = pd.DataFrame(models_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Model Accuracy")
        st.dataframe(models_df)
    
    with col2:
        st.markdown("### 📈 Performance Comparison")
        fig = px.bar(models_df, x='Model', y='Accuracy')
        st.plotly_chart(fig, use_container_width=True) 