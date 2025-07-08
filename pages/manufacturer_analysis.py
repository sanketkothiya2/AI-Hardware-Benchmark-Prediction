import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show_manufacturer_analysis():
    """Display manufacturer analysis page"""
    st.markdown("## 🏭 Manufacturer Analysis")
    
    if st.session_state.data is None:
        st.error("Dataset not loaded")
        return
    
    df = st.session_state.data
    
    # Check for required columns
    required_cols = ['Manufacturer', 'FP32_Final']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.info("Available columns: " + ", ".join(df.columns.tolist()))
        return
    
    # Market share
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Market Share")
        try:
            market_share = df['Manufacturer'].value_counts()
            
            if len(market_share) > 0:
                fig = px.pie(
                    values=market_share.values, 
                    names=market_share.index,
                    title="GPU Market Share by Manufacturer"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display market share table
                share_df = pd.DataFrame({
                    'Manufacturer': market_share.index,
                    'GPU Count': market_share.values,
                    'Market Share %': (market_share.values / market_share.sum() * 100).round(1)
                })
                st.dataframe(share_df, use_container_width=True)
            else:
                st.warning("No manufacturer data available")
        except Exception as e:
            st.error(f"Error creating market share chart: {e}")
    
    with col2:
        st.markdown("### 📈 Performance by Manufacturer")
        try:
            # Clean performance data
            perf_data = df[df['FP32_Final'].notna() & (df['FP32_Final'] > 0)]
            
            if len(perf_data) > 0:
                fig = px.box(
                    perf_data, 
                    x='Manufacturer', 
                    y='FP32_Final',
                    title="Performance Distribution by Manufacturer",
                    labels={'FP32_Final': 'FP32 Performance (GFLOPS)'}
                )
                fig.update_layout(height=400)
                fig.update_yaxis(type="log")  # Log scale for better visualization
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid performance data available")
        except Exception as e:
            st.error(f"Error creating performance chart: {e}")
    
    # Detailed manufacturer statistics
    st.markdown("### 📊 Manufacturer Statistics")
    
    try:
        # Performance statistics by manufacturer
        valid_data = df[df['FP32_Final'].notna() & (df['FP32_Final'] > 0)]
        
        if len(valid_data) > 0:
            mfg_stats = valid_data.groupby('Manufacturer')['FP32_Final'].agg([
                'count', 'mean', 'median', 'min', 'max', 'std'
            ]).round(2)
            
            mfg_stats.columns = ['GPU Count', 'Mean Performance', 'Median Performance', 
                                'Min Performance', 'Max Performance', 'Std Dev']
            
            st.dataframe(mfg_stats, use_container_width=True)
            
            # Architecture diversity by manufacturer
            if 'Architecture' in df.columns:
                st.markdown("### 🏗️ Architecture Diversity")
                
                arch_diversity = df.groupby('Manufacturer')['Architecture'].agg(['nunique', 'count']).reset_index()
                arch_diversity.columns = ['Manufacturer', 'Unique Architectures', 'Total GPUs']
                arch_diversity['Architectures per GPU'] = (arch_diversity['Unique Architectures'] / 
                                                          arch_diversity['Total GPUs']).round(3)
                
                st.dataframe(arch_diversity, use_container_width=True)
                
                # Architecture breakdown chart
                arch_breakdown = df.groupby(['Manufacturer', 'Architecture']).size().reset_index(name='Count')
                
                fig = px.sunburst(
                    arch_breakdown,
                    path=['Manufacturer', 'Architecture'],
                    values='Count',
                    title="Architecture Breakdown by Manufacturer"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in manufacturer statistics: {e}")
    
    # Performance trends over time (if date information available)
    try:
        if 'testDate' in df.columns:
            st.markdown("### 📅 Performance Trends Over Time")
            
            date_data = df[df['testDate'].notna() & df['FP32_Final'].notna()]
            
            if len(date_data) > 0:
                date_data['Year'] = pd.to_datetime(date_data['testDate'], errors='coerce').dt.year
                yearly_performance = date_data.groupby(['Year', 'Manufacturer'])['FP32_Final'].mean().reset_index()
                
                fig = px.line(
                    yearly_performance,
                    x='Year',
                    y='FP32_Final',
                    color='Manufacturer',
                    title="Average Performance Trends by Manufacturer",
                    labels={'FP32_Final': 'Average FP32 Performance (GFLOPS)'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.info("Date information not available for trend analysis") 