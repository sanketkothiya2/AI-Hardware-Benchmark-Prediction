import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def show_visualizations():
    """Display advanced visualizations page"""
    st.markdown("## 📈 Advanced Visualizations")
    
    if st.session_state.data is None:
        st.error("Dataset not loaded")
        return
    
    df = st.session_state.data
    
    # Main content area controls
    st.markdown("### 🎛️ Select Visualization Type")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        viz_type = st.selectbox(
            "Choose Visualization:",
            [
                "🔗 Correlation Analysis",
                "📊 Performance Distributions", 
                "🎯 Scatter Plot Matrix",
                "🏗️ Architecture Comparison",
                "⚡ Performance vs Efficiency",
                "💰 Price-Performance Analysis",
                "🏭 Manufacturer Trends",
                "🌡️ Technology Evolution",
                "🎮 Gaming vs AI Performance",
                "🔥 Advanced Heatmaps",
                "📈 Performance Regression",
                "🎨 3D Performance Space",
                "⚙️ Engineering Insights",
                "🚀 Benchmark Comparison"
            ]
        )
    
    with col2:
        # Description of selected visualization
        descriptions = {
            "🔗 Correlation Analysis": "Analyze relationships between different GPU performance metrics",
            "📊 Performance Distributions": "Explore statistical distributions of performance metrics",
            "🎯 Scatter Plot Matrix": "Multi-dimensional scatter plot analysis",
            "🏗️ Architecture Comparison": "Compare performance across different GPU architectures",
            "⚡ Performance vs Efficiency": "Analyze performance-per-watt relationships",
            "💰 Price-Performance Analysis": "Value analysis and price-performance ratios",
            "🏭 Manufacturer Trends": "Market share and manufacturer-specific insights",
            "🌡️ Technology Evolution": "Process node trends and technology advancement",
            "🎮 Gaming vs AI Performance": "Compare gaming and AI workload performance",
            "🔥 Advanced Heatmaps": "Performance matrices and efficiency heatmaps",
            "📈 Performance Regression": "Statistical regression analysis with ML insights",
            "🎨 3D Performance Space": "Interactive 3D visualizations with clustering",
            "⚙️ Engineering Insights": "Thermal design and architectural analysis",
            "🚀 Benchmark Comparison": "Multi-benchmark radar charts and correlation analysis"
        }
        st.info(f"**{viz_type}**: {descriptions.get(viz_type, 'Advanced visualization analysis')}")
    
    st.markdown("---")
    
    if viz_type == "🔗 Correlation Analysis":
        show_correlation_analysis(df)
    elif viz_type == "📊 Performance Distributions":
        show_distribution_analysis(df)
    elif viz_type == "🎯 Scatter Plot Matrix":
        show_scatter_matrix(df)
    elif viz_type == "🏗️ Architecture Comparison":
        show_architecture_comparison(df)
    elif viz_type == "⚡ Performance vs Efficiency":
        show_performance_efficiency(df)
    elif viz_type == "💰 Price-Performance Analysis":
        show_price_performance(df)
    elif viz_type == "🏭 Manufacturer Trends":
        show_manufacturer_trends(df)
    elif viz_type == "🌡️ Technology Evolution":
        show_technology_evolution(df)
    elif viz_type == "🎮 Gaming vs AI Performance":
        show_gaming_vs_ai_performance(df)
    elif viz_type == "🔥 Advanced Heatmaps":
        show_advanced_heatmaps(df)
    elif viz_type == "📈 Performance Regression":
        show_performance_regression(df)
    elif viz_type == "🎨 3D Performance Space":
        show_3d_performance_space(df)
    elif viz_type == "⚙️ Engineering Insights":
        show_engineering_insights(df)
    elif viz_type == "🚀 Benchmark Comparison":
        show_benchmark_comparison(df)

def show_correlation_analysis(df):
    """Display correlation analysis"""
    st.markdown("### 🔗 Correlation Analysis")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns with too many NaN values
    valid_cols = [col for col in numeric_cols if df[col].notna().sum() > len(df) * 0.5]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Select Features")
        # Only include default values that exist in valid_cols
        potential_defaults = ['FP32_Final', 'TDP', 'GFLOPS_per_Watt', 'price', 'G3Dmark']
        default_values = [col for col in potential_defaults if col in valid_cols][:5]
        
        selected_features = st.multiselect(
            "Choose features for correlation",
            valid_cols,
            default=default_values
        )
        
        correlation_method = st.selectbox(
            "Correlation Method",
            ["pearson", "spearman", "kendall"]
        )
    
    with col2:
        if len(selected_features) >= 2:
            # Calculate correlation matrix
            corr_matrix = df[selected_features].corr(method=correlation_method)
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title=f"{correlation_method.title()} Correlation Matrix"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation insights
            st.markdown("#### 🔍 Key Correlations")
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df.head(10))

def show_distribution_analysis(df):
    """Display distribution analysis"""
    st.markdown("### 📊 Performance Distributions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Feature selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = ['FP32_Final', 'TDP', 'GFLOPS_per_Watt', 'price']
        # Filter to only include existing columns
        available_features = [col for col in available_features if col in df.columns] + numeric_cols
        # Remove duplicates while preserving order
        available_features = list(dict.fromkeys(available_features))
        
        selected_feature = st.selectbox(
            "Select Feature",
            available_features
        )
        
        # Grouping option
        group_by = st.selectbox(
            "Group By",
            ["None", "Manufacturer", "Architecture", "AI_Performance_Category", "PerformanceTier"]
        )
        
        # Distribution type
        dist_type = st.selectbox(
            "Distribution Type",
            ["Histogram", "Box Plot", "Violin Plot", "Density Plot"]
        )
    
    with col2:
        if selected_feature in df.columns:
            if dist_type == "Histogram":
                if group_by != "None":
                    fig = px.histogram(
                        df, 
                        x=selected_feature, 
                        color=group_by,
                        title=f"{selected_feature} Distribution by {group_by}",
                        marginal="box"
                    )
                else:
                    fig = px.histogram(
                        df, 
                        x=selected_feature,
                        title=f"{selected_feature} Distribution",
                        marginal="box"
                    )
            
            elif dist_type == "Box Plot":
                if group_by != "None":
                    fig = px.box(
                        df, 
                        x=group_by, 
                        y=selected_feature,
                        title=f"{selected_feature} by {group_by}"
                    )
                else:
                    fig = px.box(
                        df, 
                        y=selected_feature,
                        title=f"{selected_feature} Box Plot"
                    )
            
            elif dist_type == "Violin Plot":
                if group_by != "None":
                    fig = px.violin(
                        df, 
                        x=group_by, 
                        y=selected_feature,
                        title=f"{selected_feature} by {group_by}"
                    )
                else:
                    fig = px.violin(
                        df, 
                        y=selected_feature,
                        title=f"{selected_feature} Violin Plot"
                    )
            
            else:  # Density Plot
                if group_by != "None":
                    fig = px.histogram(
                        df, 
                        x=selected_feature, 
                        color=group_by,
                        marginal="rug",
                        histnorm="density",
                        title=f"{selected_feature} Density by {group_by}"
                    )
                else:
                    fig = px.histogram(
                        df, 
                        x=selected_feature,
                        marginal="rug", 
                        histnorm="density",
                        title=f"{selected_feature} Density"
                    )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.markdown("#### 📈 Statistical Summary")
            if group_by != "None":
                summary = df.groupby(group_by)[selected_feature].describe()
                st.dataframe(summary)
            else:
                summary = df[selected_feature].describe()
                st.write(summary)

def show_scatter_matrix(df):
    """Display scatter plot matrix"""
    st.markdown("### 🎯 Scatter Plot Matrix")
    
    # Feature selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    valid_cols = [col for col in numeric_cols if df[col].notna().sum() > len(df) * 0.7]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Only include default values that exist in valid_cols
        potential_defaults = ['FP32_Final', 'TDP', 'GFLOPS_per_Watt', 'price']
        default_scatter = [col for col in potential_defaults if col in valid_cols][:4]
        
        selected_features = st.multiselect(
            "Select Features (max 5)",
            valid_cols,
            default=default_scatter
        )
        
        color_by = st.selectbox(
            "Color By",
            ["None", "Manufacturer", "Architecture", "AI_Performance_Category"]
        )
    
    with col2:
        if len(selected_features) >= 2:
            if len(selected_features) > 5:
                st.warning("Please select maximum 5 features for better visualization")
                selected_features = selected_features[:5]
            
            # Create scatter matrix
            if color_by != "None":
                fig = px.scatter_matrix(
                    df,
                    dimensions=selected_features,
                    color=color_by,
                    title="Scatter Plot Matrix"
                )
            else:
                fig = px.scatter_matrix(
                    df,
                    dimensions=selected_features,
                    title="Scatter Plot Matrix"
                )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

def show_architecture_comparison(df):
    """Show architecture comparison"""
    st.markdown("### 🏗️ Architecture Comparison")
    
    # Performance metrics by architecture
    arch_performance = df.groupby('Architecture').agg({
        'FP32_Final': ['mean', 'std', 'count'],
        'TDP': ['mean', 'std'],
        'GFLOPS_per_Watt': ['mean', 'std'],
        'price': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    arch_performance.columns = ['_'.join(col).strip() for col in arch_performance.columns]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Performance by Architecture")
        st.dataframe(arch_performance)
    
    with col2:
        # Radar chart for top architectures
        top_archs = df['Architecture'].value_counts().head(5).index
        arch_means = df[df['Architecture'].isin(top_archs)].groupby('Architecture')[
            ['FP32_Final', 'GFLOPS_per_Watt', 'TDP']
        ].mean()
        
        # Normalize for radar chart
        arch_norm = arch_means.div(arch_means.max())
        
        fig = go.Figure()
        
        for arch in arch_norm.index:
            fig.add_trace(go.Scatterpolar(
                r=arch_norm.loc[arch].values,
                theta=arch_norm.columns,
                fill='toself',
                name=arch
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Architecture Performance Radar"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_performance_efficiency(df):
    """Show performance vs efficiency analysis"""
    st.markdown("### ⚡ Performance vs Efficiency")
    
    # Create performance efficiency scatter
    fig = px.scatter(
        df,
        x='FP32_Final',
        y='GFLOPS_per_Watt',
        color='Manufacturer',
        size='TDP',
        hover_data=['gpuName', 'price'],
        title="Performance vs Efficiency",
        labels={
            'FP32_Final': 'Performance (FP32 GFLOPS)',
            'GFLOPS_per_Watt': 'Efficiency (GFLOPS/Watt)'
        }
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Efficiency leaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Efficiency Leaders")
        efficiency_leaders = df.nlargest(10, 'GFLOPS_per_Watt')[
            ['gpuName', 'Manufacturer', 'GFLOPS_per_Watt', 'FP32_Final']
        ]
        st.dataframe(efficiency_leaders)
    
    with col2:
        st.markdown("#### 🚀 Performance Leaders")
        performance_leaders = df.nlargest(10, 'FP32_Final')[
            ['gpuName', 'Manufacturer', 'FP32_Final', 'GFLOPS_per_Watt']
        ]
        st.dataframe(performance_leaders)

def show_price_performance(df):
    """Show price-performance analysis"""
    st.markdown("### 💰 Price-Performance Analysis")
    
    # Filter out entries without price
    df_priced = df.dropna(subset=['price'])
    
    if len(df_priced) == 0:
        st.warning("No price data available")
        return
    
    # Calculate price/performance ratio
    df_priced['price_per_gflops'] = df_priced['price'] / df_priced['FP32_Final']
    
    # Price vs performance scatter
    fig = px.scatter(
        df_priced,
        x='price',
        y='FP32_Final',
        color='Manufacturer',
        size='GFLOPS_per_Watt',
        hover_data=['gpuName'],
        title="Price vs Performance",
        labels={
            'price': 'Price (USD)',
            'FP32_Final': 'Performance (FP32 GFLOPS)'
        }
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Best value analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Best Value GPUs")
        best_value = df_priced.nsmallest(10, 'price_per_gflops')[
            ['gpuName', 'Manufacturer', 'price', 'FP32_Final', 'price_per_gflops']
        ]
        st.dataframe(best_value)
    
    with col2:
        st.markdown("#### 💎 Premium GPUs")
        premium_gpus = df_priced.nlargest(10, 'price')[
            ['gpuName', 'Manufacturer', 'price', 'FP32_Final', 'price_per_gflops']
        ]
        st.dataframe(premium_gpus)

def show_manufacturer_trends(df):
    """Show manufacturer trend analysis"""
    st.markdown("### 🏭 Manufacturer Trends")
    
    # Performance by manufacturer over time
    if 'testDate' in df.columns:
        df['year'] = pd.to_datetime(df['testDate'], errors='coerce').dt.year
        
        yearly_performance = df.groupby(['year', 'Manufacturer'])['FP32_Final'].mean().reset_index()
        
        fig = px.line(
            yearly_performance,
            x='year',
            y='FP32_Final',
            color='Manufacturer',
            title="Performance Trends by Manufacturer"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Market share analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Market Share")
        market_share = df['Manufacturer'].value_counts()
        
        fig = px.pie(
            values=market_share.values,
            names=market_share.index,
            title="GPU Database Distribution by Manufacturer"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Performance Distribution")
        
        fig = px.box(
            df,
            x='Manufacturer',
            y='FP32_Final',
            title="Performance Distribution by Manufacturer"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_technology_evolution(df):
    """Show technology evolution trends"""
    st.markdown("### 🌡️ Technology Evolution")
    
    # Process node evolution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔬 Process Node Evolution")
        if 'ProcessSize_nm' in df.columns:
            process_perf = df.groupby('ProcessSize_nm')['FP32_Final'].agg(['mean', 'count']).reset_index()
            process_perf = process_perf[process_perf['count'] >= 5]  # Filter for significance
            
            fig = px.scatter(
                process_perf,
                x='ProcessSize_nm',
                y='mean',
                size='count',
                title="Performance vs Process Node",
                labels={
                    'ProcessSize_nm': 'Process Node (nm)',
                    'mean': 'Average Performance (GFLOPS)',
                    'count': 'Number of GPUs'
                }
            )
            fig.update_xaxis(range=[0, 30])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Process size data not available")
    
    with col2:
        st.markdown("#### ⚡ Efficiency Evolution")
        if 'ProcessSize_nm' in df.columns and 'GFLOPS_per_Watt' in df.columns:
            eff_evolution = df.groupby('ProcessSize_nm')['GFLOPS_per_Watt'].mean().reset_index()
            
            fig = px.line(
                eff_evolution,
                x='ProcessSize_nm',
                y='GFLOPS_per_Watt',
                title="Power Efficiency vs Process Node",
                markers=True
            )
            fig.update_layout(xaxis_title="Process Node (nm)", yaxis_title="GFLOPS per Watt")
            st.plotly_chart(fig, use_container_width=True)
    
    # Memory evolution
    st.markdown("#### 💾 Memory Technology Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Memory_GB' in df.columns:
            memory_trends = df.groupby(['Manufacturer', 'Memory_GB']).size().reset_index(name='count')
            
            fig = px.sunburst(
                memory_trends,
                path=['Manufacturer', 'Memory_GB'],
                values='count',
                title="Memory Capacity Distribution by Manufacturer"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'MemoryBandwidth' in df.columns:
            bandwidth_dist = df.groupby('Manufacturer')['MemoryBandwidth'].mean().reset_index()
            
            fig = px.bar(
                bandwidth_dist,
                x='Manufacturer',
                y='MemoryBandwidth',
                title="Average Memory Bandwidth by Manufacturer",
                color='MemoryBandwidth',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

def show_gaming_vs_ai_performance(df):
    """Compare gaming vs AI performance characteristics"""
    st.markdown("### 🎮 Gaming vs AI Performance")
    
    # Create gaming and AI performance indicators
    gaming_cols = [col for col in df.columns if any(x in col.lower() for x in ['3dmark', 'fps', 'gaming', 'dx12', 'vulkan'])]
    ai_cols = [col for col in df.columns if any(x in col.lower() for x in ['ai', 'tensor', 'fp16', 'int8', 'throughput'])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎮 Gaming Performance Indicators")
        if gaming_cols:
            gaming_metric = st.selectbox("Select Gaming Metric", gaming_cols)
            
            if gaming_metric in df.columns:
                fig = px.scatter(
                    df,
                    x='FP32_Final',
                    y=gaming_metric,
                    color='Manufacturer',
                    title=f"FP32 Performance vs {gaming_metric}",
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Gaming-specific metrics not found in dataset")
    
    with col2:
        st.markdown("#### 🤖 AI Performance Indicators")
        if ai_cols:
            ai_metric = st.selectbox("Select AI Metric", ai_cols)
            
            if ai_metric in df.columns:
                fig = px.scatter(
                    df,
                    x='FP32_Final',
                    y=ai_metric,
                    color='Manufacturer',
                    title=f"FP32 Performance vs {ai_metric}",
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("AI-specific metrics not found in dataset")
    
    # Performance ratio analysis
    st.markdown("#### ⚖️ Gaming vs AI Performance Balance")
    
    # Create synthetic gaming/AI scores if specific metrics aren't available
    if 'G3Dmark' in df.columns and 'FP32_Final' in df.columns:
        df_temp = df.copy()
        df_temp['Gaming_Score'] = df_temp['G3Dmark'] / df_temp['G3Dmark'].max() * 100
        df_temp['AI_Score'] = df_temp['FP32_Final'] / df_temp['FP32_Final'].max() * 100
        df_temp['Balance_Ratio'] = df_temp['Gaming_Score'] / df_temp['AI_Score']
        
        fig = px.scatter(
            df_temp,
            x='Gaming_Score',
            y='AI_Score',
            color='Manufacturer',
            size='TDP',
            hover_data=['gpuName'],
            title="Gaming vs AI Performance Balance",
            labels={'Gaming_Score': 'Gaming Score (Normalized)', 'AI_Score': 'AI Score (Normalized)'}
        )
        
        # Add diagonal line for perfect balance
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=100, y1=100,
            line=dict(color="red", width=2, dash="dash"),
            name="Perfect Balance"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data for gaming vs AI balance analysis")

def show_advanced_heatmaps(df):
    """Show advanced heatmap visualizations"""
    st.markdown("### 🔥 Advanced Heatmaps")
    
    # Performance matrix heatmap
    st.markdown("#### 🎯 Performance Matrix by Architecture and Manufacturer")
    
    if 'Architecture' in df.columns and 'Manufacturer' in df.columns:
        # Create performance matrix
        performance_matrix = df.pivot_table(
            values='FP32_Final',
            index='Architecture',
            columns='Manufacturer',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            performance_matrix,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="RdYlBu_r",
            title="Average Performance by Architecture and Manufacturer (GFLOPS)"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Efficiency heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ Efficiency Heatmap")
        if 'GFLOPS_per_Watt' in df.columns:
            efficiency_matrix = df.pivot_table(
                values='GFLOPS_per_Watt',
                index='Architecture',
                columns='Manufacturer',
                aggfunc='mean'
            )
            
            fig = px.imshow(
                efficiency_matrix,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale="Greens",
                title="Power Efficiency (GFLOPS/W)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 💰 Price Efficiency Heatmap")
        if 'price' in df.columns:
            df_priced = df.dropna(subset=['price'])
            df_priced['price_efficiency'] = df_priced['FP32_Final'] / df_priced['price']
            
            price_eff_matrix = df_priced.pivot_table(
                values='price_efficiency',
                index='Architecture',
                columns='Manufacturer',
                aggfunc='mean'
            )
            
            fig = px.imshow(
                price_eff_matrix,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale="Blues",
                title="Price Efficiency (GFLOPS/$)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap with clustering
    st.markdown("#### 🧬 Advanced Correlation Analysis with Clustering")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    valid_cols = [col for col in numeric_cols if df[col].notna().sum() > len(df) * 0.5]
    
    if len(valid_cols) > 3:
        corr_matrix = df[valid_cols].corr()
        
        # Use seaborn with matplotlib for advanced clustering
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, annot=True, fmt='.2f', cbar_kws={"shrink": .5})
        
        plt.title("Correlation Matrix with Hierarchical Clustering")
        st.pyplot(plt)
        plt.clf()

def show_performance_regression(df):
    """Show performance regression analysis"""
    st.markdown("### 📈 Performance Regression Analysis")
    
    # Feature selection for regression
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎛️ Regression Controls")
        
        # Target variable
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_var = st.selectbox(
            "Target Variable",
            ['FP32_Final', 'GFLOPS_per_Watt', 'TDP', 'price'],
            index=0
        )
        
        # Feature variables
        available_features = [col for col in numeric_cols if col != target_var and df[col].notna().sum() > len(df) * 0.5]
        selected_features = st.multiselect(
            "Predictor Variables",
            available_features,
            default=available_features[:3] if len(available_features) >= 3 else available_features
        )
        
        # Regression type
        reg_type = st.selectbox(
            "Regression Type",
            ["Linear", "Polynomial (degree 2)", "Logarithmic"]
        )
    
    with col2:
        if len(selected_features) >= 1 and target_var in df.columns:
            # Simple regression with first feature
            feature = selected_features[0]
            
            if reg_type == "Linear":
                fig = px.scatter(
                    df,
                    x=feature,
                    y=target_var,
                    color='Manufacturer',
                    trendline="ols",
                    title=f"{target_var} vs {feature} (Linear Regression)"
                )
            elif reg_type == "Polynomial (degree 2)":
                fig = px.scatter(
                    df,
                    x=feature,
                    y=target_var,
                    color='Manufacturer',
                    trendline="ols",
                    trendline_options=dict(log_x=False),
                    title=f"{target_var} vs {feature} (Polynomial Regression)"
                )
            else:  # Logarithmic
                fig = px.scatter(
                    df,
                    x=feature,
                    y=target_var,
                    color='Manufacturer',
                    trendline="ols",
                    title=f"{target_var} vs {feature} (Log Scale)",
                    log_x=True
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Multi-variable regression visualization
            if len(selected_features) >= 2:
                st.markdown("#### 🔮 Multi-Variable Analysis")
                
                feature_x = selected_features[0]
                feature_y = selected_features[1]
                
                fig = px.scatter_3d(
                    df,
                    x=feature_x,
                    y=feature_y,
                    z=target_var,
                    color='Manufacturer',
                    title=f"3D Regression: {target_var} vs {feature_x} & {feature_y}"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    # Regression statistics
    if len(selected_features) >= 1 and target_var in df.columns:
        st.markdown("#### 📊 Regression Statistics")
        
        feature = selected_features[0]
        
        # Remove NaN values
        valid_data = df[[target_var, feature]].dropna()
        
        if len(valid_data) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_data[feature], valid_data[target_var]
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{r_value**2:.3f}")
            with col2:
                st.metric("P-value", f"{p_value:.3e}")
            with col3:
                st.metric("Slope", f"{slope:.3f}")
            with col4:
                st.metric("Std Error", f"{std_err:.3f}")

def show_3d_performance_space(df):
    """Show 3D performance space visualization"""
    st.markdown("### 🎨 3D Performance Space")
    
    # 3D Controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎛️ 3D Controls")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        valid_cols = [col for col in numeric_cols if df[col].notna().sum() > len(df) * 0.7]
        
        x_axis = st.selectbox("X-Axis", valid_cols, index=0)
        y_axis = st.selectbox("Y-Axis", valid_cols, index=1 if len(valid_cols) > 1 else 0)
        z_axis = st.selectbox("Z-Axis", valid_cols, index=2 if len(valid_cols) > 2 else 0)
        
        color_by = st.selectbox(
            "Color By",
            ["Manufacturer", "Architecture", "AI_Performance_Category", "PerformanceTier"]
        )
        
        size_by = st.selectbox(
            "Size By",
            ["None"] + valid_cols
        )
    
    with col2:
        # Create 3D scatter plot
        scatter_kwargs = {
            'data_frame': df,
            'x': x_axis,
            'y': y_axis,
            'z': z_axis,
            'color': color_by,
            'title': f"3D Performance Space: {x_axis} vs {y_axis} vs {z_axis}",
            'hover_data': ['gpuName'] if 'gpuName' in df.columns else None
        }
        
        if size_by != "None":
            scatter_kwargs['size'] = size_by
        
        fig = px.scatter_3d(**scatter_kwargs)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance clusters
    st.markdown("#### 🎯 Performance Clusters")
    
    # K-means clustering
    if len(valid_cols) >= 3:
        cluster_features = [x_axis, y_axis, z_axis]
        cluster_data = df[cluster_features].dropna()
        
        if len(cluster_data) > 10:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            n_clusters = st.slider("Number of Clusters", 2, 8, 4)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            cluster_data['Cluster'] = clusters
            
            fig = px.scatter_3d(
                cluster_data,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color='Cluster',
                title=f"Performance Clusters (K={n_clusters})",
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def show_engineering_insights(df):
    """Show engineering and technical insights"""
    st.markdown("### ⚙️ Engineering Insights")
    
    # Thermal design insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌡️ Thermal Design Analysis")
        if 'TDP' in df.columns and 'FP32_Final' in df.columns:
            df_thermal = df.copy()
            df_thermal['Performance_per_Watt'] = df_thermal['FP32_Final'] / df_thermal['TDP']
            
            fig = px.scatter(
                df_thermal,
                x='TDP',
                y='FP32_Final',
                color='Performance_per_Watt',
                size='Performance_per_Watt',
                hover_data=['gpuName'],
                title="Thermal Design vs Performance",
                color_continuous_scale='RdYlBu_r'
            )
            
            # Add efficiency contours
            fig.add_shape(
                type="line",
                x0=100, y0=5000, x1=400, y1=20000,
                line=dict(color="green", width=2, dash="dash"),
                name="High Efficiency Zone"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔧 Architectural Efficiency")
        if 'Architecture' in df.columns:
            arch_efficiency = df.groupby('Architecture').agg({
                'FP32_Final': 'mean',
                'TDP': 'mean',
                'GFLOPS_per_Watt': 'mean'
            }).round(2)
            
            # Calculate efficiency score
            arch_efficiency['Efficiency_Score'] = (
                arch_efficiency['GFLOPS_per_Watt'] / arch_efficiency['GFLOPS_per_Watt'].max() * 100
            )
            
            fig = px.bar(
                arch_efficiency.reset_index(),
                x='Architecture',
                y='Efficiency_Score',
                title="Architectural Efficiency Score",
                color='Efficiency_Score',
                color_continuous_scale='viridis'
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Memory subsystem analysis
    st.markdown("#### 💾 Memory Subsystem Analysis")
    
    if 'MemoryBandwidth' in df.columns and 'Memory_GB' in df.columns:
        fig = px.scatter(
            df,
            x='Memory_GB',
            y='MemoryBandwidth',
            color='Manufacturer',
            size='FP32_Final',
            title="Memory Capacity vs Bandwidth",
            hover_data=['gpuName']
        )
        
        # Add bandwidth efficiency lines
        fig.add_shape(
            type="line",
            x0=4, y0=200, x1=48, y1=2400,
            line=dict(color="red", width=2, dash="dash"),
            name="50 GB/s per GB"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance scaling analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Performance Scaling Laws")
        if 'ProcessSize_nm' in df.columns:
            scaling_analysis = df.groupby('ProcessSize_nm').agg({
                'FP32_Final': ['mean', 'std', 'count'],
                'TDP': 'mean',
                'GFLOPS_per_Watt': 'mean'
            }).round(2)
            
            scaling_analysis.columns = ['Perf_Mean', 'Perf_Std', 'Count', 'TDP_Mean', 'Efficiency_Mean']
            scaling_analysis = scaling_analysis[scaling_analysis['Count'] >= 3]
            
            fig = px.line(
                scaling_analysis.reset_index(),
                x='ProcessSize_nm',
                y='Efficiency_Mean',
                title="Efficiency vs Process Node",
                markers=True
            )
            fig.update_layout(xaxis_title="Process Node (nm)", yaxis_title="GFLOPS/Watt")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### ⚡ Power Efficiency Trends")
        if 'TDP' in df.columns and 'FP32_Final' in df.columns:
            # Create TDP bins
            df_power = df.copy()
            df_power['TDP_Bin'] = pd.cut(df_power['TDP'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            power_trends = df_power.groupby(['TDP_Bin', 'Manufacturer'])['GFLOPS_per_Watt'].mean().reset_index()
            
            fig = px.bar(
                power_trends,
                x='TDP_Bin',
                y='GFLOPS_per_Watt',
                color='Manufacturer',
                title="Efficiency by Power Class",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

def show_benchmark_comparison(df):
    """Show comprehensive benchmark comparison"""
    st.markdown("### 🚀 Benchmark Comparison")
    
    # Multi-benchmark radar chart
    st.markdown("#### 🎯 Multi-Benchmark Performance Radar")
    
    # Find all benchmark columns
    benchmark_cols = [col for col in df.columns if any(x in col.lower() for x in 
                     ['fps', 'score', 'mark', 'bench', 'test', 'throughput'])]
    
    if len(benchmark_cols) >= 3:
        # Select top GPUs for comparison
        top_gpus = df.nlargest(8, 'FP32_Final')['gpuName'].tolist()
        
        selected_gpus = st.multiselect(
            "Select GPUs for Comparison",
            top_gpus,
            default=top_gpus[:4] if len(top_gpus) >= 4 else top_gpus
        )
        
        if selected_gpus:
            comparison_data = df[df['gpuName'].isin(selected_gpus)]
            
            # Normalize benchmark scores
            normalized_data = comparison_data[benchmark_cols].fillna(0)
            for col in benchmark_cols:
                if normalized_data[col].max() > 0:
                    normalized_data[col] = normalized_data[col] / normalized_data[col].max() * 100
            
            # Create radar chart
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            for i, gpu in enumerate(selected_gpus):
                if gpu in comparison_data['gpuName'].values:
                    gpu_data = comparison_data[comparison_data['gpuName'] == gpu]
                    gpu_scores = normalized_data[normalized_data.index.isin(gpu_data.index)].iloc[0]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=gpu_scores.values,
                        theta=benchmark_cols,
                        fill='toself',
                        name=gpu,
                        line_color=colors[i % len(colors)]
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Multi-Benchmark Performance Comparison (Normalized)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Benchmark correlation matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔗 Benchmark Correlations")
        if len(benchmark_cols) >= 2:
            benchmark_corr = df[benchmark_cols].corr()
            
            fig = px.imshow(
                benchmark_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Benchmark Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏆 Performance Leaders")
        if 'FP32_Final' in df.columns:
            leaders = df.nlargest(10, 'FP32_Final')[
                ['gpuName', 'Manufacturer', 'FP32_Final', 'GFLOPS_per_Watt', 'TDP']
            ].round(2)
            st.dataframe(leaders, use_container_width=True)
    
    # Performance vs specific benchmarks
    st.markdown("#### 📊 Performance vs Specific Benchmarks")
    
    if benchmark_cols:
        selected_benchmark = st.selectbox("Select Benchmark for Analysis", benchmark_cols)
        
        fig = px.scatter(
            df,
            x='FP32_Final',
            y=selected_benchmark,
            color='Manufacturer',
            size='TDP',
            hover_data=['gpuName'],
            title=f"FP32 Performance vs {selected_benchmark}",
            trendline="ols"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True) 