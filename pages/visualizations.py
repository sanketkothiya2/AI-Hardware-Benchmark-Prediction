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
    
    # Better column layout for alignment
    col1, col2 = st.columns([2, 3])
    
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
            ],
            key="viz_selector"
        )
    
    with col2:
        # Description of selected visualization with better styling
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
        
        # Better aligned description with consistent height
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin-top: 1.7rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            ">
                <strong style="font-size: 1.1rem;">{viz_type}</strong><br>
                <span style="font-size: 0.9rem; opacity: 0.9;">{descriptions.get(viz_type, 'Advanced visualization analysis')}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
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
    
    # Check for required columns
    required_cols = ['Architecture', 'FP32_Final', 'TDP', 'GFLOPS_per_Watt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return
    
    # Performance metrics by architecture
    try:
        arch_performance = df.groupby('Architecture').agg({
            'FP32_Final': ['mean', 'std', 'count'],
            'TDP': ['mean', 'std'],
            'GFLOPS_per_Watt': ['mean', 'std'],
            'price': ['mean', 'std'] if 'price' in df.columns else ['count']
        }).round(2)
        
        # Flatten column names with better formatting
        arch_performance.columns = ['_'.join(col).strip() for col in arch_performance.columns]
        
        # Reset index to make Architecture a column
        arch_performance_display = arch_performance.reset_index()
        
        # Rename columns for better readability
        column_mapping = {
            'FP32_Final_mean': 'Performance (GFLOPS)',
            'FP32_Final_std': 'Perf Std Dev',
            'FP32_Final_count': 'GPU Count',
            'TDP_mean': 'Avg TDP (W)',
            'TDP_std': 'TDP Std Dev',
            'GFLOPS_per_Watt_mean': 'Efficiency (GFLOPS/W)',
            'GFLOPS_per_Watt_std': 'Eff Std Dev',
            'price_mean': 'Avg Price ($)',
            'price_std': 'Price Std Dev'
        }
        
        # Apply column renaming
        arch_performance_display = arch_performance_display.rename(columns=column_mapping)
        
        # Format numeric columns for better display
        numeric_columns = arch_performance_display.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if 'Price' in col:
                arch_performance_display[col] = arch_performance_display[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
            elif col == 'GPU Count':
                arch_performance_display[col] = arch_performance_display[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "0")
            else:
                arch_performance_display[col] = arch_performance_display[col].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "N/A")
        
        # Filter out architectures with very few GPUs
        min_gpu_count = 3
        arch_performance_filtered = arch_performance_display[
            arch_performance_display['GPU Count'].apply(lambda x: int(x.replace(',', '')) if x != "N/A" else 0) >= min_gpu_count
        ].copy()
        
        # Sort by performance
        if 'Performance (GFLOPS)' in arch_performance_filtered.columns:
            # Extract numeric values for sorting
            arch_performance_filtered['_sort_key'] = arch_performance_filtered['Performance (GFLOPS)'].apply(
                lambda x: float(x.replace(',', '')) if x != "N/A" else 0
            )
            arch_performance_filtered = arch_performance_filtered.sort_values('_sort_key', ascending=False)
            arch_performance_filtered = arch_performance_filtered.drop('_sort_key', axis=1)
        
        # Main layout with better proportions
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### Performance by Architecture")
            st.markdown(f"*Showing architectures with {min_gpu_count}+ GPUs*")
            
            # Display with improved formatting
            st.dataframe(
                arch_performance_filtered,
                use_container_width=True,
                height=450,
                hide_index=True
            )
        
        with col2:
            # Radar chart for top architectures
            st.markdown("#### 🎯 Architecture Radar")
            
            try:
                top_archs = df['Architecture'].value_counts().head(5).index
                arch_subset = df[df['Architecture'].isin(top_archs)].copy()
                
                if len(arch_subset) > 0:
                    arch_means = arch_subset.groupby('Architecture')[
                        ['FP32_Final', 'GFLOPS_per_Watt', 'TDP']
                    ].mean()
                    
                    # Invert TDP for radar (lower is better)
                    arch_means['TDP_inverted'] = arch_means['TDP'].max() - arch_means['TDP']
                    arch_means = arch_means.drop('TDP', axis=1)
                    
                    # Normalize for radar chart
                    arch_norm = arch_means.div(arch_means.max())
                    
                    if len(arch_norm) > 0:
                        fig = go.Figure()
                        
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                        
                        for i, arch in enumerate(arch_norm.index):
                            fig.add_trace(go.Scatterpolar(
                                r=arch_norm.loc[arch].values,
                                theta=['Performance', 'Efficiency', 'Power (inv)'],
                                fill='toself',
                                name=arch,
                                line=dict(color=colors[i % len(colors)])
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=True,
                            title="Top 5 Architectures",
                            height=450,
                            margin=dict(t=40, b=20, l=20, r=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient data for radar chart")
                else:
                    st.info("No architecture data available")
                    
            except Exception as e:
                st.warning(f"Could not generate radar chart: {e}")
                
                # Fallback: Simple bar chart
                try:
                    if 'FP32_Final' in df.columns:
                        arch_perf_simple = df.groupby('Architecture')['FP32_Final'].mean().sort_values(ascending=False).head(8)
                        
                        fig = px.bar(
                            x=arch_perf_simple.index,
                            y=arch_perf_simple.values,
                            title="Average Performance by Architecture",
                            labels={'x': 'Architecture', 'y': 'Performance (GFLOPS)'}
                        )
                        fig.update_layout(height=450, margin=dict(t=40, b=40, l=40, r=40))
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e2:
                    st.error(f"Could not generate fallback chart: {e2}")
        
        # Architecture insights section - full width below the columns for better alignment
        st.markdown("---")
        st.markdown("### 📊 Architecture Insights")
        
        if len(arch_performance_filtered) > 0:
            # Create four columns for better metric distribution
            col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)
            
            # Performance statistics
            top_arch = arch_performance_filtered.iloc[0]['Architecture']
            top_perf = arch_performance_filtered.iloc[0]['Performance (GFLOPS)']
            
            most_gpus_idx = arch_performance_filtered['GPU Count'].apply(lambda x: int(x.replace(',', ''))).idxmax()
            most_gpus_arch = arch_performance_filtered.loc[most_gpus_idx, 'Architecture']
            most_gpus_count = arch_performance_filtered.loc[most_gpus_idx, 'GPU Count']
            
            # Calculate additional insights
            total_archs = len(arch_performance_filtered)
            avg_efficiency = arch_performance_filtered['Efficiency (GFLOPS/W)'].apply(
                lambda x: float(x.replace(',', '')) if x != "N/A" else 0
            ).mean()
            
            with col_insight1:
                st.metric("🏆 Top Performance", f"{top_arch}", f"{top_perf}")
            
            with col_insight2:
                st.metric("📈 Most GPUs", f"{most_gpus_arch}", f"{most_gpus_count} GPUs")
            
            with col_insight3:
                st.metric("🏗️ Architectures", f"{total_archs}", "Available")
            
            with col_insight4:
                st.metric("⚡ Avg Efficiency", f"{avg_efficiency:.1f}", "GFLOPS/W")
        
    except Exception as e:
        st.error(f"Error analyzing architecture data: {e}")
        
        # Provide basic fallback analysis
        if 'Architecture' in df.columns:
            st.markdown("#### Basic Architecture Summary")
            arch_counts = df['Architecture'].value_counts()
            st.bar_chart(arch_counts.head(10))

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
    
    # Check for required columns
    required_cols = ['Manufacturer', 'FP32_Final']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return
    
    # Clean data - remove invalid manufacturers and performance values
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=['Manufacturer', 'FP32_Final'])
    df_clean = df_clean[df_clean['Manufacturer'] != '']
    df_clean = df_clean[df_clean['Manufacturer'] != 'Unknown']
    df_clean = df_clean[df_clean['FP32_Final'] > 0]
    
    if len(df_clean) == 0:
        st.warning("No valid manufacturer data available")
        return
    
    # Performance by manufacturer over time (if date column exists)
    try:
        if 'testDate' in df_clean.columns:
            st.markdown("#### 📈 Performance Trends Over Time")
            
            # Convert dates and extract year
            df_clean['year'] = pd.to_datetime(df_clean['testDate'], errors='coerce').dt.year
            df_year_clean = df_clean.dropna(subset=['year'])
            
            if len(df_year_clean) > 0 and len(df_year_clean['year'].unique()) > 1:
                yearly_performance = df_year_clean.groupby(['year', 'Manufacturer'])['FP32_Final'].mean().reset_index()
                
                # Filter to manufacturers with data across multiple years
                mfg_counts = yearly_performance.groupby('Manufacturer').size()
                valid_mfgs = mfg_counts[mfg_counts >= 2].index
                yearly_performance_filtered = yearly_performance[yearly_performance['Manufacturer'].isin(valid_mfgs)]
                
                if len(yearly_performance_filtered) > 0:
                    fig = px.line(
                        yearly_performance_filtered,
                        x='year',
                        y='FP32_Final',
                        color='Manufacturer',
                        title="Performance Trends by Manufacturer Over Time",
                        labels={'FP32_Final': 'Performance (GFLOPS)', 'year': 'Year'},
                        markers=True
                    )
                    fig.update_layout(height=400, margin=dict(t=40, b=40, l=40, r=40))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient temporal data for trend analysis")
            else:
                st.info("Insufficient date range for trend analysis")
    except Exception as e:
        st.warning(f"Could not generate trend analysis: {e}")
    
    # Market share and performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Market Share")
        try:
            # Get manufacturer counts
            market_share = df_clean['Manufacturer'].value_counts()
            
            # Filter to show only manufacturers with at least 3 GPUs
            market_share_filtered = market_share[market_share >= 3]
            
            if len(market_share_filtered) > 0:
                # Create pie chart
                fig = px.pie(
                    values=market_share_filtered.values,
                    names=market_share_filtered.index,
                    title="GPU Database Distribution by Manufacturer",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=400, margin=dict(t=40, b=40, l=40, r=40))
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show market share statistics
                total_gpus = market_share_filtered.sum()
                top_manufacturer = market_share_filtered.index[0]
                top_share = market_share_filtered.iloc[0]
                
                st.metric(
                    f"🏆 Market Leader", 
                    f"{top_manufacturer}", 
                    f"{top_share} GPUs ({top_share/total_gpus*100:.1f}%)"
                )
            else:
                st.warning("No manufacturers with sufficient data (3+ GPUs)")
                
        except Exception as e:
            st.error(f"Could not generate market share chart: {e}")
            
            # Fallback: Simple bar chart
            try:
                mfg_counts = df_clean['Manufacturer'].value_counts().head(8)
                st.bar_chart(mfg_counts)
            except Exception as e2:
                st.error(f"Fallback chart also failed: {e2}")
    
    with col2:
        st.markdown("#### 📈 Performance Distribution")
        try:
            # Filter manufacturers with enough data points
            mfg_counts = df_clean['Manufacturer'].value_counts()
            valid_manufacturers = mfg_counts[mfg_counts >= 5].index
            df_performance = df_clean[df_clean['Manufacturer'].isin(valid_manufacturers)]
            
            if len(df_performance) > 0:
                # Create box plot
                fig = px.box(
                    df_performance,
                    x='Manufacturer',
                    y='FP32_Final',
                    title="Performance Distribution by Manufacturer",
                    labels={'FP32_Final': 'Performance (GFLOPS)'},
                    color='Manufacturer',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(
                    height=400, 
                    margin=dict(t=40, b=40, l=40, r=40),
                    xaxis={'categoryorder': 'total descending'}
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance statistics
                avg_performance = df_performance.groupby('Manufacturer')['FP32_Final'].mean().sort_values(ascending=False)
                top_performer = avg_performance.index[0]
                top_avg = avg_performance.iloc[0]
                
                st.metric(
                    f"🚀 Top Performer", 
                    f"{top_performer}", 
                    f"{top_avg:,.0f} GFLOPS avg"
                )
            else:
                st.warning("Insufficient data for performance distribution (need 5+ GPUs per manufacturer)")
                
        except Exception as e:
            st.error(f"Could not generate performance distribution: {e}")
            
            # Fallback: Simple scatter plot
            try:
                avg_perf = df_clean.groupby('Manufacturer')['FP32_Final'].mean().sort_values(ascending=False).head(8)
                fig = px.bar(
                    x=avg_perf.index,
                    y=avg_perf.values,
                    title="Average Performance by Manufacturer",
                    labels={'x': 'Manufacturer', 'y': 'Average Performance (GFLOPS)'}
                )
                fig.update_layout(height=400, margin=dict(t=40, b=40, l=40, r=40))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e2:
                st.error(f"Fallback chart also failed: {e2}")
    
    # Additional manufacturer insights
    st.markdown("---")
    st.markdown("### 🔍 Manufacturer Insights")
    
    try:
        col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)
        
        # Calculate insights
        total_manufacturers = len(df_clean['Manufacturer'].unique())
        total_gpus = len(df_clean)
        avg_perf_overall = df_clean['FP32_Final'].mean()
        
        # Efficiency leaders (if available)
        efficiency_leader = "N/A"
        if 'GFLOPS_per_Watt' in df_clean.columns:
            eff_by_mfg = df_clean.groupby('Manufacturer')['GFLOPS_per_Watt'].mean().sort_values(ascending=False)
            if len(eff_by_mfg) > 0:
                efficiency_leader = eff_by_mfg.index[0]
        
        with col_insight1:
            st.metric("🏭 Total Manufacturers", f"{total_manufacturers}")
        
        with col_insight2:
            st.metric("🖥️ Total GPUs", f"{total_gpus:,}")
        
        with col_insight3:
            st.metric("📊 Avg Performance", f"{avg_perf_overall:,.0f} GFLOPS")
        
        with col_insight4:
            st.metric("⚡ Efficiency Leader", f"{efficiency_leader}")
    
    except Exception as e:
        st.warning(f"Could not generate manufacturer insights: {e}")

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
                    title=f"FP32 Performance vs {gaming_metric}"
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
                    title=f"FP32 Performance vs {ai_metric}"
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
                    title=f"{target_var} vs {feature} (Linear Regression)"
                )
            elif reg_type == "Polynomial (degree 2)":
                fig = px.scatter(
                    df,
                    x=feature,
                    y=target_var,
                    color='Manufacturer',
                    title=f"{target_var} vs {feature} (Polynomial Regression)"
                )
            else:  # Logarithmic
                fig = px.scatter(
                    df,
                    x=feature,
                    y=target_var,
                    color='Manufacturer',
                    title=f"{target_var} vs {feature} (Log Scale)",
                    log_x=True
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Multi-variable regression visualization
            if len(selected_features) >= 2:
                st.markdown("#### 🔮 Multi-Variable Analysis")
                
                try:
                    feature_x = selected_features[0]
                    feature_y = selected_features[1]
                    
                    # Clean data for 3D visualization
                    required_cols = [feature_x, feature_y, target_var]
                    available_cols = [col for col in required_cols if col in df.columns]
                    
                    if len(available_cols) == 3:
                        analysis_data = df[required_cols].dropna()
                        
                        if len(analysis_data) > 10:
                            # Check if Manufacturer column exists and add it
                            color_col = None
                            if 'Manufacturer' in df.columns:
                                manufacturer_data = df.loc[analysis_data.index, 'Manufacturer']
                                analysis_data = analysis_data.copy()
                                analysis_data['Manufacturer'] = manufacturer_data
                                color_col = 'Manufacturer'
                            
                            fig = px.scatter_3d(
                                analysis_data,
                                x=feature_x,
                                y=feature_y,
                                z=target_var,
                                color=color_col,
                                title=f"3D Regression: {target_var} vs {feature_x} & {feature_y}",
                                labels={
                                    feature_x: feature_x.replace('_', ' '),
                                    feature_y: feature_y.replace('_', ' '),
                                    target_var: target_var.replace('_', ' ')
                                }
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Insufficient data points for 3D analysis (need >10 valid data points)")
                    else:
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        st.warning(f"Missing columns for 3D analysis: {missing_cols}")
                        
                except Exception as e:
                    st.error(f"Error in multi-variable analysis: {e}")
    
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
        try:
            # Check if all required columns exist
            required_cols = [x_axis, y_axis, z_axis]
            if color_by and color_by in df.columns:
                required_cols.append(color_by)
            
            # Clean data for visualization
            plot_data = df[required_cols].dropna()
            
            if len(plot_data) > 5:
                scatter_kwargs = {
                    'data_frame': plot_data,
                    'x': x_axis,
                    'y': y_axis,
                    'z': z_axis,
                    'title': f"3D Performance Space: {x_axis} vs {y_axis} vs {z_axis}",
                    'labels': {
                        x_axis: x_axis.replace('_', ' '),
                        y_axis: y_axis.replace('_', ' '),
                        z_axis: z_axis.replace('_', ' ')
                    }
                }
                
                # Add color if column exists
                if color_by and color_by in plot_data.columns:
                    scatter_kwargs['color'] = color_by
                
                # Add hover data if available
                if 'gpuName' in df.columns:
                    gpu_names = df.loc[plot_data.index, 'gpuName']
                    plot_data = plot_data.copy()
                    plot_data['gpuName'] = gpu_names
                    scatter_kwargs['hover_data'] = ['gpuName']
                
                # Add size if specified
                if size_by != "None" and size_by in plot_data.columns:
                    scatter_kwargs['size'] = size_by
                
                fig = px.scatter_3d(**scatter_kwargs)
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for 3D visualization (need >5 valid data points)")
                
        except Exception as e:
            st.error(f"Error creating 3D visualization: {e}")
    
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
            title=f"FP32 Performance vs {selected_benchmark}"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True) 