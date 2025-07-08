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
    """Advanced dynamic hardware comparison with ML predictions"""
    st.markdown("# 🔍 Hardware Comparison Center")
    st.markdown("**Compare custom hardware configurations with AI-powered predictions**")
    
    if st.session_state.data is None:
        st.error("Dataset not loaded")
        return
    
    df = st.session_state.data
    models = getattr(st.session_state, 'models', {})
    
    # Debug section - show model status
    with st.expander("🔧 Model Status (Debug)", expanded=False):
        col_debug1, col_debug2 = st.columns([3, 1])
        
        with col_debug1:
            st.markdown("**Available Models & Preprocessors:**")
        
        with col_debug2:
            if st.button("🔄 Refresh Models", type="secondary"):
                # Clear cached data and reload
                if hasattr(st.session_state, 'models'):
                    del st.session_state.models
                st.cache_data.clear()
                st.rerun()
        
        if models:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Models:**")
                model_count = 0
                for key, value in models.items():
                    if 'model.pkl' in key:
                        model_count += 1
                        has_predict = hasattr(value, 'predict')
                        status = "✅" if has_predict else "❌"
                        st.write(f"{status} {key.replace('.pkl', '')}")
                        if not has_predict:
                            st.write(f"   Type: {type(value)}")
                
                st.write(f"**Total Models: {model_count}**")
            
            with col2:
                st.markdown("**Preprocessors:**")
                prep_count = 0
                for key, value in models.items():
                    if 'preprocessor' in key:
                        prep_count += 1
                        has_transform = hasattr(value, 'transform')
                        status = "✅" if has_transform else "❌"
                        st.write(f"{status} {key.replace('.pkl', '')}")
                        if not has_transform:
                            st.write(f"   Type: {type(value)}")
                
                st.write(f"**Total Preprocessors: {prep_count}**")
        else:
            st.error("No models loaded in session state")
            st.info("Try refreshing the page or check model files in data/models/phase3_outputs/")
            
            # Show what we're looking for
            expected_files = [
                "random_forest_FP32_Final_model.pkl",
                "preprocessor_performance.pkl",
                "efficiency_random_forest_GFLOPS_per_Watt_model.pkl", 
                "preprocessor_efficiency.pkl",
                "classification_random_forest_AI_Performance_Category_model.pkl",
                "preprocessor_classification.pkl",
                "classification_label_encoder_AI_Performance_Category.pkl"
            ]
            
            st.markdown("**Expected Model Files:**")
            for file in expected_files:
                st.write(f"• {file}")
    
    # Create tabs for different comparison modes
    tab1, tab2, tab3 = st.tabs(["🛠️ Custom Builder", "📊 Database Compare", "🎯 Use Case Optimizer"])
    
    with tab1:
        show_custom_hardware_builder(df, models)
    
    with tab2:
        show_database_comparison(df)
    
    with tab3:
        show_use_case_optimizer(df, models)

def show_custom_hardware_builder(df, models):
    """Dynamic hardware configuration builder"""
    st.markdown("### 🛠️ Build & Compare Custom Hardware Configurations")
    st.markdown("Configure up to 4 custom hardware setups and compare their predicted performance")
    
    # Configuration selector
    col1, col2 = st.columns([3, 1])
    with col1:
        num_configs = st.slider("Number of configurations to compare", 2, 4, 2)
    with col2:
        if st.button("🔄 Reset All", type="secondary"):
            for i in range(4):
                for key in [f"vendor_{i}", f"arch_{i}", f"cores_{i}", f"memory_{i}", f"tdp_{i}"]:
                    if key in st.session_state:
                        del st.session_state[key]
            st.experimental_rerun()
    
    # Get unique values from dataset
    vendors = sorted([v for v in df['Manufacturer'].unique() if v and str(v) != 'nan'])
    architectures = sorted([a for a in df['Architecture'].unique() if a and str(a) != 'nan' and a != 'Unknown'])
    
    # Configuration cards
    configs = []
    cols = st.columns(num_configs)
    
    for i in range(num_configs):
        with cols[i]:
            config = create_hardware_config_card(i, vendors, architectures, df)
            if config:
                configs.append(config)
    
    # Generate predictions and comparisons
    if len(configs) >= 2:
        st.markdown("---")
        show_configuration_comparison(configs, models, df)

def create_hardware_config_card(index: int, vendors: List[str], architectures: List[str], df: pd.DataFrame) -> Dict[str, Any]:
    """Create a hardware configuration card"""
    st.markdown(f"#### 🖥️ Configuration {index + 1}")
    
    # Create expandable configuration
    with st.expander(f"⚙️ Configure Hardware {index + 1}", expanded=True):
        
        # Vendor selection
        vendor = st.selectbox(
            "Vendor", vendors, 
            key=f"vendor_{index}",
            index=min(index, len(vendors)-1)
        )
        
        # Architecture selection (filtered by vendor)
        vendor_archs = get_vendor_architectures(df, vendor)
        if vendor_archs:
            arch_index = min(0, len(vendor_archs)-1)
        else:
            vendor_archs = architectures[:5]  # Fallback
            arch_index = 0
            
        architecture = st.selectbox(
            "Architecture", vendor_archs,
            key=f"arch_{index}",
            index=arch_index
        )
        
        # Dynamic core naming based on vendor
        if vendor == "NVIDIA":
            cores_label = "CUDA Cores"
            default_cores = 8704
            max_cores = 12000
            step_size = 128
        elif vendor == "AMD":
            cores_label = "Stream Processors" 
            default_cores = 3584
            max_cores = 12000
            step_size = 64
        elif vendor == "Intel":
            cores_label = "Execution Units"
            default_cores = 128
            max_cores = 512
            step_size = 8
        else:
            cores_label = "Compute Units"
            default_cores = 2048
            max_cores = 12000
            step_size = 128
        
        # Hardware specifications
        col1, col2 = st.columns(2)
        
        with col1:
            cores = st.number_input(
                cores_label, 
                min_value=64 if vendor == "Intel" else 500,
                max_value=max_cores,
                value=default_cores,
                step=step_size,
                key=f"cores_{index}"
            )
            
            memory_gb = st.slider(
                "Memory (GB)", 4, 48, 16,
                key=f"memory_{index}"
            )
            
            memory_bandwidth = st.number_input(
                "Memory Bandwidth (GB/s)", 
                100, 2000, 760,
                key=f"bandwidth_{index}"
            )
        
        with col2:
            tdp = st.slider(
                "TDP (W)", 50, 600, 300,
                key=f"tdp_{index}"
            )
            
            process_size = st.selectbox(
                "Process Node (nm)", [4, 5, 6, 7, 8, 12, 14, 16, 20, 22, 28],
                index=2,  # Default to 6nm
                key=f"process_{index}"
            )
            
            estimated_price = st.number_input(
                "Est. Price ($)", 200, 5000, 1000,
                key=f"price_{index}"
            )
        
        # Advanced features
        st.markdown("**🚀 Advanced Features**")
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            has_tensor = st.checkbox(
                "AI Acceleration",
                value=vendor == "NVIDIA",
                key=f"tensor_{index}",
                help=f"{'Tensor Cores' if vendor == 'NVIDIA' else 'Matrix/XMX Cores'}"
            )
        
        with feature_col2:
            supports_int8 = st.checkbox(
                "INT8 Support",
                value=True,
                key=f"int8_{index}"
            )
        
        # Configuration summary
        config = {
            "name": f"{vendor} {architecture}",
            "vendor": vendor,
            "architecture": architecture,
            "cores": cores,
            "memory_gb": memory_gb,
            "memory_bandwidth": memory_bandwidth,
            "tdp": tdp,
            "process_size": process_size,
            "estimated_price": estimated_price,
            "has_tensor": has_tensor,
            "supports_int8": supports_int8,
            "index": index
        }
        
        # Show quick preview
        efficiency = cores * 0.8 / tdp  # Simple efficiency estimate
        perf_per_dollar = (cores * memory_gb) / estimated_price
        
        preview_col1, preview_col2, preview_col3 = st.columns(3)
        with preview_col1:
            st.metric("Est. Efficiency", f"{efficiency:.1f}")
        with preview_col2:
            st.metric("Perf/$", f"{perf_per_dollar:.1f}")
        with preview_col3:
            st.metric("TDP", f"{tdp}W")
    
    return config

def get_vendor_architectures(df: pd.DataFrame, vendor: str) -> List[str]:
    """Get architectures available for a specific vendor"""
    vendor_df = df[df['Manufacturer'] == vendor]
    if not vendor_df.empty:
        archs = [a for a in vendor_df['Architecture'].unique() if a and str(a) != 'nan' and a != 'Unknown']
        return sorted(archs)
    return []

def show_configuration_comparison(configs: List[Dict], models: Dict, df: pd.DataFrame):
    """Show detailed comparison of configurations"""
    st.markdown("## 🔬 Performance Analysis & Comparison")
    
    # Generate predictions for each configuration
    predictions = []
    for config in configs:
        pred = generate_hardware_predictions(config, models, df)
        pred['config'] = config
        predictions.append(pred)
    
    # Overview metrics cards
    st.markdown("### 📊 Performance Overview")
    metric_cols = st.columns(len(configs))
    
    for i, pred in enumerate(predictions):
        with metric_cols[i]:
            config = pred['config']
            st.markdown(f"**🖥️ {config['name']}**")
            
            # Performance metrics
            if 'FP32_Performance_TFLOPS' in pred:
                st.metric(
                    "Performance", 
                    f"{pred['FP32_Performance_TFLOPS']:.1f} TFLOPS",
                    delta=None
                )
            
            if 'GFLOPS_per_Watt' in pred:
                st.metric(
                    "Efficiency",
                    f"{pred['GFLOPS_per_Watt']:.1f} GFLOPS/W"
                )
            
            if 'AI_Performance_Category' in pred:
                category = pred['AI_Performance_Category']
                color = get_category_color(category)
                st.markdown(f"**AI Category**: :{color}[{category}]")
            
            # Value metrics
            price = config['estimated_price']
            if 'FP32_Performance_TFLOPS' in pred and price > 0:
                perf_per_dollar = pred['FP32_Performance_TFLOPS'] * 1000 / price
                st.metric("Perf/$ (GFLOPS/$)", f"{perf_per_dollar:.2f}")
    
    # Detailed comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        create_performance_comparison_chart(predictions)
    
    with col2:
        create_efficiency_comparison_chart(predictions)
    
    # Feature comparison table
    st.markdown("### 📋 Feature Comparison Table")
    create_feature_comparison_table(predictions)
    
    # Radar chart comparison
    st.markdown("### 🎯 Multi-Dimensional Analysis")
    create_radar_comparison(predictions)
    
    # Recommendations
    st.markdown("### 💡 AI Recommendations")
    create_intelligent_recommendations(predictions)

def generate_hardware_predictions(config: Dict, models: Dict, df: pd.DataFrame) -> Dict:
    """Generate ML predictions for hardware configuration"""
    try:
        # Try to import prediction functions from ai_prediction module
        try:
            from pages.ai_prediction import create_feature_vector_for_prediction, generate_fallback_predictions
        except ImportError:
            st.warning("AI prediction module not available, using basic calculations")
            # Use basic fallback calculations
            base_perf = config['cores'] * 0.0012  # Simple estimate based on cores
            base_eff = base_perf * 1000 / config['tdp']
            
            return {
                'FP32_Performance_TFLOPS': max(0.1, base_perf),
                'FP32_Performance_GFLOPS': max(100, base_perf * 1000),
                'GFLOPS_per_Watt': max(1, base_eff),
                'AI_Performance_Category': 'Mid-Range'
            }
        
        # Create feature vector
        features = create_feature_vector_for_prediction(
            config['vendor'], config['architecture'], config['cores'],
            config['memory_gb'], config['memory_bandwidth'], config['tdp'],
            config['process_size'], config['has_tensor'], config['supports_int8'], df
        )
        
        predictions = {}
        
        if features is not None and models:
            # Performance prediction with validation
            try:
                perf_model_key = 'random_forest_FP32_Final_model.pkl'
                perf_prep_key = 'preprocessor_performance.pkl'
                
                if (perf_model_key in models and perf_prep_key in models and
                    hasattr(models[perf_prep_key], 'transform') and
                    hasattr(models[perf_model_key], 'predict')):
                    
                    perf_model = models[perf_model_key]
                    perf_preprocessor = models[perf_prep_key]
                    
                    # Validate preprocessor
                    if hasattr(perf_preprocessor, 'transform'):
                        X_processed = perf_preprocessor.transform([features])
                        fp32_pred = perf_model.predict(X_processed)[0]
                        predictions['FP32_Performance_GFLOPS'] = max(0, fp32_pred / 1e9)
                        predictions['FP32_Performance_TFLOPS'] = max(0, fp32_pred / 1e12)
                    else:
                        st.warning("Performance preprocessor not properly loaded")
                else:
                    st.warning("Performance model or preprocessor not available")
                    
            except Exception as e:
                st.warning(f"Performance model error: {str(e)}")
            
            # Efficiency prediction with validation
            try:
                eff_model_key = 'efficiency_random_forest_GFLOPS_per_Watt_model.pkl'
                eff_prep_key = 'preprocessor_efficiency.pkl'
                
                if (eff_model_key in models and eff_prep_key in models and
                    hasattr(models[eff_prep_key], 'transform') and
                    hasattr(models[eff_model_key], 'predict')):
                    
                    eff_model = models[eff_model_key]
                    eff_preprocessor = models[eff_prep_key]
                    
                    # Validate preprocessor
                    if hasattr(eff_preprocessor, 'transform'):
                        X_processed = eff_preprocessor.transform([features])
                        eff_pred = eff_model.predict(X_processed)[0]
                        predictions['GFLOPS_per_Watt'] = max(0, eff_pred)
                    else:
                        st.warning("Efficiency preprocessor not properly loaded")
                else:
                    st.warning("Efficiency model or preprocessor not available")
                    
            except Exception as e:
                st.warning(f"Efficiency model error: {str(e)}")
            
            # AI Category prediction with validation
            try:
                cat_model_key = 'classification_random_forest_AI_Performance_Category_model.pkl'
                cat_prep_key = 'preprocessor_classification.pkl'
                cat_enc_key = 'classification_label_encoder_AI_Performance_Category.pkl'
                
                if (cat_model_key in models and cat_prep_key in models and cat_enc_key in models and
                    hasattr(models[cat_prep_key], 'transform') and
                    hasattr(models[cat_model_key], 'predict') and
                    hasattr(models[cat_enc_key], 'inverse_transform')):
                    
                    cat_model = models[cat_model_key]
                    cat_preprocessor = models[cat_prep_key]
                    cat_encoder = models[cat_enc_key]
                    
                    # Validate preprocessor and encoder
                    if hasattr(cat_preprocessor, 'transform') and hasattr(cat_encoder, 'inverse_transform'):
                        X_processed = cat_preprocessor.transform([features])
                        cat_pred = cat_model.predict(X_processed)[0]
                        ai_category = cat_encoder.inverse_transform([cat_pred])[0]
                        predictions['AI_Performance_Category'] = ai_category
                    else:
                        st.warning("Classification preprocessor or encoder not properly loaded")
                else:
                    st.warning("Classification model, preprocessor, or encoder not available")
                    
            except Exception as e:
                st.warning(f"Classification model error: {str(e)}")
        
        # Fallback to algorithmic predictions if ML models fail or don't exist
        if not predictions:
            try:
                fallback = generate_fallback_predictions(
                    config['vendor'], config['architecture'], config['cores'],
                    config['memory_gb'], config['memory_bandwidth'], config['tdp'],
                    config['process_size'], config['has_tensor'], config['supports_int8'],
                    config['estimated_price'], df
                )
                predictions.update(fallback)
                st.info("Using algorithmic predictions (ML models not available)")
            except Exception as e:
                st.error(f"Fallback prediction error: {e}")
        
        # Ensure we have some basic predictions
        if not predictions:
            # Basic fallback calculations
            base_perf = config['cores'] * 0.0012  # Simple estimate
            base_eff = base_perf * 1000 / config['tdp']
            
            predictions = {
                'FP32_Performance_TFLOPS': max(0.1, base_perf),
                'FP32_Performance_GFLOPS': max(100, base_perf * 1000),
                'GFLOPS_per_Watt': max(1, base_eff),
                'AI_Performance_Category': 'Mid-Range'
            }
            st.info("Using basic algorithmic estimates")
        
        return predictions
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Return basic fallback predictions
        base_perf = config['cores'] * 0.0012  # More realistic estimate
        base_eff = base_perf * 1000 / config['tdp']
        
        return {
            'FP32_Performance_TFLOPS': max(0.1, base_perf),
            'FP32_Performance_GFLOPS': max(100, base_perf * 1000),
            'GFLOPS_per_Watt': max(1, base_eff),
            'AI_Performance_Category': 'Mid-Range'
        }

def create_performance_comparison_chart(predictions: List[Dict]):
    """Create performance comparison bar chart"""
    config_names = [pred['config']['name'] for pred in predictions]
    
    # Performance data
    fp32_values = [pred.get('FP32_Performance_TFLOPS', 0) for pred in predictions]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=config_names,
        y=fp32_values,
        name='FP32 Performance (TFLOPS)',
        text=[f"{val:.1f}" for val in fp32_values],
        textposition='outside',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(config_names)]
    ))
    
    fig.update_layout(
        title="🚀 Raw Performance Comparison",
        xaxis_title="Configuration",
        yaxis_title="Performance (TFLOPS)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_efficiency_comparison_chart(predictions: List[Dict]):
    """Create efficiency comparison chart"""
    config_names = [pred['config']['name'] for pred in predictions]
    
    # Efficiency data
    efficiency_values = [pred.get('GFLOPS_per_Watt', 0) for pred in predictions]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=config_names,
        y=efficiency_values,
        name='Power Efficiency',
        text=[f"{val:.1f}" for val in efficiency_values],
        textposition='outside',
        marker_color=['#2E8B57', '#FF6347', '#4682B4', '#DAA520'][:len(config_names)]
    ))
    
    fig.update_layout(
        title="⚡ Power Efficiency Comparison",
        xaxis_title="Configuration",
        yaxis_title="GFLOPS per Watt",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_feature_comparison_table(predictions: List[Dict]):
    """Create detailed feature comparison table"""
    
    # Prepare comparison data
    comparison_data = []
    
    metrics = [
        ('Hardware', 'config.vendor'),
        ('Architecture', 'config.architecture'),
        ('Compute Units', 'config.cores'),
        ('Memory (GB)', 'config.memory_gb'),
        ('TDP (W)', 'config.tdp'),
        ('Performance (TFLOPS)', 'FP32_Performance_TFLOPS'),
        ('Efficiency (GFLOPS/W)', 'GFLOPS_per_Watt'),
        ('AI Category', 'AI_Performance_Category'),
        ('Est. Price ($)', 'config.estimated_price'),
        ('Perf/$ (GFLOPS/$)', 'calculated_perf_per_dollar')
    ]
    
    for metric_name, metric_key in metrics:
        row = {'Metric': metric_name}
        
        for i, pred in enumerate(predictions):
            config = pred['config']
            
            if metric_key.startswith('config.'):
                attr = metric_key.replace('config.', '')
                value = config.get(attr, 'N/A')
            elif metric_key == 'calculated_perf_per_dollar':
                # Calculate performance per dollar
                if 'FP32_Performance_TFLOPS' in pred and config['estimated_price'] > 0:
                    value = f"{(pred['FP32_Performance_TFLOPS'] * 1000 / config['estimated_price']):.2f}"
                else:
                    value = 'N/A'
            else:
                value = pred.get(metric_key, 'N/A')
            
            # Format numeric values
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if metric_name in ['Performance (TFLOPS)', 'Efficiency (GFLOPS/W)']:
                    value = f"{value:.1f}"
                elif metric_name in ['Compute Units', 'Memory (GB)', 'TDP (W)', 'Est. Price ($)']:
                    value = f"{value:,}"
            
            row[f'Config {i+1}'] = value
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, height=400)

def create_radar_comparison(predictions: List[Dict]):
    """Create radar chart for multi-dimensional comparison"""
    
    # Define metrics for radar chart
    radar_metrics = [
        ('Performance', 'FP32_Performance_TFLOPS', 1),
        ('Efficiency', 'GFLOPS_per_Watt', 1),
        ('Memory', 'config.memory_gb', 48),  # Normalize to max 48GB
        ('Value', 'calculated_perf_per_dollar', 1),
        ('Power', 'config.tdp', 600)  # Invert TDP (lower is better)
    ]
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, pred in enumerate(predictions):
        config = pred['config']
        
        values = []
        labels = []
        
        for label, key, max_val in radar_metrics:
            if key.startswith('config.'):
                attr = key.replace('config.', '')
                raw_value = config.get(attr, 0)
            elif key == 'calculated_perf_per_dollar':
                if 'FP32_Performance_TFLOPS' in pred and config['estimated_price'] > 0:
                    raw_value = pred['FP32_Performance_TFLOPS'] * 1000 / config['estimated_price']
                else:
                    raw_value = 0
            else:
                raw_value = pred.get(key, 0)
            
            # Normalize values (0-100 scale)
            if label == 'Power':  # Invert TDP (lower is better)
                normalized = max(0, 100 - (raw_value / max_val * 100))
            else:
                normalized = min(100, (raw_value / max_val * 100)) if max_val > 0 else 0
            
            values.append(normalized)
            labels.append(label)
        
        # Close the radar chart
        values.append(values[0])
        labels.append(labels[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=f"Config {i+1}: {config['name']}",
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="🎯 Multi-Dimensional Performance Analysis",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_intelligent_recommendations(predictions: List[Dict]):
    """Generate AI-powered recommendations"""
    
    # Analyze configurations for recommendations
    best_performance = max(predictions, key=lambda x: x.get('FP32_Performance_TFLOPS', 0))
    best_efficiency = max(predictions, key=lambda x: x.get('GFLOPS_per_Watt', 0))
    
    # Calculate value scores
    value_scores = []
    for pred in predictions:
        config = pred['config']
        if 'FP32_Performance_TFLOPS' in pred and config['estimated_price'] > 0:
            value_score = pred['FP32_Performance_TFLOPS'] * 1000 / config['estimated_price']
            value_scores.append((pred, value_score))
    
    if value_scores:
        best_value = max(value_scores, key=lambda x: x[1])
    else:
        best_value = None
    
    # Display recommendations
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("#### 🏆 Best Performance")
        config = best_performance['config']
        perf = best_performance.get('FP32_Performance_TFLOPS', 0)
        st.success(f"**{config['name']}**")
        st.write(f"• {perf:.1f} TFLOPS")
        st.write(f"• {config['tdp']}W TDP")
        st.write(f"• ${config['estimated_price']:,}")
    
    with rec_col2:
        st.markdown("#### ⚡ Most Efficient")
        config = best_efficiency['config']
        eff = best_efficiency.get('GFLOPS_per_Watt', 0)
        st.info(f"**{config['name']}**")
        st.write(f"• {eff:.1f} GFLOPS/W")
        st.write(f"• {config['tdp']}W TDP")
        st.write(f"• ${config['estimated_price']:,}")
    
    with rec_col3:
        if best_value:
            st.markdown("#### 💰 Best Value")
            config = best_value[0]['config']
            value_score = best_value[1]
            st.warning(f"**{config['name']}**")
            st.write(f"• {value_score:.2f} GFLOPS/$")
            st.write(f"• {config['tdp']}W TDP")
            st.write(f"• ${config['estimated_price']:,}")
        else:
            st.markdown("#### 💰 Best Value")
            st.write("Need price info for analysis")

def get_category_color(category: str) -> str:
    """Get color for AI performance category"""
    color_map = {
        'AI_Flagship': 'violet',
        'AI_High_End': 'blue',
        'AI_Mid_Range': 'green',
        'AI_Entry': 'orange',
        'AI_Basic': 'red'
    }
    return color_map.get(category, 'gray')

def show_database_comparison(df):
    """Show comparison using existing GPUs from database"""
    st.markdown("### 📊 Compare Existing GPUs")
    st.markdown("Select and compare GPUs from our database of 2,100+ graphics cards")
    
    # GPU selection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            manufacturers = ['All'] + sorted(df['Manufacturer'].unique().tolist())
            selected_mfg = st.selectbox("Manufacturer", manufacturers)
        
        with filter_col2:
            if selected_mfg != 'All':
                vendor_df = df[df['Manufacturer'] == selected_mfg]
                architectures = ['All'] + sorted(vendor_df['Architecture'].unique().tolist())
            else:
                architectures = ['All'] + sorted(df['Architecture'].unique().tolist())
            selected_arch = st.selectbox("Architecture", architectures)
        
        with filter_col3:
            if 'PerformanceTier' in df.columns:
                tiers = ['All'] + sorted(df['PerformanceTier'].unique().tolist())
                selected_tier = st.selectbox("Performance Tier", tiers)
            else:
                selected_tier = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    if selected_mfg != 'All':
        filtered_df = filtered_df[filtered_df['Manufacturer'] == selected_mfg]
    if selected_arch != 'All':
        filtered_df = filtered_df[filtered_df['Architecture'] == selected_arch]
    if selected_tier != 'All' and 'PerformanceTier' in df.columns:
        filtered_df = filtered_df[filtered_df['PerformanceTier'] == selected_tier]
    
    with col2:
        st.metric("Available GPUs", f"{len(filtered_df):,}")
        sort_by = st.selectbox("Sort by", ['FP32_Final', 'GFLOPS_per_Watt', 'price', 'TDP'])
    
    # GPU selection
    st.markdown("#### Select GPUs to Compare")
    
    # Display GPU list with selection
    if not filtered_df.empty:
        # Sort GPUs
        if sort_by in filtered_df.columns:
            display_df = filtered_df.sort_values(sort_by, ascending=False)
        else:
            display_df = filtered_df
        
        # Create selectable list
        gpu_options = []
        for idx, row in display_df.head(50).iterrows():  # Limit to top 50 for performance
            perf = row.get('FP32_Final', 0) / 1e12 if row.get('FP32_Final', 0) > 0 else 0
            eff = row.get('GFLOPS_per_Watt', 0)
            price = row.get('price', 0)
            
            label = f"{row['gpuName']} | {perf:.1f} TFLOPS | {eff:.1f} GFLOPS/W"
            if price > 0:
                label += f" | ${price:,.0f}"
            
            gpu_options.append((label, row['gpuName']))
        
        selected_gpus = st.multiselect(
            "Choose GPUs to compare (max 4)",
            options=[opt[0] for opt in gpu_options],
            default=[],
            max_selections=4
        )
        
        if len(selected_gpus) >= 2:
            # Get selected GPU data
            selected_gpu_names = [next(opt[1] for opt in gpu_options if opt[0] == sel) for sel in selected_gpus]
            selected_gpu_data = []
            
            for gpu_name in selected_gpu_names:
                gpu_data = df[df['gpuName'] == gpu_name].iloc[0]
                selected_gpu_data.append(gpu_data)
            
            # Show comparison
            show_database_gpu_comparison(selected_gpu_data)

def show_database_gpu_comparison(gpu_data_list):
    """Show detailed comparison of selected database GPUs"""
    st.markdown("---")
    st.markdown("### 🔬 Detailed GPU Comparison")
    
    # Overview cards
    cols = st.columns(len(gpu_data_list))
    
    for i, gpu_data in enumerate(gpu_data_list):
        with cols[i]:
            st.markdown(f"#### 🖥️ {gpu_data['gpuName']}")
            
            # Key metrics
            perf = gpu_data.get('FP32_Final', 0) / 1e12 if gpu_data.get('FP32_Final', 0) > 0 else 0
            eff = gpu_data.get('GFLOPS_per_Watt', 0)
            tdp = gpu_data.get('TDP', 0)
            price = gpu_data.get('price', 0)
            
            st.metric("Performance", f"{perf:.1f} TFLOPS")
            st.metric("Efficiency", f"{eff:.1f} GFLOPS/W")
            st.metric("TDP", f"{tdp:.0f}W")
            if price > 0:
                st.metric("Price", f"${price:,.0f}")
                perf_per_dollar = (perf * 1000) / price if price > 0 else 0
                st.metric("Perf/$", f"{perf_per_dollar:.2f}")
    
    # Comparison charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Performance comparison
        gpu_names = [gpu['gpuName'] for gpu in gpu_data_list]
        performances = [gpu.get('FP32_Final', 0) / 1e12 for gpu in gpu_data_list]
        
        fig = px.bar(
            x=gpu_names,
            y=performances,
            title="🚀 Performance Comparison (TFLOPS)",
            labels={'x': 'GPU', 'y': 'Performance (TFLOPS)'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Efficiency comparison
        efficiencies = [gpu.get('GFLOPS_per_Watt', 0) for gpu in gpu_data_list]
        
        fig = px.bar(
            x=gpu_names,
            y=efficiencies,
            title="⚡ Efficiency Comparison (GFLOPS/W)",
            labels={'x': 'GPU', 'y': 'GFLOPS per Watt'},
            color=efficiencies,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    create_database_comparison_table(gpu_data_list)

def create_database_comparison_table(gpu_data_list):
    """Create comparison table for database GPUs"""
    st.markdown("### 📋 Detailed Specifications")
    
    comparison_metrics = [
        'Manufacturer', 'Architecture', 'Category', 'PerformanceTier',
        'TDP', 'Memory_GB', 'FP32_Final', 'GFLOPS_per_Watt', 
        'AI_Performance_Category', 'price'
    ]
    
    comparison_data = []
    
    for metric in comparison_metrics:
        if metric in gpu_data_list[0].index:
            row = {'Specification': metric}
            
            for i, gpu_data in enumerate(gpu_data_list):
                value = gpu_data.get(metric, 'N/A')
                
                # Format values
                if metric == 'FP32_Final' and value != 'N/A':
                    value = f"{value / 1e12:.1f} TFLOPS"
                elif metric == 'GFLOPS_per_Watt' and value != 'N/A':
                    value = f"{value:.1f}"
                elif metric in ['TDP', 'Memory_GB', 'price'] and value != 'N/A':
                    if metric == 'price':
                        value = f"${value:,.0f}" if value > 0 else 'N/A'
                    else:
                        value = f"{value:.0f}"
                
                row[f'GPU {i+1}'] = value
            
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, height=400)

def show_use_case_optimizer(df, models):
    """Show use case specific optimization"""
    st.markdown("### 🎯 Use Case Performance Optimizer")
    st.markdown("Find the optimal hardware configuration for specific AI workloads")
    
    # Use case selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        use_case = st.selectbox(
            "Select AI Use Case",
            [
                "Gaming & General Graphics",
                "Machine Learning Training",
                "AI Inference/Deployment", 
                "Computer Vision",
                "Natural Language Processing",
                "Scientific Computing",
                "Cryptocurrency Mining",
                "Content Creation & Rendering"
            ]
        )
    
    with col2:
        budget_range = st.slider("Budget Range ($)", 200, 5000, (500, 2000), step=100)
        power_limit = st.slider("Power Limit (W)", 50, 600, 300)
    
    # Show recommendations based on use case
    show_use_case_recommendations(df, use_case, budget_range, power_limit)

def show_use_case_recommendations(df, use_case, budget_range, power_limit):
    """Show use case specific recommendations"""
    
    # Filter by budget and power
    filtered_df = df.copy()
    
    if 'price' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['price'] >= budget_range[0]) & 
            (filtered_df['price'] <= budget_range[1])
        ]
    
    if 'TDP' in df.columns:
        filtered_df = filtered_df[filtered_df['TDP'] <= power_limit]
    
    st.info(f"Found {len(filtered_df)} GPUs matching your criteria")
    
    # Use case specific optimization
    use_case_weights = {
        "Gaming & General Graphics": {'FP32_Final': 0.4, 'Memory_GB': 0.3, 'price': 0.3},
        "Machine Learning Training": {'FP32_Final': 0.5, 'Memory_GB': 0.4, 'GFLOPS_per_Watt': 0.1},
        "AI Inference/Deployment": {'GFLOPS_per_Watt': 0.5, 'FP32_Final': 0.3, 'TDP': 0.2},
        "Computer Vision": {'FP32_Final': 0.4, 'Memory_GB': 0.4, 'GFLOPS_per_Watt': 0.2},
        "Natural Language Processing": {'Memory_GB': 0.5, 'FP32_Final': 0.3, 'GFLOPS_per_Watt': 0.2},
        "Scientific Computing": {'FP32_Final': 0.6, 'Memory_GB': 0.3, 'GFLOPS_per_Watt': 0.1},
        "Cryptocurrency Mining": {'GFLOPS_per_Watt': 0.7, 'FP32_Final': 0.2, 'price': 0.1},
        "Content Creation & Rendering": {'FP32_Final': 0.4, 'Memory_GB': 0.3, 'TDP': 0.3}
    }
    
    weights = use_case_weights.get(use_case, {'FP32_Final': 0.5, 'Memory_GB': 0.3, 'GFLOPS_per_Watt': 0.2})
    
    # Calculate weighted scores
    if not filtered_df.empty:
        scores = calculate_use_case_scores(filtered_df, weights)
        top_recommendations = scores.head(10)
        
        st.markdown(f"### 🏆 Top Recommendations for {use_case}")
        
        # Display top 3 in cards
        if len(top_recommendations) >= 3:
            rec_cols = st.columns(3)
            
            for i in range(min(3, len(top_recommendations))):
                with rec_cols[i]:
                    gpu_data = top_recommendations.iloc[i]
                    
                    st.markdown(f"#### 🥇 #{i+1}: {gpu_data['gpuName']}")
                    
                    # Key metrics
                    perf = gpu_data.get('FP32_Final', 0) / 1e12 if gpu_data.get('FP32_Final', 0) > 0 else 0
                    eff = gpu_data.get('GFLOPS_per_Watt', 0)
                    memory = gpu_data.get('Memory_GB', 0)
                    price = gpu_data.get('price', 0)
                    score = gpu_data.get('weighted_score', 0)
                    
                    st.metric("Score", f"{score:.2f}")
                    st.write(f"**Performance**: {perf:.1f} TFLOPS")
                    st.write(f"**Memory**: {memory:.0f} GB")
                    st.write(f"**Efficiency**: {eff:.1f} GFLOPS/W")
                    if price > 0:
                        st.write(f"**Price**: ${price:,.0f}")
        
        # Full recommendations table
        st.markdown("### 📊 Complete Rankings")
        display_cols = ['gpuName', 'Manufacturer', 'FP32_Final', 'Memory_GB', 'GFLOPS_per_Watt', 'price', 'weighted_score']
        display_recommendations = top_recommendations[display_cols].copy()
        
        # Format for display
        if 'FP32_Final' in display_recommendations.columns:
            display_recommendations['Performance (TFLOPS)'] = display_recommendations['FP32_Final'] / 1e12
            display_recommendations = display_recommendations.drop('FP32_Final', axis=1)
        
        st.dataframe(display_recommendations, use_container_width=True, height=400)

def calculate_use_case_scores(df, weights):
    """Calculate weighted scores for use case optimization"""
    scored_df = df.copy()
    
    # Normalize metrics to 0-1 scale
    for metric, weight in weights.items():
        if metric in df.columns:
            values = df[metric].fillna(0)
            
            if metric in ['TDP', 'price']:  # Lower is better
                if values.max() > 0:
                    normalized = 1 - (values / values.max())
                else:
                    normalized = pd.Series([0] * len(values))
            else:  # Higher is better
                if values.max() > 0:
                    normalized = values / values.max()
                else:
                    normalized = pd.Series([0] * len(values))
            
            scored_df[f'{metric}_normalized'] = normalized
    
    # Calculate weighted score
    scored_df['weighted_score'] = 0
    for metric, weight in weights.items():
        if f'{metric}_normalized' in scored_df.columns:
            scored_df['weighted_score'] += scored_df[f'{metric}_normalized'] * weight
    
    return scored_df.sort_values('weighted_score', ascending=False) 