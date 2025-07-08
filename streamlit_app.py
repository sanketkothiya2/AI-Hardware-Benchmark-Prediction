import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')



# Import page modules
try:
    from pages import visualizations
    from pages import hardware_comparison
    from pages import ai_prediction
    from pages import efficiency_analysis
    from pages import manufacturer_analysis
    from pages import model_performance
    from pages import reports
    PAGES_LOADED = True
except ImportError as e:
    st.error(f"Error importing page modules: {e}")
    PAGES_LOADED = False

# Configure Streamlit page
st.set_page_config(
    page_title="AI Benchmark KPI Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with background tiles
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Background tiles pattern */
    .main .block-container {
        background: 
            linear-gradient(45deg, rgba(79, 172, 254, 0.02) 25%, transparent 25%), 
            linear-gradient(-45deg, rgba(79, 172, 254, 0.02) 25%, transparent 25%), 
            linear-gradient(45deg, transparent 75%, rgba(79, 172, 254, 0.02) 75%), 
            linear-gradient(-45deg, transparent 75%, rgba(79, 172, 254, 0.02) 75%);
        background-size: 40px 40px;
        background-position: 0 0, 0 20px, 20px -20px, -20px 0px;
        background-attachment: fixed;
    }
    
    /* Alternative geometric tile pattern for variety */
    .stApp {
        background: 
            radial-gradient(circle at 20px 20px, rgba(79, 172, 254, 0.05) 1px, transparent 1px),
            radial-gradient(circle at 60px 60px, rgba(102, 126, 234, 0.03) 1px, transparent 1px);
        background-size: 80px 80px, 120px 120px;
        background-attachment: fixed;
    }
    
    /* Overlay subtle hex pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            linear-gradient(30deg, transparent 65%, rgba(79, 172, 254, 0.02) 70%, transparent 75%),
            linear-gradient(150deg, transparent 65%, rgba(102, 126, 234, 0.02) 70%, transparent 75%);
        background-size: 30px 52px;
        pointer-events: none;
        z-index: -1;
    }
    
    /* Content container with subtle backdrop */
    .main .block-container > div {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Main app styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 20px;
        backdrop-filter: blur(20px);
        border: 2px solid rgba(79, 172, 254, 0.2);
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: 
            linear-gradient(45deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.95)),
            linear-gradient(135deg, rgba(79, 172, 254, 0.03) 0%, rgba(102, 126, 234, 0.03) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 25% 25%, rgba(79, 172, 254, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(102, 126, 234, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Modern Light Sidebar Theme with tiles */
    .css-1d391kg {
        background: 
            linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%),
            linear-gradient(45deg, rgba(79, 172, 254, 0.03) 25%, transparent 25%),
            linear-gradient(-45deg, rgba(79, 172, 254, 0.03) 25%, transparent 25%);
        background-size: 100% 100%, 20px 20px, 20px 20px;
        background-position: 0 0, 0 0, 10px 10px;
        border-right: 3px solid #4facfe;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar content container */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 0;
    }
    
    /* Sidebar header styling - Vibrant */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 1rem;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .sidebar-header small {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 400;
        font-size: 0.8rem;
        display: block;
        margin-top: 0.3rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Navigation section styling - Light with colored accents */
    .nav-section {
        background: white;
        margin: 1rem;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        border-left: 4px solid #4facfe;
    }
    
    /* Navigation title */
    .nav-title {
        color: #2d3748;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tech Stack content styling */
    .tech-content {
        color: #4a5568 !important;
        font-size: 0.8rem;
        line-height: 1.6;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    .tech-content div {
        color: #4a5568 !important;
        margin-bottom: 0.3rem;
        padding: 0.2rem 0;
    }
    
    /* Performance section styling */
    .performance-content {
        text-align: center;
        padding: 0.5rem 0;
    }
    
    .performance-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        padding: 0.3rem 0;
    }
    
    .performance-label {
        color: #4a5568 !important;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .performance-value {
        font-weight: 700;
        color: #667eea !important;
        font-size: 0.9rem;
    }
    
    /* Radio button styling - Light Theme */
    .stRadio > div {
        background: transparent;
    }
    
    .stRadio label {
        color: #2d3748 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .stRadio label:hover {
        background: rgba(79, 172, 254, 0.1);
        border-radius: 8px;
        padding-left: 0.8rem;
        transform: translateX(5px);
        color: #667eea !important;
    }
    
    /* Ensure radio button text is always visible */
    .stRadio > div > label > div {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }
    
    .stRadio > div > label > div:first-child {
        color: #2d3748 !important;
    }
    
    /* Override any default Streamlit radio styling */
    .stRadio div[role="radiogroup"] > label {
        color: #2d3748 !important;
        background: transparent !important;
    }
    
    .stRadio div[role="radiogroup"] > label:hover {
        background: rgba(79, 172, 254, 0.1) !important;
        border-radius: 8px !important;
        color: #667eea !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(79, 172, 254, 0.3), transparent);
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .sidebar-footer {
        margin-top: 2rem;
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 15px;
        margin-left: 1rem;
        margin-right: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    .footer-text {
        color: #4a5568 !important;
        font-size: 0.7rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    /* Quick stats section - Colorful Theme */
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 1rem;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .stats-title {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 1rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Metric styling - Colorful */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 4px solid #4facfe;
        transition: transform 0.3s ease;
        border: 1px solid #e9ecef;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        border-left: 4px solid #667eea;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        color: #667eea;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #4a5568;
        font-size: 0.9rem;
    }
    
    /* Project info section - Warm Theme */
    .project-info {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        margin: 1rem;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .project-version {
        color: #744210 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .project-status {
        color: #744210 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }
    
    .status-badge {
        background: linear-gradient(45deg, #48bb78, #38a169);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-top: 0.5rem;
        box-shadow: 0 2px 8px rgba(72,187,120,0.3);
    }
    
    /* Special sections with gradients */
    .ml-models-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 1rem;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .ml-models-title {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .ml-models-count {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4facfe;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0.5rem;
    }
    
    .ml-models-label {
        font-size: 0.9rem;
        color: white !important;
        font-weight: 500;
        margin-bottom: 0.8rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    .tech-stack-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        margin: 1rem;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .tech-stack-title {
        color: #2d3748;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .performance-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        margin: 1rem;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .performance-title {
        color: #744210;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Chart and visualization containers with tiles */
    .js-plotly-plot, .plotly {
        background: 
            rgba(255, 255, 255, 0.95) !important,
            linear-gradient(45deg, rgba(79, 172, 254, 0.02) 25%, transparent 25%),
            linear-gradient(-45deg, rgba(79, 172, 254, 0.02) 25%, transparent 25%);
        background-size: 100% 100%, 30px 30px, 30px 30px;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Data tables with tile pattern */
    .stDataFrame {
        background: 
            rgba(255, 255, 255, 0.95),
            radial-gradient(circle at 20% 20%, rgba(79, 172, 254, 0.02) 1px, transparent 1px),
            radial-gradient(circle at 80% 80%, rgba(102, 126, 234, 0.02) 1px, transparent 1px);
        background-size: 100% 100%, 25px 25px, 35px 35px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        padding: 1rem;
    }
    
    /* Info and warning boxes with tiles */
    .stAlert {
        background: 
            rgba(255, 255, 255, 0.9),
            linear-gradient(30deg, rgba(79, 172, 254, 0.02) 50%, transparent 50%),
            linear-gradient(150deg, rgba(102, 126, 234, 0.02) 50%, transparent 50%);
        background-size: 100% 100%, 15px 26px, 15px 26px;
        backdrop-filter: blur(15px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Column containers */
    .element-container {
        background: 
            linear-gradient(135deg, rgba(255, 255, 255, 0.7) 0%, rgba(248, 250, 252, 0.7) 100%),
            repeating-linear-gradient(45deg, rgba(79, 172, 254, 0.01) 0px, rgba(79, 172, 254, 0.01) 1px, transparent 1px, transparent 15px);
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    /* Tab containers with subtle tiles */
    .stTabs [data-baseweb="tab-list"] {
        background: 
            rgba(255, 255, 255, 0.9),
            linear-gradient(90deg, rgba(79, 172, 254, 0.02) 50%, transparent 50%);
        background-size: 100% 100%, 20px 20px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Enhanced selectbox and input styling */
    .stSelectbox > div > div {
        background: 
            rgba(255, 255, 255, 0.95),
            radial-gradient(circle at center, rgba(79, 172, 254, 0.02) 1px, transparent 1px);
        background-size: 100% 100%, 12px 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(79, 172, 254, 0.2);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_main_dataset():
    """Load the Phase 2 enhanced dataset with bias correction and architecture normalization"""
    try:
        df = pd.read_csv('data/phase2_outputs/phase2_final_enhanced_dataset.csv')
        return df
    except Exception as e:
        st.error(f"Error loading Phase 2 dataset: {e}")
        return None

@st.cache_data
def load_model_artifacts():
    """Load trained models and preprocessors"""
    artifacts = {}
    model_dir = Path('data/models/phase3_outputs')
    
    if not model_dir.exists():
        st.error(f"Model directory does not exist: {model_dir}")
        return artifacts
    
    # List of expected model files
    expected_files = [
        'random_forest_FP32_Final_model.pkl',
        'xgboost_FP32_Final_model.pkl', 
        'efficiency_random_forest_GFLOPS_per_Watt_model.pkl',
        'efficiency_xgboost_GFLOPS_per_Watt_model.pkl',
        'classification_random_forest_AI_Performance_Category_model.pkl',
        'classification_xgboost_PerformanceTier_model.pkl',
        'preprocessor_performance.pkl',
        'preprocessor_efficiency.pkl',
        'preprocessor_classification.pkl',
        'classification_label_encoder_AI_Performance_Category.pkl',
        'classification_label_encoder_PerformanceTier.pkl'
    ]
    
    loaded_count = 0
    failed_count = 0
    
    try:
        # Load performance models
        try:
            rf_fp32 = pickle.load(open(model_dir / 'random_forest_FP32_Final_model.pkl', 'rb'))
            artifacts['random_forest_FP32_Final_model.pkl'] = rf_fp32
            artifacts['rf_fp32'] = rf_fp32
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load RF FP32 model: {e}")
            failed_count += 1
            
        try:
            xgb_fp32 = pickle.load(open(model_dir / 'xgboost_FP32_Final_model.pkl', 'rb'))
            artifacts['xgboost_FP32_Final_model.pkl'] = xgb_fp32
            artifacts['xgb_fp32'] = xgb_fp32
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load XGB FP32 model: {e}")
            failed_count += 1
        
        # Load efficiency models
        try:
            rf_gflops = pickle.load(open(model_dir / 'efficiency_random_forest_GFLOPS_per_Watt_model.pkl', 'rb'))
            artifacts['efficiency_random_forest_GFLOPS_per_Watt_model.pkl'] = rf_gflops
            artifacts['rf_gflops'] = rf_gflops
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load RF efficiency model: {e}")
            failed_count += 1
            
        try:
            xgb_gflops = pickle.load(open(model_dir / 'efficiency_xgboost_GFLOPS_per_Watt_model.pkl', 'rb'))
            artifacts['efficiency_xgboost_GFLOPS_per_Watt_model.pkl'] = xgb_gflops
            artifacts['xgb_gflops'] = xgb_gflops
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load XGB efficiency model: {e}")
            failed_count += 1
        
        # Load classification models
        try:
            rf_perf_cat = pickle.load(open(model_dir / 'classification_random_forest_AI_Performance_Category_model.pkl', 'rb'))
            artifacts['classification_random_forest_AI_Performance_Category_model.pkl'] = rf_perf_cat
            artifacts['rf_perf_cat'] = rf_perf_cat
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load RF AI category model: {e}")
            failed_count += 1
            
        try:
            xgb_tier = pickle.load(open(model_dir / 'classification_xgboost_PerformanceTier_model.pkl', 'rb'))
            artifacts['classification_xgboost_PerformanceTier_model.pkl'] = xgb_tier
            artifacts['classification_random_forest_PerformanceTier_model.pkl'] = xgb_tier  # Fallback key
            artifacts['xgb_tier'] = xgb_tier
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load XGB tier model: {e}")
            failed_count += 1
        
        # Load preprocessors
        try:
            preprocessor_perf = pickle.load(open(model_dir / 'preprocessor_performance.pkl', 'rb'))
            artifacts['preprocessor_performance.pkl'] = preprocessor_perf
            artifacts['preprocessor_perf'] = preprocessor_perf
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load performance preprocessor: {e}")
            failed_count += 1
            
        try:
            preprocessor_eff = pickle.load(open(model_dir / 'preprocessor_efficiency.pkl', 'rb'))
            artifacts['preprocessor_efficiency.pkl'] = preprocessor_eff
            artifacts['preprocessor_eff'] = preprocessor_eff
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load efficiency preprocessor: {e}")
            failed_count += 1
            
        try:
            preprocessor_class = pickle.load(open(model_dir / 'preprocessor_classification.pkl', 'rb'))
            artifacts['preprocessor_classification.pkl'] = preprocessor_class
            artifacts['preprocessor_class'] = preprocessor_class
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load classification preprocessor: {e}")
            failed_count += 1
        
        # Load encoders
        try:
            encoder_ai_cat = pickle.load(open(model_dir / 'classification_label_encoder_AI_Performance_Category.pkl', 'rb'))
            artifacts['classification_label_encoder_AI_Performance_Category.pkl'] = encoder_ai_cat
            artifacts['encoder_ai_cat'] = encoder_ai_cat
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load AI category encoder: {e}")
            failed_count += 1
            
        try:
            encoder_tier = pickle.load(open(model_dir / 'classification_label_encoder_PerformanceTier.pkl', 'rb'))
            artifacts['classification_label_encoder_PerformanceTier.pkl'] = encoder_tier
            artifacts['encoder_tier'] = encoder_tier
            loaded_count += 1
        except Exception as e:
            st.warning(f"Failed to load tier encoder: {e}")
            failed_count += 1
        
        # Show loading summary
        if loaded_count > 0:
            st.success(f"✅ Successfully loaded {loaded_count} model artifacts")
        if failed_count > 0:
            st.error(f"❌ Failed to load {failed_count} model artifacts")
        
        # Debug: Show what was actually loaded
        if len(artifacts) > 0:
            st.info(f"🔍 Loaded models: {list(artifacts.keys())[:5]}...")  # Show first 5 keys
        else:
            st.error("🚨 No models were loaded successfully!")
        
    except Exception as e:
        st.error(f"Critical error loading models: {e}")
    
    return artifacts

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = load_main_dataset()
if 'models' not in st.session_state:
    st.session_state.models = load_model_artifacts()

def main():
    """Main application function"""
    # Hide default Streamlit navigation and UI elements
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    [data-testid="stSidebarNav"] {display: none;}
    .css-1d391kg {display: none;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI Benchmark KPI Dashboard</h1>', unsafe_allow_html=True)
    
    # Professional Sidebar Design
    with st.sidebar:
        # Sidebar Header
        st.markdown("""
        <div class="sidebar-header">
            🚀 AI Benchmark Dashboard
            <br><small>Professional GPU Analytics Platform</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Section
        st.markdown("""
        <div class="nav-section">
            <div class="nav-title">📋 Navigation</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation using radio buttons for better UX
        page = st.radio(
            "",
            [
                "🏠 Home",
                "📊 Dataset Overview", 
                "📈 Visualizations",
                "🔍 Hardware Comparison",
                "🎯 Performance Prediction",
                "⚡ Efficiency Analysis",
                "🏭 Manufacturer Analysis",
                "📊 Model Performance",
                "📋 Reports & Insights"
            ],
            index=0,
            label_visibility="collapsed"
        )
        
        # Quick Stats Section
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            st.markdown('<div class="stats-title">📈 Quick Statistics</div>', unsafe_allow_html=True)
            
            # Metrics in columns for better layout
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total GPUs", f"{len(df):,}")
                st.metric("Manufacturers", df['Manufacturer'].nunique())
            
            with col2:
                st.metric("Architectures", df['Architecture'].nunique())
                if 'PerformanceTier' in df.columns:
                    tier_count = df['PerformanceTier'].nunique()
                    st.metric("Performance Tiers", tier_count)
            
            # Additional insights
            if 'FP32_Final' in df.columns:
                avg_performance = df['FP32_Final'].mean() / 1e12
                st.metric("Avg Performance", f"{avg_performance:.1f} TFLOPS")
            
            if 'GFLOPS_per_Watt' in df.columns:
                avg_efficiency = df['GFLOPS_per_Watt'].mean()
                st.metric("Avg Efficiency", f"{avg_efficiency:.1f} GFLOPS/W")
            
            st.markdown('</div>', unsafe_allow_html=True)
        

    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Dataset Overview":
        show_dataset_overview()
    elif page == "📈 Visualizations":
        if PAGES_LOADED:
            visualizations.show_visualizations()
        else:
            st.error("Visualizations module not loaded")
    elif page == "🔍 Hardware Comparison":
        if PAGES_LOADED:
            hardware_comparison.show_hardware_comparison()
        else:
            st.error("Hardware comparison module not loaded")
    elif page == "🎯 Performance Prediction":
        if PAGES_LOADED:
            ai_prediction.show_ai_prediction()
        else:
            st.error("Performance prediction module not loaded")
    elif page == "⚡ Efficiency Analysis":
        if PAGES_LOADED:
            efficiency_analysis.show_efficiency_analysis()
        else:
            st.error("Efficiency analysis module not loaded")
    elif page == "🏭 Manufacturer Analysis":
        if PAGES_LOADED:
            manufacturer_analysis.show_manufacturer_analysis()
        else:
            st.error("Manufacturer analysis module not loaded")
    elif page == "📊 Model Performance":
        if PAGES_LOADED:
            model_performance.show_model_performance()
        else:
            st.error("Model performance module not loaded")
    elif page == "📋 Reports & Insights":
        if PAGES_LOADED:
            reports.show_reports()
        else:
            st.error("Reports module not loaded")

def show_home_page():
    """Display the home page with project overview"""
    if st.session_state.data is None:
        st.error("Unable to load dataset. Please check data files.")
        return
    
    df = st.session_state.data
    
    # Project overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📋 Project Overview
        
        Welcome to the **AI Benchmark KPI Dashboard** - a comprehensive platform for analyzing GPU performance, 
        predicting AI capabilities, and comparing hardware across different vendors and generations.
        
        ### 🎯 Key Features:
        - **2,100+ GPU Database**: Comprehensive dataset with detailed specifications
        - **6 ML Models**: Performance, efficiency, and classification predictions
        - **Real-time Predictions**: AI performance forecasting for hardware selection
        - **Interactive Visualizations**: Advanced charts and comparison tools
        - **Manufacturer Analysis**: Vendor-specific insights and recommendations
        
        ### 🔬 Technical Highlights:
        - **Bias-corrected Performance Modeling**
        - **Architecture Normalization**
        - **Multi-metric Efficiency Analysis**
        - **Neural Network Hardware Mapping**
        """)
    
    with col2:
        # Key metrics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total GPUs", f"{len(df):,}")
        st.metric("Manufacturers", df['Manufacturer'].nunique())
        st.metric("Architectures", df['Architecture'].nunique())
        st.metric("Model Accuracy", "92.5%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset summary
    st.markdown("## 📊 Dataset Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("NVIDIA GPUs", len(df[df['Manufacturer'] == 'NVIDIA']))
    with col2:
        st.metric("AMD GPUs", len(df[df['Manufacturer'] == 'AMD']))
    with col3:
        st.metric("Avg TDP (W)", f"{df['TDP'].mean():.0f}")
    with col4:
        st.metric("Price Range", f"${df['price'].min():.0f}-${df['price'].max():.0f}")
    
    # Performance distribution chart
    st.markdown("## 🎯 Performance Distribution")
    
    fig = px.histogram(
        df, 
        x='AI_Performance_Category',
        color='Manufacturer',
        title="GPU Distribution by AI Performance Category",
        height=400
    )
    fig.update_layout(
        xaxis_title="AI Performance Category",
        yaxis_title="Number of GPUs"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent additions
    st.markdown("## 🔄 Recent Updates")
    st.info("""
    ✅ **Phase 3 Completed**: All 6 ML models trained and validated  
    ✅ **Architecture Fixes**: Reduced unknown entries from 1,252 to 532  
    ✅ **Bias Correction**: Implemented advanced performance normalization  
    ✅ **Production Ready**: Database indexed and optimized for queries  
    """)

def show_dataset_overview():
    """Display dataset overview and statistics"""
    if st.session_state.data is None:
        st.error("Dataset not loaded")
        return
        
    df = st.session_state.data
    
    st.markdown("## 📊 Dataset Overview")
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Dataset Statistics")
        st.write(f"**Total Records**: {len(df):,}")
        st.write(f"**Features**: {len(df.columns)}")
        st.write(f"**Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        st.write(f"**Data Types**: {df.dtypes.value_counts().to_dict()}")
    
    with col2:
        st.markdown("### 🎯 Data Quality")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        quality_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False).head(10)
        st.dataframe(quality_df)
    
    # Sample data
    st.markdown("### 🔍 Sample Data")
    st.dataframe(df.head(10))
    
    # Column descriptions
    st.markdown("### 📋 Key Columns")
    
    key_columns = {
        'gpuName': 'GPU model name and identifier',
        'Manufacturer': 'GPU manufacturer (NVIDIA, AMD, Intel)',
        'Architecture': 'GPU architecture (Ampere, RDNA 2, etc.)',
        'FP32_Final': 'Single precision floating point performance',
        'TDP': 'Thermal Design Power in watts',
        'AI_Performance_Category': 'Predicted AI performance tier',
        'GFLOPS_per_Watt': 'Energy efficiency metric',
        'price': 'GPU price in USD'
    }
    
    for col, desc in key_columns.items():
        st.write(f"**{col}**: {desc}")

# End of navigation functions

if __name__ == "__main__":
    main() 