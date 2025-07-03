# 🚀 AI Benchmark KPI Dashboard

A comprehensive web application for analyzing GPU performance, predicting AI capabilities, and comparing hardware across different vendors and generations.

## 📋 Features

### 🏠 Home Page
- Project overview and key metrics
- Dataset summary with interactive charts
- Recent updates and achievements

### 📊 Dataset Overview
- Comprehensive dataset statistics
- Data quality analysis
- Sample data exploration
- Column descriptions

### 📈 Visualizations
- **Correlation Analysis**: Interactive correlation matrices with multiple methods
- **Performance Distributions**: Histograms, box plots, violin plots
- **Scatter Plot Matrix**: Multi-dimensional data exploration
- **Architecture Comparison**: Performance radar charts and analysis
- **Performance vs Efficiency**: Energy efficiency analysis
- **Price-Performance Analysis**: Value optimization insights
- **Manufacturer Trends**: Market evolution over time

### 🔍 Hardware Comparison
- Advanced filtering by manufacturer, architecture, generation
- Side-by-side GPU comparisons
- Performance benchmarking tools
- Efficiency analysis
- Price-performance optimization
- Architecture deep dives

### 🤖 AI Performance Prediction
- Real-time hardware performance prediction
- Performance and efficiency forecasting
- AI category classification
- Hardware recommendation engine
- Model performance analysis

### ⚡ Efficiency Analysis
- Energy efficiency leaders
- GFLOPS/Watt and TOPs/Watt analysis
- Power consumption optimization
- Efficiency trends and insights

### 🏭 Manufacturer Analysis
- Market share analysis
- Vendor performance comparisons
- Architecture evolution
- Competitive landscape insights

### 🎯 Model Performance
- ML model accuracy metrics
- Feature importance analysis
- Cross-validation results
- Model comparison tools

### 📋 Reports & Insights
- Executive summaries
- Key performance insights
- Trend analysis
- Automated report generation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Required data files in `data/final/` directory

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements-streamlit.txt
   ```

2. **Run the Dashboard**
   ```bash
   python run_streamlit.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the Dashboard**
   - Open your browser to `http://localhost:8501`
   - The dashboard will load automatically

## 📁 Project Structure

```
AI_Benchmark_kpi/
├── streamlit_app.py              # Main Streamlit application
├── run_streamlit.py              # Dashboard runner script
├── requirements-streamlit.txt    # Python dependencies
├── pages/                        # Page modules
│   ├── __init__.py
│   ├── visualizations.py
│   ├── hardware_comparison.py
│   ├── ai_prediction.py
│   ├── efficiency_analysis.py
│   ├── manufacturer_analysis.py
│   ├── model_performance.py
│   └── reports.py
├── data/
│   ├── final/                    # Final datasets
│   └── models/                   # Trained ML models
└── STREAMLIT_README.md          # This file
```

## 📊 Data Requirements

The dashboard expects the following data files:

1. **Main Dataset**: `data/final/Ai-Benchmark-Final-enhanced-fixed.csv`
   - Enhanced GPU dataset with performance metrics
   - Architecture classifications
   - Price and efficiency data

2. **ML Models** (Optional): `data/models/phase3_outputs/`
   - Trained performance prediction models
   - Classification models
   - Preprocessors and encoders

## 🎨 Features Highlights

### Interactive Visualizations
- 📊 **Plotly Charts**: Interactive, zoomable, and exportable
- 🎯 **Real-time Filtering**: Dynamic data exploration
- 📈 **Multi-dimensional Analysis**: Correlation matrices and scatter plots
- 🌈 **Color-coded Insights**: Manufacturer and architecture groupings

### Advanced Analytics
- 🤖 **ML-powered Predictions**: Real-time performance forecasting
- 📊 **Statistical Analysis**: Comprehensive data insights
- 🔍 **Recommendation Engine**: Hardware selection optimization
- 📈 **Trend Analysis**: Historical performance evolution

### User Experience
- 🎨 **Modern UI**: Clean, professional interface
- 📱 **Responsive Design**: Works on desktop and mobile
- ⚡ **Fast Performance**: Optimized data loading and caching
- 🔧 **Customizable**: Extensive filtering and configuration options

## 🔧 Configuration

### Theme Customization
The dashboard supports custom themes through Streamlit configuration:

```toml
[theme]
base = "light"
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f0f0"
textColor = "#262730"
```

### Data Sources
Update data paths in `streamlit_app.py`:
```python
# Main dataset path
MAIN_DATASET_PATH = 'data/final/Ai-Benchmark-Final-enhanced-fixed.csv'

# Models directory
MODELS_DIR = 'data/models/phase3_outputs'
```

## 🚀 Usage Examples

### Performance Analysis
1. Navigate to **📈 Visualizations**
2. Select "Performance Distributions"
3. Choose feature and grouping options
4. Analyze patterns and insights

### Hardware Comparison
1. Go to **🔍 Hardware Comparison**
2. Use sidebar filters to narrow selection
3. Compare specifications and performance
4. Generate recommendations

### AI Prediction
1. Visit **🤖 AI Performance Prediction**
2. Input hardware specifications
3. Get real-time performance predictions
4. Compare with existing hardware

## 🤝 Contributing

To extend the dashboard:

1. **Add New Pages**: Create modules in `pages/` directory
2. **Extend Visualizations**: Add new chart types in `visualizations.py`
3. **Improve Predictions**: Enhance models in `ai_prediction.py`
4. **Custom Analytics**: Add analysis functions to relevant pages

## 📞 Support

For issues or questions:
- Check the console for error messages
- Ensure all data files are present
- Verify Python dependencies are installed
- Review the Streamlit logs for debugging

## 🎯 Performance Tips

- **Data Caching**: Large datasets are cached for faster loading
- **Incremental Updates**: Use filters to work with subsets
- **Browser Performance**: Chrome/Firefox recommended for best experience
- **Memory Usage**: Close unused tabs for optimal performance

## 📈 Roadmap

Future enhancements:
- 🔄 **Real-time Data**: Live data feeds and updates
- 📊 **Advanced ML**: Deep learning performance models
- 🌐 **Multi-language**: International localization
- 📱 **Mobile App**: Native mobile applications
- 🔗 **API Integration**: External data sources

---

**🚀 Ready to explore your AI hardware data? Run the dashboard and start discovering insights!** 