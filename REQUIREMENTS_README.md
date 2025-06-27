# AI Benchmark KPI Project - Requirements Setup

This document explains the dependency management and setup process for the AI Benchmark KPI project.

## 📋 Requirements Files Overview

### `requirements.txt` - Complete Dependencies
Comprehensive requirements file covering all project phases:
- **Phase 1**: Database setup and data processing
- **Phase 2**: Bias modeling and data optimization  
- **Phase 3**: Machine learning and prediction models
- **Phase 4**: Neural network performance modeling

### `requirements-essential.txt` - Minimal Dependencies
Essential dependencies needed to run existing functionality:
- Core data science libraries (pandas, numpy, scipy)
- Machine learning basics (scikit-learn)
- Database connectivity (PostgreSQL)
- Basic visualization (matplotlib, seaborn)

### `requirements_phase2.txt` - Phase 2 Specific
Original Phase 2 requirements (legacy file, use essential or full instead)

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
python setup_environment.py
```
This interactive script will:
- Check Python version compatibility
- Create necessary directories
- Install dependencies based on your choice
- Verify the installation

### Option 2: Manual Setup

#### Essential Dependencies Only
```bash
pip install -r requirements-essential.txt
```

#### Full Dependencies (All Phases)
```bash
pip install -r requirements.txt
```

## 📦 Dependency Categories

### Core Data Science (Always Required)
- `pandas>=1.5.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computations
- `scipy>=1.9.0` - Scientific computing

### Machine Learning
- `scikit-learn>=1.1.0` - Traditional ML algorithms
- `xgboost>=1.6.0` - Gradient boosting (Phase 3+)
- `lightgbm>=3.3.0` - Light gradient boosting (Phase 3+)

### Deep Learning (Phase 3/4)
- `tensorflow>=2.8.0` - Neural networks
- `torch>=1.11.0` - PyTorch framework
- `keras>=2.8.0` - High-level neural networks

### Database
- `psycopg2-binary>=2.9.0` - PostgreSQL adapter
- `sqlalchemy>=1.4.0` - Database ORM

### Visualization
- `matplotlib>=3.5.0` - Basic plotting
- `seaborn>=0.11.0` - Statistical visualization
- `plotly>=5.0.0` - Interactive plots (Phase 3+)
- `bokeh>=2.4.0` - Web-based visualization (Phase 3+)

### Web Development (Phase 3/4)
- `flask>=2.0.0` - Lightweight web framework
- `fastapi>=0.75.0` - Modern API framework
- `streamlit>=1.10.0` - Data app framework
- `uvicorn>=0.17.0` - ASGI server

### Development Tools
- `jupyter>=1.0.0` - Interactive notebooks
- `pytest>=7.0.0` - Testing framework
- `sphinx>=4.5.0` - Documentation generation

## 🔧 Troubleshooting

### Common Installation Issues

#### 1. PostgreSQL Adapter (psycopg2) Issues
**Windows:**
```bash
pip install psycopg2-binary
```

**Linux/Mac:**
```bash
# Install PostgreSQL development headers first
sudo apt-get install libpq-dev python3-dev  # Ubuntu/Debian
brew install postgresql  # macOS

pip install psycopg2-binary
```

#### 2. TensorFlow/PyTorch GPU Support
For GPU acceleration (optional):
```bash
# TensorFlow GPU
pip install tensorflow-gpu>=2.8.0

# PyTorch GPU (check pytorch.org for CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Memory Issues During Installation
If you encounter memory issues:
```bash
pip install --no-cache-dir -r requirements.txt
```

### Dependency Conflicts
Check for conflicts:
```bash
pip check
```

Create a virtual environment:
```bash
python -m venv ai_benchmark_env
source ai_benchmark_env/bin/activate  # Linux/Mac
ai_benchmark_env\Scripts\activate     # Windows
pip install -r requirements.txt
```

## 🧪 Environment Verification

After installation, verify your environment:

```bash
python test_environment.py
```

Or manually test key imports:
```python
import pandas as pd
import numpy as np
import sklearn
import psycopg2
import matplotlib.pyplot as plt
print("✅ Environment ready!")
```

## 📊 Project Phase Requirements

### Phase 1 ✅ Complete
- PostgreSQL database setup
- Data cleaning and normalization
- **Required**: pandas, numpy, psycopg2-binary, sqlalchemy

### Phase 2 ✅ Complete  
- Bias modeling and data optimization
- Architecture normalization
- **Required**: + scikit-learn, matplotlib, seaborn, scipy

### Phase 3 🔄 In Progress
- Machine learning prediction models
- Model validation and testing
- **Required**: + xgboost, lightgbm, plotly, joblib

### Phase 4 ⏳ Planned
- Neural network performance modeling
- Production deployment
- **Required**: + tensorflow, torch, flask/fastapi, streamlit

## 🎯 Recommendations

### For Development
Use the full `requirements.txt` to ensure all dependencies are available for future phases.

### For Production Deployment
Create a minimal requirements file with only the dependencies needed for your specific deployment.

### For CI/CD
Use `requirements-essential.txt` for faster builds and testing.

## 📞 Support

If you encounter issues with dependency installation:
1. Check Python version (3.8+ required)
2. Update pip: `pip install --upgrade pip`
3. Use virtual environment
4. Check system-specific installation guides
5. Consult the troubleshooting section above 