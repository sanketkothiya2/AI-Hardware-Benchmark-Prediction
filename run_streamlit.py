#!/usr/bin/env python3
"""
AI Benchmark KPI Dashboard Runner

This script sets up and runs the Streamlit dashboard for the AI Benchmark KPI project.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'seaborn', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements-streamlit.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_data():
    """Check if required data files exist"""
    required_files = [
        'data/final/Ai-Benchmark-Final-enhanced-fixed.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required data files found")
    return True

def run_streamlit():
    """Run the Streamlit application"""
    try:
        print("🚀 Starting AI Benchmark KPI Dashboard...")
        print("📊 Dashboard will open in your browser automatically")
        print("🔗 URL: http://localhost:8501")
        print("\n⏹️  Press Ctrl+C to stop the server\n")
        
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--theme.base', 'light',
            '--theme.primaryColor', '#1f77b4',
            '--theme.backgroundColor', '#ffffff',
            '--theme.secondaryBackgroundColor', '#f0f0f0'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped. Thank you for using AI Benchmark KPI Dashboard!")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def main():
    """Main function"""
    print("🚀 AI Benchmark KPI Dashboard")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check data files
    if not check_data():
        print("\n💡 Please ensure your data files are in the correct location")
        sys.exit(1)
    
    # Run the dashboard
    run_streamlit()

if __name__ == "__main__":
    main() 