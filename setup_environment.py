#!/usr/bin/env python3
"""
AI Benchmark KPI Project Environment Setup Script

This script helps set up the development environment for the AI Benchmark KPI project.
It provides options for installing essential dependencies or full dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        print(f"   Output: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8+ is required")
        return False
    
    print("   ✅ Python version is compatible")
    return True

def install_requirements(requirements_file):
    """Install requirements from a file"""
    if not Path(requirements_file).exists():
        print(f"   ❌ Requirements file not found: {requirements_file}")
        return False
    
    return run_command(
        f"pip install -r {requirements_file}",
        f"Installing dependencies from {requirements_file}"
    )

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/interim",
        "data/final",
        "data/phase2_outputs",
        "analysis/reports",
        "analysis/visualizations",
        "output",
        "logs"
    ]
    
    print("\n📁 Setting up project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def main():
    """Main setup function"""
    print("🚀 AI Benchmark KPI Project Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Ask user what to install
    print("\n📦 Choose installation option:")
    print("   1. Essential dependencies only (recommended for getting started)")
    print("   2. Full dependencies (includes ML, web frameworks, neural networks)")
    print("   3. Skip dependency installation")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        success = install_requirements("requirements-essential.txt")
    elif choice == "2":
        success = install_requirements("requirements.txt")
    elif choice == "3":
        print("   ⏭️  Skipping dependency installation")
        success = True
    else:
        print("   ❌ Invalid choice")
        sys.exit(1)
    
    if not success:
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    verification_commands = [
        ("python -c \"import pandas; print('Pandas:', pandas.__version__)\"", "Pandas"),
        ("python -c \"import numpy; print('NumPy:', numpy.__version__)\"", "NumPy"),
        ("python -c \"import sklearn; print('Scikit-learn:', sklearn.__version__)\"", "Scikit-learn"),
        ("python -c \"import psycopg2; print('PostgreSQL adapter: OK')\"", "PostgreSQL adapter")
    ]
    
    for command, name in verification_commands:
        if not run_command(command, f"Verifying {name}"):
            print(f"   ⚠️  {name} verification failed (might be optional)")
    
    print("\n🎉 Environment setup completed!")
    print("\n📋 Next Steps:")
    print("   1. Ensure PostgreSQL is running with the database 'AI_BENCHMARK'")
    print("   2. Run Phase 2 analysis: python scripts/phase2_modeling/run_phase2_analysis.py")
    print("   3. Check project status: python -c \"from scripts.utils.analyze_ai_benchmark_dataset import analyze_ai_benchmark_dataset; analyze_ai_benchmark_dataset()\"")
    
    # Create a simple test script
    test_script = """
# Quick test to verify environment
import pandas as pd
import numpy as np
import sklearn
print("✅ Environment is ready!")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
"""
    
    with open("test_environment.py", "w") as f:
        f.write(test_script)
    
    print("   4. Test environment: python test_environment.py")

if __name__ == "__main__":
    main() 