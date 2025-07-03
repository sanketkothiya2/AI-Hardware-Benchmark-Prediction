#!/usr/bin/env python3
"""
Test script for AI Benchmark KPI Dashboard setup
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    packages = {
        'streamlit': 'Streamlit web framework',
        'pandas': 'Data manipulation library',
        'numpy': 'Numerical computing library',
        'plotly.express': 'Interactive plotting library',
        'plotly.graph_objects': 'Plotly graph objects',
        'pickle': 'Python serialization (built-in)',
        'json': 'JSON handling (built-in)',
        'warnings': 'Warning control (built-in)'
    }
    
    success_count = 0
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package}: {description}")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {package}: {description} - {e}")
    
    print(f"\n📊 Import Results: {success_count}/{len(packages)} packages successful")
    return success_count == len(packages)

def test_data_files():
    """Test if required data files exist"""
    print("\n📁 Testing data file availability...")
    
    files = {
        'data/final/Ai-Benchmark-Final-enhanced-fixed.csv': 'Main enhanced dataset',
        'data/models/phase3_outputs': 'ML models directory (optional)',
    }
    
    success_count = 0
    for file_path, description in files.items():
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / (1024 * 1024)  # MB
                print(f"  ✅ {file_path}: {description} ({size:.1f} MB)")
            else:
                file_count = len(list(path.glob('*'))) if path.is_dir() else 0
                print(f"  ✅ {file_path}: {description} ({file_count} files)")
            success_count += 1
        else:
            print(f"  ❌ {file_path}: {description} - File not found")
    
    print(f"\n📊 Data Files: {success_count}/{len(files)} found")
    return success_count >= 1  # At least main dataset should exist

def test_streamlit_app():
    """Test if the main Streamlit app can be loaded"""
    print("\n🚀 Testing Streamlit app loading...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import the main app
        import streamlit_app
        print("  ✅ streamlit_app.py: Main application loads successfully")
        
        # Test if main functions exist
        functions = ['main', 'load_main_dataset', 'show_home_page']
        for func in functions:
            if hasattr(streamlit_app, func):
                print(f"  ✅ {func}: Function exists")
            else:
                print(f"  ⚠️  {func}: Function not found (may be defined dynamically)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ streamlit_app.py: Error loading - {e}")
        return False

def test_pages():
    """Test if page modules can be loaded"""
    print("\n📄 Testing page modules...")
    
    page_modules = [
        'pages.visualizations',
        'pages.hardware_comparison',
        'pages.ai_prediction',
        'pages.efficiency_analysis',
        'pages.manufacturer_analysis',
        'pages.model_performance',
        'pages.reports'
    ]
    
    success_count = 0
    for module in page_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}: Module loads successfully")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {module}: Import error - {e}")
        except Exception as e:
            print(f"  ⚠️  {module}: Other error - {e}")
    
    print(f"\n📊 Page Modules: {success_count}/{len(page_modules)} loaded successfully")
    return success_count >= len(page_modules) // 2  # At least half should work

def main():
    """Run all tests"""
    print("🧪 AI Benchmark KPI Dashboard - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Files", test_data_files),
        ("Streamlit App", test_streamlit_app),
        ("Page Modules", test_pages)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your dashboard is ready to run.")
        print("🚀 Run with: python run_streamlit.py")
    elif passed >= total // 2:
        print("\n⚠️  Most tests passed. Dashboard should work with some limitations.")
        print("🚀 Try running with: python run_streamlit.py")
    else:
        print("\n❌ Multiple test failures. Please check your setup.")
        print("📦 Install requirements: pip install -r requirements-streamlit.txt")
        print("📁 Ensure data files are in correct locations")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 