#!/usr/bin/env python3
"""
Quick NumPy Compatibility Fix
=============================

This script fixes NumPy compatibility issues by installing compatible versions.
"""

import subprocess
import sys

def run_command(command, description):
    """Run a command with progress indication."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    """Fix NumPy compatibility issues."""
    print("🔧 PlantNet NumPy Compatibility Fix")
    print("=" * 40)
    
    # Step 1: Upgrade NumPy to 2.x (required for OpenCV 4.12+)
    if not run_command(
        'pip install "numpy>=2.0.0,<2.3.0" --upgrade',
        "Upgrading NumPy to 2.x"
    ):
        return False
    
    # Step 2: Fix protobuf compatibility
    if not run_command(
        'pip install "protobuf>=4.21.0,<5.0.0" --force-reinstall',
        "Fixing protobuf compatibility"
    ):
        return False
    
    # Step 3: Install missing dependencies
    if not run_command(
        'pip install "FuzzyTM>=0.4.0"',
        "Installing FuzzyTM for gensim"
    ):
        return False
    
    # Step 4: Reinstall key packages with new NumPy
    packages = [
        ('"pandas>=2.0.0,<2.4.0"', "pandas"),
        ('"scikit-learn>=1.5.0"', "scikit-learn"), 
        ('"matplotlib>=3.8.0"', "matplotlib"),
        ('"seaborn>=0.13.0"', "seaborn"),
        ('"opencv-python>=4.12.0"', "opencv-python")
    ]
    
    for package_spec, package_name in packages:
        if not run_command(
            f'pip install {package_spec} --upgrade',
            f"Upgrading {package_name}"
        ):
            return False
    
    # Step 3: Test the fix
    print("\n🧪 Testing the fix...")
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Pandas: {pd.__version__}")
        print(f"✅ Scikit-learn: {sklearn.__version__}")
        print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
        print(f"✅ Seaborn: {sns.__version__}")
        
        print("\n🎉 All packages working correctly!")
        print("\n🚀 You can now run:")
        print("   python test_implementation.py")
        print("   python test_deployment.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)