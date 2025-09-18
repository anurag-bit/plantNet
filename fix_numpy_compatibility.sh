#!/bin/bash
"""
Fix NumPy Compatibility Issues
==============================

This script fixes the NumPy 2.x compatibility issues by downgrading
to NumPy 1.x which is compatible with the existing environment.
"""

echo "🔧 Fixing NumPy compatibility issues..."
echo "This will upgrade NumPy to 2.x for compatibility with OpenCV 4.12+"

# Upgrade NumPy to 2.x
echo "📦 Upgrading NumPy to 2.x..."
pip install "numpy>=2.0.0,<2.3.0" --upgrade

# Fix protobuf compatibility
echo "📦 Fixing protobuf compatibility..."
pip install "protobuf>=4.21.0,<5.0.0" --force-reinstall

# Install missing dependencies
echo "📦 Installing FuzzyTM for gensim compatibility..."
pip install "FuzzyTM>=0.4.0"

# Upgrade other packages
echo "📦 Upgrading pandas..."
pip install "pandas>=2.0.0,<2.4.0" --upgrade

echo "📦 Upgrading scikit-learn..."
pip install "scikit-learn>=1.5.0" --upgrade

echo "📦 Upgrading matplotlib..."
pip install "matplotlib>=3.8.0" --upgrade

echo "📦 Upgrading seaborn..."
pip install "seaborn>=0.13.0" --upgrade

echo "📦 Upgrading opencv-python..."
pip install "opencv-python>=4.12.0" --upgrade

# Check if the fix worked
echo "✅ Testing the fix..."
python -c "
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}') 
print(f'Scikit-learn: {sklearn.__version__}')
print('✅ All packages imported successfully!')
"

echo "🎉 Fix completed! You can now run the tests."