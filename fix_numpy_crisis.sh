#!/bin/bash

echo "🔧 Fixing NumPy 2.x compatibility crisis..."

# Step 1: Downgrade to NumPy 1.x for compatibility
echo "📉 Downgrading to NumPy 1.26.4 (last stable 1.x version)..."
pip install --force-reinstall "numpy==1.26.4"

# Step 2: Reinstall key packages that need NumPy compatibility
echo "🔄 Reinstalling NumPy-dependent packages..."
pip install --force-reinstall --no-deps \
    pandas \
    scikit-learn \
    seaborn \
    matplotlib \
    scipy

# Step 3: Reinstall PyArrow (major issue source)
echo "🏹 Reinstalling PyArrow..."
pip install --force-reinstall pyarrow

# Step 4: Reinstall bottleneck and numexpr
echo "🍼 Reinstalling bottleneck and numexpr..."
pip install --force-reinstall bottleneck numexpr

# Step 5: Test basic imports
echo "🧪 Testing critical imports..."
python -c "
import numpy as np
print(f'✅ NumPy {np.__version__} working')

import pandas as pd
print(f'✅ Pandas {pd.__version__} working')

import sklearn
print(f'✅ Scikit-learn working')

import seaborn as sns
print(f'✅ Seaborn working')

import pyarrow as pa
print(f'✅ PyArrow {pa.__version__} working')

print('🎉 All packages working with NumPy 1.x!')
"

echo "✅ NumPy compatibility fix complete!"