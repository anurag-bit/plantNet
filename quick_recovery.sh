#!/bin/bash

echo "🚀 PlantNet Quick Recovery & Training Script"
echo "This will fix dependencies and start memory-safe training"
echo "=" * 60

# Step 1: Fix NumPy compatibility
echo "🔧 Step 1: Fixing NumPy compatibility crisis..."
pip install "numpy==1.26.4" --force-reinstall --quiet
pip install pandas scikit-learn pyarrow bottleneck numexpr seaborn --force-reinstall --quiet

# Step 2: Test imports
echo "🧪 Step 2: Testing critical imports..."
python -c "import numpy; import pandas; import sklearn; import torch; print('✅ All imports working')" || {
    echo "❌ Import test failed - trying alternative fix..."
    pip install "numpy<2.0" pandas scikit-learn --force-reinstall --quiet
}

# Step 3: Check GPU status
echo "🔍 Step 3: Checking GPU status..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Step 4: Start memory-safe training
echo "🚀 Step 4: Starting memory-optimized training..."
echo "Memory settings: Batch=32, Workers=2, Single Model (ResNet50)"
echo "This should avoid OOM kills that happened before"
echo ""

# Make training script executable
chmod +x train_memory_safe.py

# Start training
python train_memory_safe.py

echo ""
echo "🎉 Training script completed!"
echo "Check the results in experiments/memory_optimized/"