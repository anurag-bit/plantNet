# PlantNet Quick Start Guide

## 🚀 Quick Start

### 1. Verify Installation
```bash
python test_implementation.py
```
This should show all tests passing ✅

### 2. Prepare Dataset
Organize your PlantVillage dataset:
```
data/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── Tomato___Bacterial_spot/
│   └── ... (38 classes total)
├── validation/
│   ├── Apple___Apple_scab/
│   └── ...
└── test/ (optional)
```

### 3. Start Training

**Option A: Quick Start (Recommended)**
```bash
# ResNet50 with transfer learning (fastest convergence)
python train.py --model_type resnet --epochs 50 --batch_size 32
```

**Option B: Custom CNN**
```bash
# Custom CNN from scratch
python train.py --model_type custom_cnn --epochs 100 --learning_rate 0.001
```

**Option C: Configuration File**
```bash
# Create and edit configuration
python train.py --create-config
# Edit config_sample.json as needed
python train.py --config config_sample.json
```

### 4. Monitor Training
```bash
# In another terminal
tensorboard --logdir results/
# Open http://localhost:6006
```

### 5. Make Predictions
```bash
# After training completes
python inference.py --model results/experiment_name/best_model.pth --image path/to/leaf.jpg
```

## 📊 Expected Results

**Training Time**: 
- ResNet50: ~2-4 hours (50 epochs, GPU)
- Custom CNN: ~4-8 hours (100 epochs, GPU)

**Accuracy**:
- ResNet50 (pretrained): 85-95%
- Custom CNN: 80-90%
- Top-5 Accuracy: >95%

## 🛠 Troubleshooting

**Out of Memory?**
```bash
python train.py --batch_size 16 --img_size 224
```

**Slow Training?**
```bash
python train.py --num_workers 0  # Disable multiprocessing
```

**Low Accuracy?**
- Use pretrained ResNet: `--model_type resnet --pretrained`
- Increase epochs: `--epochs 100`
- Enable class weights: `--use_class_weights`

## 📁 Results Structure
After training, check `results/experiment_name/`:
```
results/experiment_name/
├── best_model.pth           # Best model checkpoint
├── final_model.pth          # Final model checkpoint
├── config.json              # Training configuration
├── class_names.json         # Class mapping
├── training_history.json    # Training metrics
├── training_history.png     # Training plots
├── tensorboard_logs/        # TensorBoard logs
├── test_evaluation/         # Test results (if available)
│   ├── confusion_matrix.png
│   ├── classification_report_heatmap.png
│   └── evaluation_metrics.json
└── predictions/             # Sample predictions
    └── sample_predictions.png
```

## 🔧 Advanced Usage

### Custom Configuration
```json
{
  "model_type": "resnet",
  "model_name": "resnet50",
  "pretrained": true,
  "epochs": 100,
  "batch_size": 64,
  "learning_rate": 0.0001,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "early_stopping": true,
  "use_class_weights": true
}
```

### Hyperparameter Tuning
```bash
# Learning Rate
python train.py --learning_rate 0.0001  # Lower for fine-tuning
python train.py --learning_rate 0.01    # Higher for training from scratch

# Batch Size (adjust based on GPU memory)
python train.py --batch_size 64   # 8GB+ GPU
python train.py --batch_size 32   # 4-8GB GPU
python train.py --batch_size 16   # 2-4GB GPU

# Model Architecture
python train.py --model_type custom_cnn    # Custom architecture
python train.py --model_type resnet        # ResNet (recommended)
python train.py --model_type efficientnet  # EfficientNet (requires timm)
```

## 📈 Performance Optimization

### GPU Training
- Automatically detected if available
- Use CUDA 11.0+ for best performance
- Monitor GPU usage: `nvidia-smi`

### CPU Training
- Set `num_workers` to your CPU core count
- Use smaller batch sizes (8-16)
- Consider using smaller models

### Memory Optimization
```bash
# Reduce memory usage
python train.py --img_size 224 --batch_size 16 --num_workers 2
```

## 🎯 Next Steps

1. **Experiment with architectures**: Try ResNet18/34 for faster training
2. **Data augmentation**: Modify transforms in `utils/dataset.py`
3. **Transfer learning**: Fine-tune from different pretrained models
4. **Ensemble methods**: Combine multiple models for better accuracy
5. **Mobile deployment**: Convert to ONNX/TensorRT for mobile apps

## 📚 Learning Resources

- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Transfer Learning**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **Computer Vision**: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

Happy training! 🌱📱