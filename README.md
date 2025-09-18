# PlantNet: AMD MI300X Optimized Plant Disease Detection System

**High-Performance Plant Disease Detection System optimized for AMD MI300X GPU**

## 🚀 System Overview

PlantNet is a production-ready plant disease detection system that leverages the full potential of AMD MI300X hardware (192GB VRAM) to deliver state-of-the-art accuracy and performance. The system uses advanced ensemble models, mixed precision training, and comprehensive disease analysis to provide farmers and agricultural experts with accurate diagnoses and treatment recommendations.

### 🎯 Key Features

- **🧠 Advanced Ensemble Models**: ResNet101, EfficientNet-B4, Vision Transformer, Swin Transformer
- **⚡ MI300X Optimization**: Utilizes 192GB VRAM with large batch sizes (128-256)
- **🎨 Advanced Augmentations**: MixUp, CutMix, AutoAugment, Albumentations
- **🔍 Test-Time Augmentation**: 8 augmentation strategies for robust inference  
- **📊 Multi-Scale Processing**: 3 different scales for comprehensive analysis
- **🎯 Uncertainty Estimation**: Confidence scoring for prediction reliability
- **🖼️ GradCAM Visualization**: Visual explanations for model decisions
- **📚 Disease Database**: Comprehensive information for 38+ plant diseases
- **💾 Mixed Precision**: BFloat16 optimization for maximum performance

### 🖥️ Hardware Requirements

- **GPU**: AMD MI300X (minimum 48GB VRAM recommended)
- **CPU**: 16+ cores (optimized for 20 cores)
- **RAM**: 64GB+ (optimized for 240GB)
- **Storage**: 100GB+ SSD for dataset and models

## 📦 Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd plantNet

# Create conda environment
conda create -n plantnet python=3.11 -y
conda activate plantnet

# Install PyTorch for AMD GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install dependencies (this will install NumPy 2.x for compatibility)
pip install -r requirements.txt

# If you encounter compatibility issues, run the fix script:
python fix_numpy_compatibility.py
```

### 2. Dependencies

Create `requirements.txt`:

```
torch>=2.8.0
torchvision>=0.23.0
torchaudio
timm>=1.0.11
albumentations>=1.4.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
tensorboard>=2.13.0
grad-cam>=1.4.8
```

### 3. Dataset Preparation

Organize your PlantVillage dataset as follows:

```
plantNet/
├── data/
│   ├── train/
│   │   ├── Apple___Apple_scab/
│   │   ├── Apple___Black_rot/
│   │   ├── Apple___Cedar_apple_rust/
│   │   └── ... (other disease classes)
│   ├── val/
│   │   ├── Apple___Apple_scab/
│   │   └── ... (same structure as train)
│   └── test/
│       ├── Apple___Apple_scab/
│       └── ... (same structure as train)
```

## 🚀 Quick Start

### 1. Training on AMD MI300X

```bash
# Start training with default MI300X configuration
python train_mi300x.py

# Custom training with specific parameters
python train_mi300x.py --config config_mi300x_optimized.json --batch_size 256 --epochs 300

# Resume training from checkpoint
python train_mi300x.py --resume results_mi300x/experiment_name/best_model.pth

# Quick test run
python train_mi300x.py --test_run
```
├── results/               # Training results and saved models
├── requirements.txt       # Python dependencies
└── train.py              # Main training script
```

## Installation

1. Clone or create the project structure as shown above.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Download the PlantVillage dataset
   - Organize it in the following structure:
   ```
   data/
   ├── train/
   │   ├── Apple___Apple_scab/
   │   ├── Apple___Black_rot/
   │   ├── Apple___Cedar_apple_rust/
   │   └── ...
   ├── validation/
   │   ├── Apple___Apple_scab/
   │   └── ...
   └── test/ (optional)
       └── ...
   ```

## Usage

### Basic Training

```bash
# Train with default settings (Custom CNN, 50 epochs)
python train.py

# Train with specific parameters
python train.py --model_type resnet --epochs 100 --batch_size 64 --learning_rate 0.0001

# Train with ResNet50 pretrained model
python train.py --model_type resnet --epochs 50 --data_dir ./data
```

### Configuration File

Create a configuration file for reproducible experiments:

```bash
# Create sample configuration
python train.py --create-config

# Train with configuration file
python train.py --config config.json
```

### Advanced Usage

```python
# Example configuration (config.json)
{
  "data_dir": "data",
  "model_type": "resnet",
  "model_name": "resnet50",
  "pretrained": true,
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "optimizer": "adam",
  "scheduler": "plateau",
  "early_stopping": true,
  "early_stopping_patience": 10
}
```

## Model Architectures

### 1. Custom CNN
- Multi-layer CNN with batch normalization
- Global average pooling
- Dropout for regularization
- ~13M parameters

### 2. ResNet (Transfer Learning)
- Pre-trained on ImageNet
- Support for ResNet18/34/50/101
- Customizable backbone freezing
- Modified classifier head

### 3. EfficientNet
- State-of-the-art efficiency
- Requires `timm` library
- Compound scaling approach

## Training Features

### Data Augmentation
- Random crops and horizontal flips
- Color jittering
- Random rotation and affine transformations
- Normalization with ImageNet statistics

### Training Pipeline
- Mixed precision training support
- Learning rate scheduling (Step, ReduceLROnPlateau, CosineAnnealing)
- Early stopping with best weight restoration
- Class weight balancing for imbalanced datasets
- TensorBoard logging

### Evaluation
- Accuracy, Precision, Recall, F1-Score
- Top-k accuracy (Top-3, Top-5)
- Confusion matrix visualization
- Per-class performance analysis
- Sample prediction visualization

## Results and Monitoring

Training results are saved in the `results/` directory with:
- Model checkpoints (best and final)
- Training history plots
- Evaluation metrics and confusion matrices
- TensorBoard logs
- Configuration files
- Class names mapping

## PlantVillage Dataset

The PlantVillage dataset contains images of plant leaves with diseases. It includes:
- **38 classes** of plant diseases and healthy leaves
- **54,000+ images** across different plant species
- **Plants**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

### Expected Classes:
- Apple (Scab, Black rot, Cedar apple rust, Healthy)
- Blueberry (Healthy)
- Cherry (Powdery mildew, Healthy)
- Corn (Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy)
- Grape (Black rot, Esca, Leaf blight, Healthy)
- Orange (Haunglongbing Citrus greening)
- Peach (Bacterial spot, Healthy)
- Bell Pepper (Bacterial spot, Healthy)
- Potato (Early blight, Late blight, Healthy)
- Raspberry (Healthy)
- Soybean (Healthy)
- Squash (Powdery mildew)
- Strawberry (Leaf scorch, Healthy)
- Tomato (Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy)

## Performance Tips

1. **GPU Training**: The code automatically detects and uses GPU if available
2. **Batch Size**: Adjust based on GPU memory (32-64 typically works well)
3. **Learning Rate**: Start with 0.001, reduce if loss plateaus
4. **Data Loading**: Increase `num_workers` for faster data loading
5. **Model Choice**: ResNet50 with transfer learning often gives best results

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size or image size
2. **Dataset not found**: Check data directory structure
3. **Import errors**: Ensure all dependencies are installed
4. **Low accuracy**: Try transfer learning with pretrained models

### Memory Optimization:

```python
# Reduce memory usage
python train.py --batch_size 16 --img_size 224 --num_workers 2
```

## Contributing

Feel free to extend this implementation with:
- Additional model architectures
- New data augmentation techniques
- Advanced training strategies
- Performance optimizations

## License

This project is open source and available under the MIT License.

## Citation

If you use this code in your research, please cite:
```
@misc{plantnet2024,
  title={PlantNet: CNN-based Plant Disease Detection},
  author={Plant Disease Detection Project},
  year={2024},
  url={https://github.com/your-repo/plantnet}
}
```