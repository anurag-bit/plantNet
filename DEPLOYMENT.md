# Deployment and Model Management Documentation

## Complete PlantNet Pipeline: Dataset to Deployment

### Quick Start Guide

1. **Set up the environment** (one-time setup):
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up Git LFS
   git lfs install
   ```

2. **Automated dataset setup**:
   ```bash
   # Option 1: Create sample dataset for testing
   python setup_dataset.py --sample
   
   # Option 2: Download PlantVillage dataset automatically
   python setup_dataset.py --source auto
   
   # Option 3: Use Kaggle API (requires kaggle.json credentials)
   python setup_kaggle_dataset.py --use_kaggle
   ```

3. **Test the complete pipeline**:
   ```bash
   python test_deployment.py
   ```

4. **Complete deployment with dataset setup**:
   ```bash
   # Set your HuggingFace token
   export HF_TOKEN="your_huggingface_token_here"
   
   # Full pipeline: dataset → training → deployment
   ./deploy_pipeline.sh --setup_dataset --sample_dataset
   
   # Or deploy existing model
   ./deploy_pipeline.sh --model_path results_mi300x/best_model.pth
   ```

### Available Scripts

#### 📊 `setup_dataset.py`
Automated PlantVillage dataset download and setup with train/val/test splits.

```bash
# Create sample dataset for testing
python setup_dataset.py --sample --data_dir data

# Download from GitHub (automatic)
python setup_dataset.py --source github --data_dir data

# Verify existing dataset
python setup_dataset.py --verify_only --data_dir data
```

#### � `setup_kaggle_dataset.py`
Enhanced dataset setup with Kaggle API integration for official PlantVillage dataset.

```bash
# Download using Kaggle API (requires kaggle.json)
python setup_kaggle_dataset.py --use_kaggle --data_dir data
```

**Kaggle Setup Instructions:**
1. Go to [kaggle.com/account](https://www.kaggle.com/account)
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

#### �🔧 `compile_models.py`
Compiles trained models into multiple formats for deployment.

```bash
# Compile all formats (TorchScript, ONNX, Quantized)
python compile_models.py --model_path best_model.pth --formats all

# Compile specific format
python compile_models.py --model_path best_model.pth --formats torchscript onnx
```

#### 🤗 `upload_to_huggingface.py`
Uploads compiled models to HuggingFace Hub with proper documentation.

```bash
# Upload to HuggingFace
python upload_to_huggingface.py --model_dir compiled_models --repo_name plantnet-v1 --token $HF_TOKEN

# Create private repository
python upload_to_huggingface.py --model_dir compiled_models --repo_name plantnet-v1 --private
```

#### 🏷️ `version_manager.py`
Manages model versions with semantic versioning and Git integration.

```bash
# Increment version
python version_manager.py --increment patch --commit --tag

# Set specific version
python version_manager.py --set_version 2.0.0
```

#### 🚀 `deploy_pipeline.sh`
Complete deployment pipeline that orchestrates all steps from dataset to deployment.

```bash
# Full pipeline with dataset setup
./deploy_pipeline.sh --setup_dataset --sample_dataset

# Deploy existing model
./deploy_pipeline.sh --model_path best_model.pth --repo_name plantnet-ensemble

# Dry run (test without actual deployment)
./deploy_pipeline.sh --model_path best_model.pth --dry_run

# Skip training and deploy existing model
./deploy_pipeline.sh --model_path models/best_model.pth --skip_training
```

#### 🧪 `test_deployment.py`
Tests all deployment components including dataset setup to ensure everything works correctly.

```bash
# Run all tests (including dataset automation)
python test_deployment.py
```

### Git LFS Configuration

The `.gitattributes` file is configured to track:

- **Model files**: `*.pth`, `*.pt`, `*.safetensors`, `*.bin`, `*.ckpt`, `*.h5`
- **Optimized formats**: `*.onnx`, `*.trt`, `*.engine`
- **Large data**: `*.zip`, `*.tar.gz`, `*.hdf5`, `*.npz`
- **Media files**: `*.mp4`, `*.avi`, `*.tiff`

### Deployment Workflow

1. **Train Model** → `train_mi300x.py`
2. **Compile Models** → `compile_models.py`
3. **Version Management** → `version_manager.py`
4. **Git Operations** → Commit, tag, push
5. **HuggingFace Upload** → `upload_to_huggingface.py`

### Environment Variables

- `HF_TOKEN`: HuggingFace API token for model uploads
- `WANDB_API_KEY`: Weights & Biases API key (optional)
- `GITHUB_TOKEN`: GitHub token for automated releases (CI/CD)

### Model Repository Structure

```
your-hf-repo/
├── README.md              # Generated model card
├── config.json            # HuggingFace model configuration
├── model.pt               # TorchScript model
├── model.onnx             # ONNX model
├── model_quantized.pt     # Quantized model
├── inference.py           # Inference wrapper
├── requirements.txt       # Model dependencies
└── metadata.json          # Comprehensive model metadata
```

### CI/CD Integration

GitHub Actions workflow (`.github/workflows/model-deployment.yml`) provides:

- Automated model validation
- Multi-format compilation
- Version management
- Automated HuggingFace uploads
- Release creation

### Troubleshooting

**Git LFS Issues:**
```bash
# Reinstall Git LFS
git lfs install --force

# Check LFS status
git lfs status

# Track additional file types
git lfs track "*.custom_extension"
```

**HuggingFace Upload Issues:**
```bash
# Login to HuggingFace CLI
huggingface-cli login

# Test connection
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

**Large File Issues:**
```bash
# Check file sizes
find . -type f -size +100M

# Add to Git LFS
git lfs track "large_file.pth"
git add .gitattributes large_file.pth
git commit -m "Track large file with LFS"
```

### Performance Tips

1. **Use Git LFS** for all model files > 10MB
2. **Compress models** when possible (quantization, pruning)
3. **Batch uploads** to HuggingFace for multiple formats
4. **Version incrementally** to track changes properly
5. **Test locally** with `--dry_run` before actual deployment

### Security Considerations

- **Never commit tokens** to Git repository
- **Use environment variables** for sensitive information
- **Create private repos** for proprietary models
- **Review model cards** before public release
- **Validate uploaded models** after deployment