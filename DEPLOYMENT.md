# Deployment and Model Management Documentation

## Git LFS and HuggingFace Integration

### Quick Start Guide

1. **Set up Git LFS** (one-time setup):
   ```bash
   git lfs install
   ```

2. **Test the complete pipeline**:
   ```bash
   python test_deployment.py
   ```

3. **Deploy your trained model**:
   ```bash
   # Set your HuggingFace token
   export HF_TOKEN="your_huggingface_token_here"
   
   # Run deployment pipeline
   ./deploy_pipeline.sh --model_path results_mi300x/best_model.pth
   ```

### Available Scripts

#### 🔧 `compile_models.py`
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
Complete deployment pipeline that orchestrates all steps.

```bash
# Full deployment
./deploy_pipeline.sh --model_path best_model.pth --repo_name plantnet-ensemble

# Dry run (test without actual deployment)
./deploy_pipeline.sh --model_path best_model.pth --dry_run
```

#### 🧪 `test_deployment.py`
Tests all deployment components to ensure everything works correctly.

```bash
# Run all tests
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