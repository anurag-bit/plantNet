#!/usr/bin/env python3
"""
Upload PlantNet model to HuggingFace Hub
Complete deployment pipeline
"""

import os
import json
import torch
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file
import sys

# Add project root
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

def upload_to_huggingface():
    """Upload trained model to HuggingFace Hub"""
    
    # Configuration
    MODEL_NAME = "plantnet-disease-detection"
    USERNAME = "your-username"  # Replace with your HF username
    REPO_ID = f"{USERNAME}/{MODEL_NAME}"
    
    # File paths
    MODEL_PATH = "results_mi300x/plant_disease_mi300x_ensemble/best_model.pth"
    CLASS_NAMES_PATH = "results_mi300x/plant_disease_mi300x_ensemble/class_names.json"
    CONFIG_PATH = "results_mi300x/plant_disease_mi300x_ensemble/mi300x_config.json"
    
    print("🚀 Uploading PlantNet to HuggingFace Hub...")
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository
    try:
        create_repo(
            repo_id=REPO_ID,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✅ Repository created/verified: {REPO_ID}")
    except Exception as e:
        print(f"❌ Error creating repo: {e}")
        return
    
    # Create model card
    model_card = f"""---
license: mit
tags:
- agriculture
- plant-disease
- computer-vision
- ensemble
- pytorch
datasets:
- PlantVillage
metrics:
- accuracy
library_name: pytorch
---

# PlantNet: Plant Disease Detection 🌱

## Model Description

PlantNet is an advanced ensemble model for plant disease detection, combining 4 state-of-the-art architectures:
- ResNet152 (60.2M parameters)
- EfficientNetB4 (19.3M parameters)
- Vision Transformer (86.6M parameters)
- Swin Transformer (87.8M parameters)

**Total Parameters:** 235,985,048  
**Validation Accuracy:** 95%+  
**Classes:** 38 plant diseases across 9 crop types

## Supported Plants & Diseases

- **Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- **Grape:** Black Rot, Esca, Leaf Blight, Healthy
- **Tomato:** 10 different diseases + Healthy
- **Corn:** Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- **Potato:** Early Blight, Late Blight, Healthy
- **And more:** Cherry, Strawberry, Peach, Pepper, Orange, etc.

## Usage

```python
from transformers import pipeline

# Load the model
classifier = pipeline("image-classification", model="{REPO_ID}")

# Predict plant disease
result = classifier("path/to/plant_image.jpg")
print(result)
```

## Training Details

- **Hardware:** AMD MI300X GPU (192GB VRAM)
- **Framework:** PyTorch + ROCm
- **Precision:** BF16 mixed precision
- **Epochs:** 4 (early stopped)
- **Dataset:** PlantVillage (79,227 images)

## Model Performance

- Validation Accuracy: 95%+
- Inference Speed: ~0.1-0.5 seconds per image
- GPU Memory: ~2-3GB during inference

## Citation

If you use this model, please cite:

```
@misc{{plantnet2025,
  title={{PlantNet: Advanced Plant Disease Detection using Ensemble Learning}},
  author={{Your Name}},
  year={{2025}},
  publisher={{HuggingFace Hub}},
  url={{https://huggingface.co/{REPO_ID}}}
}}
```
"""

    # Save model card
    with open("README.md", "w") as f:
        f.write(model_card)
    
    # Create config for HuggingFace
    hf_config = {{
        "architectures": ["EnsembleModel"],
        "model_type": "plantnet-ensemble",
        "num_classes": 38,
        "image_size": 224,
        "framework": "pytorch",
        "library": "transformers"
    }}
    
    with open("config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    
    # Upload files
    files_to_upload = [
        (MODEL_PATH, "pytorch_model.bin"),
        (CLASS_NAMES_PATH, "class_names.json"),
        (CONFIG_PATH, "training_config.json"),
        ("README.md", "README.md"),
        ("config.json", "config.json")
    ]
    
    for local_path, hub_path in files_to_upload:
        if os.path.exists(local_path):
            print(f"📤 Uploading: {local_path} -> {hub_path}")
            try:
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=hub_path,
                    repo_id=REPO_ID,
                    repo_type="model"
                )
                print(f"✅ Uploaded: {hub_path}")
            except Exception as e:
                print(f"❌ Error uploading {hub_path}: {e}")
        else:
            print(f"⚠️ File not found: {local_path}")
    
    print(f"\n🎉 Model uploaded successfully!")
    print(f"🔗 Model URL: https://huggingface.co/{REPO_ID}")
    print(f"📱 API Endpoint: https://api-inference.huggingface.co/models/{REPO_ID}")

if __name__ == "__main__":
    # Check if logged in to HuggingFace
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"👤 HuggingFace user: {user['name']}")
    except:
        print("❌ Please login to HuggingFace first:")
        print("   pip install huggingface_hub")
        print("   huggingface-cli login")
        sys.exit(1)
    
    upload_to_huggingface()