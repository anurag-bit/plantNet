#!/usr/bin/env python3
"""
Quick 5-epoch training script for PlantVillage dataset on MI300X GPU
"""

import os
import sys
import time
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from utils.dataset import create_data_loaders
from models.cnn_models import create_model
from utils.config import ModelConfig
from utils.advanced_trainer import AdvancedTrainer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

def main():
    # Disable PyTorch compilation for stability
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    
    print("🚀 Starting 5-epoch training on MI300X...")
    
    # Configuration for quick training
    config = ModelConfig(
        data_dir="dataset",
        experiment_dir="experiments/quick_5epoch",
        model_type="resnet50",  # Single model for speed
        batch_size=128,
        epochs=5,
        learning_rate=3e-4,
        weight_decay=0.01,
        mixed_precision=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=8,
        pin_memory=True,
        channels_last=True,
        log_interval=10,
        save_best=True,
        save_last=True,
        save_intermediate=False,
        # Simplified augmentation for stability
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        label_smoothing=0.1,
        gradient_clip_norm=1.0
    )
    
    # Create experiment directory
    os.makedirs(config.experiment_dir, exist_ok=True)
    
    # Create data loaders
    print("📁 Loading dataset...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        channels_last=config.channels_last
    )
    
    print(f"🎯 Training batches: {len(train_loader)}")
    print(f"📈 Validation batches: {len(val_loader)}")
    print(f"🧪 Test batches: {len(test_loader)}")
    print(f"🏷️ Classes: {len(class_names)}")
    
    # Create model
    print("🧠 Creating model...")
    model = create_model(
        model_type=config.model_type,
        num_classes=len(class_names),
        pretrained=True,
        channels_last=config.channels_last
    ).to(config.device)
    
    print(f"✅ Model created: {config.model_type}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and loss
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    criterion = CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    # Simple learning rate schedule (no warmup for quick training)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        save_dir=config.experiment_dir,
        log_interval=config.log_interval,
        class_names=class_names,
        mixed_precision=config.mixed_precision,
        compile_model=False,  # Disabled for stability
        gradient_clip_norm=config.gradient_clip_norm,
        mixup_alpha=config.mixup_alpha,
        cutmix_alpha=config.cutmix_alpha,
        label_smoothing=config.label_smoothing
    )
    
    # Start training
    print("🚀 Starting training...")
    print(f"⚡ Device: {config.device}")
    print(f"🎯 Epochs: {config.epochs}")
    print(f"📦 Batch size: {config.batch_size}")
    print(f"⚖️ Mixed precision: {'bf16' if config.mixed_precision else 'fp32'}")
    
    start_time = time.time()
    
    try:
        history = trainer.train(
            num_epochs=config.epochs,
            early_stopping=None,  # No early stopping for quick training
            save_best=config.save_best,
            save_last=config.save_last,
            save_intermediate=config.save_intermediate
        )
        
        training_time = time.time() - start_time
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️ Total time: {training_time/60:.1f} minutes")
        
        # Final validation
        print("\n🧪 Running final validation...")
        val_loss, val_acc = trainer.validate_epoch()
        print(f"📊 Final validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        print(f"💾 Models saved in: {config.experiment_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())