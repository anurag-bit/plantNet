#!/usr/bin/env python3
"""
Memory-Optimized PlantNet Training
Designed to avoid OOM kills and train successfully
"""

import os
import sys
import time
import torch
import argparse
from pathlib import Path

# Critical memory optimizations
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['OMP_NUM_THREADS'] = '4'

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

def create_memory_efficient_config():
    """Create memory-optimized configuration"""
    class MemoryOptimizedConfig:
        def __init__(self):
            # Critical memory settings
            self.data_dir = "dataset"
            self.experiment_dir = "experiments/memory_optimized"
            self.model_type = "resnet50"  # Single model instead of ensemble
            
            # Reduced memory settings
            self.batch_size = 32  # Reduced from 128
            self.epochs = 5
            self.learning_rate = 1e-4
            self.weight_decay = 0.01
            
            # Memory optimizations
            self.num_workers = 2  # Reduced from 8
            self.pin_memory = False  # Disabled to save memory
            self.channels_last = False  # Disabled for memory
            self.mixed_precision = True  # Keep for GPU memory efficiency
            
            # Training settings
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_interval = 20
            self.save_best = True
            self.save_last = True
            self.save_intermediate = False
            
            # Simplified augmentations
            self.mixup_alpha = 0.0
            self.cutmix_alpha = 0.0
            self.label_smoothing = 0.1
            self.gradient_clip_norm = 1.0
            
    return MemoryOptimizedConfig()

def main():
    print("🚀 Starting Memory-Optimized PlantNet Training")
    print("⚠️  Designed to avoid OOM kills")
    print("=" * 50)
    
    config = create_memory_efficient_config()
    
    # Create experiment directory
    os.makedirs(config.experiment_dir, exist_ok=True)
    
    # Import here to avoid early memory allocation
    from utils.dataset import create_data_loaders
    from models.cnn_models import create_model
    from utils.advanced_trainer import AdvancedTrainer
    from torch.optim import AdamW
    from torch.nn import CrossEntropyLoss
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    print("📁 Loading dataset with memory optimizations...")
    try:
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            channels_last=config.channels_last
        )
        
        print(f"✅ Dataset loaded successfully")
        print(f"🎯 Training batches: {len(train_loader)}")
        print(f"📈 Validation batches: {len(val_loader)}")
        print(f"🧪 Test batches: {len(test_loader)}")
        print(f"🏷️ Classes: {len(class_names)}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return 1
    
    print("\n🧠 Creating memory-efficient model...")
    try:
        model = create_model(
            model_type=config.model_type,
            num_classes=len(class_names),
            pretrained=True,
            channels_last=config.channels_last
        ).to(config.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {config.model_type}")
        print(f"📊 Parameters: {total_params:,} (much smaller than ensemble)")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return 1
    
    # Create optimizer and scheduler
    print("\n⚡ Setting up training components...")
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    criterion = CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    # Create trainer with memory optimizations
    print("🎯 Creating memory-optimized trainer...")
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
    
    # Memory monitoring
    print("\n💾 Memory Status Before Training:")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved()/1e9:.1f}GB reserved")
    
    # Start training
    print(f"\n🚀 Starting {config.epochs}-epoch training...")
    print(f"📦 Batch size: {config.batch_size} (reduced for memory)")
    print(f"👥 Workers: {config.num_workers} (reduced for memory)")
    print(f"🔧 Model: {config.model_type} (single model, not ensemble)")
    
    start_time = time.time()
    
    try:
        history = trainer.train(
            num_epochs=config.epochs,
            early_stopping=None,  # No early stopping for guaranteed training
            save_best=config.save_best,
            save_last=config.save_last,
            save_intermediate=config.save_intermediate
        )
        
        training_time = time.time() - start_time
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️ Total time: {training_time/60:.1f} minutes")
        print(f"💾 Model saved in: {config.experiment_dir}")
        
        # Final validation
        print("\n🧪 Running final validation...")
        val_loss, val_acc = trainer.validate_epoch()
        print(f"📊 Final validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)