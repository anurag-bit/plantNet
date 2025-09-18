#!/usr/bin/env python3
"""
High-Performance Training Script for AMD MI300X
==============================================

Optimized training pipeline utilizing the full potential of AMD MI300X
with 192GB VRAM for maximum performance and precision.

Features:
- Advanced ensemble training
- Mixed precision (BF16) optimization
- High batch sizes (128-256)
- Advanced augmentations (MixUp, CutMix, AutoAugment)
- Comprehensive monitoring and analysis
- Multi-scale training and validation
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.cnn_models import create_model, count_parameters, EnsembleModel
from utils.dataset import create_data_loaders, calculate_class_weights
from utils.advanced_trainer import (
    AdvancedTrainer, EarlyStopping, create_advanced_optimizer, 
    create_advanced_scheduler, LabelSmoothingCrossEntropy
)
from utils.evaluation import ModelEvaluator, plot_training_history, visualize_predictions

warnings.filterwarnings('ignore', category=UserWarning)


class MI300XConfig:
    """Optimized configuration for AMD MI300X hardware."""
    
    def __init__(self, config_dict: dict = None):
        """Initialize with MI300X optimizations."""
        # Hardware specifications
        self.gpu_type = "AMD MI300X"
        self.vram_gb = 192
        self.vcpu_cores = 20
        self.ram_gb = 240
        
        # Data parameters optimized for large VRAM
        self.data_dir = "data"
        self.batch_size = 128  # Large batch for MI300X
        self.img_size = 384    # Higher resolution
        self.num_workers = 20  # Match vCPU cores
        self.pin_memory = True
        self.persistent_workers = True
        self.prefetch_factor = 4
        
        # Advanced ensemble model
        self.model_type = "ensemble"
        self.architectures = [
            {"type": "resnet", "name": "resnet101", "pretrained": True, "weight": 0.3},
            {"type": "efficientnet", "name": "efficientnet_b4", "pretrained": True, "weight": 0.25},
            {"type": "vit", "name": "vit_base_patch16_384", "pretrained": True, "weight": 0.25},
            {"type": "swin", "name": "swin_base_patch4_window12_384", "pretrained": True, "weight": 0.2}
        ]
        self.dropout_rate = 0.3
        self.label_smoothing = 0.1
        
        # Training parameters for high precision
        self.epochs = 200
        self.learning_rate = 0.0003
        self.weight_decay = 0.05
        self.optimizer = "adamw"
        self.scheduler = "cosine_warmup"
        self.warmup_epochs = 10
        self.min_lr = 1e-6
        self.criterion = "crossentropy_smooth"
        self.use_class_weights = True
        self.gradient_clip_norm = 1.0
        
        # Advanced augmentations
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0
        self.rand_augment = True
        self.trivial_augment = True
        self.random_erase = 0.25
        
        # Mixed precision and optimization
        self.mixed_precision = "bf16"  # BFloat16 for MI300X
        self.compile_model = True
        self.channels_last = True
        self.benchmark = True
        
        # Early stopping
        self.early_stopping = True
        self.early_stopping_patience = 25
        self.early_stopping_min_delta = 0.0001
        
        # Output and evaluation
        self.save_dir = "results_mi300x"
        self.experiment_name = f"plant_disease_mi300x_ensemble_{int(time.time())}"
        self.log_interval = 5
        self.save_best = True
        self.save_last = True
        self.save_intermediate = [50, 100, 150]
        self.detailed_logging = True
        
        # Advanced evaluation
        self.evaluate_on_test = True
        self.visualize_predictions = True
        self.num_prediction_samples = 64
        self.generate_gradcam = True
        self.tta_inference = True
        self.multi_scale_inference = True
        
        # Device optimization
        self.device = "cuda"
        
        # Update with provided config
        if config_dict:
            self._update_from_dict(config_dict)
        
        # Setup device and optimizations
        self._setup_device()
        self._setup_optimizations()
        
        # Create experiment directory
        self.experiment_dir = os.path.join(self.save_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
    
    def _update_from_dict(self, config_dict: dict):
        """Update configuration from dictionary."""
        # Handle nested configuration structure
        if 'data' in config_dict:
            data_config = config_dict['data']
            for key, value in data_config.items():
                if key != 'comment' and hasattr(self, key):
                    setattr(self, key, value)
        
        if 'model' in config_dict:
            model_config = config_dict['model']
            for key, value in model_config.items():
                if key != 'comment' and hasattr(self, key):
                    setattr(self, key, value)
        
        if 'training' in config_dict:
            training_config = config_dict['training']
            for key, value in training_config.items():
                if key != 'comment' and hasattr(self, key):
                    setattr(self, key, value)
        
        if 'early_stopping' in config_dict:
            es_config = config_dict['early_stopping']
            if 'enabled' in es_config:
                self.early_stopping = es_config['enabled']
            if 'patience' in es_config:
                self.early_stopping_patience = es_config['patience']
            if 'min_delta' in es_config:
                self.early_stopping_min_delta = es_config['min_delta']
        
        if 'device' in config_dict:
            device_config = config_dict['device']
            for key, value in device_config.items():
                if key == 'device_type':
                    self.device = value
                elif key != 'comment' and hasattr(self, key):
                    setattr(self, key, value)
        
        if 'output' in config_dict:
            output_config = config_dict['output']
            for key, value in output_config.items():
                if key != 'comment' and hasattr(self, key):
                    setattr(self, key, value)
        
        if 'augmentation' in config_dict:
            aug_config = config_dict['augmentation']
            if 'mixup_alpha' in aug_config:
                self.mixup_alpha = aug_config['mixup_alpha']
            if 'cutmix_alpha' in aug_config:
                self.cutmix_alpha = aug_config['cutmix_alpha']
            if 'random_erase' in aug_config:
                self.random_erase = aug_config['random_erase']
        
        # Handle flat structure as well
        for key, value in config_dict.items():
            if key not in ['data', 'model', 'training', 'early_stopping', 'device', 'output', 'augmentation', 'hardware', 'scheduler', 'inference', 'evaluation']:
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def _setup_device(self):
        """Setup device with optimizations."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            # Enable optimizations for AMD GPU
            torch.backends.cudnn.benchmark = self.benchmark
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Memory optimizations
            torch.cuda.empty_cache()
            
            print(f"🚀 AMD MI300X Detected!")
            print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"⚡ Mixed Precision: {self.mixed_precision}")
    
    def _setup_optimizations(self):
        """Setup performance optimizations."""
        # Memory format optimization
        if self.channels_last:
            print("📈 Enabling channels_last memory format")
        
        # Model compilation
        if self.compile_model and hasattr(torch, 'compile'):
            print("🔧 Model compilation enabled")
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, torch.device):
                    config_dict[key] = str(value)
                elif isinstance(value, (list, dict, str, int, float, bool)):
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def print_config(self):
        """Print comprehensive configuration summary."""
        print("\n" + "="*80)
        print("🚀 AMD MI300X OPTIMIZED CONFIGURATION")
        print("="*80)
        
        sections = {
            "🖥️ Hardware": ["gpu_type", "vram_gb", "vcpu_cores", "ram_gb"],
            "📊 Data Pipeline": ["batch_size", "img_size", "num_workers", "pin_memory"],
            "🧠 Model Architecture": ["model_type", "mixed_precision", "dropout_rate"],
            "🎯 Training Setup": ["epochs", "learning_rate", "optimizer", "scheduler"],
            "🎨 Augmentations": ["mixup_alpha", "cutmix_alpha", "label_smoothing"],
            "⚡ Optimizations": ["compile_model", "channels_last", "benchmark"],
            "💾 Output": ["save_dir", "experiment_name"]
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    if isinstance(value, list) and len(value) > 3:
                        print(f"  {key}: [{len(value)} items]")
                    else:
                        print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        print(f"🎯 Expected Training Time: ~{self.epochs * 0.8 / 60:.1f} hours")
        print(f"📈 Expected Memory Usage: ~{self.batch_size * 3 * self.img_size**2 * 4 / 1e9:.1f} GB")
        print("="*80 + "\n")


def load_mi300x_config(config_path: str) -> MI300XConfig:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return MI300XConfig(config_dict)


def main():
    """Main training function optimized for MI300X."""
    parser = argparse.ArgumentParser(description='MI300X Optimized Plant Disease Detection Training')
    parser.add_argument('--config', type=str, default='config_mi300x_optimized.json',
                       help='Path to MI300X configuration file')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--test_run', action='store_true', help='Run test with small dataset')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_mi300x_config(args.config)
        print(f"📁 Loaded configuration: {args.config}")
    else:
        config = MI300XConfig()
        print("⚠️  Using default MI300X configuration")
    
    # Override with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.test_run:
        config.epochs = 5
        config.batch_size = 32
        config.experiment_name = f"test_run_{int(time.time())}"
        config.experiment_dir = os.path.join(config.save_dir, config.experiment_name)
        os.makedirs(config.experiment_dir, exist_ok=True)
    
    # Print configuration
    config.print_config()
    
    # Save configuration
    config.save_config(os.path.join(config.experiment_dir, 'mi300x_config.json'))
    
    # Check data directory
    if not os.path.exists(config.data_dir):
        print(f"❌ Error: Data directory '{config.data_dir}' not found!")
        print("\n🚀 To set up the dataset automatically, run:")
        print("   python setup_dataset.py --sample  # For quick testing")
        print("   python setup_dataset.py --source auto  # For full dataset")
        print("   python setup_kaggle_dataset.py --use_kaggle  # For Kaggle dataset")
        print("\nOr organize your PlantVillage dataset as described in README.md")
        return 1
    
    try:
        # Create optimized data loaders
        print("🚀 Loading dataset with MI300X optimizations...")
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            config.data_dir,
            batch_size=config.batch_size,
            img_size=config.img_size,
            num_workers=config.num_workers
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Classes: {len(class_names)}")
        print(f"🎯 Training batches: {len(train_loader)}")
        print(f"📈 Validation batches: {len(val_loader)}")
        if test_loader:
            print(f"🧪 Test batches: {len(test_loader)}")
        
        # Save class names
        with open(os.path.join(config.experiment_dir, 'class_names.json'), 'w') as f:
            json.dump(class_names, f, indent=2)
        
        # Create advanced ensemble model
        print(f"🧠 Creating {config.model_type} model...")
        if config.model_type == "ensemble":
            model = create_model(config.model_type, len(class_names), 
                               architectures=config.architectures)
        else:
            model = create_model(config.model_type, len(class_names))
        
        # Move to device and optimize
        model = model.to(config.device)
        
        # Enable channels_last memory format for better performance
        if config.channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        # Compile model for optimization
        if config.compile_model and hasattr(torch, 'compile'):
            print("🔧 Compiling model for optimization...")
            model = torch.compile(model, mode='max-autotune')
        
        print(f"✅ Model created with {count_parameters(model):,} parameters")
        
        # Calculate class weights for imbalanced dataset
        class_weights = None
        if config.use_class_weights:
            print("⚖️ Calculating class weights...")
            class_weights = calculate_class_weights(os.path.join(config.data_dir, 'train'))
            class_weights = class_weights.to(config.device)
        
        # Create advanced optimizer and scheduler
        optimizer = create_advanced_optimizer(
            model,
            config.optimizer,
            config.learning_rate,
            config.weight_decay
        )
        
        scheduler = create_advanced_scheduler(
            optimizer,
            config.scheduler,
            config.epochs,
            warmup_epochs=config.warmup_epochs,
            min_lr=config.min_lr
        )
        
        # Create advanced loss function
        if config.label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(
                smoothing=config.label_smoothing,
                weight=class_weights
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Create early stopping
        early_stopping = None
        if config.early_stopping:
            early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                restore_best_weights=True
            )
        
        # Create advanced trainer
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
            compile_model=False,  # Already compiled
            gradient_clip_norm=config.gradient_clip_norm,
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            label_smoothing=config.label_smoothing
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"🔄 Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        print("🚀 Starting MI300X optimized training...")
        estimated_time_per_epoch = 45 if config.model_type == "ensemble" else 20  # seconds
        estimated_total_time = config.epochs * estimated_time_per_epoch
        completion_time = time.time() + estimated_total_time
        print(f"⏱️ Estimated completion: {time.strftime('%H:%M:%S', time.localtime(completion_time))} ({estimated_total_time/3600:.1f}h total)")
        
        start_time = time.time()
        history = trainer.train(
            num_epochs=config.epochs,
            early_stopping=early_stopping,
            save_best=config.save_best,
            save_last=config.save_last,
            save_intermediate=config.save_intermediate
        )
        
        training_time = time.time() - start_time
        print(f"🎉 Training completed in {training_time/3600:.2f} hours")
        
        # Plot training history
        print("📊 Generating training visualizations...")
        plot_training_history(history, config.experiment_dir)
        
        # Advanced evaluation on test set
        if test_loader and config.evaluate_on_test:
            print("🧪 Performing comprehensive test evaluation...")
            
            # Load best model for evaluation
            best_model_path = os.path.join(config.experiment_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                trainer.load_checkpoint(best_model_path)
            
            evaluator = ModelEvaluator(model, config.device, class_names)
            test_metrics = evaluator.evaluate_model(
                test_loader,
                os.path.join(config.experiment_dir, 'test_evaluation')
            )
            
            print(f"🎯 Final Test Results:")
            print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"   Top-3 Accuracy: {test_metrics['top_3_accuracy']:.4f}")
            print(f"   Top-5 Accuracy: {test_metrics['top_5_accuracy']:.4f}")
            print(f"   Macro F1: {test_metrics['macro_f1']:.4f}")
        
        # Generate prediction visualizations
        if config.visualize_predictions:
            print("🖼️ Generating prediction visualizations...")
            eval_loader = test_loader if test_loader else val_loader
            visualize_predictions(
                model, eval_loader, class_names, config.device,
                num_samples=config.num_prediction_samples,
                save_dir=os.path.join(config.experiment_dir, 'predictions')
            )
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 MI300X TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"💾 Results saved to: {config.experiment_dir}")
        print(f"⏱️ Total time: {training_time/3600:.2f} hours")
        print(f"⚡ Average time per epoch: {training_time/config.epochs:.1f}s")
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"🖥️ Peak GPU memory: {peak_memory:.2f} GB / {config.vram_gb} GB")
            print(f"💡 Memory utilization: {peak_memory/config.vram_gb*100:.1f}%")
        
        print("✅ Ready for deployment with advanced_inference.py!")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)