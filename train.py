#!/usr/bin/env python3
"""
PlantNet: Plant Disease Detection using CNN
==========================================

A comprehensive implementation of CNN-based plant disease detection
using the PlantVillage dataset.

Usage:
    python train.py --config config.json
    python train.py --data_dir ./data --model_type resnet --epochs 50

Author: Generated for PlantNet Project
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

from models.cnn_models import create_model, count_parameters
from utils.dataset import create_data_loaders, calculate_class_weights
from utils.trainer import Trainer, EarlyStopping, create_optimizer, create_scheduler, create_criterion
from utils.evaluation import ModelEvaluator, plot_training_history, visualize_predictions

warnings.filterwarnings('ignore', category=UserWarning)


class Config:
    """Configuration class for training parameters."""
    
    def __init__(self, config_dict: dict = None):
        """Initialize configuration with default values."""
        # Data parameters
        self.data_dir = "data"
        self.batch_size = 32
        self.img_size = 224
        self.num_workers = 4
        
        # Model parameters
        self.model_type = "custom_cnn"  # custom_cnn, resnet, efficientnet
        self.model_name = "resnet50"    # For pretrained models
        self.pretrained = True
        self.freeze_backbone = False
        self.dropout_rate = 0.5
        
        # Training parameters
        self.epochs = 50
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.optimizer = "adam"  # adam, adamw, sgd
        self.scheduler = "plateau"  # step, plateau, cosine, none
        self.criterion = "crossentropy"
        self.use_class_weights = True
        
        # Scheduler parameters
        self.step_size = 10
        self.gamma = 0.1
        self.patience = 5
        self.factor = 0.5
        
        # Early stopping
        self.early_stopping = True
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 0.001
        
        # Output and logging
        self.save_dir = "results"
        self.experiment_name = f"plant_disease_{int(time.time())}"
        self.log_interval = 10
        self.save_best = True
        self.save_last = True
        
        # Evaluation
        self.evaluate_on_test = True
        self.visualize_predictions = True
        self.num_prediction_samples = 16
        
        # Device
        self.device = "auto"  # auto, cpu, cuda
        
        # Update with provided config
        if config_dict:
            self._update_from_dict(config_dict)
        
        # Setup device
        self._setup_device()
        
        # Create experiment directory
        self.experiment_dir = os.path.join(self.save_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
    
    def _update_from_dict(self, config_dict: dict):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration key: {key}")
    
    def _setup_device(self):
        """Setup compute device."""
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)
        
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = {k: str(v) if isinstance(v, torch.device) else v 
                      for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def print_config(self):
        """Print configuration summary."""
        print("\n" + "="*50)
        print("CONFIGURATION SUMMARY")
        print("="*50)
        
        sections = {
            "Data": ["data_dir", "batch_size", "img_size", "num_workers"],
            "Model": ["model_type", "model_name", "pretrained", "dropout_rate"],
            "Training": ["epochs", "learning_rate", "optimizer", "scheduler"],
            "Device": ["device"]
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                if hasattr(self, key):
                    print(f"  {key}: {getattr(self, key)}")
        
        print("="*50 + "\n")


def load_config(config_path: str) -> Config:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return Config(config_dict)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Plant Disease Detection Training')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_type', type=str, default='custom_cnn', 
                       choices=['custom_cnn', 'resnet', 'efficientnet'],
                       help='Model type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='results', help='Save directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()
        # Override with command line arguments
        if args.data_dir != 'data':
            config.data_dir = args.data_dir
        if args.model_type != 'custom_cnn':
            config.model_type = args.model_type
        if args.epochs != 50:
            config.epochs = args.epochs
        if args.batch_size != 32:
            config.batch_size = args.batch_size
        if args.learning_rate != 0.001:
            config.learning_rate = args.learning_rate
        if args.save_dir != 'results':
            config.save_dir = args.save_dir
            config.experiment_dir = os.path.join(config.save_dir, config.experiment_name)
            os.makedirs(config.experiment_dir, exist_ok=True)
    
    # Print configuration
    config.print_config()
    
    # Save configuration
    config.save_config(os.path.join(config.experiment_dir, 'config.json'))
    
    # Check if data directory exists
    if not os.path.exists(config.data_dir):
        print(f"Error: Data directory '{config.data_dir}' not found!")
        print("Please organize your PlantVillage dataset as:")
        print(f"{config.data_dir}/")
        print("├── train/")
        print("│   ├── class1/")
        print("│   └── class2/")
        print("├── validation/")
        print("└── test/ (optional)")
        return
    
    try:
        # Create data loaders
        print("Loading dataset...")
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            config.data_dir, 
            batch_size=config.batch_size,
            img_size=config.img_size,
            num_workers=config.num_workers
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Number of classes: {len(class_names)}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        if test_loader:
            print(f"Test batches: {len(test_loader)}")
        
        # Save class names
        with open(os.path.join(config.experiment_dir, 'class_names.json'), 'w') as f:
            json.dump(class_names, f, indent=2)
        
        # Create model
        print(f"Creating {config.model_type} model...")
        model_kwargs = {
            'dropout_rate': config.dropout_rate
        }
        
        if config.model_type == 'resnet':
            model_kwargs.update({
                'model_name': config.model_name,
                'pretrained': config.pretrained,
                'freeze_backbone': config.freeze_backbone
            })
        elif config.model_type == 'efficientnet':
            model_kwargs.update({
                'model_name': config.model_name,
                'pretrained': config.pretrained
            })
        
        model = create_model(config.model_type, len(class_names), **model_kwargs)
        model = model.to(config.device)
        
        print(f"Model created with {count_parameters(model):,} trainable parameters")
        
        # Calculate class weights if requested
        class_weights = None
        if config.use_class_weights:
            print("Calculating class weights...")
            class_weights = calculate_class_weights(os.path.join(config.data_dir, 'train'))
            class_weights = class_weights.to(config.device)
            print("Class weights calculated")
        
        # Create optimizer, scheduler, and criterion
        optimizer = create_optimizer(
            model, 
            config.optimizer, 
            config.learning_rate, 
            config.weight_decay
        )
        
        scheduler_kwargs = {}
        if config.scheduler == 'step':
            scheduler_kwargs = {'step_size': config.step_size, 'gamma': config.gamma}
        elif config.scheduler == 'plateau':
            scheduler_kwargs = {'patience': config.patience, 'factor': config.factor}
        
        scheduler = create_scheduler(optimizer, config.scheduler, **scheduler_kwargs)
        criterion = create_criterion(config.criterion, class_weights)
        
        # Create early stopping
        early_stopping = None
        if config.early_stopping:
            early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                restore_best_weights=True
            )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.device,
            save_dir=config.experiment_dir,
            log_interval=config.log_interval,
            class_names=class_names
        )
        
        # Train model
        print("Starting training...")
        start_time = time.time()
        
        history = trainer.train(
            num_epochs=config.epochs,
            early_stopping=early_stopping,
            save_best=config.save_best,
            save_last=config.save_last
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Plot training history
        print("Plotting training history...")
        plot_training_history(history, config.experiment_dir)
        
        # Evaluate on test set if available
        if test_loader and config.evaluate_on_test:
            print("Evaluating on test set...")
            
            # Load best model
            best_model_path = os.path.join(config.experiment_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                checkpoint = trainer.load_checkpoint(best_model_path)
            
            evaluator = ModelEvaluator(model, config.device, class_names)
            test_metrics = evaluator.evaluate_model(
                test_loader, 
                os.path.join(config.experiment_dir, 'test_evaluation')
            )
            
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test Top-3 Accuracy: {test_metrics['top_3_accuracy']:.4f}")
            print(f"Test Top-5 Accuracy: {test_metrics['top_5_accuracy']:.4f}")
        
        # Visualize predictions
        if config.visualize_predictions:
            print("Visualizing predictions...")
            eval_loader = test_loader if test_loader else val_loader
            visualize_predictions(
                model, eval_loader, class_names, config.device,
                num_samples=config.num_prediction_samples,
                save_dir=os.path.join(config.experiment_dir, 'predictions')
            )
        
        print(f"Training completed! Results saved to: {config.experiment_dir}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def create_sample_config():
    """Create a sample configuration file."""
    config = Config()
    config.save_config('config_sample.json')
    print("Sample configuration saved to 'config_sample.json'")


if __name__ == "__main__":
    # Check if user wants to create sample config
    if len(sys.argv) > 1 and sys.argv[1] == '--create-config':
        create_sample_config()
    else:
        exit_code = main()
        sys.exit(exit_code)