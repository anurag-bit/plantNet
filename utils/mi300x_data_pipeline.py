"""
High-Performance Data Pipeline for AMD MI300X
===========================================

Optimized data loading and preprocessing pipeline designed to maximize
AMD MI300X performance with large batch sizes and multi-CPU processing.

Features:
- Optimized data loading for 20 vCPU cores
- Large batch processing (128-256 samples)
- Advanced augmentation pipelines
- Memory-efficient prefetching
- Multi-scale training support
"""

import os
import json
import warnings
from typing import Tuple, Optional, Dict, List, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import autoaugment, functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore', category=UserWarning)


class MI300XDataPipeline:
    """High-performance data pipeline optimized for AMD MI300X."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MI300X optimized data pipeline."""
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.batch_size = config.get('batch_size', 128)
        self.img_size = config.get('img_size', 384)
        self.num_workers = config.get('num_workers', 20)
        self.pin_memory = config.get('pin_memory', True)
        self.persistent_workers = config.get('persistent_workers', True)
        self.prefetch_factor = config.get('prefetch_factor', 4)
        
        # Advanced augmentation settings
        self.use_autoaugment = config.get('rand_augment', True)
        self.use_trivialaugment = config.get('trivial_augment', True)
        self.use_albumentations = config.get('use_albumentations', True)
        self.random_erase_prob = config.get('random_erase', 0.25)
        
        # Multi-scale training
        self.multi_scale_training = config.get('multi_scale_training', True)
        self.scale_sizes = config.get('scale_sizes', [320, 384, 448])
        
        # Class balancing
        self.use_weighted_sampling = config.get('use_weighted_sampling', True)
        
        print(f"🚀 Initializing MI300X Data Pipeline")
        print(f"📊 Batch Size: {self.batch_size}")
        print(f"🖼️ Image Size: {self.img_size}x{self.img_size}")
        print(f"⚡ Workers: {self.num_workers}")
        print(f"💾 Pin Memory: {self.pin_memory}")
    
    def get_train_transforms(self) -> transforms.Compose:
        """Create optimized training transforms for MI300X."""
        transform_list = []
        
        # Multi-scale resizing for robustness
        if self.multi_scale_training:
            transform_list.append(
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33),
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        else:
            transform_list.extend([
                transforms.Resize((int(self.img_size * 1.1), int(self.img_size * 1.1))),
                transforms.RandomCrop(self.img_size)
            ])
        
        # Basic augmentations
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15, interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        
        # Advanced color augmentations
        transform_list.extend([
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomEqualize(p=0.1),
        ])
        
        # Add AutoAugment for better generalization
        if self.use_autoaugment:
            transform_list.append(
                autoaugment.AutoAugment(
                    policy=autoaugment.AutoAugmentPolicy.IMAGENET
                )
            )
        
        if self.use_trivialaugment:
            transform_list.append(
                autoaugment.TrivialAugmentWide()
            )
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Random erasing for regularization
        if self.random_erase_prob > 0:
            transform_list.append(
                transforms.RandomErasing(
                    p=self.random_erase_prob,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value=0,
                    inplace=False
                )
            )
        
        return transforms.Compose(transform_list)
    
    def get_albumentations_train_transforms(self) -> A.Compose:
        """Create Albumentations transforms for additional augmentation."""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, 
                    contrast_limit=0.3, 
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            A.OneOf([
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(p=1.0),
                A.ElasticTransform(p=1.0),
            ], p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=self.img_size // 8,
                max_width=self.img_size // 8,
                fill_value=0,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def get_val_transforms(self) -> transforms.Compose:
        """Create validation transforms."""
        return transforms.Compose([
            transforms.Resize((int(self.img_size * 1.1), int(self.img_size * 1.1))),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_multi_scale_transforms(self) -> List[transforms.Compose]:
        """Create transforms for multi-scale inference."""
        multi_scale_transforms = []
        
        for size in self.scale_sizes:
            transform = transforms.Compose([
                transforms.Resize((int(size * 1.1), int(size * 1.1))),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            multi_scale_transforms.append(transform)
        
        return multi_scale_transforms
    
    def calculate_class_weights(self, train_dir: str) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets."""
        print("⚖️ Calculating class weights for balanced training...")
        
        class_counts = {}
        total_samples = 0
        
        # Count samples per class
        for class_name in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[class_name] = count
                total_samples += count
        
        # Calculate weights (inverse frequency)
        num_classes = len(class_counts)
        class_weights = []
        
        sorted_classes = sorted(class_counts.keys())
        for class_name in sorted_classes:
            weight = total_samples / (num_classes * class_counts[class_name])
            class_weights.append(weight)
        
        class_weights = torch.FloatTensor(class_weights)
        
        print(f"📊 Class distribution:")
        for i, class_name in enumerate(sorted_classes):
            print(f"   {class_name}: {class_counts[class_name]} samples (weight: {class_weights[i]:.3f})")
        
        return class_weights
    
    def create_weighted_sampler(self, dataset: Dataset) -> WeightedRandomSampler:
        """Create weighted sampler for balanced batch sampling."""
        print("🎯 Creating weighted sampler for balanced batches...")
        
        # Get class counts
        class_counts = {}
        for _, label in dataset.samples:
            class_name = dataset.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Calculate sample weights
        sample_weights = []
        total_samples = len(dataset)
        num_classes = len(dataset.classes)
        
        for _, label in dataset.samples:
            class_name = dataset.classes[label]
            class_count = class_counts[class_name]
            weight = total_samples / (num_classes * class_count)
            sample_weights.append(weight)
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def create_data_loaders(self, data_dir: str) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
        """Create optimized data loaders for MI300X."""
        print(f"🚀 Creating MI300X optimized data loaders...")
        
        # Verify data structure
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')
        
        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")
        if not os.path.exists(val_dir):
            raise ValueError(f"Validation directory not found: {val_dir}")
        
        # Get transforms
        if self.use_albumentations:
            # Use Albumentations for training (more advanced)
            train_transform = self.get_albumentations_train_transforms()
            # Create custom dataset that uses Albumentations
            train_dataset = AlbumentationsDataset(train_dir, train_transform)
        else:
            train_transform = self.get_train_transforms()
            train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        
        val_transform = self.get_val_transforms()
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        
        # Test dataset (optional)
        test_dataset = None
        if os.path.exists(test_dir):
            test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
        
        # Get class names
        class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else val_dataset.classes
        
        print(f"✅ Found {len(class_names)} classes:")
        for i, class_name in enumerate(class_names[:10]):  # Show first 10
            print(f"   {i}: {class_name}")
        if len(class_names) > 10:
            print(f"   ... and {len(class_names) - 10} more")
        
        # Create weighted sampler for training if enabled
        sampler = None
        shuffle = True
        if self.use_weighted_sampling and hasattr(train_dataset, 'samples'):
            sampler = self.create_weighted_sampler(train_dataset)
            shuffle = False  # Don't shuffle when using sampler
        
        # Create data loaders with MI300X optimizations
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True  # For stable training
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor
        )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor
            )
        
        # Memory optimization
        print("💾 Optimizing data loaders for memory efficiency...")
        if torch.cuda.is_available():
            # Pre-allocate GPU memory for stable performance
            torch.cuda.empty_cache()
        
        print(f"✅ Data loaders created successfully!")
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📈 Validation samples: {len(val_dataset)}")
        if test_dataset:
            print(f"🧪 Test samples: {len(test_dataset)}")
        
        print(f"🚀 Performance optimizations:")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Workers: {self.num_workers}")
        print(f"   Pin memory: {self.pin_memory}")
        print(f"   Persistent workers: {self.persistent_workers}")
        print(f"   Prefetch factor: {self.prefetch_factor}")
        
        return train_loader, val_loader, test_loader, class_names


class AlbumentationsDataset(Dataset):
    """Custom dataset using Albumentations transforms."""
    
    def __init__(self, root_dir: str, transform: A.Compose):
        """Initialize dataset with Albumentations transforms."""
        self.root_dir = root_dir
        self.transform = transform
        
        # Build file list and class mappings
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            self.classes.append(class_name)
            self.class_to_idx[class_name] = class_idx
            
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(class_path, filename)
                    self.samples.append((file_path, class_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(file_path).convert('RGB')
        image = np.array(image)
        
        # Apply Albumentations transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def create_data_loaders(data_dir: str, batch_size: int = 128, img_size: int = 384, 
                       num_workers: int = 20) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    """Convenience function to create MI300X optimized data loaders."""
    config = {
        'data_dir': data_dir,
        'batch_size': batch_size,
        'img_size': img_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,
        'use_albumentations': True,
        'use_weighted_sampling': True,
        'multi_scale_training': True,
        'rand_augment': True,
        'trivial_augment': True,
        'random_erase': 0.25
    }
    
    pipeline = MI300XDataPipeline(config)
    return pipeline.create_data_loaders(data_dir)


def calculate_class_weights(train_dir: str) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset."""
    config = {'data_dir': train_dir}
    pipeline = MI300XDataPipeline(config)
    return pipeline.calculate_class_weights(train_dir)


if __name__ == "__main__":
    # Test the data pipeline
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MI300X Data Pipeline')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("Please organize your PlantVillage dataset first.")
        exit(1)
    
    try:
        print("🧪 Testing MI300X Data Pipeline...")
        
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            args.data_dir, 
            batch_size=args.batch_size,
            num_workers=4  # Reduced for testing
        )
        
        print(f"✅ Pipeline test successful!")
        print(f"📊 Batch shape: {next(iter(train_loader))[0].shape}")
        print(f"🎯 Classes: {len(class_names)}")
        
        # Test batch loading speed
        import time
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            if i >= 5:  # Test 5 batches
                break
        
        elapsed_time = time.time() - start_time
        print(f"⚡ Loading speed: {elapsed_time/5:.3f}s per batch")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()