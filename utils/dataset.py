import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional


class PlantVillageDataset(Dataset):
    """
    Custom Dataset class for PlantVillage dataset.
    Expected directory structure:
    data/
    ├── train/
    │   ├── class1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class2/
    │   └── ...
    ├── validation/
    └── test/
    """
    
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir (str): Directory with all the images organized by class
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Build file paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(class_dir, filename)
                    self.samples.append((file_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(img_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Get data transforms for training or validation/testing.
    
    Args:
        img_size (int): Target image size
        is_training (bool): Whether to apply training augmentations
        
    Returns:
        transforms.Compose: Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(data_dir: str, batch_size: int = 32, img_size: int = 224, 
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir (str): Root directory containing train, validation, and test folders
        batch_size (int): Batch size for data loaders
        img_size (int): Target image size
        num_workers (int): Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    # Get transforms
    train_transform = get_transforms(img_size, is_training=True)
    val_test_transform = get_transforms(img_size, is_training=False)
    
    # Create datasets
    train_dataset = PlantVillageDataset(train_dir, transform=train_transform)
    val_dataset = PlantVillageDataset(val_dir, transform=val_test_transform)
    
    # Check if test directory exists
    test_loader = None
    if os.path.exists(test_dir):
        test_dataset = PlantVillageDataset(test_dir, transform=val_test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    
    class_names = train_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names


def calculate_class_weights(data_dir: str) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        data_dir (str): Directory containing class folders
        
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = {}
    total_samples = 0
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
            total_samples += count
    
    # Calculate inverse frequency weights
    num_classes = len(class_counts)
    weights = []
    
    for class_name in sorted(class_counts.keys()):
        weight = total_samples / (num_classes * class_counts[class_name])
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def get_dataset_stats(data_loader: DataLoader) -> dict:
    """
    Calculate dataset statistics (mean, std, class distribution).
    
    Args:
        data_loader (DataLoader): Data loader to analyze
        
    Returns:
        dict: Dataset statistics
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    class_counts = {}
    
    for data, labels in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
        
        # Count classes
        for label in labels:
            label = label.item()
            class_counts[label] = class_counts.get(label, 0) + 1
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'total_samples': total_samples,
        'class_distribution': class_counts
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing PlantVillage dataset loader...")
    
    # Note: This assumes you have the PlantVillage dataset in the expected structure
    data_dir = "data"
    
    if os.path.exists(os.path.join(data_dir, "train")):
        try:
            train_loader, val_loader, test_loader, class_names = create_data_loaders(
                data_dir, batch_size=8, img_size=224, num_workers=2
            )
            
            print(f"Number of classes: {len(class_names)}")
            print(f"Classes: {class_names[:10]}...")  # Show first 10 classes
            print(f"Training batches: {len(train_loader)}")
            print(f"Validation batches: {len(val_loader)}")
            
            # Test loading a batch
            for images, labels in train_loader:
                print(f"Batch shape: {images.shape}")
                print(f"Labels shape: {labels.shape}")
                break
                
        except Exception as e:
            print(f"Error: {e}")
            print("Please ensure the PlantVillage dataset is organized in the expected structure:")
            print("data/train/, data/validation/, data/test/ with class subfolders")
    else:
        print("Dataset directory not found. Please organize your PlantVillage dataset as:")
        print("data/")
        print("├── train/")
        print("│   ├── class1/")
        print("│   └── class2/")
        print("├── validation/")
        print("└── test/")