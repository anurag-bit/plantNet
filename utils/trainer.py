import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best_weights: bool = True):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved
            min_delta (float): Minimum change in monitored quantity to qualify as improvement
            restore_best_weights (bool): Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to potentially save weights from
            
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class Trainer:
    """Training pipeline for plant disease classification."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
        device: torch.device = torch.device('cpu'),
        save_dir: str = 'results',
        log_interval: int = 10,
        class_names: List[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion (nn.Module): Loss function
            optimizer (optim.Optimizer): Optimizer
            scheduler: Learning rate scheduler
            device (torch.device): Device to train on
            save_dir (str): Directory to save results
            log_interval (int): Interval for logging training progress
            class_names (List[str]): List of class names
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.class_names = class_names or [f"Class_{i}" for i in range(model.num_classes)]
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(os.path.join(save_dir, 'tensorboard_logs'))
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(
        self,
        num_epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        save_best: bool = True,
        save_last: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs (int): Number of epochs to train
            early_stopping (EarlyStopping): Early stopping callback
            save_best (bool): Whether to save the best model
            save_last (bool): Whether to save the last model
            
        Returns:
            Dict: Training history
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(
                    epoch + 1,
                    val_loss,
                    val_acc,
                    filename='best_model.pth'
                )
                print(f"New best validation accuracy: {val_acc:.2f}%")
            
            # Early stopping
            if early_stopping:
                if early_stopping(val_loss, self.model):
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                    break
        
        # Save final model
        if save_last:
            self.save_checkpoint(
                num_epochs,
                val_loss,
                val_acc,
                filename='final_model.pth'
            )
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save training history
        self.save_history()
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_acc: float,
        filename: str = 'checkpoint.pth'
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'class_names': self.class_names,
            'model_config': {
                'num_classes': len(self.class_names),
                'model_type': self.model.__class__.__name__
            }
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            
        Returns:
            Dict: Checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        return checkpoint


def create_optimizer(model: nn.Module, optimizer_name: str = 'adam', 
                    learning_rate: float = 0.001, weight_decay: float = 1e-4, 
                    **kwargs) -> optim.Optimizer:
    """
    Create optimizer for training.
    
    Args:
        model (nn.Module): Model to optimize
        optimizer_name (str): Name of optimizer ('adam', 'sgd', 'adamw')
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        **kwargs: Additional optimizer parameters
        
    Returns:
        optim.Optimizer: Configured optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, 
                         weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, 
                          weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'sgd':
        momentum = kwargs.pop('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=learning_rate, 
                        momentum=momentum, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(optimizer: optim.Optimizer, scheduler_name: str = 'step', 
                    **kwargs) -> Optional[object]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer (optim.Optimizer): Optimizer to schedule
        scheduler_name (str): Name of scheduler ('step', 'plateau', 'cosine')
        **kwargs: Additional scheduler parameters
        
    Returns:
        Learning rate scheduler
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'step':
        step_size = kwargs.pop('step_size', 10)
        gamma = kwargs.pop('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs)
    elif scheduler_name == 'plateau':
        patience = kwargs.pop('patience', 5)
        factor = kwargs.pop('factor', 0.5)
        return ReduceLROnPlateau(optimizer, mode='min', patience=patience, 
                                factor=factor, verbose=True, **kwargs)
    elif scheduler_name == 'cosine':
        T_max = kwargs.pop('T_max', 50)
        return CosineAnnealingLR(optimizer, T_max=T_max, **kwargs)
    elif scheduler_name == 'none' or scheduler_name is None:
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def create_criterion(criterion_name: str = 'crossentropy', 
                    class_weights: Optional[torch.Tensor] = None, 
                    **kwargs) -> nn.Module:
    """
    Create loss function.
    
    Args:
        criterion_name (str): Name of loss function
        class_weights (torch.Tensor): Class weights for imbalanced datasets
        **kwargs: Additional criterion parameters
        
    Returns:
        nn.Module: Loss function
    """
    criterion_name = criterion_name.lower()
    
    if criterion_name == 'crossentropy':
        return nn.CrossEntropyLoss(weight=class_weights, **kwargs)
    elif criterion_name == 'focal':
        # Focal loss implementation would go here
        # For now, fall back to CrossEntropy
        print("Warning: Focal loss not implemented, using CrossEntropy")
        return nn.CrossEntropyLoss(weight=class_weights, **kwargs)
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")


if __name__ == "__main__":
    # Example usage
    print("Training pipeline components ready!")
    print("Available optimizers: adam, adamw, sgd")
    print("Available schedulers: step, plateau, cosine")
    print("Available criterions: crossentropy")