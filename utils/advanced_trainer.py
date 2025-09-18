import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
import os
import math
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import json
import warnings


class CosineWarmupScheduler:
    """Cosine annealing scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 min_lr: float = 1e-6, warmup_start_lr: float = 1e-8):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Warmup phase
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.warmup_start_lr + (self.base_lrs[i] - self.warmup_start_lr) * epoch / self.warmup_epochs
                param_group['lr'] = lr
        else:
            # Cosine annealing phase
            for i, param_group in enumerate(self.optimizer.param_groups):
                cosine_epochs = self.max_epochs - self.warmup_epochs
                current_epoch = epoch - self.warmup_epochs
                lr = self.min_lr + (self.base_lrs[i] - self.min_lr) * \
                     (1 + math.cos(math.pi * current_epoch / cosine_epochs)) / 2
                param_group['lr'] = lr


class MixUpLoss(nn.Module):
    """Mixup loss function."""
    
    def __init__(self, criterion):
        super(MixUpLoss, self).__init__()
        self.criterion = criterion
        
    def forward(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss."""
    
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred, target):
        log_prob = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        if self.weight is not None:
            nll_loss = nll_loss * self.weight[target]
        
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class AdvancedTrainer:
    """Advanced training pipeline optimized for high-end hardware."""
    
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
        class_names: List[str] = None,
        mixed_precision: str = 'bf16',
        compile_model: bool = True,
        gradient_clip_norm: float = 1.0,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        label_smoothing: float = 0.1
    ):
        """
        Initialize the advanced trainer.
        
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
            mixed_precision (str): Mixed precision mode ('fp16', 'bf16', 'none')
            compile_model (bool): Whether to compile model for optimization
            gradient_clip_norm (float): Gradient clipping norm
            mixup_alpha (float): Mixup alpha parameter
            cutmix_alpha (float): CutMix alpha parameter
            label_smoothing (float): Label smoothing factor
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
        self.gradient_clip_norm = gradient_clip_norm
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        
        # Mixed precision setup
        self.mixed_precision = mixed_precision
        if mixed_precision in ['fp16', 'bf16']:
            self.scaler = GradScaler()
            self.autocast_dtype = torch.float16 if mixed_precision == 'fp16' else torch.bfloat16
        else:
            self.scaler = None
            self.autocast_dtype = torch.float32
        
        # Model compilation for optimization
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                print("✓ Model compiled for optimization")
            except Exception as e:
                print(f"Warning: Model compilation failed: {e}")
        
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
            'learning_rates': [],
            'gpu_memory': []
        }
        
        # Setup advanced loss functions
        if label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        if mixup_alpha > 0:
            self.mixup_loss = MixUpLoss(self.criterion)
    
    def mixup_data(self, x, y, alpha=1.0):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(self, x, y, alpha=1.0):
        """Apply CutMix augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        
        # Generate random box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch with advanced augmentations."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Track GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1e9
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # Apply data augmentations
            use_mixup = np.random.random() < 0.5 and self.mixup_alpha > 0
            use_cutmix = np.random.random() < 0.5 and self.cutmix_alpha > 0 and not use_mixup
            
            if use_mixup:
                data, target_a, target_b, lam = self.mixup_data(data, target, self.mixup_alpha)
            elif use_cutmix:
                data, target_a, target_b, lam = self.cutmix_data(data, target, self.cutmix_alpha)
            else:
                target_a, target_b, lam = target, target, 1.0
            
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision forward pass
            if self.mixed_precision in ['fp16', 'bf16']:
                with autocast(dtype=self.autocast_dtype):
                    output = self.model(data)
                    if use_mixup or use_cutmix:
                        loss = self.mixup_loss(output, target_a, target_b, lam)
                    else:
                        loss = self.criterion(output, target)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                if use_mixup or use_cutmix:
                    loss = self.mixup_loss(output, target_a, target_b, lam)
                else:
                    loss = self.criterion(output, target)
                
                loss.backward()
                
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.optimizer.step()
            
            # Calculate accuracy
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            
            if use_mixup or use_cutmix:
                correct += (lam * predicted.eq(target_a.data).cpu().sum().float() + 
                           (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())
            else:
                correct += predicted.eq(target.data).cpu().sum()
            
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%',
                    'LR': f'{current_lr:.6f}'
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        # Track GPU memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            self.history['gpu_memory'].append(peak_memory)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for data, target in pbar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                if self.mixed_precision in ['fp16', 'bf16']:
                    with autocast(dtype=self.autocast_dtype):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum()
                
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
        early_stopping = None,
        save_best: bool = True,
        save_last: bool = True,
        save_intermediate: List[int] = None
    ) -> Dict:
        """Train the model with advanced optimizations."""
        print(f"🚀 Training on device: {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"⚡ Mixed precision: {self.mixed_precision}")
        print(f"🎯 Batch size: {self.train_loader.batch_size}")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\n{'='*60}")
            print(f"🌱 Epoch {epoch + 1}/{num_epochs}")
            print('='*60)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step(epoch)
                elif hasattr(self.scheduler, 'step') and hasattr(self.scheduler, 'metric'):
                    self.scheduler.step(val_loss)
            
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
            
            if torch.cuda.is_available():
                self.writer.add_scalar('GPU_Memory_GB', self.history['gpu_memory'][-1], epoch)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"🎯 Learning Rate: {current_lr:.8f}")
            print(f"⏱️  Epoch Time: {epoch_time:.2f}s")
            
            if torch.cuda.is_available():
                print(f"🖥️  GPU Memory: {self.history['gpu_memory'][-1]:.2f} GB")
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(
                    epoch + 1,
                    val_loss,
                    val_acc,
                    filename='best_model.pth'
                )
                print(f"🏆 New best validation accuracy: {val_acc:.2f}%")
            
            # Save intermediate models
            if save_intermediate and (epoch + 1) in save_intermediate:
                self.save_checkpoint(
                    epoch + 1,
                    val_loss,
                    val_acc,
                    filename=f'model_epoch_{epoch+1}.pth'
                )
                print(f"💾 Intermediate model saved at epoch {epoch + 1}")
            
            # Early stopping
            if early_stopping:
                if early_stopping(val_loss, self.model):
                    print(f"⏹️  Early stopping triggered after epoch {epoch + 1}")
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
        print(f"\n🎉 Training completed in {total_time:.2f}s ({total_time/3600:.2f}h)")
        print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save training history
        self.save_history()
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.history
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, filename: str = 'checkpoint.pth'):
        """Save model checkpoint with comprehensive information."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'class_names': self.class_names,
            'history': self.history,
            'model_config': {
                'num_classes': len(self.class_names),
                'model_type': self.model.__class__.__name__,
                'mixed_precision': self.mixed_precision
            },
            'training_config': {
                'mixup_alpha': self.mixup_alpha,
                'cutmix_alpha': self.cutmix_alpha,
                'gradient_clip_norm': self.gradient_clip_norm
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


def create_advanced_optimizer(model: nn.Module, optimizer_name: str = 'adamw', 
                            learning_rate: float = 0.0003, weight_decay: float = 0.05, 
                            **kwargs) -> optim.Optimizer:
    """Create advanced optimizers optimized for high-end training."""
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, 
                          weight_decay=weight_decay, betas=(0.9, 0.999), 
                          eps=1e-8, **kwargs)
    elif optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, 
                         weight_decay=weight_decay, betas=(0.9, 0.999), **kwargs)
    elif optimizer_name == 'sgd':
        momentum = kwargs.pop('momentum', 0.9)
        nesterov = kwargs.pop('nesterov', True)
        return optim.SGD(model.parameters(), lr=learning_rate, 
                        momentum=momentum, weight_decay=weight_decay, 
                        nesterov=nesterov, **kwargs)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_advanced_scheduler(optimizer: optim.Optimizer, scheduler_name: str = 'cosine_warmup', 
                            epochs: int = 200, **kwargs):
    """Create advanced learning rate schedulers."""
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'cosine_warmup':
        warmup_epochs = kwargs.pop('warmup_epochs', 10)
        min_lr = kwargs.pop('min_lr', 1e-6)
        return CosineWarmupScheduler(optimizer, warmup_epochs, epochs, min_lr)
    elif scheduler_name == 'onecycle':
        max_lr = kwargs.pop('max_lr', 0.01)
        return OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, 
                         steps_per_epoch=kwargs.get('steps_per_epoch', 100))
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        # Fall back to original schedulers
        from .trainer import create_scheduler
        return create_scheduler(optimizer, scheduler_name, **kwargs)


if __name__ == "__main__":
    print("🚀 Advanced training pipeline ready for AMD MI300X!")
    print("✅ Features: Mixed precision, model compilation, advanced augmentations")
    print("⚡ Optimizations: CosineWarmup scheduler, AdamW optimizer, gradient clipping")