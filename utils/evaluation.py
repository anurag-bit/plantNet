import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import os
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str]):
        """
        Initialize the evaluator.
        
        Args:
            model (nn.Module): Trained model
            device (torch.device): Device to run evaluation on
            class_names (List[str]): List of class names
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate_model(self, data_loader, save_dir: str = None) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            data_loader: Data loader for evaluation
            save_dir (str): Directory to save evaluation results
            
        Returns:
            Dict: Evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        metrics['average_loss'] = total_loss / len(data_loader)
        
        # Generate visualizations
        if save_dir:
            self._save_evaluation_results(
                all_labels, all_predictions, all_probabilities, 
                metrics, save_dir
            )
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic accuracy
        accuracy = np.mean(y_true == y_pred)
        metrics['accuracy'] = accuracy
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['per_class_precision'] = precision.tolist()
        metrics['per_class_recall'] = recall.tolist()
        metrics['per_class_f1'] = f1.tolist()
        metrics['per_class_support'] = support.tolist()
        
        # Average metrics
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        metrics['weighted_precision'] = np.average(precision, weights=support)
        metrics['weighted_recall'] = np.average(recall, weights=support)
        metrics['weighted_f1'] = np.average(f1, weights=support)
        
        # Top-k accuracy (for top-3 and top-5)
        metrics['top_3_accuracy'] = self._calculate_topk_accuracy(y_true, y_prob, k=3)
        metrics['top_5_accuracy'] = self._calculate_topk_accuracy(y_true, y_prob, k=5)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def _calculate_topk_accuracy(self, y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy."""
        if k >= self.num_classes:
            return 1.0
        
        top_k_pred = np.argsort(y_prob, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_pred[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def _save_evaluation_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_prob: np.ndarray, metrics: Dict, save_dir: str):
        """Save evaluation results and generate visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate and save plots
        self._plot_confusion_matrix(y_true, y_pred, save_dir)
        self._plot_classification_report(metrics, save_dir)
        self._plot_per_class_metrics(metrics, save_dir)
        
        # Save detailed classification report
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, 
            output_dict=False, zero_division=0
        )
        report_path = os.path.join(save_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_dir: str):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_classification_report(self, metrics: Dict, save_dir: str):
        """Plot classification report as heatmap."""
        # Prepare data for heatmap
        data = []
        labels = []
        
        for i, class_name in enumerate(self.class_names):
            data.append([
                metrics['per_class_precision'][i],
                metrics['per_class_recall'][i],
                metrics['per_class_f1'][i]
            ])
            labels.append(class_name)
        
        # Add macro and weighted averages
        data.append([metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1']])
        data.append([metrics['weighted_precision'], metrics['weighted_recall'], metrics['weighted_f1']])
        labels.extend(['Macro Avg', 'Weighted Avg'])
        
        data = np.array(data)
        
        plt.figure(figsize=(8, max(10, len(labels) * 0.5)))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=['Precision', 'Recall', 'F1-Score'],
                   yticklabels=labels, cbar_kws={'label': 'Score'})
        plt.title('Classification Report Heatmap')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'classification_report_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, metrics: Dict, save_dir: str):
        """Plot per-class metrics as bar charts."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Precision
        axes[0, 0].bar(range(len(self.class_names)), metrics['per_class_precision'])
        axes[0, 0].set_title('Per-Class Precision')
        axes[0, 0].set_xlabel('Classes')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_xticks(range(len(self.class_names)))
        axes[0, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[0, 1].bar(range(len(self.class_names)), metrics['per_class_recall'])
        axes[0, 1].set_title('Per-Class Recall')
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_xticks(range(len(self.class_names)))
        axes[0, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1-Score
        axes[1, 0].bar(range(len(self.class_names)), metrics['per_class_f1'])
        axes[1, 0].set_title('Per-Class F1-Score')
        axes[1, 0].set_xlabel('Classes')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_xticks(range(len(self.class_names)))
        axes[1, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Support (number of samples per class)
        axes[1, 1].bar(range(len(self.class_names)), metrics['per_class_support'])
        axes[1, 1].set_title('Per-Class Support')
        axes[1, 1].set_xlabel('Classes')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_xticks(range(len(self.class_names)))
        axes[1, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def plot_training_history(history: Dict, save_dir: str = None):
    """
    Plot training history.
    
    Args:
        history (Dict): Training history containing losses and accuracies
        save_dir (str): Directory to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training and validation loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training and validation accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-', label='Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss difference (overfitting indicator)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 1].plot(epochs, loss_diff, 'm-', label='Val Loss - Train Loss')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Overfitting Indicator')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show() if save_dir is None else plt.close()


def visualize_predictions(model: nn.Module, data_loader, class_names: List[str], 
                         device: torch.device, num_samples: int = 16, 
                         save_dir: str = None, correct_only: bool = False):
    """
    Visualize model predictions on sample images.
    
    Args:
        model (nn.Module): Trained model
        data_loader: Data loader
        class_names (List[str]): List of class names
        device (torch.device): Device to run inference on
        num_samples (int): Number of samples to visualize
        save_dir (str): Directory to save the visualization
        correct_only (bool): Show only correct predictions
    """
    model.eval()
    
    images_shown = 0
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalization parameters (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                if images_shown >= num_samples:
                    break
                
                is_correct = predicted[i] == labels[i]
                if correct_only and not is_correct:
                    continue
                
                # Denormalize image
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = img * std + mean
                img = np.clip(img, 0, 1)
                
                row = images_shown // cols
                col = images_shown % cols
                
                axes[row, col].imshow(img)
                
                true_label = class_names[labels[i]]
                pred_label = class_names[predicted[i]]
                confidence = probabilities[i][predicted[i]].item() * 100
                
                # Set title color based on correctness
                color = 'green' if is_correct else 'red'
                title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
                axes[row, col].set_title(title, color=color, fontsize=10)
                axes[row, col].axis('off')
                
                images_shown += 1
            
            if images_shown >= num_samples:
                break
    
    # Hide unused subplots
    for i in range(images_shown, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = 'correct_predictions.png' if correct_only else 'sample_predictions.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    
    plt.show() if save_dir is None else plt.close()


if __name__ == "__main__":
    # Example usage
    print("Model evaluation utilities ready!")
    print("Available functions:")
    print("- ModelEvaluator: Comprehensive model evaluation")
    print("- plot_training_history: Plot training metrics")
    print("- visualize_predictions: Visualize model predictions")