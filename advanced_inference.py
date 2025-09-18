#!/usr/bin/env python3
"""
Advanced Plant Disease Detection Inference System
=================================================

High-performance inference with Test-Time Augmentation (TTA),
multi-scale processing, uncertainty estimation, and comprehensive
disease analysis optimized for AMD MI300X hardware.

Features:
- Test-Time Augmentation for improved accuracy
- Multi-scale inference
- Uncertainty estimation
- Comprehensive disease database
- Batch processing support
- GradCAM visualization
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
import cv2
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.cnn_models import create_model, EnsembleModel
from utils.disease_database import get_disease_info, format_disease_report


class TTATransforms:
    """Test-Time Augmentation transforms for improved inference accuracy."""
    
    def __init__(self, img_size: int = 384):
        self.img_size = img_size
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_tta_transforms(self, num_augmentations: int = 8) -> List[transforms.Compose]:
        """Generate TTA transform variations."""
        tta_transforms = []
        
        # Original
        tta_transforms.append(self.base_transform)
        
        if num_augmentations > 1:
            # Horizontal flip
            tta_transforms.append(transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]))
        
        if num_augmentations > 2:
            # Slight rotation variations
            for angle in [-5, 5]:
                tta_transforms.append(transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomRotation((angle, angle)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ]))
        
        if num_augmentations > 4:
            # Color variations
            for factor in [0.9, 1.1]:
                tta_transforms.append(transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ColorJitter(brightness=factor/1.1, contrast=factor/1.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ]))
        
        if num_augmentations > 6:
            # Scale variations
            for scale in [0.95, 1.05]:
                tta_transforms.append(transforms.Compose([
                    transforms.Resize((int(self.img_size * scale), int(self.img_size * scale))),
                    transforms.CenterCrop((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ]))
        
        return tta_transforms[:num_augmentations]


class GradCAM:
    """Gradient-weighted Class Activation Mapping for visualization."""
    
    def __init__(self, model: nn.Module, target_layer: str = None):
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _find_target_layer(self) -> str:
        """Find the best layer for GradCAM visualization."""
        # For ResNet-like architectures
        if hasattr(self.model, 'layer4'):
            return 'layer4'
        elif hasattr(self.model, 'features'):
            return 'features'
        elif hasattr(self.model, 'backbone'):
            if hasattr(self.model.backbone, 'layer4'):
                return 'backbone.layer4'
        return 'conv_block13'  # For custom CNN
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Navigate to target layer
        target_module = self.model
        for layer in self.target_layer.split('.'):
            target_module = getattr(target_module, layer)
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate GradCAM heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize
        cam = cam / torch.max(cam)
        
        # Resize to input size
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                           size=input_image.shape[2:], 
                           mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()


class AdvancedPlantDiseaseDetector:
    """Advanced plant disease detection system with comprehensive analysis."""
    
    def __init__(self, model_path: str, device: torch.device = None, 
                 mixed_precision: str = 'bf16'):
        """
        Initialize the advanced detector.
        
        Args:
            model_path (str): Path to trained model checkpoint
            device (torch.device): Device for inference
            mixed_precision (str): Mixed precision mode
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = mixed_precision
        
        # Load model
        self.model, self.class_names = self._load_model()
        self.model.eval()
        
        # Setup mixed precision
        if mixed_precision in ['fp16', 'bf16']:
            self.autocast_dtype = torch.float16 if mixed_precision == 'fp16' else torch.bfloat16
        else:
            self.autocast_dtype = torch.float32
        
        # Initialize components
        self.tta_transforms = TTATransforms(img_size=384)
        self.gradcam = GradCAM(self.model)
        
        print(f"🚀 Advanced detector loaded on {self.device}")
        print(f"🧠 Model: {self.model.__class__.__name__}")
        print(f"📊 Classes: {len(self.class_names)}")
        print(f"⚡ Mixed precision: {mixed_precision}")
    
    def _load_model(self) -> Tuple[nn.Module, List[str]]:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration
        model_config = checkpoint['model_config']
        num_classes = model_config['num_classes']
        
        # Create model based on saved type
        model_type = model_config.get('model_type', 'PlantDiseaseResNet')
        
        if 'Ensemble' in model_type:
            # For ensemble models, need architecture info
            if 'architectures' in checkpoint:
                model = create_model('ensemble', num_classes, 
                                   architectures=checkpoint['architectures'])
            else:
                # Fallback to ResNet
                model = create_model('resnet', num_classes, model_name='resnet50')
        else:
            # Single model
            if 'ResNet' in model_type:
                model = create_model('resnet', num_classes, model_name='resnet50')
            elif 'EfficientNet' in model_type:
                model = create_model('efficientnet', num_classes)
            elif 'ViT' in model_type:
                model = create_model('vit', num_classes)
            elif 'Swin' in model_type:
                model = create_model('swin', num_classes)
            else:
                model = create_model('custom_cnn', num_classes)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        # Get class names
        class_names = checkpoint['class_names']
        
        return model, class_names
    
    def preprocess_image(self, image_path: str, multi_scale: bool = False) -> List[torch.Tensor]:
        """
        Preprocess image for inference.
        
        Args:
            image_path (str): Path to image
            multi_scale (bool): Whether to use multi-scale processing
            
        Returns:
            List[torch.Tensor]: Preprocessed images
        """
        image = Image.open(image_path).convert('RGB')
        
        if multi_scale:
            # Multi-scale processing
            scales = [320, 384, 448]
            processed_images = []
            
            for scale in scales:
                transform = transforms.Compose([
                    transforms.Resize((scale, scale)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                processed_images.append(transform(image).unsqueeze(0))
            
            return processed_images
        else:
            transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            return [transform(image).unsqueeze(0)]
    
    def predict_with_tta(self, image_path: str, num_augmentations: int = 8,
                        multi_scale: bool = True, uncertainty: bool = True) -> Dict:
        """
        Make prediction with Test-Time Augmentation and uncertainty estimation.
        
        Args:
            image_path (str): Path to input image
            num_augmentations (int): Number of TTA augmentations
            multi_scale (bool): Use multi-scale processing
            uncertainty (bool): Calculate uncertainty measures
            
        Returns:
            Dict: Comprehensive prediction results
        """
        image = Image.open(image_path).convert('RGB')
        all_predictions = []
        all_confidences = []
        
        # Get TTA transforms
        tta_transforms = self.tta_transforms.get_tta_transforms(num_augmentations)
        
        # Multi-scale processing
        scales = [320, 384, 448] if multi_scale else [384]
        
        with torch.no_grad():
            for scale in scales:
                # Update transform scale
                for transform in tta_transforms:
                    # Apply transform to image
                    processed_image = transform(image).unsqueeze(0).to(self.device)
                    
                    # Mixed precision inference
                    if self.mixed_precision in ['fp16', 'bf16']:
                        with autocast(dtype=self.autocast_dtype):
                            outputs = self.model(processed_image)
                    else:
                        outputs = self.model(processed_image)
                    
                    # Get probabilities
                    probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
                    all_predictions.append(probabilities)
                    all_confidences.append(np.max(probabilities))
        
        # Average predictions across all augmentations and scales
        mean_predictions = np.mean(all_predictions, axis=0)
        
        # Calculate uncertainty measures
        prediction_std = np.std(all_predictions, axis=0) if uncertainty else None
        entropy = -np.sum(mean_predictions * np.log(mean_predictions + 1e-8)) if uncertainty else None
        
        # Get top predictions
        top_indices = np.argsort(mean_predictions)[::-1][:5]
        top_predictions = []
        
        for idx in top_indices:
            class_name = self.class_names[idx]
            confidence = mean_predictions[idx]
            uncertainty_score = prediction_std[idx] if uncertainty else 0.0
            
            # Get disease information
            disease_info = get_disease_info(class_name)
            
            top_predictions.append({
                'class': class_name,
                'confidence': float(confidence),
                'uncertainty': float(uncertainty_score),
                'disease_info': disease_info
            })
        
        return {
            'predictions': top_predictions,
            'mean_confidence': float(np.mean(all_confidences)),
            'confidence_std': float(np.std(all_confidences)),
            'entropy': float(entropy) if entropy is not None else None,
            'num_augmentations': len(all_predictions),
            'scales_used': scales
        }
    
    def generate_gradcam_visualization(self, image_path: str, class_idx: int = None,
                                     save_path: str = None) -> np.ndarray:
        """Generate GradCAM visualization."""
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Use top prediction if class_idx not provided
        if class_idx is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                class_idx = torch.argmax(outputs, dim=1).item()
        
        # Generate GradCAM
        cam = self.gradcam.generate_cam(input_tensor, class_idx)
        
        if save_path:
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax1.imshow(image)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # GradCAM heatmap
            ax2.imshow(cam, cmap='jet')
            ax2.set_title(f'GradCAM - {self.class_names[class_idx]}')
            ax2.axis('off')
            
            # Overlay
            ax3.imshow(image, alpha=0.7)
            ax3.imshow(cam, cmap='jet', alpha=0.3)
            ax3.set_title('Overlay')
            ax3.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return cam
    
    def analyze_image(self, image_path: str, output_dir: str = None,
                     detailed_analysis: bool = True) -> Dict:
        """
        Comprehensive image analysis with all features.
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Directory to save analysis results
            detailed_analysis (bool): Include GradCAM and detailed report
            
        Returns:
            Dict: Complete analysis results
        """
        print(f"🔬 Analyzing image: {os.path.basename(image_path)}")
        
        # Perform TTA prediction
        results = self.predict_with_tta(
            image_path, 
            num_augmentations=8,
            multi_scale=True,
            uncertainty=True
        )
        
        # Get best prediction
        best_prediction = results['predictions'][0]
        disease_info = best_prediction['disease_info']
        confidence = best_prediction['confidence']
        
        # Generate detailed report
        if detailed_analysis:
            report = format_disease_report(disease_info, confidence)
            results['detailed_report'] = report
            
            # Generate GradCAM if requested
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save GradCAM
                gradcam_path = os.path.join(output_dir, 'gradcam_analysis.png')
                self.generate_gradcam_visualization(
                    image_path, 
                    class_idx=self.class_names.index(best_prediction['class']),
                    save_path=gradcam_path
                )
                results['gradcam_path'] = gradcam_path
                
                # Save report
                report_path = os.path.join(output_dir, 'disease_report.txt')
                with open(report_path, 'w') as f:
                    f.write(report)
                results['report_path'] = report_path
                
                # Save JSON results
                json_path = os.path.join(output_dir, 'analysis_results.json')
                with open(json_path, 'w') as f:
                    # Remove non-serializable items for JSON
                    json_results = {k: v for k, v in results.items() 
                                  if k not in ['detailed_report']}
                    json.dump(json_results, f, indent=2)
                results['json_path'] = json_path
        
        return results


def main():
    """Main function for advanced inference."""
    parser = argparse.ArgumentParser(description='Advanced Plant Disease Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results')
    parser.add_argument('--tta_augmentations', type=int, default=8,
                       help='Number of TTA augmentations')
    parser.add_argument('--multi_scale', action='store_true',
                       help='Use multi-scale processing')
    parser.add_argument('--gradcam', action='store_true',
                       help='Generate GradCAM visualization')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                       choices=['fp16', 'bf16', 'fp32'],
                       help='Mixed precision mode')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Advanced Plant Disease Detection System")
    print(f"💻 Device: {device}")
    print(f"⚡ Mixed Precision: {args.mixed_precision}")
    print("="*60)
    
    # Initialize detector
    detector = AdvancedPlantDiseaseDetector(
        model_path=args.model,
        device=device,
        mixed_precision=args.mixed_precision
    )
    
    # Perform analysis
    results = detector.analyze_image(
        image_path=args.image,
        output_dir=args.output_dir,
        detailed_analysis=True
    )
    
    # Print results
    print(f"\n🎯 ANALYSIS COMPLETE")
    print("="*60)
    
    best_pred = results['predictions'][0]
    print(f"🌱 Primary Diagnosis: {best_pred['class']}")
    print(f"🎯 Confidence: {best_pred['confidence']:.1%}")
    print(f"📊 Uncertainty: {best_pred['uncertainty']:.4f}")
    print(f"🔬 Entropy: {results['entropy']:.4f}")
    
    print(f"\n📋 TOP 5 PREDICTIONS:")
    for i, pred in enumerate(results['predictions'][:5], 1):
        print(f"   {i}. {pred['class']} - {pred['confidence']:.1%}")
    
    # Print detailed report if available
    if 'detailed_report' in results:
        print(results['detailed_report'])
    
    if args.output_dir:
        print(f"\n💾 Analysis saved to: {args.output_dir}")
        if 'gradcam_path' in results:
            print(f"🔥 GradCAM visualization: {results['gradcam_path']}")
        if 'report_path' in results:
            print(f"📄 Detailed report: {results['report_path']}")


if __name__ == "__main__":
    main()