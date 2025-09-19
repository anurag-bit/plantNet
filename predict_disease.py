#!/usr/bin/env python3
"""
PlantNet Inference Script
Use your trained ensemble model to detect plant diseases from images
"""

import os
import sys
import torch
import json
import argparse
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from models.cnn_models import create_model

class PlantDiseasePredictor:
    def __init__(self, model_path, class_names_path, device='cuda'):
        """Initialize the plant disease predictor"""
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        print(f"🌱 Loading PlantNet model from: {model_path}")
        print(f"📱 Device: {self.device}")
        print(f"🏷️ Classes: {len(self.class_names)} disease types")
        
        # Load the trained model
        self.model = self._load_model(model_path)
        
        # Define image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("✅ Model loaded successfully!")
    
    def _load_model(self, model_path):
        """Load the trained ensemble model"""
        # Create model architecture (ensemble)
        model = create_model(
            model_type="ensemble",
            num_classes=len(self.class_names),
            pretrained=False  # We're loading trained weights
        ).to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📊 Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_acc' in checkpoint:
                print(f"🎯 Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def predict(self, image_path, top_k=5):
        """Predict plant disease from image"""
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))
            
            results = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_name = self.class_names[idx.item()]
                confidence = prob.item() * 100
                results.append({
                    'disease': class_name,
                    'confidence': confidence,
                    'plant': class_name.split('___')[0],
                    'condition': class_name.split('___')[1] if '___' in class_name else class_name
                })
        
        return results
    
    def predict_and_display(self, image_path, top_k=5):
        """Predict and display results in a nice format"""
        print(f"\n🔍 Analyzing image: {image_path}")
        print("=" * 50)
        
        results = self.predict(image_path, top_k)
        
        if results is None:
            return
        
        print(f"🌱 Plant Disease Detection Results:")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            plant = result['plant'].replace('_', ' ').title()
            condition = result['condition'].replace('_', ' ').title()
            confidence = result['confidence']
            
            # Add emoji based on condition
            if 'healthy' in condition.lower():
                emoji = "✅"
            else:
                emoji = "🦠"
            
            print(f"{i}. {emoji} {plant} - {condition}")
            print(f"   Confidence: {confidence:.1f}%")
            
            if i == 1:  # Highlight top prediction
                print(f"   >>> PRIMARY DIAGNOSIS <<<")
            print()

def main():
    parser = argparse.ArgumentParser(description="PlantNet Disease Detection")
    parser.add_argument("image_path", help="Path to plant image")
    parser.add_argument("--model_path", 
                       default="results_mi300x/plant_disease_mi300x_ensemble/best_model.pth",
                       help="Path to trained model")
    parser.add_argument("--class_names", 
                       default="results_mi300x/plant_disease_mi300x_ensemble/class_names.json",
                       help="Path to class names JSON")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Show top K predictions")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return
    
    if not os.path.exists(args.class_names):
        print(f"❌ Class names file not found: {args.class_names}")
        return
    
    # Create predictor and run inference
    try:
        predictor = PlantDiseasePredictor(args.model_path, args.class_names, args.device)
        predictor.predict_and_display(args.image_path, args.top_k)
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()