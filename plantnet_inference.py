#!/usr/bin/env python3
"""
PlantNet Inference - Production Ready
Complete inference pipeline using your trained ensemble model
"""

import os
import sys
import time
import torch
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Disable PyTorch compilation for stability
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Add project root to path  
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from models.cnn_models import EnsembleModel

class PlantNetInference:
    def __init__(self, model_path=None, class_names_path=None):
        """Initialize PlantNet inference system"""
        
        # Default paths
        if model_path is None:
            model_path = "results_mi300x/plant_disease_mi300x_ensemble/best_model.pth"
        if class_names_path is None:
            class_names_path = "results_mi300x/plant_disease_mi300x_ensemble/class_names.json"
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("🚀 PlantNet Disease Detection System")
        print(f"📱 Device: {self.device}")
        print(f"💾 Model: {model_path}")
        
        # Load class names
        print("📋 Loading class names...")
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        print(f"🏷️ Classes: {len(self.class_names)} disease types")
        
        # Create and load model
        print("🧠 Loading ensemble model...")
        self.model = self._create_and_load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("✅ PlantNet ready for inference!")
        print("-" * 50)
    
    def _create_and_load_model(self, model_path):
        """Create ensemble model and load trained weights"""
        
        # Create ensemble model architecture
        model = EnsembleModel(num_classes=len(self.class_names))
        model = model.to(self.device)
        
        # Load checkpoint
        print(f"📦 Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load model weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"📊 Loaded from epoch: {checkpoint['epoch']}")
            if 'val_acc' in checkpoint:
                print(f"🎯 Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()  # Set to evaluation mode
        return model
    
    def predict_image(self, image_path, top_k=5, show_confidence=True):
        """Predict plant disease from image path"""
        
        # Load and validate image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        try:
            image = Image.open(image_path).convert('RGB')
            print(f"🖼️ Image loaded: {image.size}")
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        print("🔍 Running inference...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        inference_time = time.time() - start_time
        print(f"⚡ Inference time: {inference_time:.3f} seconds")
        
        # Get top predictions
        top_k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Format results
        results = []
        for i in range(top_k):
            prob = top_probs[0][i].item()
            idx = top_indices[0][i].item()
            class_name = self.class_names[idx]
            
            # Parse class name
            parts = class_name.split('___')
            plant = parts[0].replace('_', ' ').title()
            condition = parts[1].replace('_', ' ').title() if len(parts) > 1 else class_name
            
            results.append({
                'plant': plant,
                'condition': condition,
                'full_name': class_name,
                'confidence': prob * 100,
                'rank': i + 1
            })
        
        return results
    
    def display_results(self, results, image_path):
        """Display prediction results in a nice format"""
        
        if not results:
            print("❌ No results to display")
            return
        
        print("\n" + "="*60)
        print(f"🌱 PLANTNET DISEASE DETECTION RESULTS")
        print(f"📸 Image: {Path(image_path).name}")
        print("="*60)
        
        # Primary diagnosis (top prediction)
        top_result = results[0]
        is_healthy = 'healthy' in top_result['condition'].lower()
        primary_emoji = "✅" if is_healthy else "🦠"
        
        print(f"\n{primary_emoji} PRIMARY DIAGNOSIS:")
        print(f"   Plant: {top_result['plant']}")
        print(f"   Condition: {top_result['condition']}")
        print(f"   Confidence: {top_result['confidence']:.1f}%")
        
        if is_healthy:
            print(f"   Status: HEALTHY PLANT ✅")
        else:
            print(f"   Status: DISEASE DETECTED ⚠️")
        
        # Alternative diagnoses
        if len(results) > 1:
            print(f"\n🔍 Alternative Diagnoses:")
            for result in results[1:]:
                alt_emoji = "✅" if 'healthy' in result['condition'].lower() else "🦠"
                print(f"   {result['rank']}. {alt_emoji} {result['plant']} - {result['condition']}")
                print(f"      Confidence: {result['confidence']:.1f}%")
        
        print("\n" + "="*60)
    
    def batch_predict(self, image_folder, output_file=None):
        """Predict multiple images in a folder"""
        
        image_folder = Path(image_folder)
        if not image_folder.exists():
            print(f"❌ Folder not found: {image_folder}")
            return
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in image_folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No image files found in {image_folder}")
            return
        
        print(f"📁 Found {len(image_files)} images")
        
        all_results = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")
            
            results = self.predict_image(str(image_file), top_k=1, show_confidence=False)
            if results:
                result = results[0]
                result['image_file'] = image_file.name
                all_results.append(result)
                
                status = "HEALTHY" if 'healthy' in result['condition'].lower() else "DISEASED"
                print(f"   Result: {result['plant']} - {result['condition']} ({result['confidence']:.1f}%) - {status}")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        
        return all_results

def main():
    """Main function for command-line usage"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python plantnet_inference.py <image_path>")
        print("  Batch images: python plantnet_inference.py <folder_path> --batch")
        print("\nExample:")
        print("  python plantnet_inference.py test_image.jpg")
        print("  python plantnet_inference.py test_images/ --batch")
        return
    
    # Initialize PlantNet
    try:
        plantnet = PlantNetInference()
    except Exception as e:
        print(f"❌ Failed to initialize PlantNet: {e}")
        return
    
    image_path = sys.argv[1]
    
    # Check if batch mode
    if len(sys.argv) > 2 and sys.argv[2] == '--batch':
        # Batch prediction
        results = plantnet.batch_predict(image_path)
        print(f"\n📊 Processed {len(results)} images successfully!")
        
    else:
        # Single image prediction
        results = plantnet.predict_image(image_path)
        
        if results:
            plantnet.display_results(results, image_path)
        else:
            print("❌ Prediction failed")

if __name__ == "__main__":
    main()