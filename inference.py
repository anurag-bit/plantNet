#!/usr/bin/env python3
"""
Inference script for plant disease detection.
Use trained models to make predictions on new images.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.cnn_models import create_model


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint['model_config']
    num_classes = model_config['num_classes']
    
    # Create model (assuming ResNet for now, can be extended)
    model = create_model('resnet', num_classes, model_name='resnet50', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get class names
    class_names = checkpoint['class_names']
    
    return model, class_names


def get_transform(img_size: int = 224):
    """Get preprocessing transform for inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def predict_image(model, image_path: str, class_names: list, 
                 device: torch.device, top_k: int = 5):
    """Make prediction on a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()
        
        predictions = []
        for i in range(top_k):
            if top_k == 1:
                idx = top_indices.item()
                prob = top_probs.item()
            else:
                idx = top_indices[i]
                prob = top_probs[i]
            
            predictions.append({
                'class': class_names[idx],
                'probability': float(prob),
                'confidence': f"{prob * 100:.2f}%"
            })
    
    return predictions


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Plant Disease Detection Inference')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return 1
    
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return 1
    
    try:
        # Load model
        print("Loading model...")
        model, class_names = load_model(args.model, device)
        print(f"Model loaded with {len(class_names)} classes")
        
        # Make prediction
        print(f"Analyzing image: {args.image}")
        predictions = predict_image(model, args.image, class_names, device, args.top_k)
        
        # Display results
        print(f"\nTop {args.top_k} Predictions:")
        print("-" * 50)
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['class']} - {pred['confidence']}")
        
        # Display best prediction prominently
        best_pred = predictions[0]
        print(f"\nBest Prediction: {best_pred['class']} ({best_pred['confidence']})")
        
        # Provide interpretation
        if best_pred['probability'] > 0.8:
            confidence_level = "Very High"
        elif best_pred['probability'] > 0.6:
            confidence_level = "High"
        elif best_pred['probability'] > 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        print(f"Confidence Level: {confidence_level}")
        
        # Disease vs Healthy interpretation
        if "healthy" in best_pred['class'].lower():
            print("Status: Plant appears to be healthy ✓")
        else:
            print("Status: Plant disease detected ⚠")
            print("Recommendation: Consider consulting with agricultural experts")
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)