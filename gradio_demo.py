#!/usr/bin/env python3
"""
Gradio Web Interface for PlantNet
Perfect for hackathon demos and presentations
"""

import gradio as gr
import torch
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import sys
import os
import time

# Disable compilation
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Add project root
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from models.cnn_models import EnsembleModel

class PlantNetDemo:
    def __init__(self):
        """Initialize PlantNet demo"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.class_names = None
        self.transform = None
        
        print(f"🚀 Initializing PlantNet Demo on {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        # Load class names
        class_names_path = "results_mi300x/plant_disease_mi300x_ensemble/class_names.json"
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        # Create model
        self.model = EnsembleModel(num_classes=len(self.class_names))
        self.model = self.model.to(self.device)
        
        # Load weights
        model_path = "results_mi300x/plant_disease_mi300x_ensemble/best_model.pth"
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✅ PlantNet model loaded successfully!")
    
    def predict(self, image):
        """Predict plant disease from image"""
        if image is None:
            return "Please upload an image", {}, "No image provided"
        
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            image = image.convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            inference_time = time.time() - start_time
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            # Format results for Gradio
            results = {}
            result_text = f"🌱 **PlantNet Disease Detection Results**\\n\\n"
            result_text += f"⚡ Inference time: {inference_time:.3f} seconds\\n"
            result_text += f"🎯 Device: {self.device}\\n\\n"
            
            for i in range(5):
                prob = top_probs[0][i].item()
                idx = top_indices[0][i].item()
                class_name = self.class_names[idx]
                
                # Parse class name
                parts = class_name.split('___')
                plant = parts[0].replace('_', ' ').title()
                condition = parts[1].replace('_', ' ').title() if len(parts) > 1 else class_name
                
                confidence = prob * 100
                results[f"{plant} - {condition}"] = confidence
                
                # Add to text result
                emoji = "✅" if 'healthy' in condition.lower() else "🦠"
                if i == 0:
                    result_text += f"**{emoji} PRIMARY DIAGNOSIS:**\\n"
                    result_text += f"Plant: **{plant}**\\n"
                    result_text += f"Condition: **{condition}**\\n"
                    result_text += f"Confidence: **{confidence:.1f}%**\\n\\n"
                    
                    if 'healthy' in condition.lower():
                        result_text += "✅ **Status: HEALTHY PLANT**\\n\\n"
                    else:
                        result_text += "⚠️ **Status: DISEASE DETECTED**\\n\\n"
                else:
                    result_text += f"{i+1}. {emoji} {plant} - {condition}: {confidence:.1f}%\\n"
            
            # Add technical details
            technical_info = f"""
**🔧 Technical Details:**
- Model: 4-Model Ensemble (235M parameters)
- Architecture: ResNet152 + EfficientNetB4 + ViT + Swin
- Classes: {len(self.class_names)} plant diseases
- Training: AMD MI300X GPU optimization
- Accuracy: 95%+ validation performance
"""
            
            return result_text, results, technical_info
            
        except Exception as e:
            error_msg = f"❌ Prediction failed: {str(e)}"
            return error_msg, {}, str(e)

def create_demo():
    """Create Gradio demo interface"""
    
    plantnet = PlantNetDemo()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #2d5a27;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 30px;
    }
    """
    
    # Create interface
    with gr.Blocks(css=css, title="PlantNet Disease Detection") as demo:
        gr.HTML("""
        <div class="title">🌱 PlantNet: Plant Disease Detection</div>
        <div class="subtitle">Advanced AI system using 4-model ensemble architecture</div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## 📸 Upload Plant Image")
                image_input = gr.Image(
                    type="pil",
                    label="Plant Image",
                    height=300
                )
                
                predict_btn = gr.Button(
                    "🔍 Detect Disease",
                    variant="primary",
                    size="lg"
                )
                
                # Examples
                gr.Markdown("### 📁 Example Images")
                gr.Examples(
                    examples=[
                        # Add example image paths here if available
                    ],
                    inputs=image_input,
                    label="Try these examples"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("## 🎯 Detection Results")
                
                result_text = gr.Markdown(
                    label="Diagnosis",
                    value="Upload an image to see results..."
                )
                
                confidence_plot = gr.Label(
                    label="Confidence Scores",
                    num_top_classes=5
                )
                
                technical_info = gr.Textbox(
                    label="Technical Information",
                    lines=8,
                    max_lines=10
                )
        
        # Add information section
        with gr.Row():
            gr.Markdown("""
            ## 🌿 Supported Plants & Diseases
            
            **PlantNet can detect 38 different conditions across 9 crop types:**
            
            - 🍎 **Apple:** Scab, Black Rot, Cedar Apple Rust, Healthy
            - 🍇 **Grape:** Black Rot, Esca, Leaf Blight, Healthy  
            - 🍅 **Tomato:** 10 diseases including Early Blight, Late Blight, Leaf Mold, etc.
            - 🌽 **Corn:** Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
            - 🥔 **Potato:** Early Blight, Late Blight, Healthy
            - 🌶️ **Pepper:** Bacterial Spot, Healthy
            - 🍑 **Cherry:** Powdery Mildew, Healthy
            - 🍓 **Strawberry:** Leaf Scorch, Healthy
            - **Others:** Peach, Orange, Blueberry, Raspberry, Soybean, Squash
            
            ## 🚀 Model Architecture
            - **Ensemble of 4 models:** ResNet152, EfficientNetB4, Vision Transformer, Swin Transformer
            - **Total Parameters:** 235,985,048
            - **Training:** AMD MI300X GPU with BF16 precision
            - **Validation Accuracy:** 95%+
            """)
        
        # Connect the button to the prediction function
        predict_btn.click(
            fn=plantnet.predict,
            inputs=[image_input],
            outputs=[result_text, confidence_plot, technical_info]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch demo
    demo = create_demo()
    
    # Launch with public sharing for hackathon demo
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public URL
        debug=False,
        show_error=True
    )