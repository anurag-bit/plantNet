#!/usr/bin/env python3
"""
FastAPI Deployment for PlantNet
High-performance inference API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import json
import io
import uvicorn
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import sys
import os
import time

# Disable compilation for stability
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Add project root
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from models.cnn_models import EnsembleModel

# Initialize FastAPI
app = FastAPI(
    title="PlantNet Disease Detection API",
    description="Advanced plant disease detection using ensemble learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model instance
model = None
class_names = None
device = None
transform = None

def load_model():
    """Load the trained PlantNet model"""
    global model, class_names, device, transform
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Loading PlantNet on {device}")
    
    # Load class names
    class_names_path = "results_mi300x/plant_disease_mi300x_ensemble/class_names.json"
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    # Create and load model
    model = EnsembleModel(num_classes=len(class_names))
    model = model.to(device)
    
    # Load weights
    model_path = "results_mi300x/plant_disease_mi300x_ensemble/best_model.pth"
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("✅ PlantNet model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "PlantNet Disease Detection API",
        "version": "1.0.0",
        "status": "online",
        "model_classes": len(class_names) if class_names else 0,
        "endpoints": {
            "/predict": "POST - Upload image for disease detection",
            "/health": "GET - API health check",
            "/classes": "GET - List all supported disease classes",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": str(device) if device else None,
        "classes_count": len(class_names) if class_names else 0
    }

@app.get("/classes")
async def get_classes():
    """Get all supported disease classes"""
    if not class_names:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Group by plant type
    plants = {}
    for class_name in class_names:
        parts = class_name.split('___')
        plant = parts[0].replace('_', ' ').title()
        condition = parts[1].replace('_', ' ').title() if len(parts) > 1 else class_name
        
        if plant not in plants:
            plants[plant] = []
        plants[plant].append(condition)
    
    return {
        "total_classes": len(class_names),
        "plants_count": len(plants),
        "plants": plants,
        "all_classes": class_names
    }

@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    top_k: int = 5,
    confidence_threshold: float = 0.1
):
    """Predict plant disease from uploaded image"""
    
    if not model or not class_names:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        start_time = time.time()
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        inference_time = time.time() - start_time
        
        # Get top predictions
        top_k = min(top_k, len(class_names))
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Format results
        predictions = []
        for i in range(top_k):
            prob = top_probs[0][i].item()
            
            if prob < confidence_threshold:
                break
                
            idx = top_indices[0][i].item()
            class_name = class_names[idx]
            
            # Parse class name
            parts = class_name.split('___')
            plant = parts[0].replace('_', ' ').title()
            condition = parts[1].replace('_', ' ').title() if len(parts) > 1 else class_name
            
            is_healthy = 'healthy' in condition.lower()
            
            predictions.append({
                "rank": i + 1,
                "plant": plant,
                "condition": condition,
                "confidence": round(prob * 100, 2),
                "is_healthy": is_healthy,
                "class_name": class_name
            })
        
        # Determine overall status
        primary = predictions[0] if predictions else None
        status = "healthy" if primary and primary["is_healthy"] else "diseased"
        
        return {
            "success": True,
            "status": status,
            "primary_diagnosis": primary,
            "predictions": predictions,
            "metadata": {
                "inference_time": round(inference_time, 3),
                "image_size": f"{image.size[0]}x{image.size[1]}",
                "model_device": str(device),
                "total_classes": len(class_names)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction for multiple images"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Reuse single prediction logic
            result = await predict_disease(file, top_k=3, confidence_threshold=0.1)
            result["image_index"] = i
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                "image_index": i,
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "batch_size": len(files),
        "results": results,
        "summary": {
            "successful": len([r for r in results if r.get("success", False)]),
            "failed": len([r for r in results if not r.get("success", False)])
        }
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )