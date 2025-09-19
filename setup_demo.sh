#!/bin/bash
# PlantNet Demo Setup Script
# Run this before your hackathon presentation

echo "🌱 Setting up PlantNet Demo..."

# Install requirements
echo "📦 Installing dependencies..."
pip install gradio fastapi uvicorn python-multipart pillow

# Make scripts executable
chmod +x gradio_demo.py api_server.py plantnet_inference.py

echo "🚀 Starting Gradio Demo..."
echo "📱 This will create a public URL for your demo!"
echo "🎯 Perfect for hackathon presentations!"

# Launch Gradio demo
python gradio_demo.py

echo "✅ Demo ready! Share the public URL with judges!"