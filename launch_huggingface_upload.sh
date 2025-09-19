#!/bin/bash
"""
Automated HuggingFace Upload Launcher
===================================

This script ensures all dependencies are installed and launches the
automated upload process for PlantNet models to HuggingFace Hub.
"""

set -e  # Exit on any error

echo "🌱 PlantNet - Automated HuggingFace Upload Launcher"
echo "=================================================="

# Change to script directory
cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.8 or later."
    exit 1
fi

print_status "Python found: $(python --version)"

# Install required packages
print_status "Installing required dependencies..."

pip install --upgrade pip --quiet

# Core dependencies
pip install torch torchvision --quiet --index-url https://download.pytorch.org/whl/cpu
pip install huggingface-hub --quiet
pip install pillow --quiet
pip install "numpy>=1.24.0,<2.0.0" --quiet

print_status "All dependencies installed successfully"

# Run the automated upload
echo ""
echo "🚀 Starting automated upload process..."
echo ""

python automated_huggingface_upload.py "$@"

upload_result=$?

if [ $upload_result -eq 0 ]; then
    echo ""
    print_status "Upload completed successfully!"
    echo ""
    echo "🎉 Your PlantNet model is now available on HuggingFace Hub!"
    echo "💡 Next steps:"
    echo "   1. Visit your model page to verify everything looks correct"
    echo "   2. Test the model using the provided inference examples"
    echo "   3. Share your model with the community!"
else
    echo ""
    print_error "Upload failed. Please check the error messages above."
    exit 1
fi