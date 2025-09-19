#!/bin/bash
"""
HuggingFace Credentials Setup Script
===================================

This script helps you set up HuggingFace credentials for the automated upload script.
"""

echo "🌱 PlantNet - HuggingFace Credentials Setup"
echo "=========================================="
echo ""

# Check if token is already set
if [ -n "$HUGGINGFACE_TOKEN" ] || [ -n "$HF_TOKEN" ]; then
    echo "✅ HuggingFace token is already set in environment"
else
    echo "❌ HuggingFace token not found in environment variables"
fi

echo ""
echo "🔧 Setting up HuggingFace credentials..."
echo ""
echo "1. First, get your token from: https://huggingface.co/settings/tokens"
echo "   (Make sure to create a token with 'write' permissions!)"
echo ""

# Prompt for token
read -p "📝 Enter your HuggingFace token: " hf_token
if [ -z "$hf_token" ]; then
    echo "❌ Token cannot be empty!"
    exit 1
fi

# Prompt for username (optional)
read -p "👤 Enter your HuggingFace username (optional, press Enter to skip): " hf_username

# Export for current session
export HUGGINGFACE_TOKEN="$hf_token"
if [ -n "$hf_username" ]; then
    export HUGGINGFACE_USERNAME="$hf_username"
fi

echo ""
echo "✅ Environment variables set for current session!"

# Ask if user wants to make it permanent
echo ""
read -p "💾 Do you want to make these settings permanent? (y/N): " make_permanent

if [[ $make_permanent =~ ^[Yy]$ ]]; then
    # Determine which shell config file to use
    if [ -n "$ZSH_VERSION" ]; then
        config_file="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        config_file="$HOME/.bashrc"
    else
        config_file="$HOME/.profile"
    fi
    
    echo ""
    echo "📝 Adding to $config_file..."
    
    # Add to config file
    echo "" >> "$config_file"
    echo "# HuggingFace credentials for PlantNet" >> "$config_file"
    echo "export HUGGINGFACE_TOKEN=\"$hf_token\"" >> "$config_file"
    
    if [ -n "$hf_username" ]; then
        echo "export HUGGINGFACE_USERNAME=\"$hf_username\"" >> "$config_file"
    fi
    
    echo "✅ Credentials saved permanently!"
    echo "🔄 Run 'source $config_file' or restart your terminal to apply changes"
else
    echo "⚠️  Credentials are only set for this session"
fi

echo ""
echo "🚀 You can now run the automated upload script:"
echo "   python automated_huggingface_upload.py"
echo ""
echo "🧪 Test your setup:"
echo "   python -c \"import os; print('Token set!' if os.getenv('HUGGINGFACE_TOKEN') else 'Token not found')\""
echo ""