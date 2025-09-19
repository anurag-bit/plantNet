#!/usr/bin/env python3
"""
ROCm 6.0 System Verification Script
Comprehensive check for PyTorch ROCm 6.0 setup
"""

import torch
import subprocess
import sys

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_rocm_system():
    """Check ROCm system status"""
    print("🚀 ROCm 6.0 System Verification")
    print("=" * 50)
    
    # Check ROCm installation
    print("\n📦 ROCm Installation:")
    rocm_smi = run_command("rocm-smi --version")
    if rocm_smi:
        print(f"✅ ROCm SMI: {rocm_smi}")
    else:
        print("❌ ROCm SMI not found")
    
    # Check GPU detection
    print("\n🔍 GPU Detection:")
    gpu_info = run_command("rocm-smi --showid")
    if "GPU" in gpu_info:
        print("✅ GPU detected by ROCm")
        print(gpu_info)
    else:
        print("❌ No GPU detected")
    
    # Check PyTorch
    print(f"\n🐍 PyTorch Information:")
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ Device count: {torch.cuda.device_count()}")
        print(f"✅ Current device: {torch.cuda.current_device()}")
        print(f"✅ Device name: {torch.cuda.get_device_name(0)}")
        
        # Memory info
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1e9
        print(f"✅ Total memory: {total_memory:.1f} GB")
        print(f"✅ Compute capability: {props.major}.{props.minor}")
        
        # Test tensor operations
        print("\n⚡ Testing Tensor Operations:")
        try:
            x = torch.randn(100, 100, device='cuda')
            y = torch.randn(100, 100, device='cuda')
            z = torch.mm(x, y)
            print("✅ Basic tensor operations working")
            
            # Test mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                result = torch.mm(x, y)
            print("✅ BF16 mixed precision working")
            
        except Exception as e:
            print(f"❌ Tensor operations failed: {e}")
    
    else:
        print("❌ PyTorch CUDA/ROCm not available")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_rocm_system()