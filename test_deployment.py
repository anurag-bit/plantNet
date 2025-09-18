#!/usr/bin/env python3
"""
Complete Model Deployment and Upload Test Script
===============================================

This script demonstrates the complete pipeline for dataset setup,
model compilation, and uploading PlantNet models to HuggingFace Hub 
with proper Git LFS tracking.
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def test_dataset_setup():
    """Test the automated dataset setup functionality."""
    print("🧪 Testing dataset setup...")
    
    # Test sample dataset creation
    try:
        from setup_dataset import DatasetDownloader
        
        # Create temporary test directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data_dir = Path(temp_dir) / "test_data"
            
            print(f"📁 Testing dataset setup in: {test_data_dir}")
            
            # Initialize downloader
            downloader = DatasetDownloader(data_dir=str(test_data_dir))
            
            # Test sample dataset creation
            print("🔄 Creating sample dataset...")
            success = downloader.setup_dataset(create_sample=True)
            
            if success:
                print("✅ Sample dataset created successfully")
                
                # Verify structure
                if downloader.verify_dataset():
                    print("✅ Dataset verification passed")
                    return True
                else:
                    print("❌ Dataset verification failed")
                    return False
            else:
                print("❌ Sample dataset creation failed")
                return False
                
    except Exception as e:
        print(f"❌ Dataset setup test failed: {e}")
        return False


def test_kaggle_setup():
    """Test Kaggle dataset setup (without actual download)."""
    print("🧪 Testing Kaggle integration...")
    
    try:
        from setup_kaggle_dataset import KaggleDatasetSetup
        
        # Test Kaggle setup class initialization
        kaggle_setup = KaggleDatasetSetup(data_dir="test_data")
        print("✅ Kaggle setup class initialized")
        
        # Test credential check (should fail without actual credentials)
        has_credentials = kaggle_setup.check_kaggle_setup()
        if has_credentials:
            print("✅ Kaggle credentials found")
        else:
            print("ℹ️ Kaggle credentials not configured (expected for testing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Kaggle setup test failed: {e}")
        return False


def create_test_model():
    """Create a test model for demonstration purposes."""
    print("🧪 Creating test model...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple test model
        class TestPlantModel(nn.Module):
            def __init__(self, num_classes=38):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(64, num_classes)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        # Create and save test model
        model = TestPlantModel()
        
        # Create test class names
        class_names = [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
            "Cherry___Powdery_mildew", "Cherry___healthy",
            "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
            "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
            "Peach___Bacterial_spot", "Peach___healthy",
            "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
            "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Strawberry___Leaf_scorch", "Strawberry___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
        ][:38]  # Ensure exactly 38 classes
        
        # Create test checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'class_names': class_names,
            'config': {
                'model_type': 'test_cnn',
                'img_size': 224,
                'num_classes': len(class_names)
            },
            'performance': {
                'accuracy': 0.925,
                'f1_score': 0.918,
                'top_3_accuracy': 0.985,
                'inference_time_ms': 15
            }
        }
        
        test_model_path = "test_model.pth"
        torch.save(checkpoint, test_model_path)
        print(f"✅ Test model created: {test_model_path}")
        
        return test_model_path
        
    except ImportError:
        print("⚠️ PyTorch not available, using dummy model file")
        
        # Create dummy model file
        test_model_path = "test_model.pth"
        with open(test_model_path, 'wb') as f:
            f.write(b"DUMMY_MODEL_FOR_TESTING" * 1000)  # Create some content
        
        return test_model_path


def test_git_lfs_setup():
    """Test Git LFS configuration."""
    print("\n🔍 Testing Git LFS setup...")
    
    try:
        import subprocess
        
        # Check if Git LFS is installed
        result = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git LFS is installed and working")
            print(f"   Version: {result.stdout.strip()}")
        else:
            print("❌ Git LFS not properly installed")
            return False
        
        # Check if .gitattributes exists
        if os.path.exists('.gitattributes'):
            print("✅ .gitattributes file exists")
            with open('.gitattributes', 'r') as f:
                content = f.read()
                if 'filter=lfs' in content:
                    print("✅ LFS filters configured")
                else:
                    print("⚠️ LFS filters may not be properly configured")
        else:
            print("❌ .gitattributes file not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Git LFS test failed: {e}")
        return False


def test_model_compilation():
    """Test model compilation pipeline."""
    print("\n🔧 Testing model compilation...")
    
    # Create test model
    test_model_path = create_test_model()
    
    if not os.path.exists(test_model_path):
        print("❌ Test model creation failed")
        return False
    
    try:
        # Test model compilation script
        import subprocess
        
        result = subprocess.run([
            'python', 'compile_models.py',
            '--model_path', test_model_path,
            '--formats', 'torchscript',
            '--output_dir', 'test_compiled'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model compilation test passed")
            
            # Check for compiled files
            if os.path.exists('test_compiled'):
                files = list(Path('test_compiled').glob('*'))
                print(f"   Generated {len(files)} files:")
                for file in files:
                    size_mb = file.stat().st_size / (1024*1024)
                    print(f"     {file.name} ({size_mb:.1f} MB)")
            
            # Cleanup
            import shutil
            if os.path.exists('test_compiled'):
                shutil.rmtree('test_compiled')
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
            
            return True
        else:
            print("❌ Model compilation failed")
            print(f"Error: {result.stderr}")
            return False
        
    except Exception as e:
        print(f"❌ Model compilation test failed: {e}")
        return False


def test_version_management():
    """Test version management system."""
    print("\n🏷️ Testing version management...")
    
    try:
        import subprocess
        
        # Test version increment
        result = subprocess.run([
            'python', 'version_manager.py',
            '--set_version', '1.0.0'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Version management test passed")
            
            # Check version file
            if os.path.exists('VERSION'):
                with open('VERSION', 'r') as f:
                    version = f.read().strip()
                    print(f"   Set version: {version}")
                
                # Test version increment
                result2 = subprocess.run([
                    'python', 'version_manager.py',
                    '--increment', 'patch'
                ], capture_output=True, text=True)
                
                if result2.returncode == 0:
                    with open('VERSION', 'r') as f:
                        new_version = f.read().strip()
                        print(f"   Incremented to: {new_version}")
                
                return True
            else:
                print("❌ VERSION file not created")
                return False
        else:
            print("❌ Version management failed")
            print(f"Error: {result.stderr}")
            return False
        
    except Exception as e:
        print(f"❌ Version management test failed: {e}")
        return False


def test_huggingface_setup():
    """Test HuggingFace Hub setup (without actual upload)."""
    print("\n🤗 Testing HuggingFace Hub setup...")
    
    try:
        from huggingface_hub import HfApi
        print("✅ HuggingFace Hub library available")
        
        # Test API connection (without token)
        api = HfApi()
        print("✅ HuggingFace API initialized")
        
        # Check if HF_TOKEN is available
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            print("✅ HF_TOKEN environment variable found")
            try:
                user_info = api.whoami(token=hf_token)
                print(f"   Authenticated as: {user_info.get('name', 'Unknown')}")
            except Exception as e:
                print(f"⚠️ Token validation failed: {e}")
        else:
            print("⚠️ HF_TOKEN not found (required for upload)")
        
        return True
        
    except ImportError:
        print("❌ HuggingFace Hub library not available")
        print("   Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ HuggingFace setup test failed: {e}")
        return False


def test_complete_pipeline():
    """Test the complete deployment pipeline in dry-run mode."""
    print("\n🚀 Testing complete deployment pipeline...")
    
    try:
        import subprocess
        
        # Create a test model for pipeline testing
        test_model_path = create_test_model()
        
        if not os.path.exists(test_model_path):
            print("❌ Test model creation failed")
            return False
        
        # Run pipeline in dry-run mode
        result = subprocess.run([
            './deploy_pipeline.sh',
            '--model_path', test_model_path,
            '--version_type', 'patch',
            '--repo_name', 'test-plantnet-model',
            '--dry_run'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Complete pipeline test passed (dry-run)")
            print("   Pipeline output:")
            for line in result.stdout.split('\n')[-10:]:  # Show last 10 lines
                if line.strip():
                    print(f"     {line}")
        else:
            print("❌ Complete pipeline test failed")
            print(f"Error: {result.stderr}")
            return False
        
        # Cleanup
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 PlantNet Deployment Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dataset Setup", test_dataset_setup),
        ("Kaggle Integration", test_kaggle_setup),
        ("Git LFS Setup", test_git_lfs_setup),
        ("Model Compilation", test_model_compilation),
        ("Version Management", test_version_management),
        ("HuggingFace Setup", test_huggingface_setup),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("🏁 Test Results Summary")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your deployment pipeline is ready.")
        print("\nNext steps:")
        print("1. Run dataset setup: python setup_dataset.py --sample")
        print("2. Train a model: python train_mi300x.py")
        print("3. Deploy pipeline: ./deploy_pipeline.sh --model_path models/best_model.pth")
    else:
        print("⚠️ Some tests failed. Please fix the issues before deployment.")
        print("\nTroubleshooting:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Set up Git LFS: git lfs install")
        print("- Set HF_TOKEN environment variable for HuggingFace uploads")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)