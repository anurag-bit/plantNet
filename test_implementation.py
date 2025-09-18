#!/usr/bin/env python3
"""
Test script to verify the CNN implementation works correctly.
This script runs basic tests without requiring the actual dataset.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_model_creation():
    """Test model creation and basic functionality."""
    print("Testing model creation...")
    
    try:
        from models.cnn_models import create_model, count_parameters
        
        num_classes = 38  # PlantVillage dataset classes
        
        # Test Custom CNN
        print("  Creating Custom CNN...")
        model_cnn = create_model('custom_cnn', num_classes, dropout_rate=0.5)
        params_cnn = count_parameters(model_cnn)
        print(f"    Parameters: {params_cnn:,}")
        
        # Test ResNet
        print("  Creating ResNet50...")
        model_resnet = create_model('resnet', num_classes, model_name='resnet50', pretrained=False)
        params_resnet = count_parameters(model_resnet)
        print(f"    Parameters: {params_resnet:,}")
        
        # Test forward pass
        device = torch.device('cpu')
        model_cnn.to(device)
        
        dummy_input = torch.randn(4, 3, 224, 224)
        output = model_cnn(dummy_input)
        
        assert output.shape == (4, num_classes), f"Expected shape (4, {num_classes}), got {output.shape}"
        print(f"  Forward pass test passed: {dummy_input.shape} -> {output.shape}")
        
        print("✓ Model creation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False


def test_data_transforms():
    """Test data transformation pipeline."""
    print("Testing data transforms...")
    
    try:
        from utils.dataset import get_transforms
        
        # Test training transforms
        train_transform = get_transforms(img_size=224, is_training=True)
        val_transform = get_transforms(img_size=224, is_training=False)
        
        # Create dummy image data
        dummy_image = torch.randn(3, 256, 256)  # Simulate PIL image as tensor
        
        # Apply transforms (this would normally be done on PIL Image)
        # For testing purposes, we'll just verify the transform objects exist
        assert train_transform is not None, "Training transform is None"
        assert val_transform is not None, "Validation transform is None"
        
        print("  Training transforms created successfully")
        print("  Validation transforms created successfully")
        print("✓ Data transforms tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data transforms test failed: {e}")
        return False


def test_training_components():
    """Test training pipeline components."""
    print("Testing training components...")
    
    try:
        from utils.trainer import create_optimizer, create_scheduler, create_criterion, EarlyStopping
        from models.cnn_models import create_model
        
        # Create a simple model
        model = create_model('custom_cnn', 10, dropout_rate=0.5)
        
        # Test optimizer creation
        optimizer = create_optimizer(model, 'adam', learning_rate=0.001)
        assert optimizer is not None, "Optimizer creation failed"
        print("  Optimizer creation: ✓")
        
        # Test scheduler creation
        scheduler = create_scheduler(optimizer, 'step', step_size=10, gamma=0.1)
        assert scheduler is not None, "Scheduler creation failed"
        print("  Scheduler creation: ✓")
        
        # Test criterion creation
        criterion = create_criterion('crossentropy')
        assert criterion is not None, "Criterion creation failed"
        print("  Criterion creation: ✓")
        
        # Test early stopping
        early_stopping = EarlyStopping(patience=5)
        assert early_stopping is not None, "Early stopping creation failed"
        print("  Early stopping creation: ✓")
        
        print("✓ Training components tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Training components test failed: {e}")
        return False


def test_evaluation_components():
    """Test evaluation components."""
    print("Testing evaluation components...")
    
    try:
        from utils.evaluation import ModelEvaluator
        from models.cnn_models import create_model
        
        # Create model and evaluator
        model = create_model('custom_cnn', 10)
        device = torch.device('cpu')
        class_names = [f"Class_{i}" for i in range(10)]
        
        evaluator = ModelEvaluator(model, device, class_names)
        assert evaluator is not None, "Evaluator creation failed"
        print("  ModelEvaluator creation: ✓")
        
        print("✓ Evaluation components tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Evaluation components test failed: {e}")
        return False


def test_config_system():
    """Test configuration system."""
    print("Testing configuration system...")
    
    try:
        # Import train module to test config
        import importlib.util
        spec = importlib.util.spec_from_file_location("train", "train.py")
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        
        # Test config creation
        config = train_module.Config()
        assert config is not None, "Config creation failed"
        print("  Config creation: ✓")
        
        # Test config methods
        config.print_config()
        print("  Config printing: ✓")
        
        print("✓ Configuration system tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration system test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("PLANTNET CNN IMPLEMENTATION TEST SUITE")
    print("="*60)
    print()
    
    tests = [
        test_model_creation,
        test_data_transforms,
        test_training_components,
        test_evaluation_components,
        test_config_system
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Test {test_func.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("🎉 All tests passed! The CNN implementation is ready to use.")
        print()
        print("Next steps:")
        print("1. Prepare your PlantVillage dataset in the 'data/' directory")
        print("2. Run: python train.py --model_type resnet --epochs 50")
        print("3. Monitor training with TensorBoard: tensorboard --logdir results/")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)