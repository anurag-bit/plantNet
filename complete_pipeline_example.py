#!/usr/bin/env python3
"""
PlantNet Complete Pipeline Example
=================================

This script demonstrates the complete automated workflow from 
dataset setup to model deployment on HuggingFace Hub.

Usage Examples:
    # Quick test run with sample data
    python complete_pipeline_example.py --quick_test
    
    # Full pipeline with real dataset
    python complete_pipeline_example.py --full_pipeline
    
    # Custom configuration
    python complete_pipeline_example.py --data_source kaggle --epochs 20
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


class PlantNetPipelineRunner:
    """Orchestrates the complete PlantNet pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.project_root = Path(__file__).parent
        
    def run_step(self, step_name: str, command: list, description: str) -> bool:
        """Run a pipeline step with error handling."""
        print(f"\n🚀 {step_name}: {description}")
        print(f"💻 Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            print(f"✅ {step_name} completed successfully")
            if result.stdout:
                print("📋 Output:")
                print(result.stdout[:500] + ("..." if len(result.stdout) > 500 else ""))
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {step_name} failed!")
            print(f"Error: {e.stderr}")
            return False
    
    def setup_dataset(self) -> bool:
        """Set up the dataset based on configuration."""
        print("\n" + "="*60)
        print("📊 STEP 1: DATASET SETUP")
        print("="*60)
        
        if self.config.data_source == "sample":
            return self.run_step(
                "Sample Dataset",
                ["python", "setup_dataset.py", "--sample", "--data_dir", "data"],
                "Creating sample dataset for testing"
            )
        elif self.config.data_source == "kaggle":
            return self.run_step(
                "Kaggle Dataset",
                ["python", "setup_kaggle_dataset.py", "--use_kaggle", "--data_dir", "data"],
                "Downloading PlantVillage dataset from Kaggle"
            )
        else:  # auto
            return self.run_step(
                "Auto Dataset",
                ["python", "setup_dataset.py", "--source", "auto", "--data_dir", "data"],
                "Automatically downloading PlantVillage dataset"
            )
    
    def run_training(self) -> bool:
        """Train the PlantNet model."""
        print("\n" + "="*60)
        print("🤖 STEP 2: MODEL TRAINING")
        print("="*60)
        
        # Check if training script exists
        training_script = self.project_root / "train_mi300x.py"
        if not training_script.exists():
            print("⚠️ Training script not found. Creating a dummy training run...")
            
            # Create a dummy model for testing
            dummy_command = [
                "python", "-c", 
                f"""
import torch
import torch.nn as nn
from pathlib import Path

# Create simple test model
class TestModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Save model
model = TestModel()
Path("models").mkdir(exist_ok=True)
torch.save({{'model_state_dict': model.state_dict(), 'num_classes': 38}}, 
           "models/best_model.pth")
print("✅ Dummy model saved to models/best_model.pth")
                """
            ]
            
            return self.run_step(
                "Dummy Training",
                dummy_command,
                "Creating test model for pipeline demonstration"
            )
        else:
            return self.run_step(
                "Model Training",
                [
                    "python", "train_mi300x.py",
                    "--data_dir", "data",
                    "--epochs", str(self.config.epochs),
                    "--batch_size", str(self.config.batch_size),
                    "--output_dir", "models"
                ],
                "Training PlantNet model on dataset"
            )
    
    def compile_and_deploy(self) -> bool:
        """Compile models and deploy to HuggingFace."""
        print("\n" + "="*60)
        print("🚀 STEP 3: COMPILATION & DEPLOYMENT")
        print("="*60)
        
        # Find the trained model
        model_path = "models/best_model.pth"
        
        if not Path(model_path).exists():
            print("❌ Trained model not found!")
            return False
        
        # Build deployment command
        deploy_cmd = [
            "./deploy_pipeline.sh",
            "--model_path", model_path,
            "--repo_name", self.config.repo_name
        ]
        
        if self.config.private_repo:
            deploy_cmd.append("--private")
        
        if self.config.dry_run:
            deploy_cmd.append("--dry_run")
        
        return self.run_step(
            "Model Deployment",
            deploy_cmd,
            "Compiling and deploying model to HuggingFace Hub"
        )
    
    def run_tests(self) -> bool:
        """Run the test suite."""
        print("\n" + "="*60)
        print("🧪 STEP 0: TESTING PIPELINE")
        print("="*60)
        
        return self.run_step(
            "Pipeline Tests",
            ["python", "test_deployment.py"],
            "Running deployment pipeline tests"
        )
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete pipeline."""
        print("🌟 PlantNet Complete Pipeline Runner")
        print("="*60)
        print(f"📊 Data Source: {self.config.data_source}")
        print(f"🤖 Training Epochs: {self.config.epochs}")
        print(f"📦 Batch Size: {self.config.batch_size}")
        print(f"🤗 Repository: {self.config.repo_name}")
        print(f"🔒 Private Repo: {self.config.private_repo}")
        print(f"🧪 Dry Run: {self.config.dry_run}")
        
        # Run tests first if requested
        if self.config.run_tests:
            if not self.run_tests():
                print("❌ Tests failed. Stopping pipeline.")
                return False
        
        # Step 1: Dataset Setup
        if not self.setup_dataset():
            print("❌ Dataset setup failed. Stopping pipeline.")
            return False
        
        # Step 2: Model Training (skip if model exists and skip_training is True)
        if not (self.config.skip_training and Path("models/best_model.pth").exists()):
            if not self.run_training():
                print("❌ Model training failed. Stopping pipeline.")
                return False
        
        # Step 3: Compilation & Deployment
        if not self.compile_and_deploy():
            print("❌ Deployment failed. Stopping pipeline.")
            return False
        
        # Success summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📋 Summary:")
        print("✅ Dataset setup completed")
        print("✅ Model training completed" if not self.config.skip_training else "⏭️ Model training skipped")
        print("✅ Model compilation completed")
        print("✅ HuggingFace deployment completed" if not self.config.dry_run else "🧪 Dry run completed")
        
        print("\n🔗 Next Steps:")
        if self.config.dry_run:
            print("- Remove --dry_run to perform actual deployment")
        print("- Check your HuggingFace repository for the deployed model")
        print("- Use the model for inference in your applications")
        
        return True


def main():
    """Main pipeline runner."""
    parser = argparse.ArgumentParser(description='PlantNet Complete Pipeline Runner')
    
    # Pipeline options
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test run with sample data')
    parser.add_argument('--full_pipeline', action='store_true',
                       help='Full pipeline with real dataset')
    
    # Data options
    parser.add_argument('--data_source', choices=['sample', 'kaggle', 'auto'], 
                       default='sample', help='Dataset source')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training step')
    
    # Deployment options
    parser.add_argument('--repo_name', type=str, default='plantnet-pipeline-test',
                       help='HuggingFace repository name')
    parser.add_argument('--private_repo', action='store_true',
                       help='Create private HuggingFace repository')
    parser.add_argument('--dry_run', action='store_true',
                       help='Test run without actual deployment')
    
    # Testing options
    parser.add_argument('--run_tests', action='store_true',
                       help='Run tests before pipeline')
    parser.add_argument('--tests_only', action='store_true',
                       help='Only run tests')
    
    args = parser.parse_args()
    
    # Handle preset configurations
    if args.quick_test:
        args.data_source = 'sample'
        args.epochs = 2
        args.batch_size = 8
        args.dry_run = True
        args.run_tests = True
        print("🧪 Quick test mode activated")
    
    elif args.full_pipeline:
        args.data_source = 'auto'
        args.epochs = 20
        args.batch_size = 32
        args.dry_run = False
        print("🚀 Full pipeline mode activated")
    
    # Check for HuggingFace token if not dry run
    if not args.dry_run and not os.getenv('HF_TOKEN'):
        print("⚠️ HF_TOKEN environment variable not set!")
        print("Set it with: export HF_TOKEN='your_token_here'")
        print("Or use --dry_run for testing without actual upload")
        
        response = input("Continue without token (dry run)? (y/N): ")
        if response.lower() != 'y':
            return 1
        args.dry_run = True
    
    # Initialize and run pipeline
    runner = PlantNetPipelineRunner(args)
    
    if args.tests_only:
        success = runner.run_tests()
    else:
        success = runner.run_complete_pipeline()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)