#!/usr/bin/env python3
"""
Enhanced Dataset Setup with Kaggle Integration
==============================================

This script provides enhanced dataset download capabilities including
Kaggle API integration for accessing the official PlantVillage dataset.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


class KaggleDatasetSetup:
    """Enhanced dataset setup with Kaggle integration."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.kaggle_dataset = "abdallahalidev/plantvillage-dataset"
        
    def check_kaggle_setup(self) -> bool:
        """Check if Kaggle API is properly configured."""
        try:
            # Try importing kaggle (will install if needed)
            try:
                import kaggle  # type: ignore
            except ImportError:
                print("📦 Installing Kaggle package...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
                import kaggle  # type: ignore
            
            kaggle.api.authenticate()
            print("✅ Kaggle API is configured")
            return True
        except Exception as e:
            print(f"❌ Kaggle API not configured: {e}")
            return False
    
    def setup_kaggle_credentials(self) -> bool:
        """Help user set up Kaggle credentials."""
        print("\n🔑 Kaggle API Setup Required")
        print("=" * 40)
        print("To download the official PlantVillage dataset, you need Kaggle API credentials.")
        print("\nSetup steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json file")
        print("4. Place it at: ~/.kaggle/kaggle.json")
        print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        
        # Try to install kaggle package
        try:
            print("\n📦 Installing Kaggle package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            print("✅ Kaggle package installed")
        except Exception as e:
            print(f"❌ Failed to install kaggle package: {e}")
            return False
        
        # Check if kaggle.json exists
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_file = kaggle_dir / "kaggle.json"
        
        if not kaggle_file.exists():
            print(f"\n⚠️ Kaggle credentials not found at: {kaggle_file}")
            print("Please complete the setup steps above and run this script again.")
            return False
        
        # Set proper permissions
        try:
            import stat
            os.chmod(kaggle_file, stat.S_IRUSR | stat.S_IWUSR)
            print("✅ Kaggle credentials permissions set")
        except Exception as e:
            print(f"⚠️ Could not set permissions: {e}")
        
        return self.check_kaggle_setup()
    
    def download_from_kaggle(self) -> bool:
        """Download dataset from Kaggle."""
        try:
            # Import kaggle with fallback installation
            try:
                import kaggle  # type: ignore
            except ImportError:
                print("📦 Installing Kaggle package...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
                import kaggle  # type: ignore
            
            print(f"📥 Downloading dataset: {self.kaggle_dataset}")
            
            # Create download directory
            download_dir = self.data_dir / "raw"
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            kaggle.api.dataset_download_files(
                self.kaggle_dataset,
                path=str(download_dir),
                unzip=True
            )
            
            print("✅ Dataset downloaded from Kaggle successfully!")
            
            # Find the downloaded data
            extracted_dirs = [d for d in download_dir.iterdir() if d.is_dir()]
            if extracted_dirs:
                print(f"📁 Dataset extracted to: {extracted_dirs[0]}")
                return True
            else:
                # Check for zip files
                zip_files = list(download_dir.glob("*.zip"))
                if zip_files:
                    print(f"📦 Dataset zip found: {zip_files[0]}")
                    return True
            
            return True
            
        except Exception as e:
            print(f"❌ Kaggle download failed: {e}")
            return False
    
    def setup_with_kaggle(self) -> bool:
        """Set up dataset using Kaggle API."""
        print("🚀 Setting up PlantVillage dataset via Kaggle...")
        
        # Check if Kaggle is configured
        if not self.check_kaggle_setup():
            if not self.setup_kaggle_credentials():
                return False
        
        # Download from Kaggle
        if self.download_from_kaggle():
            print("✅ Kaggle dataset setup completed!")
            return True
        else:
            print("❌ Kaggle dataset setup failed!")
            return False


def main():
    """Main function for enhanced dataset setup."""
    parser = argparse.ArgumentParser(description='Enhanced Dataset Setup with Kaggle')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to store dataset')
    parser.add_argument('--use_kaggle', action='store_true',
                       help='Use Kaggle API to download dataset')
    
    args = parser.parse_args()
    
    if args.use_kaggle:
        kaggle_setup = KaggleDatasetSetup(args.data_dir)
        success = kaggle_setup.setup_with_kaggle()
        return 0 if success else 1
    else:
        print("Use --use_kaggle flag to download from Kaggle, or use setup_dataset.py for alternative sources.")
        return 0


if __name__ == "__main__":
    sys.exit(main())