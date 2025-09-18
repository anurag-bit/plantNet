#!/usr/bin/env python3
"""
Model Versioning and CI/CD Integration Script
===========================================

Automated pipeline for model versioning, Git tagging, and CI/CD integration
for PlantNet model deployments to HuggingFace Hub.

Features:
- Semantic versioning for models
- Automated Git tagging and releases
- CI/CD pipeline integration
- Changelog generation
- Version comparison and metrics tracking
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


class ModelVersionManager:
    """Handle model versioning and release management."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize version manager."""
        self.repo_path = Path(repo_path)
        self.version_file = self.repo_path / "VERSION"
        self.changelog_file = self.repo_path / "CHANGELOG.md"
        self.releases_dir = self.repo_path / "releases"
        self.releases_dir.mkdir(exist_ok=True)
        
        print("🏷️ Model Version Manager initialized")
        print(f"📁 Repository: {self.repo_path}")
    
    def get_current_version(self) -> str:
        """Get current model version."""
        if self.version_file.exists():
            return self.version_file.read_text().strip()
        else:
            return "0.1.0"
    
    def increment_version(self, version_type: str = "patch") -> str:
        """Increment version number following semantic versioning."""
        current_version = self.get_current_version()
        major, minor, patch = map(int, current_version.split('.'))
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        elif version_type == "patch":
            patch += 1
        else:
            raise ValueError("version_type must be 'major', 'minor', or 'patch'")
        
        new_version = f"{major}.{minor}.{patch}"
        return new_version
    
    def set_version(self, version: str):
        """Set the current version."""
        self.version_file.write_text(version)
        print(f"📌 Version set to: {version}")
    
    def create_release_metadata(self, version: str, model_path: str, 
                              performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive release metadata."""
        print(f"📋 Creating release metadata for v{version}")
        
        # Calculate model size
        model_size_mb = 0
        if os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024**2)
        
        # Get git information
        git_commit = self._run_command("git rev-parse HEAD", capture_output=True)
        git_branch = self._run_command("git branch --show-current", capture_output=True)
        
        release_metadata = {
            "version": version,
            "release_date": datetime.now().isoformat(),
            "git_commit": git_commit,
            "git_branch": git_branch,
            "model": {
                "path": model_path,
                "size_mb": round(model_size_mb, 2),
                "format": Path(model_path).suffix if model_path else None
            },
            "performance": performance_metrics,
            "system": {
                "python_version": sys.version,
                "platform": os.name,
                "timestamp": int(time.time())
            },
            "build": {
                "automated": True,
                "ci_cd": os.getenv('CI', 'false').lower() == 'true'
            }
        }
        
        # Save release metadata
        metadata_path = self.releases_dir / f"v{version}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(release_metadata, f, indent=2)
        
        print(f"✅ Release metadata saved: {metadata_path}")
        return release_metadata
    
    def update_changelog(self, version: str, changes: List[str], 
                        performance_improvements: Dict[str, str] = None):
        """Update changelog with new release information."""
        print(f"📝 Updating changelog for v{version}")
        
        # Read existing changelog
        existing_content = ""
        if self.changelog_file.exists():
            existing_content = self.changelog_file.read_text()
        
        # Create new entry
        date_str = datetime.now().strftime("%Y-%m-%d")
        new_entry = f"""## [v{version}] - {date_str}

### 🚀 New Features
{chr(10).join(f"- {change}" for change in changes if change.startswith("Add") or change.startswith("Implement"))}

### 🐛 Bug Fixes
{chr(10).join(f"- {change}" for change in changes if change.startswith("Fix") or change.startswith("Resolve"))}

### ⚡ Performance Improvements
{chr(10).join(f"- {change}" for change in changes if change.startswith("Improve") or change.startswith("Optimize"))}
"""
        
        if performance_improvements:
            new_entry += "\n### 📊 Performance Metrics\n"
            for metric, value in performance_improvements.items():
                new_entry += f"- {metric}: {value}\n"
        
        new_entry += "\n---\n\n"
        
        # Combine with existing content
        full_content = f"""# PlantNet Model Changelog

All notable changes to PlantNet models will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

{new_entry}{existing_content}"""
        
        # Write updated changelog
        self.changelog_file.write_text(full_content)
        print(f"✅ Changelog updated: {self.changelog_file}")
    
    def create_git_tag(self, version: str, message: str = None):
        """Create and push git tag for release."""
        print(f"🏷️ Creating git tag v{version}")
        
        if not message:
            message = f"Release v{version} - PlantNet model update"
        
        try:
            # Create tag
            self._run_command(f'git tag -a v{version} -m "{message}"')
            print(f"✅ Git tag created: v{version}")
            
            # Push tag (if remote exists)
            try:
                self._run_command(f"git push origin v{version}")
                print(f"✅ Tag pushed to remote: v{version}")
            except subprocess.CalledProcessError:
                print("⚠️ Could not push tag (no remote or permission issues)")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create git tag: {e}")
    
    def compare_versions(self, old_metadata_path: str, 
                        new_metadata_path: str) -> Dict[str, Any]:
        """Compare two model versions and generate comparison report."""
        print("📊 Comparing model versions...")
        
        # Load metadata files
        with open(old_metadata_path, 'r') as f:
            old_metadata = json.load(f)
        with open(new_metadata_path, 'r') as f:
            new_metadata = json.load(f)
        
        # Compare performance metrics
        old_perf = old_metadata.get('performance', {})
        new_perf = new_metadata.get('performance', {})
        
        comparison = {
            "version_comparison": {
                "old_version": old_metadata.get('version'),
                "new_version": new_metadata.get('version')
            },
            "performance_changes": {},
            "model_changes": {
                "size_change_mb": (new_metadata.get('model', {}).get('size_mb', 0) - 
                                 old_metadata.get('model', {}).get('size_mb', 0))
            },
            "summary": {
                "improvements": [],
                "regressions": [],
                "neutral_changes": []
            }
        }
        
        # Compare each performance metric
        for metric in ['accuracy', 'f1_score', 'top_3_accuracy', 'inference_time_ms']:
            if metric in old_perf and metric in new_perf:
                old_val = old_perf[metric]
                new_val = new_perf[metric]
                change = new_val - old_val
                change_percent = (change / old_val) * 100 if old_val != 0 else 0
                
                comparison['performance_changes'][metric] = {
                    "old": old_val,
                    "new": new_val,
                    "change": change,
                    "change_percent": change_percent
                }
                
                # Categorize changes
                if metric == 'inference_time_ms':
                    # Lower is better for inference time
                    if change < -1:
                        comparison['summary']['improvements'].append(f"Faster inference: {abs(change):.1f}ms improvement")
                    elif change > 1:
                        comparison['summary']['regressions'].append(f"Slower inference: {change:.1f}ms regression")
                else:
                    # Higher is better for accuracy metrics
                    if change > 0.001:
                        comparison['summary']['improvements'].append(f"Better {metric}: +{change:.3f} ({change_percent:.1f}%)")
                    elif change < -0.001:
                        comparison['summary']['regressions'].append(f"Lower {metric}: {change:.3f} ({change_percent:.1f}%)")
        
        # Save comparison report
        comparison_path = self.releases_dir / f"comparison_v{old_metadata.get('version')}_to_v{new_metadata.get('version')}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✅ Version comparison saved: {comparison_path}")
        return comparison
    
    def _run_command(self, command: str, capture_output: bool = False) -> Optional[str]:
        """Run shell command and optionally capture output."""
        try:
            if capture_output:
                result = subprocess.run(command, shell=True, capture_output=True, 
                                      text=True, cwd=self.repo_path)
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    return None
            else:
                subprocess.run(command, shell=True, check=True, cwd=self.repo_path)
                return None
        except subprocess.CalledProcessError:
            return None


def create_github_actions_workflow():
    """Create GitHub Actions workflow for automated model deployment."""
    workflow_content = """name: Model Training and Deployment

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'models/**'
      - 'utils/**' 
      - 'train_mi300x.py'
      - 'config_mi300x_optimized.json'
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      version_type:
        description: 'Version increment type'
        required: true
        default: 'patch'
        type: choice
        options:
          - patch
          - minor
          - major

env:
  PYTHON_VERSION: '3.11'
  PYTORCH_VERSION: '2.8.0'

jobs:
  model-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        lfs: true
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install onnx onnxruntime huggingface_hub
    
    - name: Validate model architecture
      run: |
        python models/cnn_models.py --test_models
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
  
  model-compilation:
    needs: model-validation
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
      with:
        lfs: true
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install onnx onnxruntime huggingface_hub
    
    - name: Download latest model
      run: |
        # Download trained model from artifacts or model registry
        # This would be customized based on your model storage
        echo "Downloading latest trained model..."
    
    - name: Compile models
      run: |
        python compile_models.py --model_path results_mi300x/best_model.pth --formats all
    
    - name: Version and tag
      run: |
        python version_manager.py --increment patch --commit --tag
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Upload to HuggingFace
      run: |
        python upload_to_huggingface.py --model_dir compiled_models --repo_name plantnet-ensemble-v1
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: v${{ steps.version.outputs.version }}
        release_name: PlantNet v${{ steps.version.outputs.version }}
        body_path: CHANGELOG.md
        draft: false
        prerelease: false

  deploy-demo:
    needs: model-compilation
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to demo environment
      run: |
        echo "Deploying to demo environment..."
        # Add your deployment scripts here
"""
    
    # Create .github/workflows directory
    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    # Write workflow file
    workflow_path = workflow_dir / "model-deployment.yml"
    workflow_path.write_text(workflow_content)
    
    print(f"✅ GitHub Actions workflow created: {workflow_path}")
    return str(workflow_path)


def main():
    """Main version management script."""
    parser = argparse.ArgumentParser(description='PlantNet Model Version Manager')
    parser.add_argument('--increment', choices=['major', 'minor', 'patch'],
                       help='Increment version number')
    parser.add_argument('--set_version', type=str,
                       help='Set specific version number')
    parser.add_argument('--model_path', type=str,
                       help='Path to model file for release')
    parser.add_argument('--performance_metrics', type=str,
                       help='Path to performance metrics JSON file')
    parser.add_argument('--changes', type=str, nargs='+',
                       help='List of changes for changelog')
    parser.add_argument('--commit', action='store_true',
                       help='Commit changes to git')
    parser.add_argument('--tag', action='store_true',
                       help='Create git tag')
    parser.add_argument('--create_workflow', action='store_true',
                       help='Create GitHub Actions workflow')
    parser.add_argument('--compare', type=str, nargs=2,
                       help='Compare two model versions (old_metadata new_metadata)')
    
    args = parser.parse_args()
    
    try:
        # Initialize version manager
        version_manager = ModelVersionManager()
        
        # Create GitHub Actions workflow
        if args.create_workflow:
            create_github_actions_workflow()
            return 0
        
        # Compare versions
        if args.compare:
            old_metadata, new_metadata = args.compare
            comparison = version_manager.compare_versions(old_metadata, new_metadata)
            print("\\n📊 Version Comparison Summary:")
            for improvement in comparison['summary']['improvements']:
                print(f"  ✅ {improvement}")
            for regression in comparison['summary']['regressions']:
                print(f"  ⚠️ {regression}")
            return 0
        
        # Set or increment version
        if args.set_version:
            new_version = args.set_version
            version_manager.set_version(new_version)
        elif args.increment:
            new_version = version_manager.increment_version(args.increment)
            version_manager.set_version(new_version)
        else:
            current_version = version_manager.get_current_version()
            print(f"📌 Current version: {current_version}")
            return 0
        
        # Load performance metrics
        performance_metrics = {}
        if args.performance_metrics and os.path.exists(args.performance_metrics):
            with open(args.performance_metrics, 'r') as f:
                performance_metrics = json.load(f)
        
        # Create release metadata
        if args.model_path:
            version_manager.create_release_metadata(
                new_version, 
                args.model_path, 
                performance_metrics
            )
        
        # Update changelog
        if args.changes:
            version_manager.update_changelog(
                new_version, 
                args.changes,
                performance_metrics.get('performance', {})
            )
        
        # Commit changes
        if args.commit:
            version_manager._run_command("git add .")
            version_manager._run_command(f'git commit -m "Release v{new_version}"')
            print("✅ Changes committed to git")
        
        # Create git tag
        if args.tag:
            version_manager.create_git_tag(new_version)
        
        print(f"\\n🎉 Version management completed!")
        print(f"📌 Version: {new_version}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Version management failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)