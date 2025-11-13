"""
Setup script for RL-STIR environment.
Handles Conda environment creation and dependency installation.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA is available: {device_name}")
            return True
        else:
            print("❌ CUDA is not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def setup_conda_env():
    """Set up the Conda environment"""
    print("🚀 Setting up RL-STIR Conda environment...")
    
    # Check if conda is available
    result = run_command("conda --version", check=False)
    if result.returncode != 0:
        print("❌ Conda not found. Please install Anaconda or Miniconda first.")
        sys.exit(1)
    
    print("✅ Conda found")
    
    # Create environment from yml file
    env_file = Path("environment.yml")
    if not env_file.exists():
        print("❌ environment.yml not found")
        sys.exit(1)
    
    print("📦 Creating Conda environment from environment.yml...")
    run_command("conda env create -f environment.yml")
    
    print("✅ Environment created successfully!")


def install_package():
    """Install the RL-STIR package in development mode"""
    print("📦 Installing RL-STIR package in development mode...")
    run_command("pip install -e .")
    print("✅ Package installed successfully!")


def verify_installation():
    """Verify the installation"""
    print("🔍 Verifying installation...")
    
    # Test imports
    try:
        import torch
        import torchvision
        import pandas
        import numpy
        import gymnasium
        import torch_geometric
        import transformers
        import sentencepiece
        import pytorch_lightning
        print("✅ All core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check CUDA
    cuda_ok = check_cuda()
    
    # Test RL-STIR imports
    try:
        from data.schemas import LogRecord, LogChannel
        from modeling.tokenizer import RLSTIRTokenizer
        from envs.sim.gen import SyntheticEpisodeGenerator
        print("✅ RL-STIR modules imported successfully")
    except ImportError as e:
        print(f"❌ RL-STIR import error: {e}")
        return False
    
    return cuda_ok


def main():
    """Main setup function"""
    print("🎯 RL-STIR Environment Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("environment.yml").exists():
        print("❌ Please run this script from the RL-STIR project root directory")
        sys.exit(1)
    
    # Set up environment
    setup_conda_env()
    
    # Install package
    install_package()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate the environment: conda activate rl-stir")
        print("2. Generate sample data: python -m envs.sim.gen")
        print("3. Start training: python scripts/train_supervised.py configs/supervised_4060.yaml")
    else:
        print("\n⚠️  Setup completed with warnings. Please check the output above.")


if __name__ == "__main__":
    main()
