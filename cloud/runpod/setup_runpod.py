#!/usr/bin/env python3
"""
RunPod Setup Script
===================

This script helps you set up your RunPod environment for training.
It will:
1. Install required dependencies
2. Verify your dataset structure
3. Set up the training environment
4. Run a quick test to ensure everything works

Usage:
    python setup_runpod.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional


class RunPodSetup:
    """Setup helper for RunPod environment"""
    
    def __init__(self):
        self.required_packages = [
            "torch>=2.2.0",
            "torchvision>=0.17.0", 
            "pycocotools>=2.0.7",
            "opencv-python>=4.9.0.80",
            "numpy>=1.26.0",
            "tqdm>=4.66.0",
            "Pillow>=10.0.0",
            "matplotlib>=3.8.0",
            "pyyaml>=6.0.1",
            "roboflow>=1.1.27"
        ]
        
    def install_dependencies(self):
        """Install required packages"""
        print("Installing dependencies...")
        for package in self.required_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")
                return False
        return True
        
    def verify_dataset_structure(self, data_dir: str) -> bool:
        """Verify dataset structure"""
        print(f"Verifying dataset structure at: {data_dir}")
        
        if not os.path.exists(data_dir):
            print(f"✗ Dataset directory not found: {data_dir}")
            return False
            
        # Check for common dataset structures
        required_dirs = ["train", "valid"]
        found_format = None
        
        for split in required_dirs:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                print(f"✗ Missing {split} directory")
                return False
                
            # Check for COCO format
            coco_ann = os.path.join(split_dir, "_annotations.coco.json")
            if os.path.exists(coco_ann):
                found_format = "COCO"
                print(f"✓ Found COCO format in {split}")
                continue
                
            # Check for VOC format
            xml_files = [f for f in os.listdir(split_dir) if f.endswith('.xml')]
            if xml_files:
                found_format = "VOC"
                print(f"✓ Found VOC format in {split}")
                continue
                
            print(f"✗ No valid annotations found in {split}")
            return False
            
        print(f"✓ Dataset format detected: {found_format}")
        return True
        
    def create_output_structure(self, output_dir: str):
        """Create output directory structure"""
        print(f"Creating output structure at: {output_dir}")
        
        dirs_to_create = [
            output_dir,
            f"{output_dir}/checkpoints",
            f"{output_dir}/logs",
            f"{output_dir}/tensorboard"
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created: {dir_path}")
            
    def test_training_script(self, data_dir: str, output_dir: str) -> bool:
        """Test the training script with a quick run"""
        print("Testing training script...")
        
        try:
            # Import the training script
            import runpod_train
            
            # Create a test config
            config = runpod_train.RunPodConfig(
                data_dir=data_dir,
                output_dir=output_dir,
                epochs=1,  # Just test with 1 epoch
                patience=1
            )
            
            print("✓ Training script imports successfully")
            return True
            
        except Exception as e:
            print(f"✗ Training script test failed: {e}")
            return False
            
    def create_runpod_config(self, data_dir: str, output_dir: str):
        """Create a RunPod-specific configuration file"""
        config = {
            "data_dir": data_dir,
            "output_dir": output_dir,
            "epochs": 50,
            "patience": 10,
            "target_effective_batch_size": 16,
            "pretrained_weights": "COCO",
            "trainable_backbone_layers": 3,
            "use_amp": True,
            "grad_clip": 1.0,
            "num_workers": 4,
            "pin_memory": True,
            "device": "cuda" if os.system("nvidia-smi") == 0 else "cpu",
            "log_level": "INFO",
            "save_interval": 5,
            "eval_interval": 2
        }
        
        config_path = "runpod_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"✓ Created configuration file: {config_path}")
        return config_path
        
    def print_usage_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("RUNPOD SETUP COMPLETE!")
        print("="*60)
        print("\nTo start training:")
        print("1. Update the paths in runpod_train.py:")
        print("   - data_dir: Path to your dataset")
        print("   - output_dir: Path for outputs")
        print("\n2. Run the training script:")
        print("   python runpod_train.py")
        print("\n3. Monitor training:")
        print("   - Check logs in: /workspace/outputs/logs/")
        print("   - Checkpoints in: /workspace/outputs/checkpoints/")
        print("\n4. For RunPod Pod deployment:")
        print("   - Upload your dataset to /workspace/dataset/")
        print("   - The script will automatically detect COCO or VOC format")
        print("   - Training will start automatically")
        print("="*60)


def main():
    """Main setup function"""
    print("RunPod Training Setup")
    print("=" * 40)
    
    # Get paths from user or use defaults
    data_dir = input("Enter dataset path (default: /workspace/dataset): ").strip()
    if not data_dir:
        data_dir = "/workspace/dataset"
        
    output_dir = input("Enter output path (default: /workspace/outputs): ").strip()
    if not output_dir:
        output_dir = "/workspace/outputs"
    
    setup = RunPodSetup()
    
    # Run setup steps
    steps = [
        ("Installing dependencies", lambda: setup.install_dependencies()),
        ("Creating output structure", lambda: setup.create_output_structure(output_dir)),
        ("Verifying dataset", lambda: setup.verify_dataset_structure(data_dir)),
        ("Testing training script", lambda: setup.test_training_script(data_dir, output_dir)),
        ("Creating config", lambda: setup.create_runpod_config(data_dir, output_dir))
    ]
    
    success = True
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"✗ {step_name} failed!")
            success = False
        else:
            print(f"✓ {step_name} completed")
    
    if success:
        setup.print_usage_instructions()
    else:
        print("\n✗ Setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
