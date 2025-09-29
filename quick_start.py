#!/usr/bin/env python3
"""
Quick Start Script for RunPod Training
======================================

This script provides a simple interface to start training with minimal configuration.
Just update the paths and run!

Usage:
    python quick_start.py
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runpod_train import RunPodConfig, RunPodTrainer


def get_user_input():
    """Get configuration from user"""
    print("🚀 RunPod Object Detection Training - Quick Start")
    print("=" * 60)
    
    # Get dataset path
    while True:
        data_dir = input("\n📁 Enter your dataset path: ").strip()
        if os.path.exists(data_dir):
            break
        print("❌ Path does not exist. Please try again.")
    
    # Get output path
    output_dir = input("📁 Enter output directory (default: ./outputs): ").strip()
    if not output_dir:
        output_dir = "./outputs"
    
    # Get training parameters
    try:
        epochs = int(input("🔄 Number of epochs (default: 50): ") or "50")
        patience = int(input("⏹️  Early stopping patience (default: 10): ") or "10")
    except ValueError:
        print("❌ Invalid input. Using defaults.")
        epochs, patience = 50, 10
    
    return data_dir, output_dir, epochs, patience


def main():
    """Main quick start function"""
    try:
        # Get user configuration
        data_dir, output_dir, epochs, patience = get_user_input()
        
        # Create configuration
        config = RunPodConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs,
            patience=patience,
            # Use optimal settings for cloud
            target_effective_batch_size=16,
            pretrained_weights="COCO",
            trainable_backbone_layers=3,
            use_amp=True,
            grad_clip=1.0,
            num_workers=4,
            pin_memory=True,
            device="cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
        )
        
        print(f"\n✅ Configuration:")
        print(f"   Dataset: {config.data_dir}")
        print(f"   Output: {config.output_dir}")
        print(f"   Epochs: {config.epochs}")
        print(f"   Patience: {config.patience}")
        print(f"   Device: {config.device}")
        
        # Confirm before starting
        confirm = input("\n🚀 Start training? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Training cancelled.")
            return
        
        # Create trainer and start
        print("\n🏃 Starting training...")
        trainer = RunPodTrainer(config)
        trainer.train()
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Check your outputs in: {config.output_dir}")
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
