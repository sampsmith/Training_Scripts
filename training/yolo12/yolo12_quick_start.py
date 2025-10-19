#!/usr/bin/env python3
"""
YOLO12 Quick Start Script
========================

Simple interface for YOLO12 training with minimal configuration.
Automatically detects dataset format and creates optimal settings.

Usage:
    python yolo12_quick_start.py
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolo12_train import YOLO12Config, YOLO12Trainer, create_dataset_yaml


def detect_dataset_format(data_dir: str) -> str:
    """Detect dataset format (COCO, YOLO, VOC)"""
    train_dir = os.path.join(data_dir, "train")
    
    if not os.path.exists(train_dir):
        return "unknown"
    
    # Check for COCO format
    coco_ann = os.path.join(train_dir, "_annotations.coco.json")
    if os.path.exists(coco_ann):
        return "coco"
    
    # Check for YOLO format
    if os.path.exists(os.path.join(train_dir, "labels")):
        return "yolo"
    
    # Check for VOC format
    xml_files = [f for f in os.listdir(train_dir) if f.endswith('.xml')]
    if xml_files:
        return "voc"
    
    return "unknown"


def get_optimal_settings(model_size: str, dataset_size: str) -> dict:
    """Get optimal settings based on model and dataset size"""
    settings = {
        "yolo12n": {"batch": 32, "epochs": 100, "imgsz": 640},
        "yolo12s": {"batch": 24, "epochs": 100, "imgsz": 640},
        "yolo12m": {"batch": 16, "epochs": 100, "imgsz": 640},
        "yolo12l": {"batch": 12, "epochs": 100, "imgsz": 640},
        "yolo12x": {"batch": 8, "epochs": 100, "imgsz": 640},
    }
    
    base_settings = settings.get(model_size, settings["yolo12n"])
    
    # Adjust for dataset size
    if dataset_size == "small":
        base_settings["epochs"] = 50
    elif dataset_size == "large":
        base_settings["epochs"] = 200
        
    return base_settings


def main():
    """Main quick start function"""
    print("🚀 YOLO12 Quick Start Training")
    print("=" * 50)
    
    # Get dataset path
    while True:
        data_dir = input("\n📁 Enter your dataset directory: ").strip()
        if os.path.exists(data_dir):
            break
        print("❌ Directory does not exist. Please try again.")
    
    # Detect dataset format
    format_type = detect_dataset_format(data_dir)
    print(f"📊 Detected dataset format: {format_type.upper()}")
    
    if format_type == "unknown":
        print("❌ Unknown dataset format. Please ensure your dataset has:")
        print("   - COCO: _annotations.coco.json files")
        print("   - YOLO: labels/ directory with .txt files")
        print("   - VOC: .xml annotation files")
        return
    
    # Get model size
    print("\n🤖 Choose model size:")
    print("1. YOLO12n (nano) - Fastest, smallest")
    print("2. YOLO12s (small) - Good balance")
    print("3. YOLO12m (medium) - Better accuracy")
    print("4. YOLO12l (large) - High accuracy")
    print("5. YOLO12x (extra large) - Best accuracy")
    
    model_choice = input("Choose (1-5, default: 2): ").strip()
    model_map = {"1": "yolo12n", "2": "yolo12s", "3": "yolo12m", "4": "yolo12l", "5": "yolo12x"}
    model_size = model_map.get(model_choice, "yolo12s")
    
    # Get dataset size
    print("\n📊 Estimate your dataset size:")
    print("1. Small (< 1000 images)")
    print("2. Medium (1000-10000 images)")
    print("3. Large (> 10000 images)")
    
    size_choice = input("Choose (1-3, default: 2): ").strip()
    size_map = {"1": "small", "2": "medium", "3": "large"}
    dataset_size = size_map.get(size_choice, "medium")
    
    # Get optimal settings
    optimal = get_optimal_settings(model_size, dataset_size)
    
    # Get training parameters
    try:
        epochs = int(input(f"\n🔄 Number of epochs (default: {optimal['epochs']}): ") or optimal['epochs'])
        batch = int(input(f"📦 Batch size (default: {optimal['batch']}): ") or optimal['batch'])
        imgsz = int(input(f"🖼️  Image size (default: {optimal['imgsz']}): ") or optimal['imgsz'])
    except ValueError:
        print("❌ Invalid input. Using defaults.")
        epochs, batch, imgsz = optimal['epochs'], optimal['batch'], optimal['imgsz']
    
    # Get output settings
    project = input("\n📁 Project directory (default: runs/train): ").strip() or "runs/train"
    name = input("📝 Experiment name (default: yolo12): ").strip() or "yolo12"
    
    # Create dataset YAML if needed
    if format_type in ["coco", "voc"]:
        print(f"\n📝 Creating dataset YAML for {format_type.upper()} format...")
        dataset_yaml = create_dataset_yaml(data_dir, f"{name}_dataset.yaml")
    else:
        dataset_yaml = os.path.join(data_dir, "dataset.yaml")
        if not os.path.exists(dataset_yaml):
            print("❌ No dataset.yaml found. Please create one or use COCO/VOC format.")
            return
    
    # Advanced options
    print("\n⚙️  Advanced options:")
    use_flash = input("Enable FlashAttention? (y/N): ").strip().lower() in ['y', 'yes']
    device = input("Device (default: auto): ").strip() or ""
    
    # Create configuration
    config = YOLO12Config(
        model=model_size,
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        use_flash_attention=use_flash,
    )
    
    # Show configuration
    print("\n📋 Configuration:")
    print("-" * 30)
    print(f"Model: {config.model}")
    print(f"Dataset: {config.data}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch}")
    print(f"Image size: {config.imgsz}")
    print(f"Device: {config.device or 'auto'}")
    print(f"FlashAttention: {config.use_flash_attention}")
    print(f"Output: {config.project}/{config.name}")
    
    # Confirm and start
    print(f"\n🚀 Ready to start training!")
    confirm = input("Start training now? (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        print("\n🏃 Starting YOLO12 training...")
        try:
            trainer = YOLO12Trainer(config)
            results = trainer.train_model()
            
            print("\n✅ Training completed successfully!")
            print(f"📁 Results saved to: {config.project}/{config.name}")
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n📝 To start training later, run:")
        print(f"python yolo12_train.py --model {model_size} --data {dataset_yaml} --epochs {epochs} --batch {batch}")


if __name__ == "__main__":
    main()
