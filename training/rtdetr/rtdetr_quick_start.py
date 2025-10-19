#!/usr/bin/env python3
"""
RT-DETR Quick Start Script
==========================

Simple interface for RT-DETR training with minimal configuration.
Automatically detects dataset format and creates optimal settings.

RT-DETR (Real-Time Detection Transformer) is a Vision Transformer-based
detector that provides:
- Real-time performance with high accuracy
- NMS-free detection framework
- Efficient hybrid encoder
- IoU-aware query selection

Usage:
    python rtdetr_quick_start.py
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtdetr_train import RTDETRConfig, RTDETRTrainer, create_dataset_yaml


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
        "rtdetr-l": {"batch": 16, "epochs": 100, "imgsz": 640, "lr0": 0.001},
        "rtdetr-x": {"batch": 12, "epochs": 100, "imgsz": 640, "lr0": 0.0008},
    }
    
    base_settings = settings.get(model_size, settings["rtdetr-l"])
    
    # Adjust for dataset size
    if dataset_size == "small":
        base_settings["epochs"] = 50
        base_settings["warmup_epochs"] = 3
    elif dataset_size == "large":
        base_settings["epochs"] = 200
        base_settings["warmup_epochs"] = 10
    else:
        base_settings["warmup_epochs"] = 5
        
    return base_settings


def main():
    """Main quick start function"""
    print("RT-DETR Quick Start Training")
    print("=" * 50)
    print("Real-Time Detection Transformer")
    print("Vision Transformer-based object detector")
    print("=" * 50)
    
    # Get dataset path
    while True:
        data_dir = input("\nEnter your dataset directory: ").strip()
        if os.path.exists(data_dir):
            break
        print("Directory does not exist. Please try again.")
    
    # Detect dataset format
    format_type = detect_dataset_format(data_dir)
    print(f"\nDetected dataset format: {format_type.upper()}")
    
    if format_type == "unknown":
        print("\nUnknown dataset format. Please ensure your dataset has:")
        print("   - COCO: _annotations.coco.json files")
        print("   - YOLO: labels/ directory with .txt files")
        print("   - VOC: .xml annotation files")
        return
    
    # Get model size
    print("\nChoose RT-DETR model:")
    print("1. RT-DETR-L (Large) - 53.0% AP, 114 FPS on T4 GPU")
    print("2. RT-DETR-X (Extra Large) - 54.8% AP, 74 FPS on T4 GPU")
    
    model_choice = input("Choose (1-2, default: 1): ").strip()
    model_map = {"1": "rtdetr-l", "2": "rtdetr-x"}
    model_size = model_map.get(model_choice, "rtdetr-l")
    
    # Get dataset size
    print("\nEstimate your dataset size:")
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
        epochs = int(input(f"\nNumber of epochs (default: {optimal['epochs']}): ") or optimal['epochs'])
        batch = int(input(f"Batch size (default: {optimal['batch']}): ") or optimal['batch'])
        imgsz = int(input(f"Image size (default: {optimal['imgsz']}): ") or optimal['imgsz'])
    except ValueError:
        print("Invalid input. Using defaults.")
        epochs, batch, imgsz = optimal['epochs'], optimal['batch'], optimal['imgsz']
    
    # Get output settings
    project = input("\nProject directory (default: runs/train): ").strip() or "runs/train"
    name = input("Experiment name (default: rtdetr): ").strip() or "rtdetr"
    
    # Create dataset YAML if needed
    if format_type in ["coco", "voc"]:
        print(f"\nCreating dataset YAML for {format_type.upper()} format...")
        dataset_yaml = create_dataset_yaml(data_dir, f"{name}_dataset.yaml")
    else:
        dataset_yaml = os.path.join(data_dir, "dataset.yaml")
        if not os.path.exists(dataset_yaml):
            print("No dataset.yaml found. Please create one or use COCO/VOC format.")
            return
    
    # Advanced options
    print("\nAdvanced options:")
    device = input("Device (default: auto): ").strip() or ""
    
    # RT-DETR specific options
    print("\nRT-DETR configuration:")
    decoder_layers = input("Decoder layers (default: 6, fewer=faster): ").strip()
    decoder_layers = int(decoder_layers) if decoder_layers else 6
    
    num_queries = input("Number of queries (default: 300): ").strip()
    num_queries = int(num_queries) if num_queries else 300
    
    use_amp = input("Enable mixed precision training? (Y/n): ").strip().lower()
    use_amp = use_amp != 'n'
    
    # Create configuration
    config = RTDETRConfig(
        model=model_size,
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        lr0=optimal['lr0'],
        warmup_epochs=optimal['warmup_epochs'],
        decoder_layers=decoder_layers,
        num_queries=num_queries,
        amp=use_amp,
    )
    
    # Show configuration
    print("\nConfiguration Summary:")
    print("-" * 50)
    print(f"Model: {config.model}")
    print(f"Dataset: {config.data}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch}")
    print(f"Image size: {config.imgsz}")
    print(f"Learning rate: {config.lr0}")
    print(f"Warmup epochs: {config.warmup_epochs}")
    print(f"Device: {config.device or 'auto'}")
    print(f"Decoder layers: {config.decoder_layers}")
    print(f"Number of queries: {config.num_queries}")
    print(f"Mixed precision: {config.amp}")
    print(f"Output: {config.project}/{config.name}")
    print("-" * 50)
    
    # RT-DETR benefits information
    print("\nRT-DETR Benefits:")
    print("  - NMS-free detection (no post-processing bottleneck)")
    print("  - Efficient hybrid encoder for multiscale features")
    print("  - IoU-aware query selection for better accuracy")
    print("  - Adaptable inference speed via decoder layers")
    print("  - Excellent performance with TensorRT acceleration")
    
    # Confirm and start
    print(f"\nReady to start training!")
    confirm = input("Start training now? (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        print("\nStarting RT-DETR training...")
        try:
            trainer = RTDETRTrainer(config)
            results = trainer.train_model()
            
            print("\nTraining completed successfully!")
            print(f"Results saved to: {config.project}/{config.name}")
            
            # Provide next steps
            print("\nNext steps:")
            print(f"  1. Validate: python rtdetr_train.py --model {config.model} --data {dataset_yaml} --validate")
            print(f"  2. Export to TensorRT for optimal inference speed")
            print(f"  3. Adjust decoder layers for different speed/accuracy trade-offs")
            
        except Exception as e:
            print(f"\nTraining failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nTo start training later, run:")
        print(f"python rtdetr_train.py --model {model_size} --data {dataset_yaml} --epochs {epochs} --batch {batch}")


if __name__ == "__main__":
    main()

