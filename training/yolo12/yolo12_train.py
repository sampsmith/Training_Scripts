#!/usr/bin/env python3
"""
YOLO12 Training Script - Local Training
======================================

This script provides comprehensive training for YOLO12 models with:
- Support for all YOLO12 variants (n, s, m, l, x)
- Area Attention mechanism and R-ELAN architecture
- FlashAttention optimization (optional)
- Multiple dataset formats (COCO, VOC, YOLO)
- Comprehensive monitoring and logging
- Advanced training features

Based on: https://docs.ultralytics.com/models/yolo12/

Usage:
    python yolo12_train.py --model yolo12n --data coco8.yaml --epochs 100
"""

import os
import sys
import time
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml

# YOLO12 imports
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER, colorstr
    from ultralytics.utils.torch_utils import select_device
    from ultralytics.data import build_dataloader
    from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
    from ultralytics.utils.plotting import plot_results
    from ultralytics.utils.callbacks import default_callbacks
except ImportError as e:
    print("❌ Error: ultralytics package not found!")
    print("Install with: pip install ultralytics")
    sys.exit(1)


@dataclass
class YOLO12Config:
    """Configuration for YOLO12 training"""
    # Model settings
    model: str = "yolo12n"  # yolo12n, yolo12s, yolo12m, yolo12l, yolo12x
    pretrained: bool = True
    freeze: Optional[List[int]] = None
    
    # Data settings
    data: str = "coco8.yaml"  # Path to dataset yaml
    imgsz: int = 640
    batch: int = 16
    workers: int = 8
    
    # Training settings
    epochs: int = 100
    patience: int = 50
    save_period: int = 10
    
    # Optimization
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    
    # YOLO12 specific
    use_flash_attention: bool = False
    area_attention_regions: int = 4
    r_elan_blocks: int = 3
    
    # Hardware
    device: str = ""
    multi_scale: bool = False
    single_cls: bool = False
    
    # Logging
    project: str = "runs/train"
    name: str = "yolo12"
    exist_ok: bool = False
    verbose: bool = True
    seed: int = 0
    
    # Validation
    val: bool = True
    split: str = "val"
    save_json: bool = False
    save_hybrid: bool = False
    conf: float = 0.001
    iou: float = 0.6
    max_det: int = 300
    half: bool = False
    dnn: bool = False
    
    # Export
    format: str = "torchscript"
    optimize: bool = False
    int8: bool = False
    dynamic: bool = False
    simplify: bool = False
    opset: Optional[int] = None
    workspace: int = 4
    nms: bool = False


class YOLO12Trainer:
    """Advanced YOLO12 trainer with comprehensive features"""
    
    def __init__(self, config: YOLO12Config):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Training state
        self.model = None
        self.trainer = None
        self.start_time = None
        self.best_fitness = 0.0
        self.results = {}
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path(self.config.project) / self.config.name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.project, exist_ok=True)
        
    def log_system_info(self):
        """Log system information"""
        self.logger.info("=" * 80)
        self.logger.info("YOLO12 TRAINING SESSION STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"GPU count: {torch.cuda.device_count()}")
            self.logger.info(f"Current GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.logger.info(f"Model: {self.config.model}")
        self.logger.info(f"Dataset: {self.config.data}")
        self.logger.info(f"Image size: {self.config.imgsz}")
        self.logger.info(f"Batch size: {self.config.batch}")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info(f"Device: {self.config.device or 'auto'}")
        self.logger.info("=" * 80)
        
    def check_flash_attention(self):
        """Check FlashAttention availability and requirements"""
        if not self.config.use_flash_attention:
            self.logger.info("FlashAttention disabled")
            return True
            
        try:
            import flash_attn
            self.logger.info("✅ FlashAttention available")
            return True
        except ImportError:
            self.logger.warning("⚠️  FlashAttention not available")
            self.logger.info("To enable FlashAttention:")
            self.logger.info("1. Install: pip install flash-attn")
            self.logger.info("2. Requires compatible GPU (Turing, Ampere, Ada Lovelace, or Hopper)")
            return False
            
    def validate_dataset(self, data_path: str) -> bool:
        """Validate dataset configuration"""
        if not os.path.exists(data_path):
            self.logger.error(f"Dataset file not found: {data_path}")
            return False
            
        try:
            with open(data_path, 'r') as f:
                data_config = yaml.safe_load(f)
                
            # Check required fields
            required_fields = ['path', 'train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data_config:
                    self.logger.error(f"Missing required field '{field}' in dataset config")
                    return False
                    
            # Check if paths exist
            base_path = data_config['path']
            for split in ['train', 'val']:
                split_path = os.path.join(base_path, data_config[split])
                if not os.path.exists(split_path):
                    self.logger.error(f"Split path not found: {split_path}")
                    return False
                    
            self.logger.info(f"✅ Dataset validation passed")
            self.logger.info(f"Classes: {data_config['nc']}")
            self.logger.info(f"Class names: {data_config['names']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False
            
    def create_model(self) -> YOLO:
        """Create YOLO12 model"""
        self.logger.info(f"Creating YOLO12 model: {self.config.model}")
        
        try:
            # Load model
            model = YOLO(f"{self.config.model}.pt" if self.config.pretrained else f"{self.config.model}.yaml")
            
            # Configure model for training
            if hasattr(model.model, 'area_attention_regions'):
                model.model.area_attention_regions = self.config.area_attention_regions
                
            if hasattr(model.model, 'r_elan_blocks'):
                model.model.r_elan_blocks = self.config.r_elan_blocks
                
            self.logger.info("✅ Model created successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            raise
            
    def train_model(self) -> Dict:
        """Train the YOLO12 model"""
        try:
            self.log_system_info()
            self.check_flash_attention()
            
            # Validate dataset
            if not self.validate_dataset(self.config.data):
                raise ValueError("Dataset validation failed")
                
            # Create model
            self.model = self.create_model()
            
            # Start training
            self.logger.info("Starting YOLO12 training...")
            self.start_time = time.time()
            
            # Train the model
            results = self.model.train(
                # Model
                model=self.model,
                
                # Data
                data=self.config.data,
                imgsz=self.config.imgsz,
                batch=self.config.batch,
                workers=self.config.workers,
                
                # Training
                epochs=self.config.epochs,
                patience=self.config.patience,
                save_period=self.config.save_period,
                
                # Optimization
                lr0=self.config.lr0,
                lrf=self.config.lrf,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                warmup_epochs=self.config.warmup_epochs,
                warmup_momentum=self.config.warmup_momentum,
                warmup_bias_lr=self.config.warmup_bias_lr,
                
                # Augmentation
                hsv_h=self.config.hsv_h,
                hsv_s=self.config.hsv_s,
                hsv_v=self.config.hsv_v,
                degrees=self.config.degrees,
                translate=self.config.translate,
                scale=self.config.scale,
                shear=self.config.shear,
                perspective=self.config.perspective,
                flipud=self.config.flipud,
                fliplr=self.config.fliplr,
                mosaic=self.config.mosaic,
                mixup=self.config.mixup,
                copy_paste=self.config.copy_paste,
                
                # Hardware
                device=self.config.device,
                multi_scale=self.config.multi_scale,
                single_cls=self.config.single_cls,
                
                # Logging
                project=self.config.project,
                name=self.config.name,
                exist_ok=self.config.exist_ok,
                verbose=self.config.verbose,
                seed=self.config.seed,
                
                # Validation
                val=self.config.val,
                split=self.config.split,
                save_json=self.config.save_json,
                save_hybrid=self.config.save_hybrid,
                conf=self.config.conf,
                iou=self.config.iou,
                max_det=self.config.max_det,
                half=self.config.half,
                dnn=self.config.dnn,
            )
            
            # Training completed
            total_time = time.time() - self.start_time
            self.logger.info(f"✅ Training completed in {total_time/3600:.2f} hours")
            
            # Log final results
            if hasattr(results, 'results_dict'):
                self.logger.info("Final Results:")
                for key, value in results.results_dict.items():
                    self.logger.info(f"  {key}: {value:.4f}")
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
            
    def validate_model(self) -> Dict:
        """Validate the trained model"""
        if not self.model:
            raise ValueError("No model available for validation")
            
        self.logger.info("Running validation...")
        
        try:
            # Run validation
            results = self.model.val(
                data=self.config.data,
                imgsz=self.config.imgsz,
                batch=self.config.batch,
                conf=self.config.conf,
                iou=self.config.iou,
                max_det=self.config.max_det,
                half=self.config.half,
                device=self.config.device,
                save_json=self.config.save_json,
                save_hybrid=self.config.save_hybrid,
                verbose=self.config.verbose
            )
            
            self.logger.info("Validation Results:")
            if hasattr(results, 'results_dict'):
                for key, value in results.results_dict.items():
                    self.logger.info(f"  {key}: {value:.4f}")
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise
            
    def export_model(self, format: str = "torchscript") -> str:
        """Export the trained model"""
        if not self.model:
            raise ValueError("No model available for export")
            
        self.logger.info(f"Exporting model to {format}...")
        
        try:
            export_path = self.model.export(
                format=format,
                imgsz=self.config.imgsz,
                optimize=self.config.optimize,
                int8=self.config.int8,
                dynamic=self.config.dynamic,
                simplify=self.config.simplify,
                opset=self.config.opset,
                workspace=self.config.workspace,
                nms=self.config.nms,
                device=self.config.device,
                verbose=self.config.verbose
            )
            
            self.logger.info(f"✅ Model exported to: {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise


def create_dataset_yaml(data_dir: str, output_path: str = "dataset.yaml"):
    """Create a dataset YAML file for YOLO training"""
    # Detect dataset format and create appropriate YAML
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    
    # Check if train/valid directories exist, if not, use the root directory
    if not os.path.exists(train_dir):
        train_dir = data_dir
    if not os.path.exists(val_dir):
        val_dir = data_dir
    
    # Detect classes from directory structure or annotation files
    classes = []
    
    # Try to detect from COCO format
    coco_ann = os.path.join(train_dir, "_annotations.coco.json")
    if os.path.exists(coco_ann):
        import json
        with open(coco_ann, 'r') as f:
            coco_data = json.load(f)
        classes = [cat['name'] for cat in coco_data['categories']]
    else:
        # Try to detect from YOLO format
        label_dir = os.path.join(train_dir, "labels")
        if os.path.exists(label_dir):
            # Read classes from first label file
            for file in os.listdir(label_dir):
                if file.endswith('.txt'):
                    with open(os.path.join(label_dir, file), 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            class_id = int(line.split()[0])
                            if class_id >= len(classes):
                                classes.extend([f"class_{i}" for i in range(len(classes), class_id + 1)])
                    break
    
    if not classes:
        # Default classes if none detected
        classes = [f"class_{i}" for i in range(10)]
    
    # Determine the correct paths based on actual directory structure
    train_path = "train/images" if os.path.exists(os.path.join(data_dir, "train", "images")) else "."
    val_path = "valid/images" if os.path.exists(os.path.join(data_dir, "valid", "images")) else "."
    
    # Create YAML content
    yaml_content = {
        'path': data_dir,
        'train': train_path,
        'val': val_path,
        'nc': len(classes),
        'names': classes
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"✅ Created dataset YAML: {output_path}")
    print(f"Classes: {classes}")
    return output_path


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="YOLO12 Training Script")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="yolo12n", 
                       choices=["yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x"],
                       help="YOLO12 model variant")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device (cpu, 0, 1, 2, ...)")
    
    # Training arguments
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory")
    parser.add_argument("--name", type=str, default="yolo12", help="Experiment name")
    
    # YOLO12 specific
    parser.add_argument("--flash-attention", action="store_true", help="Enable FlashAttention")
    parser.add_argument("--area-attention-regions", type=int, default=4, help="Area attention regions")
    
    # Dataset creation
    parser.add_argument("--create-dataset-yaml", type=str, help="Create dataset YAML from directory")
    
    args = parser.parse_args()
    
    # Create dataset YAML if requested
    if args.create_dataset_yaml:
        dataset_yaml = create_dataset_yaml(args.create_dataset_yaml)
        args.data = dataset_yaml
    
    # Create configuration
    config = YOLO12Config(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        patience=args.patience,
        project=args.project,
        name=args.name,
        use_flash_attention=args.flash_attention,
        area_attention_regions=args.area_attention_regions,
    )
    
    # Create trainer and run
    trainer = YOLO12Trainer(config)
    
    try:
        # Train model
        results = trainer.train_model()
        
        # Validate model
        val_results = trainer.validate_model()
        
        # Export model
        export_path = trainer.export_model()
        
        print("\n" + "="*80)
        print("🎉 YOLO12 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Model: {config.model}")
        print(f"Results saved to: {config.project}/{config.name}")
        print(f"Exported model: {export_path}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
