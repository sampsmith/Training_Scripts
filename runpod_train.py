#!/usr/bin/env python3
"""
RunPod Optimized Training Script for Object Detection
====================================================

This script is specifically optimized for RunPod cloud training with:
- Hardcoded optimal settings for cloud GPUs
- Automatic batch size detection
- Memory optimization
- Robust error handling and recovery
- Comprehensive logging and monitoring

Usage:
    python runpod_train.py

The script will automatically:
1. Detect your dataset format (COCO/VOC)
2. Find optimal batch size for your GPU
3. Train with best practices for cloud training
4. Save checkpoints and logs
"""

import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add detection module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from detection.datasets.factory import build_datasets
from detection.utils.transforms import Compose, ToTensor, RandomHorizontalFlip, collate_fn
from detection.utils.coco_eval import coco_evaluate


@dataclass
class RunPodConfig:
    """Hardcoded optimal settings for RunPod cloud training"""
    # Paths (update these for your specific dataset)
    data_dir: str = "/workspace/dataset"  # Update this path
    output_dir: str = "/workspace/outputs"
    
    # Training hyperparameters (optimized for cloud)
    epochs: int = 50
    patience: int = 10
    target_effective_batch_size: int = 16
    
    # Model settings
    pretrained_weights: str = "COCO"  # Best for detection
    trainable_backbone_layers: int = 3
    
    # Cloud optimizations
    use_amp: bool = True  # Mixed precision for speed
    grad_clip: float = 1.0  # Gradient clipping for stability
    num_workers: int = 4  # Optimized for cloud
    pin_memory: bool = True
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_level: str = "INFO"
    save_interval: int = 5  # Save every 5 epochs
    eval_interval: int = 2  # Evaluate every 2 epochs


class RunPodTrainer:
    """Optimized trainer for RunPod cloud training"""
    
    def __init__(self, config: RunPodConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Training state
        self.best_metric = -1.0
        self.patience_counter = 0
        self.start_time = None
        
    def setup_logging(self):
        """Setup comprehensive logging for cloud training"""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(f"{self.config.output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.config.output_dir}/logs", exist_ok=True)
        
    def log_system_info(self):
        """Log system information for debugging"""
        self.logger.info("=" * 60)
        self.logger.info("RUNPOD TRAINING SESSION STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"GPU count: {torch.cuda.device_count()}")
            self.logger.info(f"Current GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        self.logger.info(f"Data directory: {self.config.data_dir}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info("=" * 60)
        
    def build_model(self, num_classes: int) -> nn.Module:
        """Build optimized Faster R-CNN model"""
        self.logger.info(f"Building model for {num_classes} classes...")
        
        if self.config.pretrained_weights.upper() == "COCO":
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            backbone_weights = None
        elif self.config.pretrained_weights.upper() == "IMAGENET":
            weights = None
            backbone_weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
            backbone_weights = None
            
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=backbone_weights,
            trainable_backbone_layers=self.config.trainable_backbone_layers,
            box_detections_per_img=300,
        )
        
        # Replace classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        return model
        
    def find_optimal_batch_size(self, model: nn.Module, train_dataset) -> Tuple[int, int]:
        """Automatically find optimal batch size for the GPU"""
        self.logger.info("Finding optimal batch size...")
        
        device = torch.device(self.config.device)
        model.to(device)
        
        # Test different batch sizes
        candidates = [32, 16, 8, 4, 2, 1]
        
        for batch_size in candidates:
            try:
                self.logger.info(f"Testing batch size: {batch_size}")
                
                # Create test loader
                test_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,  # Use fewer workers for testing
                    collate_fn=collate_fn,
                    pin_memory=self.config.pin_memory
                )
                
                # Test forward pass
                model.train()
                images, targets = next(iter(test_loader))
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                with autocast(enabled=self.config.use_amp):
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                
                # Test backward pass
                loss.backward()
                model.zero_grad(set_to_none=True)
                
                # Clear cache
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                # Calculate accumulation steps
                accumulation_steps = max(1, self.config.target_effective_batch_size // batch_size)
                
                self.logger.info(f"✓ Batch size {batch_size} works! Using accumulation: {accumulation_steps}")
                return batch_size, accumulation_steps
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"✗ Batch size {batch_size} failed: OOM")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                    
        # Fallback to batch size 1
        self.logger.warning("All batch sizes failed, using batch size 1")
        return 1, self.config.target_effective_batch_size
        
    def create_data_loaders(self, train_dataset, valid_dataset, batch_size: int):
        """Create optimized data loaders"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.config.pin_memory,
            drop_last=True  # For consistent batch sizes
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, valid_loader
        
    def train_epoch(self, model: nn.Module, optimizer, train_loader: DataLoader, 
                   scaler: GradScaler, accumulation_steps: int, epoch: int) -> Dict[str, float]:
        """Train one epoch with optimizations"""
        model.train()
        device = torch.device(self.config.device)
        
        total_losses = {}
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for step, (images, targets) in enumerate(pbar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                
            # Backward pass
            if scaler.is_enabled():
                scaler.scale(loss / accumulation_steps).backward()
            else:
                (loss / accumulation_steps).backward()
                
            # Update weights
            if (step + 1) % accumulation_steps == 0:
                if self.config.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad(set_to_none=True)
                
            # Track losses
            for k, v in loss_dict.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.detach().item()
                
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.6f}"
            })
            
        # Average losses
        for k in total_losses:
            total_losses[k] /= len(train_loader)
            
        return total_losses
        
    def evaluate_model(self, model: nn.Module, valid_loader: DataLoader, 
                      train_dataset, valid_dataset, eval_type: str) -> float:
        """Evaluate model and return metric"""
        model.eval()
        device = torch.device(self.config.device)
        
        all_outputs = []
        all_image_ids = []
        
        with torch.no_grad():
            for images, targets in tqdm(valid_loader, desc="Evaluating"):
                images = [img.to(device) for img in images]
                outputs = model(images)
                outputs = [{k: v.to("cpu") for k, v in o.items()} for o in outputs]
                all_outputs.extend(outputs)
                all_image_ids.extend([int(t["image_id"].item()) for t in targets])
                
        # Calculate metric
        if eval_type == "coco":
            _, cont_to_cat = train_dataset.get_cat_mappings()
            metrics, _ = coco_evaluate(valid_dataset.get_coco_api(), all_outputs, all_image_ids, cont_to_cat)
            metric = metrics.get("AP", 0.0)
        else:
            # Simple metric for VOC
            if all_outputs:
                scores = torch.cat([o["scores"] for o in all_outputs])
                metric = float(torch.mean(scores[:100]).item())
            else:
                metric = 0.0
                
        return metric
        
    def save_checkpoint(self, model: nn.Module, optimizer, epoch: int, 
                       metric: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        if epoch % self.config.save_interval == 0:
            checkpoint_path = f"{self.config.output_dir}/checkpoints/epoch_{epoch:03d}.pth"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
        # Save best model
        if is_best:
            best_path = f"{self.config.output_dir}/checkpoints/best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")
            
    def train(self):
        """Main training loop"""
        try:
            self.log_system_info()
            self.start_time = time.time()
            
            # Load datasets
            self.logger.info("Loading datasets...")
            train_transforms = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
            valid_transforms = Compose([ToTensor()])
            
            train_dataset, valid_dataset, eval_type = build_datasets(
                self.config.data_dir, train_transforms, valid_transforms
            )
            
            self.logger.info(f"Train samples: {len(train_dataset)}")
            self.logger.info(f"Valid samples: {len(valid_dataset)}")
            self.logger.info(f"Number of classes: {train_dataset.num_classes}")
            self.logger.info(f"Evaluation type: {eval_type}")
            
            # Build model
            model = self.build_model(train_dataset.num_classes)
            device = torch.device(self.config.device)
            model.to(device)
            
            # Find optimal batch size
            batch_size, accumulation_steps = self.find_optimal_batch_size(model, train_dataset)
            self.logger.info(f"Using batch size: {batch_size}, accumulation: {accumulation_steps}")
            
            # Create data loaders
            train_loader, valid_loader = self.create_data_loaders(
                train_dataset, valid_dataset, batch_size
            )
            
            # Setup optimizer and scheduler
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(
                params, lr=0.005, momentum=0.9, weight_decay=0.0005
            )
            
            # Learning rate scheduler
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=8, gamma=0.1
            )
            
            # Mixed precision scaler
            scaler = GradScaler(enabled=self.config.use_amp)
            
            # Training loop
            self.logger.info("Starting training...")
            
            for epoch in range(1, self.config.epochs + 1):
                epoch_start = time.time()
                
                # Train
                train_losses = self.train_epoch(
                    model, optimizer, train_loader, scaler, accumulation_steps, epoch
                )
                
                # Update learning rate
                lr_scheduler.step()
                
                # Evaluate
                if epoch % self.config.eval_interval == 0:
                    metric = self.evaluate_model(
                        model, valid_loader, train_dataset, valid_dataset, eval_type
                    )
                    
                    is_best = metric > self.best_metric
                    if is_best:
                        self.best_metric = metric
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        
                    # Save checkpoint
                    self.save_checkpoint(model, optimizer, epoch, metric, is_best)
                    
                    # Log results
                    epoch_time = time.time() - epoch_start
                    self.logger.info(
                        f"Epoch {epoch}/{self.config.epochs} | "
                        f"Metric: {metric:.4f} | "
                        f"Best: {self.best_metric:.4f} | "
                        f"Time: {epoch_time:.1f}s | "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )
                    
                    # Early stopping
                    if self.patience_counter >= self.config.patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                else:
                    # Save checkpoint without evaluation
                    self.save_checkpoint(model, optimizer, epoch, 0.0, False)
                    
            # Training complete
            total_time = time.time() - self.start_time
            self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
            self.logger.info(f"Best metric: {self.best_metric:.4f}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise


def main():
    """Main entry point"""
    # Configuration - UPDATE THESE PATHS FOR YOUR SETUP
    config = RunPodConfig(
        data_dir="/workspace/dataset",  # UPDATE: Path to your dataset
        output_dir="/workspace/outputs",  # UPDATE: Path for outputs
        epochs=50,
        patience=10,
    )
    
    # Create trainer and run
    trainer = RunPodTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()



