import argparse
import datetime
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from detection.datasets.factory import build_datasets
from detection.utils.transforms import Compose, ToTensor, RandomHorizontalFlip, collate_fn
from detection.utils.engine import train_one_epoch, evaluate
from detection.utils.coco_eval import coco_evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, backbone_weights: str = "IMAGENET", trainable_backbone_layers: int = 3):
    # torchvision 0.17 API: weights/backbone_weights
    if backbone_weights.upper() == "IMAGENET":
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        backbone_weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    elif backbone_weights.upper() == "NONE":
        weights = None
        backbone_weights_enum = None
    else:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        backbone_weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=backbone_weights_enum,
        trainable_backbone_layers=trainable_backbone_layers,
        box_detections_per_img=300,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN ResNet50 on COCO-format (Roboflow) dataset")
    # Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Roboflow COCO dataset root")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save checkpoints and logs")

    # Optimiser and scheduler
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lr_step_size", type=int, default=8)
    parser.add_argument("--lr_gamma", type=float, default=0.1)

    # Model/backbone
    parser.add_argument("--backbone_weights", type=str, default="IMAGENET", choices=["IMAGENET", "NONE"])
    parser.add_argument("--trainable_backbone_layers", type=int, default=3, choices=[0,1,2,3,4,5])
    parser.add_argument("--box_score_thresh", type=float, default=0.05)
    parser.add_argument("--box_nms_thresh", type=float, default=0.5)
    parser.add_argument("--box_detections_per_img", type=int, default=300)

    # Data
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--flip_prob", type=float, default=0.5)

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--accumulation_steps", type=int, default=1)

    # Eval
    parser.add_argument("--eval_interval", type=int, default=1)

    # Checkpointing
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=1)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Datasets and loaders
    train_transforms = Compose([ToTensor(), RandomHorizontalFlip(args.flip_prob)])
    valid_transforms = Compose([ToTensor()])

    train_dataset, valid_dataset, eval_type = build_datasets(args.data_dir, train_transforms, valid_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    num_classes = train_dataset.num_classes
    model = build_model(
        num_classes=num_classes,
        backbone_weights=args.backbone_weights,
        trainable_backbone_layers=args.trainable_backbone_layers,
    )
    model.roi_heads.score_thresh = args.box_score_thresh
    model.roi_heads.nms_thresh = args.box_nms_thresh
    model.roi_heads.detections_per_img = args.box_detections_per_img

    device = torch.device(args.device)
    model.to(device)

    # Optimiser and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Resume
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"]) if "optimizer" in checkpoint else None
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"]) if "lr_scheduler" in checkpoint else None
        if "scaler" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])  # type: ignore[arg-type]
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    # Training loop
    best_ap = -1.0
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        loss_stats = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            scaler,
            max_norm=args.grad_clip,
            gradient_accumulation_steps=args.accumulation_steps,
        )

        # Step LR
        lr_scheduler.step()

        # Save checkpoint
        if (epoch % args.save_interval) == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"model_epoch_{epoch:03d}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "args": vars(args),
                },
                ckpt_path,
            )

        # Evaluate
        if (epoch % args.eval_interval) == 0:
            outputs, image_ids = evaluate(model, valid_loader, device)
            if eval_type == "coco":
                _, cont_to_cat = train_dataset.get_cat_mappings()
                metrics, _ = coco_evaluate(
                    valid_dataset.get_coco_api(), outputs, image_ids, cont_to_cat
                )
                ap = metrics.get("AP", 0.0)
            else:
                # Simple metric: mean score above threshold when VOC
                ap = float(torch.mean(torch.cat([o["scores"] for o in outputs])[:100]).item()) if len(outputs) else 0.0
            is_best = ap > best_ap
            best_ap = max(best_ap, ap)

            if is_best:
                best_path = os.path.join(args.output_dir, "model_best.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "best_ap": best_ap,
                        "args": vars(args),
                    },
                    best_path,
                )

        # Simple log
        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs} | LR {lr_current:.6f} | Losses: "
            + ", ".join([f"{k}:{v:.4f}" for k, v in loss_stats.items()])
            + (f" | Val AP: {best_ap:.4f}" if best_ap >= 0 else "")
        )

    print("Training complete.")


if __name__ == "__main__":
    main()


