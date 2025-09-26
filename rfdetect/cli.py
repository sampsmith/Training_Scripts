import argparse
import os
import sys
from typing import Any, Dict, List

import torch
import torchvision
from torch.utils.data import DataLoader

from detection.datasets.factory import build_datasets
from detection.utils.transforms import Compose, ToTensor, RandomHorizontalFlip, collate_fn
from detection.utils.engine import train_one_epoch, evaluate
from detection.utils.coco_eval import coco_evaluate


def build_model(num_classes: int, pretrained_weights: str = "COCO", trainable_backbone_layers: int = 3,
                box_score_thresh: float = 0.05, box_nms_thresh: float = 0.5, box_detections_per_img: int = 300):
    """
    Build Faster R-CNN model with pretrained weights.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained_weights: "COCO" (default, best for detection), "IMAGENET" (backbone only), or "NONE"
        trainable_backbone_layers: Number of backbone layers to fine-tune (0-5)
    """
    if pretrained_weights.upper() == "COCO":
        # Use COCO-pretrained Faster R-CNN (best for detection tasks)
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        backbone_weights_enum = None  # Backbone already pretrained in COCO weights
    elif pretrained_weights.upper() == "IMAGENET":
        # Use ImageNet-pretrained backbone only
        weights = None
        backbone_weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    elif pretrained_weights.upper() == "NONE":
        # No pretrained weights
        weights = None
        backbone_weights_enum = None
    else:
        # Default to COCO
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        backbone_weights_enum = None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=backbone_weights_enum,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Replace classifier head for custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    # Set detection thresholds
    model.roi_heads.score_thresh = box_score_thresh
    model.roi_heads.nms_thresh = box_nms_thresh
    model.roi_heads.detections_per_img = box_detections_per_img
    return model


def add_train_args(p: argparse.ArgumentParser):
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--lr_step_size", type=int, default=8)
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--pretrained_weights", type=str, default="COCO", choices=["COCO", "IMAGENET", "NONE"]) 
    p.add_argument("--trainable_backbone_layers", type=int, default=3, choices=[0,1,2,3,4,5])
    p.add_argument("--box_score_thresh", type=float, default=0.05)
    p.add_argument("--box_nms_thresh", type=float, default=0.5)
    p.add_argument("--box_detections_per_img", type=int, default=300)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--flip_prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_interval", type=int, default=1)


def add_eval_args(p: argparse.ArgumentParser):
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


def add_predict_args(p: argparse.ArgumentParser):
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--score_thresh", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


def cmd_train(args: argparse.Namespace):
    import numpy as np
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    train_transforms = Compose([ToTensor(), RandomHorizontalFlip(args.flip_prob)])
    valid_transforms = Compose([ToTensor()])

    train_dataset, valid_dataset, eval_type = build_datasets(args.data_dir, train_transforms, valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = build_model(
        num_classes=train_dataset.num_classes,
        pretrained_weights=args.pretrained_weights,
        trainable_backbone_layers=args.trainable_backbone_layers,
        box_score_thresh=args.box_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        box_detections_per_img=args.box_detections_per_img,
    )

    device = torch.device(args.device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])  # type: ignore[arg-type]
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])  # type: ignore[arg-type]
        if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])  # type: ignore[arg-type]
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    best_ap = -1.0
    for epoch in range(start_epoch, args.epochs + 1):
        loss_stats = train_one_epoch(
            model, optimizer, train_loader, device, epoch, scaler,
            max_norm=args.grad_clip, gradient_accumulation_steps=args.accumulation_steps
        )

        lr_scheduler.step()

        if (epoch % args.save_interval) == 0:
            path = os.path.join(args.output_dir, f"model_epoch_{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "args": vars(args),
            }, path)

        if (epoch % args.eval_interval) == 0:
            outputs, image_ids = evaluate(model, valid_loader, device)
            if eval_type == "coco":
                _, cont_to_cat = train_dataset.get_cat_mappings()
                metrics, _ = coco_evaluate(valid_dataset.get_coco_api(), outputs, image_ids, cont_to_cat)
                ap = metrics.get("AP", 0.0)
            else:
                ap = float(torch.mean(torch.cat([o["scores"] for o in outputs])[:100]).item()) if len(outputs) else 0.0
            if ap > best_ap:
                best_ap = ap
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_ap": best_ap,
                    "args": vars(args),
                }, os.path.join(args.output_dir, "model_best.pth"))

        lr_current = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} | LR {lr_current:.6f} | " + ", ".join([f"{k}:{v:.4f}" for k, v in loss_stats.items()]))


def cmd_eval(args: argparse.Namespace):
    device = torch.device(args.device)

    train_dataset, valid_dataset, eval_type = build_datasets(args.data_dir, Compose([ToTensor()]), Compose([ToTensor()]))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = build_model(num_classes=valid_dataset.num_classes)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"]) 
    model.to(device)
    model.eval()

    outputs, image_ids = evaluate(model, valid_loader, device)
    if eval_type == "coco":
        _, cont_to_cat = valid_dataset.get_cat_mappings()
        metrics, _ = coco_evaluate(valid_dataset.get_coco_api(), outputs, image_ids, cont_to_cat)
        print(metrics)
    else:
        ap = float(torch.mean(torch.cat([o["scores"] for o in outputs])[:100]).item()) if len(outputs) else 0.0
        print({"score_mean_top100": ap})


def cmd_predict(args: argparse.Namespace):
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(args.output_dir, exist_ok=True)

    dummy_dataset = COCODetectionDataset(os.path.dirname(args.images_dir), split="train", transforms=Compose([ToTensor()]))
    # We only need category mapping and num_classes; model weights come from checkpoint.
    model = build_model(num_classes=dummy_dataset.num_classes)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"]) 
    model.to(torch.device(args.device))
    model.eval()

    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for name in os.listdir(args.images_dir):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(args.images_dir, name)
        image = Image.open(path).convert("RGB")
        tensor = ToTensor()(image, {})[0].unsqueeze(0).to(args.device)
        with torch.no_grad():
            pred = model(tensor)[0]

        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        keep = scores >= args.score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        draw = ImageDraw.Draw(image)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            txt = f"{int(label.item())}:{score:.2f}"
            if font is not None:
                draw.text((x1, max(0, y1 - 10)), txt, fill="red", font=font)
            else:
                draw.text((x1, max(0, y1 - 10)), txt, fill="red")

        out_path = os.path.join(args.output_dir, name)
        image.save(out_path)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(prog="rfdetect", description="Roboflow COCO Faster R-CNN (ResNet-50 FPN) CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train model on dataset")
    add_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("eval", help="Evaluate checkpoint on validation set")
    add_eval_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_pred = subparsers.add_parser("predict", help="Run inference on a folder of images and save visualisations")
    add_predict_args(p_pred)
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


