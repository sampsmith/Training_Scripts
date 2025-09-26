import os
from typing import Tuple

from .coco import COCODetectionDataset
from .voc import VOCDataset


def detect_format(root_dir: str) -> str:
    # Heuristics: COCO has _annotations.coco.json under split dirs; VOC has .xml pairs
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue
        coco_ann = os.path.join(split_dir, "_annotations.coco.json")
        if os.path.isfile(coco_ann):
            return "coco"
        # Look for any .xml files
        for name in os.listdir(split_dir):
            if name.lower().endswith(".xml"):
                return "voc"
    # default to VOC if xml present anywhere
    return "voc"


def build_datasets(root_dir: str, transforms_train, transforms_valid):
    fmt = detect_format(root_dir)
    if fmt == "coco":
        train_ds = COCODetectionDataset(root_dir, split="train", transforms=transforms_train)
        valid_ds = COCODetectionDataset(root_dir, split="valid", transforms=transforms_valid)
        eval_type = "coco"
    else:
        train_ds = VOCDataset(root_dir, split="train", transforms=transforms_train)
        valid_ds = VOCDataset(root_dir, split="valid", transforms=transforms_valid)
        eval_type = "voc"
    return train_ds, valid_ds, eval_type


