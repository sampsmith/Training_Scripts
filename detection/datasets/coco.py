import os
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch
from pycocotools.coco import COCO


class COCODetectionDataset(torch.utils.data.Dataset):
    """
    Minimal COCO-format dataset for object detection compatible with torchvision's
    Faster R-CNN. Expects Roboflow COCO export structure, e.g.:

    root_dir/
      train/
        _annotations.coco.json
        image1.jpg
        ...
      valid/
        _annotations.coco.json
        ...
      test/ (optional)
        _annotations.coco.json
        ...
    """

    def __init__(self, root_dir: str, split: str, transforms=None) -> None:
        super().__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.transforms = transforms

        self.split_dir = os.path.join(self.root_dir, split)
        ann_file = os.path.join(self.split_dir, "_annotations.coco.json")
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(f"COCO annotations not found: {ann_file}")

        self.coco: COCO = COCO(ann_file)
        self.img_ids: List[int] = self.coco.getImgIds()

        # Map category ids to contiguous labels [1..num_classes]
        self.cat_ids: List[int] = sorted(self.coco.getCatIds())
        self.cat_id_to_contiguous: Dict[int, int] = {
            cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)
        }
        self.contiguous_to_cat_id: Dict[int, int] = {
            v: k for k, v in self.cat_id_to_contiguous.items()
        }

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_image(self, image_info: Dict[str, Any]) -> Image.Image:
        image_path = os.path.join(self.split_dir, image_info["file_name"]) 
        if not os.path.isfile(image_path):
            # Some Roboflow exports include nested directories; try nested path
            nested_path = os.path.join(self.split_dir, os.path.basename(image_info["file_name"]))
            if os.path.isfile(nested_path):
                image_path = nested_path
            else:
                raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return image

    def _convert_annotations(self, ann_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in ann_list:
            if ann.get("iscrowd", 0) == 1:
                # Skip crowd annotations for training
                continue
            x, y, w, h = ann["bbox"]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_contiguous[ann["category_id"]])
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target: Dict[str, Any] = {}
        if len(boxes) == 0:
            # Torchvision expects empty tensors with proper shape when no boxes
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        else:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["area"] = torch.as_tensor(areas, dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        return target

    def __getitem__(self, index: int) -> Tuple[Image.Image, Dict[str, Any]]:
        img_id = self.img_ids[index]
        image_info = self.coco.loadImgs([img_id])[0]
        image = self._load_image(image_info)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_list = self.coco.loadAnns(ann_ids)
        target = self._convert_annotations(ann_list)

        target["image_id"] = torch.tensor([img_id])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @property
    def num_classes(self) -> int:
        # +1 for background class expected by Faster R-CNN
        return len(self.cat_ids) + 1

    def get_coco_api(self) -> COCO:
        return self.coco

    def get_cat_mappings(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        return self.cat_id_to_contiguous, self.contiguous_to_cat_id


