import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch


class VOCDataset(torch.utils.data.Dataset):
    """
    Simple Pascal VOC (Roboflow VOC export) dataset for torchvision Faster R-CNN.
    Expects structure:

    root_dir/
      train/
        image1.jpg
        image1.xml
        ...
      valid/
        ...
      test/ (optional)
        ...
    """

    def __init__(self, root_dir: str, split: str, transforms=None) -> None:
        super().__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.transforms = transforms
        self.split_dir = os.path.join(self.root_dir, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Build index from files with matching .jpg/.png and .xml
        files = sorted(os.listdir(self.split_dir))
        stem_to_img = {}
        stem_to_xml = {}
        for f in files:
            path = os.path.join(self.split_dir, f)
            stem, ext = os.path.splitext(f)
            ext_low = ext.lower()
            if ext_low in [".jpg", ".jpeg", ".png"]:
                stem_to_img[stem] = path
            elif ext_low == ".xml":
                stem_to_xml[stem] = path

        self.items: List[Tuple[str, str]] = []
        for stem, img_path in stem_to_img.items():
            if stem in stem_to_xml:
                self.items.append((img_path, stem_to_xml[stem]))

        if len(self.items) == 0:
            raise RuntimeError(f"No image-xml pairs found in {self.split_dir}")

        # Collect categories across split to build label mapping
        self.class_to_label: Dict[str, int] = {}
        for _, xml_path in self.items:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text  # type: ignore[assignment]
                if name not in self.class_to_label:
                    self.class_to_label[name] = len(self.class_to_label) + 1  # start at 1; 0 is background

        self.label_to_class: Dict[int, str] = {v: k for k, v in self.class_to_label.items()}

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        return image

    def _parse_xml(self, xml_path: str) -> Dict[str, Any]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes: List[List[float]] = []
        labels: List[int] = []
        iscrowd: List[int] = []
        areas: List[float] = []

        for obj in root.findall("object"):
            name_el = obj.find("name")
            bnd = obj.find("bndbox")
            if name_el is None or bnd is None:
                continue
            cls_name = name_el.text
            if cls_name is None:
                continue
            xmin = float(bnd.findtext("xmin", default="0"))
            ymin = float(bnd.findtext("ymin", default="0"))
            xmax = float(bnd.findtext("xmax", default="0"))
            ymax = float(bnd.findtext("ymax", default="0"))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_label[cls_name])
            iscrowd.append(0)
            areas.append((xmax - xmin) * (ymax - ymin))

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
                "area": torch.as_tensor(areas, dtype=torch.float32),
            }
        return target

    def __getitem__(self, index: int):
        image_path, xml_path = self.items[index]
        image = self._load_image(image_path)
        target = self._parse_xml(xml_path)
        # For consistency with COCO, set an image_id; here we use index
        target["image_id"] = torch.tensor([index])
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    @property
    def num_classes(self) -> int:
        return len(self.class_to_label) + 1


