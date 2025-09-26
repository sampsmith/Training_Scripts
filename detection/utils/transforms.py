from typing import Any, Callable, Dict, List, Tuple

import random
from PIL import Image
import torch
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: Dict[str, Any]):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image: Image.Image, target: Dict[str, Any]):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, flip_prob: float = 0.5) -> None:
        self.flip_prob = flip_prob

    def __call__(self, image: torch.Tensor, target: Dict[str, Any]):
        if random.random() < self.flip_prob:
            if isinstance(image, torch.Tensor):
                _, height, width = image.shape
            else:
                width, height = image.size
            image = F.hflip(image)

            boxes = target.get("boxes")
            if boxes is not None and boxes.numel() > 0:
                x_min = boxes[:, 0]
                y_min = boxes[:, 1]
                x_max = boxes[:, 2]
                y_max = boxes[:, 3]
                new_x_min = width - x_max
                new_x_max = width - x_min
                boxes = torch.stack([new_x_min, y_min, new_x_max, y_max], dim=1)
                target["boxes"] = boxes
        return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


