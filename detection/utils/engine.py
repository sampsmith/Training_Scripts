from typing import Any, Dict, Iterable, List, Tuple

import math
import time

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def reduce_dict(input_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    # No DDP: simply detach to float
    return {k: float(v.detach().item()) for k, v in input_dict.items()}


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    max_norm: float = 0.0,
    gradient_accumulation_steps: int = 1,
    progress_callback=None,
) -> Dict[str, float]:
    model.train()

    header = f"Epoch [{epoch}]"
    loss_dict_avg: Dict[str, float] = {}

    optimizer.zero_grad(set_to_none=True)
    total_batches = len(data_loader)
    
    for step, (images, targets) in enumerate(tqdm(data_loader, desc=header)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.detach()

        # backward
        if scaler is not None:
            scaler.scale(losses / gradient_accumulation_steps).backward()
        else:
            (losses / gradient_accumulation_steps).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if max_norm and max_norm > 0.0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # aggregate losses for logging
        reduced_losses = reduce_dict(loss_dict)
        for k, v in reduced_losses.items():
            loss_dict_avg[k] = loss_dict_avg.get(k, 0.0) + v

        # Progress callback
        if progress_callback is not None:
            progress_callback(step + 1, total_batches, reduced_losses)

    # average over steps
    num_steps = max(1, step + 1)
    for k in list(loss_dict_avg.keys()):
        loss_dict_avg[k] /= num_steps

    return loss_dict_avg


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: Iterable, device: torch.device) -> List[Dict[str, Any]]:
    model.eval()
    outputs_all: List[Dict[str, Any]] = []
    image_ids: List[int] = []

    for images, targets in tqdm(data_loader, desc="Eval"):
        images = [img.to(device) for img in images]
        outputs = model(images)
        outputs = [{k: v.to("cpu") for k, v in o.items()} for o in outputs]
        outputs_all.extend(outputs)
        image_ids.extend([int(t["image_id"].item()) for t in targets])

    return outputs_all, image_ids


