from __future__ import annotations

from typing import Iterable

import torch


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of parameters; update with decay close to 1.0.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.ema = self._clone_model(model)
        self.ema.eval()
        self.decay = decay

    @torch.no_grad()
    def _clone_model(self, model: torch.nn.Module) -> torch.nn.Module:
        clone = type(model)  # type: ignore[assignment]
        ema_model = model.__class__(**{})  # may not support constructor args; fallback below
        # Safer: deep-copy state dict onto a copy of model
        ema_model = model.__class__.__new__(model.__class__)  # type: ignore[misc]
        ema_model.__dict__ = model.__dict__.copy()
        for p in ema_model.parameters():
            p.detach_()
        ema_model.load_state_dict(model.state_dict(), strict=True)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        return ema_model

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:
                continue
            v.copy_(v * d + msd[k] * (1.0 - d))

    def state_dict(self):
        return self.ema.state_dict()


