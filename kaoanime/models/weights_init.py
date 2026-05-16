from __future__ import annotations

import torch.nn as nn
from torch.nn import init


def init_weights(module: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    """Apply weight initialization to *module* and all its children in-place.

    init_type:
        "normal"  — Normal(0, gain). CycleGAN paper standard for generators
                    and discriminators that use instance/batch norm.
        "kaiming" — Kaiming-normal, a=0.2, fan_in. `gain` is ignored.
                    For LeakyReLU(0.2) networks without normalisation layers.

    Handles nn.utils.spectral_norm wrappers by targeting weight_orig
    instead of the normalised weight property.
    """

    def _init_func(m: nn.Module) -> None:
        if not isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            return
        weight_attr = "weight_orig" if hasattr(m, "weight_orig") else "weight"
        w = getattr(m, weight_attr)
        if init_type == "normal":
            init.normal_(w.data, 0.0, gain)
        elif init_type == "kaiming":
            init.kaiming_normal_(w.data, a=0.2, mode="fan_in")
        else:
            raise ValueError(f"Unknown init_type: {init_type!r}. Use 'normal' or 'kaiming'.")
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    module.apply(_init_func)
