from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator with spectral norm."""

    def __init__(self, in_channels: int = 3, num_filters: int = 64) -> None:
        super().__init__()
        nf = num_filters
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(nf * 8, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
