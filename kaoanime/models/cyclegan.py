from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_filters: int = 64,
        num_residual_blocks: int = 9,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, num_filters, kernel_size=7, bias=False),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True),
        ]

        # Downsample ×2
        nf = num_filters
        for _ in range(2):
            nf_next = min(nf * 2, 256)
            layers += [
                nn.Conv2d(nf, nf_next, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(nf_next),
                nn.ReLU(inplace=True),
            ]
            nf = nf_next

        # Residual blocks
        for _ in range(num_residual_blocks):
            layers.append(_ResBlock(nf))

        # Upsample ×2
        for _ in range(2):
            nf_prev = nf // 2
            layers += [
                nn.ConvTranspose2d(
                    nf,
                    nf_prev,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(nf_prev),
                nn.ReLU(inplace=True),
            ]
            nf = nf_prev

        # Output
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator."""

    def __init__(self, in_channels: int = 3, num_filters: int = 64) -> None:
        super().__init__()
        nf = num_filters
        self.model = nn.Sequential(
            # Layer 0 — no norm
            nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 1
            nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3 — stride 1
            nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output — stride 1
            nn.Conv2d(nf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
