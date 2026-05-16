from __future__ import annotations

import torch
import torch.nn as nn

from kaoanime.models.weights_init import init_weights


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
        enc_channels = [num_filters]
        nf = num_filters
        for _ in range(3):
            nf = min(nf * 2, 512)
            enc_channels.append(nf)

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, enc_channels[0], kernel_size=7, bias=False),
            nn.InstanceNorm2d(enc_channels[0]),
            nn.ReLU(inplace=True),
        ]
        for i in range(3):
            layers += [
                nn.Conv2d(enc_channels[i], enc_channels[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(enc_channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        nf = enc_channels[-1]
        for _ in range(num_residual_blocks):
            layers.append(_ResBlock(nf))
        for i in range(3, 0, -1):
            layers += [
                nn.ConvTranspose2d(enc_channels[i], enc_channels[i - 1], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(enc_channels[i - 1]),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(enc_channels[0], out_channels, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
        init_weights(self, "normal", gain=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _ResBlockD(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, res_ratio: float = 0.1) -> None:
        super().__init__()
        self.res_ratio = res_ratio
        sn = nn.utils.spectral_norm
        self.block = nn.Sequential(
            sn(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)),
        )
        self.shortcut = (
            sn(nn.Conv2d(in_channels, out_channels, 1, bias=False))
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shortcut(x) + self.res_ratio * self.block(x))


class ResNetDiscriminator(nn.Module):
    """ResNet-style global discriminator (WGAN-QC style) with spectral norm."""

    def __init__(self, in_channels: int = 3, num_filters: int = 64) -> None:
        super().__init__()
        nf = num_filters
        ch = [in_channels, nf, nf * 2, nf * 4, min(nf * 8, 512), min(nf * 8, 512)]
        blocks: list[nn.Module] = []
        for i in range(len(ch) - 1):
            blocks.append(_ResBlockD(ch[i], ch[i + 1]))
            if i < len(ch) - 2:
                blocks.append(nn.AvgPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.utils.spectral_norm(nn.Linear(ch[-1], 1))
        init_weights(self, "kaiming")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)
