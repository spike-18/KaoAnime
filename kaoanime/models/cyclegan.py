from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Build encoder channel sequence so the decoder can mirror it exactly.
        enc_channels = [num_filters]
        nf = num_filters
        for _ in range(3):
            nf = min(nf * 2, 512)
            enc_channels.append(nf)
        # enc_channels: [num_filters, stage1_out, stage2_out, stage3_out]

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, enc_channels[0], kernel_size=7, bias=False),
            nn.InstanceNorm2d(enc_channels[0]),
            nn.ReLU(inplace=True),
        ]

        # Downsample ×3
        for i in range(3):
            layers += [
                nn.Conv2d(
                    enc_channels[i],
                    enc_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(enc_channels[i + 1]),
                nn.ReLU(inplace=True),
            ]

        nf = enc_channels[-1]

        # Residual blocks
        for _ in range(num_residual_blocks):
            layers.append(_ResBlock(nf))

        # Upsample ×3 — mirror encoder channel sequence in reverse
        for i in range(3, 0, -1):
            layers += [
                nn.ConvTranspose2d(
                    enc_channels[i],
                    enc_channels[i - 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(enc_channels[i - 1]),
                nn.ReLU(inplace=True),
            ]

        nf = enc_channels[0]

        # Output
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None) -> None:
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_filters: int = 64,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        f = num_filters
        factor = 2 if bilinear else 1
        self.inc   = _DoubleConv(in_channels, f)
        self.down1 = _Down(f,     2 * f)
        self.down2 = _Down(2 * f, 4 * f)
        self.down3 = _Down(4 * f, 8 * f)
        self.down4 = _Down(8 * f, 16 * f // factor)
        self.up1   = _Up(16 * f,  8 * f // factor, bilinear)
        self.up2   = _Up(8 * f,   4 * f // factor, bilinear)
        self.up3   = _Up(4 * f,   2 * f // factor, bilinear)
        self.up4   = _Up(2 * f,   f,               bilinear)
        self.outc  = nn.Sequential(nn.Conv2d(f, out_channels, kernel_size=1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator."""

    def __init__(self, in_channels: int = 3, num_filters: int = 64) -> None:
        super().__init__()
        nf = num_filters
        self.model = nn.Sequential(
            # Layer 0 — no norm
            nn.utils.spectral_norm(nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 1
            nn.utils.spectral_norm(nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.utils.spectral_norm(nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3 — stride 1
            nn.utils.spectral_norm(nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4 — extra stride-1 conv
            nn.utils.spectral_norm(nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # Output — stride 1
            nn.utils.spectral_norm(nn.Conv2d(nf * 8, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _ResBlockD(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        sn = nn.utils.spectral_norm
        self.block = nn.Sequential(
            sn(nn.Conv2d(in_channels,  out_channels, 3, padding=1, bias=False)),
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
        return self.act(self.shortcut(x) + self.block(x))


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
        self.features   = nn.Sequential(*blocks)
        self.classifier = nn.utils.spectral_norm(nn.Linear(ch[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)