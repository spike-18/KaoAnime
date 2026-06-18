from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kaoanime.models.weights_init import init_weights


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None) -> None:
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
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
        self.inc = _DoubleConv(in_channels, f)
        self.down1 = _Down(f, 2 * f)
        self.down2 = _Down(2 * f, 4 * f)
        self.down3 = _Down(4 * f, 8 * f)
        self.down4 = _Down(8 * f, 16 * f // factor)
        self.up1 = _Up(16 * f, 8 * f // factor, bilinear)
        self.up2 = _Up(8 * f, 4 * f // factor, bilinear)
        self.up3 = _Up(4 * f, 2 * f // factor, bilinear)
        self.up4 = _Up(2 * f, f, bilinear)
        self.outc = nn.Sequential(nn.Conv2d(f, out_channels, kernel_size=1), nn.Tanh())
        init_weights(self, "normal", gain=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
