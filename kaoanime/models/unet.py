from __future__ import annotations

import torch
import torch.nn as nn


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize
            )
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_filters: int = 64,
        num_down: int = 7,
    ) -> None:
        super().__init__()
        enc_chs = [min(num_filters * (2**i), 512) for i in range(num_down)]

        self.enc = nn.ModuleList()
        for i in range(num_down):
            # No BN on first block (standard) or bottleneck (1×1 spatial fails BatchNorm)
            normalize = i not in (0, num_down - 1)
            in_ch = in_channels if i == 0 else enc_chs[i - 1]
            self.enc.append(_DownBlock(in_ch, enc_chs[i], normalize=normalize))

        self.dec = nn.ModuleList()
        for i in range(num_down - 1):
            out_ch = enc_chs[num_down - 2 - i]
            in_ch = enc_chs[-1] if i == 0 else 2 * enc_chs[num_down - 1 - i]
            self.dec.append(_UpBlock(in_ch, out_ch, dropout=(i < 3)))

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(
                2 * enc_chs[0], out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_feats: list[torch.Tensor] = []
        feat = x
        for block in self.enc:
            feat = block(feat)
            enc_feats.append(feat)

        out = enc_feats[-1]
        for i, block in enumerate(self.dec):
            out = block(out)
            out = torch.cat([out, enc_feats[len(self.enc) - 2 - i]], dim=1)

        return self.out_conv(out)
