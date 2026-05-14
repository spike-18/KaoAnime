from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlockNOT(nn.Module):
    """Residual block without BN (reference ResNetBlock, bn=False).

    res_ratio=0.1 scales the residual branch to stabilise early training.
    """

    def __init__(self, fin: int, fout: int, res_ratio: float = 0.1) -> None:
        super().__init__()
        self.res_ratio = res_ratio
        self.learned_shortcut = fin != fout
        fhidden = min(fin, fout)
        self.conv_0 = nn.Conv2d(fin, fhidden, 3, padding=1, bias=True)
        self.conv_1 = nn.Conv2d(fhidden, fout, 3, padding=1, bias=True)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, 1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.relu(self.conv_0(x))
        dx = self.conv_1(dx)
        return self.relu(x_s + self.res_ratio * dx)


class NOTPotential(nn.Module):
    """Kantorovich potential f for Neural Optimal Transport.

    ResNet_D architecture (no BN, no spectral norm) from Korotin et al. ICLR 2023.
    Outputs raw scalar per image (B, 1) — no final activation.

    Architecture: Conv+ReLU → ResBlock×2 → [AvgPool → ResBlock×2] × 4 → GlobalAvgPool → Linear
    Channel sequence (num_filters=64): 3→64→64→64→128→256→512→512
    """

    def __init__(self, in_channels: int = 3, num_filters: int = 64) -> None:
        super().__init__()
        nf = num_filters
        ch = [nf, nf, nf * 2, nf * 4, min(nf * 8, 512), min(nf * 8, 512)]

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        blocks: list[nn.Module] = [
            _ResBlockNOT(ch[0], ch[0]),
            _ResBlockNOT(ch[0], ch[1]),
        ]
        for i in range(1, len(ch) - 1):
            blocks.append(nn.AvgPool2d(3, stride=2, padding=1))
            blocks.append(_ResBlockNOT(ch[i], ch[i]))
            blocks.append(_ResBlockNOT(ch[i], ch[i + 1]))
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Linear(ch[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.features(x)
        x = x.mean(dim=[2, 3])      # global avg pool → (B, ch[-1])
        return self.classifier(x)   # (B, 1), no activation
