# Neural Optimal Transport (NOT) Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the codebase so CycleGAN and NOT live as parallel, clean implementations sharing building blocks (resnet.py / unet.py), then add the NOT model.

**Architecture:** Split `models/cyclegan.py` by building-block type (resnet.py, unet.py, patch_discriminator.py). Split `config.py` into `config_cyclegan.py` + `config_not.py` + thin unified `config.py`. Rename `model.py` → `model_cyclegan.py`, add `model_not.py` alongside. One unified `train.py` switches on `cfg.model_type`.

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra, MLflow, uv. Always invoke Python as `uv run python`.

---

## Algorithm: NOT (strong OT)

```
# Per step:
# T inner loop (t_iters=10 times):
T_loss = MSE(X, T(X))  −  f(T(X)).mean()
T_opt.step()

# f update (once):
f_loss = f(T(X)).mean()  −  f(Y).mean()
f_opt.step()
```

Key: MSE cost makes transport "efficient" (penalises large pixel changes). No cycle-consistency needed.

---

## File Map

| Before                        | After                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| `kaoanime/models/cyclegan.py` | **deleted** — split into 3 files below                                                            |
| _(new)_                       | `kaoanime/models/resnet.py` — `_ResBlock`, `ResNetGenerator`, `_ResBlockD`, `ResNetDiscriminator` |
| _(new)_                       | `kaoanime/models/unet.py` — `_DoubleConv`, `_Down`, `_Up`, `UNetGenerator`                        |
| _(new)_                       | `kaoanime/models/patch_discriminator.py` — `PatchDiscriminator`                                   |
| _(new)_                       | `kaoanime/models/not_potential.py` — `_ResBlockNOT`, `NOTPotential`                               |
| `kaoanime/models/__init__.py` | Updated re-exports                                                                                |
| `kaoanime/config.py`          | Content split — CycleGAN-specific moved out; DataConfig/EvalConfig/unified `Config` stay          |
| _(new)_                       | `kaoanime/config_cyclegan.py` — `TrainConfig`, `ModelConfig`                                      |
| _(new)_                       | `kaoanime/config_not.py` — `NOTConfig`                                                            |
| `kaoanime/model.py`           | **renamed** → `kaoanime/model_cyclegan.py`                                                        |
| _(new)_                       | `kaoanime/model_not.py` — `NOTModel`                                                              |
| `train.py`                    | Add `model_type` switch; update imports                                                           |
| `eval.py`                     | Update import `kaoanime.model` → `kaoanime.model_cyclegan`                                        |
| `infer.py`                    | Update import `kaoanime.model` → `kaoanime.model_cyclegan`                                        |
| `tests/test_model.py`         | Update import                                                                                     |
| `tests/test_inference.py`     | Update import                                                                                     |

---

## Task 1: Commit and push current changes

- [ ] **Step 1.1: Stage pending changes**

```bash
git add kaoanime/config.py kaoanime/utils/align.py kaoanime/utils/transforms.py
git add notebooks/03_alignment_tuning.ipynb
git add notebooks/04_data_pipeline.ipynb notebooks/05_celeba_pipeline.ipynb
git add pyproject.toml uv.lock
git add docs/superpowers/plans/
```

- [ ] **Step 1.2: Commit and push**

```bash
git commit -m "$(cat <<'EOF'
feat: adopt reference Resize preprocessing, re-tune alignment, add pipeline notebooks

- CelebA: direct Resize((128,128)) instead of Resize(158)+RandomCrop(128)
- _CELEBA_REF_128 re-measured for new preprocessing
- Anime crop tuned: scale=1.20, shift_y=-0.02 (resolution-independent fractions)
- Add notebooks/04 (data pipeline) and notebooks/05 (CelebA processing comparison)
- UNetGenerator + ResNetDiscriminator set as model defaults in config
- Test splits: 1000 held-out images per domain in separate directories

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 2: Create feature branch

- [ ] **Step 2.1:**

```bash
git checkout -b feat/not-model
```

---

## Task 3: Reorganize kaoanime/models/ — split cyclegan.py by building-block type

**Files:**

- Create: `kaoanime/models/resnet.py`
- Create: `kaoanime/models/unet.py`
- Create: `kaoanime/models/patch_discriminator.py`
- Delete: `kaoanime/models/cyclegan.py`
- Modify: `kaoanime/models/__init__.py`

- [ ] **Step 3.1: Create `kaoanime/models/resnet.py`**

```python
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)
```

- [ ] **Step 3.2: Create `kaoanime/models/unet.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
```

- [ ] **Step 3.3: Create `kaoanime/models/patch_discriminator.py`**

```python
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
```

- [ ] **Step 3.4: Update `kaoanime/models/__init__.py`**

```python
from .patch_discriminator import PatchDiscriminator
from .resnet import ResNetDiscriminator, ResNetGenerator
from .unet import UNetGenerator
from .not_potential import NOTPotential

__all__ = [
    "PatchDiscriminator",
    "ResNetDiscriminator",
    "ResNetGenerator",
    "UNetGenerator",
    "NOTPotential",
]
```

- [ ] **Step 3.5: Delete the old `kaoanime/models/cyclegan.py`**

```bash
git rm kaoanime/models/cyclegan.py
```

- [ ] **Step 3.6: Verify all models still import**

```bash
uv run python -c "
from kaoanime.models import PatchDiscriminator, ResNetGenerator, ResNetDiscriminator, UNetGenerator
import torch
x = torch.randn(2, 3, 128, 128)
assert ResNetGenerator()(x).shape == (2, 3, 128, 128)
assert ResNetDiscriminator()(x).shape == (2, 1)
assert UNetGenerator()(x).shape == (2, 3, 128, 128)
print('models/__init__.py OK — all shapes correct')
"
```

Expected: `models/__init__.py OK — all shapes correct`

---

## Task 4: Add NOTPotential

**Files:**

- Create: `kaoanime/models/not_potential.py`

`NOTPotential` = reference ResNet_D adapted for our codebase. No BN, no spectral norm, `res_ratio=0.1` weighted residual. Global avg pool → Linear(1) for scalar output.

- [ ] **Step 4.1: Create `kaoanime/models/not_potential.py`**

```python
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
```

- [ ] **Step 4.2: Verify NOTPotential shape and no activation clipping**

```bash
uv run python -c "
from kaoanime.models import NOTPotential
import torch

f = NOTPotential(num_filters=64)
x = torch.randn(2, 3, 128, 128)
out = f(x)
assert out.shape == (2, 1), f'Wrong shape: {out.shape}'
params = sum(p.numel() for p in f.parameters())
print(f'NOTPotential OK — shape {out.shape}, params {params:,}')
"
```

Expected: `NOTPotential OK — shape torch.Size([2, 1]), params ~5M-10M`

---

## Task 5: Restructure config

**Files:**

- Create: `kaoanime/config_cyclegan.py` (TrainConfig + ModelConfig, moved from current config.py)
- Create: `kaoanime/config_not.py` (NOTConfig, new)
- Modify: `kaoanime/config.py` (keep DataConfig + EvalConfig; add `model_type`; import from the two new files; rewrite `Config` + `register_configs`)

The split: CycleGAN-specific hyperparameters (training schedule, generator architecture) go to `config_cyclegan.py`. NOT-specific go to `config_not.py`. Shared data/eval config stays in `config.py`. All existing import sites (`from kaoanime.config import Config, register_configs`) continue to work unchanged.

- [ ] **Step 5.1: Create `kaoanime/config_cyclegan.py`**

```python
# kaoanime/config_cyclegan.py
from dataclasses import dataclass


@dataclass
class TrainConfig:
    max_epochs: int = 100
    lr: float = 2e-4
    lr_decay_start_epoch: int = 30
    beta1: float = 0.5
    precision: str = "16-mixed"
    log_every_n_steps: int = 50
    mlflow_tracking_uri: str = "http://10.0.111.233:9999"
    gen_steps: int = 1
    disc_steps: int = 1
    log_image_every_n_steps: int = 5000
    fid_every_n_steps: int = 2000
    fid_num_images: int = 512


@dataclass
class ModelConfig:
    num_filters: int = 64
    num_residual_blocks: int = 9
    generator: str = "unet"       # "resnet" or "unet"
    discriminator: str = "resnet"  # "patch" or "resnet"
    lambda_cycle: float = 10.0
    lambda_identity: float = 1.0
```

- [ ] **Step 5.2: Create `kaoanime/config_not.py`**

```python
# kaoanime/config_not.py
from dataclasses import dataclass


@dataclass
class NOTConfig:
    t_iters: int = 10              # T inner-loop updates per f update (matches reference)
    t_lr: float = 1e-4             # transport map (UNet) learning rate
    f_lr: float = 1e-4             # potential (NOTPotential) learning rate
    t_filters: int = 48            # UNet num_filters (reference: base_factor=48)
    f_filters: int = 64            # NOTPotential num_filters
    max_steps: int = 100_001
    precision: str = "16-mixed"
    log_every_n_steps: int = 50
    mlflow_tracking_uri: str = "http://10.0.111.233:9999"
    log_image_every_n_steps: int = 1000
    fid_every_n_steps: int = 5000
    fid_num_images: int = 512
```

- [ ] **Step 5.3: Rewrite `kaoanime/config.py`**

Replace the entire file with:

```python
# kaoanime/config.py — unified config entry point
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from kaoanime.config_cyclegan import ModelConfig, TrainConfig
from kaoanime.config_not import NOTConfig


@dataclass
class DataConfig:
    root_a: str = "/beta/home/madorskii/datasets/CelebA/img_align_celeba/img_align_celeba"
    root_b: str = "/beta/home/madorskii/datasets/alignedanimefaces/safebooru_jpeg"
    test_a: str = "/beta/home/madorskii/datasets/CelebA/test"
    test_b: str = "/beta/home/madorskii/datasets/alignedanimefaces/test"
    extra_roots_a: list[str] = field(default_factory=list)
    extra_roots_b: list[str] = field(default_factory=list)
    batch_size: int = 8
    image_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    align_a: bool = False
    anime_scale  : float = 1.20
    anime_shift_x: float = 0.00
    anime_shift_y: float = -0.02


@dataclass
class EvalConfig:
    checkpoint: str = ""
    output_dir: str = "outputs/eval"
    input: str = ""
    direction: str = "a2b"
    align: bool = False


@dataclass
class Config:
    model_type: str = "cyclegan"   # "cyclegan" or "not" — selects which model train.py runs
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # CycleGAN-specific (ignored when model_type="not")
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # NOT-specific (ignored when model_type="cyclegan")
    not_: NOTConfig = field(default_factory=NOTConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
```

- [ ] **Step 5.4: Verify config imports and defaults**

```bash
uv run python -c "
from kaoanime.config import Config, DataConfig, EvalConfig, register_configs
from kaoanime.config_cyclegan import TrainConfig, ModelConfig
from kaoanime.config_not import NOTConfig

cfg = Config()
assert cfg.model_type == 'cyclegan'
assert cfg.data.image_size == 128
assert cfg.not_.t_iters == 10
assert cfg.train.max_epochs == 100
print('Config OK')
print(f'  model_type={cfg.model_type}')
print(f'  data.batch_size={cfg.data.batch_size}')
print(f'  train.lr={cfg.train.lr}')
print(f'  not_.t_iters={cfg.not_.t_iters}')
"
```

Expected:

```
Config OK
  model_type=cyclegan
  data.batch_size=8
  train.lr=0.0002
  not_.t_iters=10
```

---

## Task 6: Rename model.py → model_cyclegan.py; update all import sites

**Files:**

- Rename: `kaoanime/model.py` → `kaoanime/model_cyclegan.py`
- Modify: `train.py`, `eval.py`, `infer.py`, `tests/test_model.py`, `tests/test_inference.py`

- [ ] **Step 6.1: Rename the file**

```bash
git mv kaoanime/model.py kaoanime/model_cyclegan.py
```

- [ ] **Step 6.2: Update `eval.py` — change import**

In `eval.py`, replace:

```python
from kaoanime.model import KaoAnimeModel
```

with:

```python
from kaoanime.model_cyclegan import KaoAnimeModel
```

- [ ] **Step 6.3: Update `infer.py` — change import**

In `infer.py`, replace:

```python
from kaoanime.model import KaoAnimeModel
```

with:

```python
from kaoanime.model_cyclegan import KaoAnimeModel
```

- [ ] **Step 6.4: Update `tests/test_model.py` — change import**

In `tests/test_model.py`, replace:

```python
from kaoanime.model import KaoAnimeModel
```

with:

```python
from kaoanime.model_cyclegan import KaoAnimeModel
```

- [ ] **Step 6.5: Update `tests/test_inference.py` — change import**

In `tests/test_inference.py`, replace:

```python
from kaoanime.model import KaoAnimeModel
```

with:

```python
from kaoanime.model_cyclegan import KaoAnimeModel
```

- [ ] **Step 6.6: Verify KaoAnimeModel still imports and forward pass works**

```bash
uv run python -c "
import torch
from kaoanime.config import Config
from kaoanime.model_cyclegan import KaoAnimeModel

cfg = Config()
m = KaoAnimeModel(cfg)
x = torch.randn(1, 3, 128, 128)
out = m(x)
assert out.shape == (1, 3, 128, 128)
print('model_cyclegan.py OK')
"
```

Expected: `model_cyclegan.py OK`

---

## Task 7: Create NOTModel

**Files:**

- Create: `kaoanime/model_not.py`

`NOTModel` uses `UNetGenerator` as transport map T and `NOTPotential` as Kantorovich potential f.

`toggle_optimizer()` in Lightning sets `requires_grad=False` on the other model's parameters — equivalent to `freeze(f)` / `unfreeze(f)` in the reference code.

- [ ] **Step 7.1: Create `kaoanime/model_not.py`**

```python
from __future__ import annotations

import numpy as np
import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from torchmetrics.image.fid import FrechetInceptionDistance

from kaoanime.config import Config
from kaoanime.models import NOTPotential, UNetGenerator


def _tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) tensor in [-1, 1] to (H, W, 3) uint8 array."""
    img = t.float().clamp(-1.0, 1.0).add(1.0).div(2.0)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


class NOTModel(pl.LightningModule):
    """Neural Optimal Transport (strong OT) Lightning module.

    T: X→Y transport map (UNet).
    f: Kantorovich potential — raw scalar output, no activation (NOTPotential).

    T-loss: MSE(X, T(X)) − f(T(X)).mean()   ← minimise transport cost + fool potential
    f-loss: f(T(X)).mean() − f(Y).mean()    ← maximise Kantorovich separation
    """

    automatic_optimization = False

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "cfg": OmegaConf.to_container(
                    OmegaConf.structured(cfg), resolve=True, throw_on_missing=False
                )
            }
        )
        self.cfg = cfg
        self.T = UNetGenerator(num_filters=cfg.not_.t_filters)
        self.f = NOTPotential(num_filters=cfg.not_.f_filters)
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
        self._fid_images_seen = 0
        self._train_step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.T(x)

    def training_step(self, batch: dict, batch_idx: int) -> None:
        if batch_idx == 0 and not hasattr(self, "_log_batch"):
            self._log_batch = {k: v[:1].detach().cpu() for k, v in batch.items()}
        real_a, real_b = batch["A"], batch["B"]
        opt_t, opt_f = self.optimizers()

        # --- T inner loop (t_iters updates) ---
        # toggle_optimizer freezes f parameters while updating T.
        self.toggle_optimizer(opt_t)
        for _ in range(self.cfg.not_.t_iters):
            T_X = self.T(real_a)
            t_loss = F.mse_loss(real_a, T_X) - self.f(T_X).mean()
            opt_t.zero_grad()
            self.manual_backward(t_loss)
            opt_t.step()
        self.untoggle_optimizer(opt_t)
        # T_X from last inner step — used for FID below

        # --- f update (once) ---
        # T is frozen; recompute T(real_a) without tracking T's gradients.
        self.toggle_optimizer(opt_f)
        with torch.no_grad():
            T_X = self.T(real_a)
        f_loss = self.f(T_X).mean() - self.f(real_b).mean()
        opt_f.zero_grad()
        self.manual_backward(f_loss)
        opt_f.step()
        self.untoggle_optimizer(opt_f)

        self.log_dict(
            {"train/t_loss": t_loss, "train/f_loss": f_loss},
            on_step=True,
            on_epoch=True,
        )

        # --- FID accumulation ---
        n_fid = self.cfg.not_.fid_every_n_steps
        fid_limit = self.cfg.not_.fid_num_images
        if n_fid > 0 and self._fid_images_seen < fid_limit:
            take = min(real_a.shape[0], fid_limit - self._fid_images_seen)
            with torch.no_grad():
                real_f = real_a[:take].float().add(1).div(2).clamp(0, 1)
                fake_f = T_X[:take].float().add(1).div(2).clamp(0, 1)
            self.fid.update(real_f, real=True)
            self.fid.update(fake_f, real=False)
            self._fid_images_seen += take

        self._train_step += 1

        if n_fid > 0 and self._train_step % n_fid == 0 and self._fid_images_seen > 0:
            score = self.fid.compute()
            if isinstance(self.logger, MLFlowLogger):
                self.logger.experiment.log_metric(
                    self.logger.run_id, "val/fid", score.item(), step=self._train_step
                )
            self.fid.reset()
            self._fid_images_seen = 0

        n = self.cfg.not_.log_image_every_n_steps
        if (
            isinstance(self.logger, MLFlowLogger)
            and hasattr(self, "_log_batch")
            and self._train_step % n == 0
        ):
            with torch.no_grad():
                log_a = self._log_batch["A"].to(self.device)
                log_tb = self.T(log_a)
            run_id = self.logger.run_id
            if self._train_step == n:
                self.logger.experiment.log_image(
                    run_id, _tensor_to_image(log_a[0]), "images/input.png"
                )
            self.logger.experiment.log_image(
                run_id, _tensor_to_image(log_tb[0]), f"images/{self._train_step:06d}_output.png"
            )

    def configure_optimizers(self):
        opt_t = torch.optim.Adam(
            self.T.parameters(), lr=self.cfg.not_.t_lr, weight_decay=1e-10
        )
        opt_f = torch.optim.Adam(
            self.f.parameters(), lr=self.cfg.not_.f_lr, weight_decay=1e-10
        )
        return [opt_t, opt_f]
```

- [ ] **Step 7.2: Verify NOTModel forward pass**

```bash
uv run python -c "
import torch
from kaoanime.config import Config
from kaoanime.model_not import NOTModel

cfg = Config()
m = NOTModel(cfg)
x = torch.randn(1, 3, 128, 128)
out = m(x)
assert out.shape == (1, 3, 128, 128)
assert out.min() >= -1.0 and out.max() <= 1.0
print('model_not.py OK')
"
```

---

## Task 8: Update train.py for model selection

**Files:**

- Modify: `train.py`

One `train.py` reads `cfg.model_type` and instantiates the right model. CycleGAN uses `max_epochs`; NOT uses `max_steps`. Each model reads its own sub-config for precision/lr/etc.

- [ ] **Step 8.1: Rewrite `train.py`**

```python
# train.py
import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from kaoanime.config import Config, register_configs
from kaoanime.model_cyclegan import KaoAnimeModel
from kaoanime.model_not import NOTModel
from kaoanime.utils import UnpairedImageDataset, create_dataloader

torch.set_float32_matmul_precision("medium")
register_configs()


def _make_model(cfg: Config) -> pl.LightningModule:
    if cfg.model_type == "cyclegan":
        return KaoAnimeModel(cfg)
    if cfg.model_type == "not":
        return NOTModel(cfg)
    raise ValueError(f"Unknown model_type {cfg.model_type!r}. Choose 'cyclegan' or 'not'.")


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config) -> None:
    dataset = UnpairedImageDataset(
        cfg.data.root_a,
        cfg.data.root_b,
        cfg.data.image_size,
        extra_roots_a=list(cfg.data.extra_roots_a),
        extra_roots_b=list(cfg.data.extra_roots_b),
        align_a=cfg.data.align_a,
        anime_scale=cfg.data.anime_scale,
        anime_shift_x=cfg.data.anime_shift_x,
        anime_shift_y=cfg.data.anime_shift_y,
    )

    train_dl = create_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = _make_model(cfg)

    if cfg.model_type == "not":
        logger = MLFlowLogger(
            experiment_name="kaoanime-not",
            tracking_uri=cfg.not_.mlflow_tracking_uri,
        )
        trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            max_steps=cfg.not_.max_steps,
            precision=cfg.not_.precision,
            log_every_n_steps=cfg.not_.log_every_n_steps,
            logger=logger,
            callbacks=[ModelCheckpoint(filename="step{step:06d}", save_last=True, save_top_k=0)],
        )
    else:
        logger = MLFlowLogger(
            experiment_name="kaoanime",
            tracking_uri=cfg.train.mlflow_tracking_uri,
        )
        trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            max_epochs=cfg.train.max_epochs,
            precision=cfg.train.precision,
            log_every_n_steps=cfg.train.log_every_n_steps,
            logger=logger,
            callbacks=[ModelCheckpoint(filename="epoch{epoch:03d}", save_last=True, save_top_k=0)],
        )

    trainer.fit(model, train_dl)

    ckpt_path = trainer.checkpoint_callback.last_model_path
    if ckpt_path:
        logger.experiment.log_artifact(logger.run_id, ckpt_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Verify train.py imports cleanly**

```bash
uv run python -c "import train; print('train.py import OK')"
```

Expected: `train.py import OK`

---

## Task 9: Shape verification and GPU smoke tests

- [ ] **Step 9.1: Full shape check for all new models**

```bash
uv run python -c "
import torch
from kaoanime.models import (
    ResNetGenerator, ResNetDiscriminator,
    UNetGenerator, PatchDiscriminator,
    NOTPotential,
)

x = torch.randn(2, 3, 128, 128)

# Generators
rg = ResNetGenerator(num_filters=64, num_residual_blocks=9)
ug = UNetGenerator(num_filters=48)
assert rg(x).shape == (2, 3, 128, 128)
assert ug(x).shape == (2, 3, 128, 128)

# CycleGAN discriminators
pd = PatchDiscriminator(num_filters=64)
rd = ResNetDiscriminator(num_filters=64)
assert pd(x).shape[1] == 1           # patch map
assert rd(x).shape == (2, 1)         # global scalar

# NOT potential
f = NOTPotential(num_filters=64)
assert f(x).shape == (2, 1)          # global scalar

print('All shape checks PASSED')
print(f'  ResNetGenerator params:    {sum(p.numel() for p in rg.parameters()):,}')
print(f'  UNetGenerator(48) params:  {sum(p.numel() for p in ug.parameters()):,}')
print(f'  NOTPotential params:       {sum(p.numel() for p in f.parameters()):,}')
"
```

- [ ] **Step 9.2: CycleGAN GPU smoke test (3 steps)**

```bash
uv run python -c "
import torch, lightning as pl
from kaoanime.config import Config
from kaoanime.model_cyclegan import KaoAnimeModel

cfg = Config()
m = KaoAnimeModel(cfg)

class FakeDL:
    def __iter__(self):
        yield {'A': torch.randn(2, 3, 128, 128), 'B': torch.randn(2, 3, 128, 128)}
    def __len__(self):
        return 1

trainer = pl.Trainer(max_steps=3, accelerator='auto', devices=1,
                     precision=cfg.train.precision, logger=False, enable_checkpointing=False)
trainer.fit(m, FakeDL())
print('CycleGAN smoke test PASSED')
"
```

- [ ] **Step 9.3: NOT GPU smoke test (3 steps)**

```bash
uv run python -c "
import torch, lightning as pl
from kaoanime.config import Config
from kaoanime.model_not import NOTModel

cfg = Config()
m = NOTModel(cfg)

class FakeDL:
    def __iter__(self):
        yield {'A': torch.randn(2, 3, 128, 128), 'B': torch.randn(2, 3, 128, 128)}
    def __len__(self):
        return 1

trainer = pl.Trainer(max_steps=3, accelerator='auto', devices=1,
                     precision=cfg.not_.precision, logger=False, enable_checkpointing=False)
trainer.fit(m, FakeDL())
print('NOT smoke test PASSED')
"
```

- [ ] **Step 9.4: Real-dataset NOT smoke test (3 steps)**

```bash
uv run python -c "
import torch, lightning as pl
from kaoanime.config import Config
from kaoanime.model_not import NOTModel
from kaoanime.utils import UnpairedImageDataset, create_dataloader

cfg = Config()
dataset = UnpairedImageDataset(
    cfg.data.root_a, cfg.data.root_b, cfg.data.image_size,
    align_a=cfg.data.align_a,
    anime_scale=cfg.data.anime_scale,
    anime_shift_x=cfg.data.anime_shift_x,
    anime_shift_y=cfg.data.anime_shift_y,
)
dl = create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0)
m = NOTModel(cfg)
trainer = pl.Trainer(max_steps=3, accelerator='auto', devices=1,
                     precision=cfg.not_.precision, logger=False, enable_checkpointing=False)
trainer.fit(m, dl)
print('NOT real-dataset smoke test PASSED')
"
```

---

## Task 10: Commit

- [ ] **Step 10.1: Stage all changes**

```bash
git add kaoanime/models/resnet.py
git add kaoanime/models/unet.py
git add kaoanime/models/patch_discriminator.py
git add kaoanime/models/not_potential.py
git add kaoanime/models/__init__.py
git add kaoanime/config.py
git add kaoanime/config_cyclegan.py
git add kaoanime/config_not.py
git add kaoanime/model_cyclegan.py   # was model.py (git mv preserves history)
git add kaoanime/model_not.py
git add train.py
git add eval.py
git add infer.py
git add tests/test_model.py
git add tests/test_inference.py
```

- [ ] **Step 10.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor: restructure codebase for parallel CycleGAN + NOT models

Building blocks:
- models/cyclegan.py split into resnet.py, unet.py, patch_discriminator.py
- models/not_potential.py: NOTPotential (ResNet_D, no BN/SN, res_ratio=0.1)

Config:
- config_cyclegan.py: TrainConfig, ModelConfig (CycleGAN hyperparameters)
- config_not.py: NOTConfig (t_iters=10, t_lr/f_lr=1e-4, t_filters=48)
- config.py: DataConfig, EvalConfig, unified Config(model_type) — existing
  import sites unchanged

Models:
- model.py → model_cyclegan.py (KaoAnimeModel, unchanged logic)
- model_not.py: NOTModel — T inner loop × t_iters, f outer update
  T-loss = MSE(X, T(X)) - f(T(X)), f-loss = f(T(X)) - f(Y)

Train:
- train.py: one entry point, switches on cfg.model_type
  CycleGAN: uv run python train.py
  NOT:      uv run python train.py model_type=not

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Running training

**CycleGAN** (default):

```bash
uv run python train.py
```

**NOT**:

```bash
uv run python train.py model_type=not
```

**Override NOT hyperparameters** via Hydra CLI:

```bash
uv run python train.py model_type=not not_.t_iters=5 not_.t_lr=5e-5
```

Monitor both runs in MLflow at `http://10.0.111.233:9999` — CycleGAN logs to experiment `kaoanime`, NOT to `kaoanime-not`.

---

## Notes

- **`config.py` stays as the unified import point** — `from kaoanime.config import Config, register_configs` works everywhere; `from kaoanime.config_cyclegan import TrainConfig` also works if you want the explicit submodule
- **`toggle_optimizer()` = `freeze/unfreeze`** — Lightning disables requires_grad on the non-active model, matching the reference's explicit freeze pattern
- **NOT has no cycle loss** — the MSE cost `MSE(X, T(X))` in the T-loss serves the same role: it forces T to be "efficient" rather than arbitrary
- **`t_iters=10` is load-bearing** — too low makes T purely adversarial; the 10:1 ratio between T and f updates is from the paper
- **UNet `num_filters=48`** — matches reference `base_factor=48`; the default 64 in existing UNetGenerator is heavier; NOT config sets `t_filters=48`
