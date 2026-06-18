# Weight Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply principled weight initialization to all five model classes so that training starts from a well-conditioned state rather than PyTorch's default Kaiming-uniform.

**Architecture:** A single `init_weights(module, init_type, gain)` utility lives in `kaoanime/models/weights_init.py`. Each model's `__init__` calls `self.apply(...)` with the type appropriate to its activation function and normalization scheme. Two types are used: `"normal"` (CycleGAN paper standard, Normal(0, 0.02)) for generators and PatchDiscriminator; `"kaiming"` (Kaiming-normal with a=0.2 for LeakyReLU) for NOTPotential and ResNetDiscriminator which have no batch/instance-norm layers.

**Why two different init types:**

- Generators and PatchDiscriminator use InstanceNorm layers that re-centre activations, so the exact weight scale matters less — Normal(0, 0.02) is the established CycleGAN-paper standard.
- NOTPotential and ResNetDiscriminator have no normalization (only LeakyReLU), so proper Kaiming scaling is needed to keep variance stable across the network depth.

**Spectral norm handling:** `nn.utils.spectral_norm` stores the actual weight in `weight_orig`. The utility checks for this attribute and targets it directly so the normalised view is consistent after init.

**Tech Stack:** Python, PyTorch, pytest. Always invoke Python as `uv run python`, pytest as `uv run pytest`.

---

## File Map

| File                                     | Change                                                                      |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| `kaoanime/models/weights_init.py`        | Create: `init_weights` utility function                                     |
| `kaoanime/models/__init__.py`            | Export `init_weights`                                                       |
| `kaoanime/models/unet.py`                | Apply `init_weights(self, "normal")` in `UNetGenerator.__init__`            |
| `kaoanime/models/resnet.py`              | Apply `"normal"` to `ResNetGenerator`, `"kaiming"` to `ResNetDiscriminator` |
| `kaoanime/models/patch_discriminator.py` | Apply `init_weights(self, "normal")` in `PatchDiscriminator.__init__`       |
| `kaoanime/models/not_potential.py`       | Apply `init_weights(self, "kaiming")` in `NOTPotential.__init__`            |
| `tests/test_weights_init.py`             | Create: tests for `init_weights` and per-model weight statistics            |

---

## Task 1 — Create `init_weights` utility and tests

**Files:**

- Create: `kaoanime/models/weights_init.py`
- Create: `tests/test_weights_init.py`

- [ ] **Step 1.1: Write failing tests**

Create `tests/test_weights_init.py`:

```python
import torch
import torch.nn as nn

from kaoanime.models.weights_init import init_weights


def test_normal_init_sets_weight_std():
    m = nn.Conv2d(8, 16, 3, padding=1)
    init_weights(m, "normal", gain=0.02)
    std = m.weight.data.std().item()
    assert 0.005 < std < 0.04, f"Expected std≈0.02, got {std:.4f}"


def test_normal_init_zeros_bias():
    m = nn.Conv2d(8, 16, 3, padding=1)  # bias=True by default
    init_weights(m, "normal", gain=0.02)
    assert m.bias.data.abs().max().item() == 0.0


def test_normal_init_no_bias_layer():
    m = nn.Conv2d(8, 16, 3, padding=1, bias=False)
    init_weights(m, "normal", gain=0.02)  # must not crash
    std = m.weight.data.std().item()
    assert 0.005 < std < 0.04


def test_kaiming_init_sets_weight_std():
    # Conv2d(64, 128, 3): fan_in = 64*3*3 = 576
    # kaiming_normal (a=0.2, fan_in) → std = sqrt(2 / (576*(1+0.04))) ≈ 0.057
    m = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    init_weights(m, "kaiming")
    std = m.weight.data.std().item()
    assert 0.03 < std < 0.15, f"Expected kaiming std ~0.057, got {std:.4f}"


def test_normal_and_kaiming_differ():
    torch.manual_seed(0)
    m1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    init_weights(m1, "normal", gain=0.02)

    torch.manual_seed(0)
    m2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    init_weights(m2, "kaiming")

    assert not torch.allclose(m1.weight.data, m2.weight.data)


def test_spectral_norm_layer_init():
    m = nn.utils.spectral_norm(nn.Conv2d(8, 16, 3, padding=1))
    init_weights(m, "normal", gain=0.02)
    assert hasattr(m, "weight_orig")
    std = m.weight_orig.data.std().item()
    assert 0.005 < std < 0.04, f"weight_orig std should be ≈0.02, got {std:.4f}"


def test_init_weights_applies_to_sequential():
    seq = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1, bias=False),
        nn.Conv2d(16, 8, 3, padding=1, bias=False),
    )
    init_weights(seq, "normal", gain=0.02)
    for layer in seq:
        std = layer.weight.data.std().item()
        assert 0.005 < std < 0.04
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_weights_init.py -v
```

Expected: all FAIL with `ModuleNotFoundError: No module named 'kaoanime.models.weights_init'`.

- [ ] **Step 1.3: Create `kaoanime/models/weights_init.py`**

```python
from __future__ import annotations

import torch.nn as nn
from torch.nn import init


def init_weights(module: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    """Apply weight initialization to *module* and all its children in-place.

    init_type:
        "normal"  — Normal(0, gain). CycleGAN paper standard for generators
                    and discriminators that use instance/batch norm.
        "kaiming" — Kaiming-normal, a=0.2, fan_in. For LeakyReLU(0.2)
                    networks without normalisation layers (NOTPotential,
                    ResNetDiscriminator).

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
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    module.apply(_init_func)
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_weights_init.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add kaoanime/models/weights_init.py tests/test_weights_init.py
git commit -m "feat: add init_weights utility (normal/kaiming, spectral-norm-safe)"
```

---

## Task 2 — Apply initialization to all model classes

**Files:**

- Modify: `kaoanime/models/unet.py`
- Modify: `kaoanime/models/resnet.py`
- Modify: `kaoanime/models/patch_discriminator.py`
- Modify: `kaoanime/models/not_potential.py`
- Modify: `kaoanime/models/__init__.py`
- Test: `tests/test_weights_init.py`

- [ ] **Step 2.1: Write failing per-model tests**

Append to `tests/test_weights_init.py`:

```python
from kaoanime.models import (
    NOTPotential,
    PatchDiscriminator,
    ResNetDiscriminator,
    ResNetGenerator,
    UNetGenerator,
)


def _conv_weight_std(model: nn.Module) -> float:
    """Concatenate all Conv2d/ConvTranspose2d weights and return their std."""
    weights = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
            weights.append(w.data.flatten())
    assert weights, "No conv layers found"
    return torch.cat(weights).std().item()


def test_unet_generator_normal_init():
    g = UNetGenerator(num_filters=16)
    std = _conv_weight_std(g)
    assert std < 0.04, f"UNetGenerator std should be ≈0.02, got {std:.4f}"


def test_resnet_generator_normal_init():
    g = ResNetGenerator(num_filters=16, num_residual_blocks=2)
    std = _conv_weight_std(g)
    assert std < 0.04, f"ResNetGenerator std should be ≈0.02, got {std:.4f}"


def test_patch_discriminator_normal_init():
    d = PatchDiscriminator(num_filters=16)
    std = _conv_weight_std(d)
    assert std < 0.04, f"PatchDiscriminator std should be ≈0.02, got {std:.4f}"


def test_resnet_discriminator_kaiming_init():
    d = ResNetDiscriminator(num_filters=16)
    std = _conv_weight_std(d)
    # kaiming for (16→32, 3×3) fan_in=16*9=144 → std≈sqrt(2/(144*1.04))≈0.115
    # kaiming for (3→16, 3×3) fan_in=3*9=27 → std≈sqrt(2/(27*1.04))≈0.267
    # mixed: should be clearly > 0.04 and < 0.5
    assert std > 0.04, f"ResNetDiscriminator kaiming std should be >0.04, got {std:.4f}"


def test_not_potential_kaiming_init():
    f = NOTPotential(num_filters=16)
    std = _conv_weight_std(f)
    assert std > 0.04, f"NOTPotential kaiming std should be >0.04, got {std:.4f}"
```

- [ ] **Step 2.2: Run failing tests**

```bash
uv run pytest tests/test_weights_init.py::test_unet_generator_normal_init \
              tests/test_weights_init.py::test_resnet_generator_normal_init \
              tests/test_weights_init.py::test_patch_discriminator_normal_init \
              tests/test_weights_init.py::test_resnet_discriminator_kaiming_init \
              tests/test_weights_init.py::test_not_potential_kaiming_init \
              -v
```

Expected: some PASS (models whose default Kaiming-uniform already satisfies the loose bound), some FAIL. The normal-init tests (UNet, ResNet generator, PatchDisc) will FAIL because PyTorch's default gives std ≈ 0.18–0.27, much larger than 0.04.

- [ ] **Step 2.3: Update `kaoanime/models/unet.py`**

Add import at the top (after `import torch.nn.functional as F`):

```python
from kaoanime.models.weights_init import init_weights
```

Add one line at the end of `UNetGenerator.__init__`, after `self.outc = ...`:

```python
        init_weights(self, "normal", gain=0.02)
```

Full `__init__` after the change:

```python
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
        init_weights(self, "normal", gain=0.02)
```

- [ ] **Step 2.4: Update `kaoanime/models/resnet.py`**

Add import at the top (after `import torch.nn as nn`):

```python
from kaoanime.models.weights_init import init_weights
```

In `ResNetGenerator.__init__`, add one line at the end (after `self.model = nn.Sequential(*layers)`):

```python
        init_weights(self, "normal", gain=0.02)
```

In `ResNetDiscriminator.__init__`, add one line at the end (after `self.classifier = ...`):

```python
        init_weights(self, "kaiming")
```

- [ ] **Step 2.5: Update `kaoanime/models/patch_discriminator.py`**

Add import at the top (after `import torch.nn as nn`):

```python
from kaoanime.models.weights_init import init_weights
```

In `PatchDiscriminator.__init__`, add one line at the end (after `self.model = nn.Sequential(...)`):

```python
        init_weights(self, "normal", gain=0.02)
```

- [ ] **Step 2.6: Update `kaoanime/models/not_potential.py`**

Add import at the top (after `import torch.nn as nn`):

```python
from kaoanime.models.weights_init import init_weights
```

In `NOTPotential.__init__`, add one line at the end (after `self.classifier = nn.Linear(ch[-1], 1)`):

```python
        init_weights(self, "kaiming")
```

- [ ] **Step 2.7: Update `kaoanime/models/__init__.py`**

Replace the file content with:

```python
from .not_potential import NOTPotential
from .patch_discriminator import PatchDiscriminator
from .resnet import ResNetDiscriminator, ResNetGenerator
from .unet import UNetGenerator
from .weights_init import init_weights

__all__ = [
    "NOTPotential",
    "PatchDiscriminator",
    "ResNetDiscriminator",
    "ResNetGenerator",
    "UNetGenerator",
    "init_weights",
]
```

- [ ] **Step 2.8: Run per-model tests**

```bash
uv run pytest tests/test_weights_init.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 2.9: Run full test suite to catch regressions**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2.10: Verify shapes and outputs unchanged**

```bash
uv run python -c "
import torch
from kaoanime.models import UNetGenerator, ResNetGenerator, PatchDiscriminator, ResNetDiscriminator, NOTPotential

x = torch.randn(2, 3, 128, 128)

g = UNetGenerator(num_filters=48)
out = g(x)
assert out.shape == (2, 3, 128, 128) and out.min() >= -1.0 and out.max() <= 1.0
print(f'UNetGenerator OK — {out.shape}')

g2 = ResNetGenerator(num_filters=48, num_residual_blocks=9)
out2 = g2(x)
assert out2.shape == (2, 3, 128, 128)
print(f'ResNetGenerator OK — {out2.shape}')

d = PatchDiscriminator(num_filters=48)
out3 = d(x)
assert out3.shape[1] == 1
print(f'PatchDiscriminator OK — {out3.shape}')

d2 = ResNetDiscriminator(num_filters=48)
out4 = d2(x)
assert out4.shape == (2, 1)
print(f'ResNetDiscriminator OK — {out4.shape}')

f = NOTPotential(num_filters=48)
out5 = f(x)
assert out5.shape == (2, 1)
print(f'NOTPotential OK — {out5.shape}')
"
```

Expected: all 5 lines print "OK".

- [ ] **Step 2.11: Commit**

```bash
git add kaoanime/models/unet.py kaoanime/models/resnet.py \
        kaoanime/models/patch_discriminator.py kaoanime/models/not_potential.py \
        kaoanime/models/__init__.py tests/test_weights_init.py
git commit -m "feat: apply weight initialization to all model classes (normal for generators+PatchDisc, kaiming for NOTPotential+ResNetDisc)"
```

---

## Self-review

**Spec coverage:**

- ✅ `init_weights` utility: Task 1
- ✅ `"normal"` for UNetGenerator: Task 2
- ✅ `"normal"` for ResNetGenerator: Task 2
- ✅ `"normal"` for PatchDiscriminator: Task 2
- ✅ `"kaiming"` for ResNetDiscriminator: Task 2
- ✅ `"kaiming"` for NOTPotential: Task 2
- ✅ Spectral norm safe: Task 1 test `test_spectral_norm_layer_init`
- ✅ Export from `__init__.py`: Task 2 step 2.7

**Placeholder scan:** No TBDs, all code blocks complete.

**Type consistency:** `init_weights(module, init_type, gain)` signature used identically across all call sites. `_conv_weight_std` helper defined once in test file, used by all per-model tests.
