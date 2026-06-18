# NOT Training Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix NOT model training instability by switching UNetGenerator to BatchNorm (matching Korotin reference) and adding gradient norm clipping to the T inner loop.

**Architecture:** Two independent commits. Task 1 changes `kaoanime/models/unet.py` only — two `InstanceNorm2d` → `BatchNorm2d` substitutions in `_DoubleConv`. Task 2 adds one config field to `kaoanime/config_not.py` and one `clip_grad_norm_` call inside the T inner loop in `kaoanime/model_not.py`.

**Tech Stack:** PyTorch, Lightning, Python 3.x. Always invoke as `uv run python` / `uv run pytest`.

---

## File Map

| File | Change |
|---|---|
| `kaoanime/models/unet.py` | Replace both `InstanceNorm2d` → `BatchNorm2d` in `_DoubleConv.net` |
| `kaoanime/config_not.py` | Add `t_grad_clip: float = 100.0` |
| `kaoanime/model_not.py` | Add `clip_grad_norm_` call after `manual_backward` in T inner loop |
| `tests/test_unet.py` | New file: tests for BatchNorm presence and output shape/range |
| `tests/test_model_not.py` | Add two tests: config field exists, clip called during training |

---

### Task 1: Switch UNetGenerator from InstanceNorm to BatchNorm

`kaoanime/models/unet.py` line 10–24 defines `_DoubleConv`. It currently uses
`nn.InstanceNorm2d` twice. The Korotin NOT reference uses `BatchNorm2d`. InstanceNorm
normalizes per-image (stripping global batch statistics), which conflicts with the
quadratic MSE cost that T must minimize across the batch.

**Files:**
- Modify: `kaoanime/models/unet.py`
- Create: `tests/test_unet.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_unet.py`:

```python
import torch
import torch.nn as nn

from kaoanime.models import UNetGenerator


def test_unet_uses_batchnorm():
    model = UNetGenerator(num_filters=8)
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    has_in = any(isinstance(m, nn.InstanceNorm2d) for m in model.modules())
    assert has_bn, "UNetGenerator must use BatchNorm2d (Korotin reference)"
    assert not has_in, "UNetGenerator must not use InstanceNorm2d"


def test_unet_output_shape_and_range():
    model = UNetGenerator(num_filters=8)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3, 32, 32), f"Expected (2,3,32,32), got {out.shape}"
    assert out.min().item() >= -1.0, "Output below -1 (Tanh broken?)"
    assert out.max().item() <= 1.0, "Output above 1 (Tanh broken?)"
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
uv run pytest tests/test_unet.py -v
```

Expected: `test_unet_uses_batchnorm` FAILS (`has_bn=False, has_in=True`).
`test_unet_output_shape_and_range` PASSES (shape/Tanh already correct).

- [ ] **Step 1.3: Replace `InstanceNorm2d` with `BatchNorm2d` in `_DoubleConv`**

In `kaoanime/models/unet.py`, replace the body of `_DoubleConv.__init__` so `self.net` reads:

```python
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
```

`bias=False` is kept — BatchNorm has its own learnable bias, so the Conv bias would be redundant.

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
uv run pytest tests/test_unet.py -v
```

Expected: both tests PASS.

- [ ] **Step 1.5: Run the full test suite to catch regressions**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass. If `test_cyclegan_model.py` or `test_model.py` fails with a
shape error, the CycleGAN model uses `ResNetGenerator` from `resnet.py` (not affected by
this change) — investigate before proceeding.

- [ ] **Step 1.6: Commit**

```bash
git add kaoanime/models/unet.py tests/test_unet.py
git commit -m "fix: switch UNetGenerator from InstanceNorm2d to BatchNorm2d (Korotin reference)"
```

---

### Task 2: Add gradient clipping to the T inner loop

`kaoanime/model_not.py` `training_step` runs a T inner loop (`t_iters` iterations) then
one f update. Observed symptom: MSE drifts upward over 80k steps, meaning T takes
oversized gradient steps. Add `clip_grad_norm_` after `manual_backward` and before
`opt_t.step()` in every T inner iteration. The clip norm is configurable via a new
`t_grad_clip` field in `NOTConfig`.

`_tiny_cfg()` and `_PairedDataset` are already defined in `tests/test_model_not.py`.

**Files:**
- Modify: `kaoanime/config_not.py`
- Modify: `kaoanime/model_not.py`
- Modify: `tests/test_model_not.py`

- [ ] **Step 2.1: Write the failing tests**

In `tests/test_model_not.py`, update the existing mock import at line 1 to include `patch`:

```python
from unittest.mock import MagicMock, patch
```

Add these two tests at the bottom of the file:

```python
def test_not_config_has_t_grad_clip():
    cfg = Config()
    assert hasattr(cfg.not_, "t_grad_clip"), "NOTConfig must have t_grad_clip field"
    assert cfg.not_.t_grad_clip == 100.0


def test_grad_clip_called_in_t_loop(tmp_path):
    """clip_grad_norm_ must be called exactly t_iters times per training_step."""
    cfg = _tiny_cfg()  # t_iters=2
    model = NOTModel(cfg)
    model.fid.update = MagicMock()
    model.fid.reset = MagicMock()
    model.fid.compute = MagicMock(return_value=torch.tensor(0.5))

    loader = DataLoader(_PairedDataset(), batch_size=2, num_workers=0)

    with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
        trainer = Trainer(
            max_steps=cfg.not_.t_iters + 1,  # (t_iters + 1) Lightning steps = 1 training_step
            max_epochs=-1,
            accelerator="cpu",
            precision="32-true",
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_dataloaders=loader)

    assert mock_clip.call_count == cfg.not_.t_iters, (
        f"Expected clip_grad_norm_ called {cfg.not_.t_iters} times "
        f"(once per T inner iter), got {mock_clip.call_count}"
    )
    clip_value = mock_clip.call_args_list[0].args[1]
    assert clip_value == cfg.not_.t_grad_clip, (
        f"Expected clip value {cfg.not_.t_grad_clip}, got {clip_value}"
    )
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
uv run pytest tests/test_model_not.py::test_not_config_has_t_grad_clip tests/test_model_not.py::test_grad_clip_called_in_t_loop -v
```

Expected: both FAIL — `AttributeError: 'NOTConfig' object has no attribute 't_grad_clip'`
for the first; same error (or `call_count == 0`) for the second.

- [ ] **Step 2.3: Add `t_grad_clip` to `NOTConfig`**

In `kaoanime/config_not.py`, add one field at the end of the dataclass (after `resume_from_checkpoint`):

```python
from dataclasses import dataclass


@dataclass
class NOTConfig:
    t_iters: int = 10
    t_lr: float = 1e-4
    f_lr: float = 1e-4
    t_filters: int = 48
    f_filters: int = 64
    max_steps: int = 150001
    precision: str = "bf16-mixed"
    log_every_n_steps: int = 50
    mlflow_tracking_uri: str = "http://10.0.111.233:9999"
    log_image_every_n_steps: int = 100
    fid_every_n_steps: int = 500
    fid_num_images: int = 512
    resume_from_checkpoint: str = ""
    t_grad_clip: float = 100.0
```

- [ ] **Step 2.4: Add `clip_grad_norm_` to the T inner loop**

In `kaoanime/model_not.py`, modify the T inner loop. The full loop (currently lines ~62–70) must become:

```python
        self.toggle_optimizer(opt_t)
        for _ in range(self.cfg.not_.t_iters):
            T_X = self.T(real_a)
            t_loss = F.mse_loss(real_a, T_X) - self.f(T_X).mean()
            opt_t.zero_grad()
            self.manual_backward(t_loss)
            torch.nn.utils.clip_grad_norm_(self.T.parameters(), self.cfg.not_.t_grad_clip)
            opt_t.step()
        self.untoggle_optimizer(opt_t)
```

No other lines in `training_step` change.

- [ ] **Step 2.5: Run tests to verify they pass**

```bash
uv run pytest tests/test_model_not.py::test_not_config_has_t_grad_clip tests/test_model_not.py::test_grad_clip_called_in_t_loop -v
```

Expected: both PASS.

- [ ] **Step 2.6: Run the full test suite to catch regressions**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 2.7: Commit**

```bash
git add kaoanime/config_not.py kaoanime/model_not.py tests/test_model_not.py
git commit -m "feat: add T gradient clipping in NOT training (t_grad_clip=100)"
```

---

## Final Verification

```bash
uv run python -c "
import torch, torch.nn as nn
from kaoanime.models import UNetGenerator
from kaoanime.config import Config

g = UNetGenerator(num_filters=48)
x = torch.randn(2, 3, 128, 128)
out = g(x)
assert out.shape == (2, 3, 128, 128)
assert out.min() >= -1.0 and out.max() <= 1.0
assert not any(isinstance(m, nn.InstanceNorm2d) for m in g.modules())
assert any(isinstance(m, nn.BatchNorm2d) for m in g.modules())
params = sum(p.numel() for p in g.parameters())
print(f'UNetGenerator OK — params {params:,}')

cfg = Config()
assert cfg.not_.t_grad_clip == 100.0
print('NOTConfig OK')
"
```

Expected:
```
UNetGenerator OK — params <N>
NOTConfig OK
```
