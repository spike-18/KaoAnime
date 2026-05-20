# NOT Training Stability — Design

**Date:** 2026-05-20
**Status:** Approved

## Problem

The NOT model (Neural Optimal Transport, Korotin et al. ICLR 2023) fails to converge on the
CelebA → anime domain pair. Symptoms observed at 80k steps:

- `f_on_real` and `f_on_fake` both decrease from ~1 to ~0 — the Kantorovich potential f
  collapses to a near-zero constant, losing the ability to separate the two distributions.
- `f_loss` oscillates around 0 — no meaningful gradient signal to T from f.
- `t_loss` increases from −0.1 to +0.2 while `train/mse` increases from 0.2 to 0.3 —
  T is drifting away from real_a without a coherent target, driven by noisy gradients from the
  collapsing f.

A previous run had weights explode at 90k steps.

## Root Cause Analysis

Comparison against the Korotin reference implementation
(https://github.com/iamalexkorotin/NeuralOptimalTransport) identified two differences:

### 1. InstanceNorm vs BatchNorm in T

The reference UNet uses `BatchNorm2d`. Our `kaoanime/models/unet.py` uses `InstanceNorm2d`.

In NOT, the quadratic cost `c(x, T(x)) = ||x − T(x)||²` is computed in pixel space. T needs to
track global pixel statistics across the batch to minimize this coherently. InstanceNorm
normalizes each feature map per image — it strips away the global batch-level statistics that the
MSE cost depends on. BatchNorm preserves batch statistics, making the optimization landscape
consistent with the cost function.

InstanceNorm is well-suited for style transfer (CycleGAN, Ulyanov et al.) where per-image
normalization removes unwanted texture — but that is precisely the wrong inductive bias for NOT,
where T must transport mass according to a batch-level quadratic cost.

### 2. T gradient drift (engineering gap)

The reference benchmarks use simpler distribution pairs (synthetic, color transfer) where T
updates are small. On CelebA → anime, the distributional gap is large and f frequently sends
noisy gradient signals to T (especially early in training and during f-collapse episodes). With
no gradient norm bound, these signals cause T to take oversized steps, which is why MSE drifts
upward over 80k steps instead of stabilizing.

The reference does not clip gradients because its distributions do not require it; this is an
engineering fix needed for the harder domain pair, not a change to the mathematical formulation.

## Solution

### Change 1 — Switch `_DoubleConv` to BatchNorm (`kaoanime/models/unet.py`)

Replace both `nn.InstanceNorm2d` calls in `_DoubleConv.net` with `nn.BatchNorm2d`.

The `track_running_stats` default (True) is kept — running stats are used at inference time,
matching standard BatchNorm behavior.

**Note:** BatchNorm requires `batch_size > 1`. The current training config uses `bs=32`, so
this is never an issue. No config guard is added.

### Change 2 — Gradient clipping for T (`kaoanime/model_not.py` + `kaoanime/config_not.py`)

Add `t_grad_clip: float = 100.0` to `NOTConfig`.

Inside the T inner loop in `training_step`, after `self.manual_backward(t_loss)` and before
`opt_t.step()`, add:

```python
torch.nn.utils.clip_grad_norm_(self.T.parameters(), self.cfg.not_.t_grad_clip)
```

The clip norm of 100 matches the scale used in comparable adversarial training literature and is
large enough not to interfere with normal gradient updates (typical T gradient norms are O(1–10))
while preventing the occasional large spikes that drive MSE drift.

## Files Changed

| File | Change |
|---|---|
| `kaoanime/models/unet.py` | `_DoubleConv`: replace `InstanceNorm2d` → `BatchNorm2d` (2 lines) |
| `kaoanime/config_not.py` | Add `t_grad_clip: float = 100.0` |
| `kaoanime/model_not.py` | Add `clip_grad_norm_` call inside T inner loop |

No other files change. The CycleGAN model (`kaoanime/models/cyclegan.py`) has its own separate
UNet that is unaffected — the fix targets only `kaoanime/models/unet.py` which is used by
`NOTModel`.

**Note on CycleGAN generator:** `kaoanime/models/resnet.py` (`ResNetGenerator`) also uses
`InstanceNorm2d`. CycleGAN is an adversarial style-transfer model where InstanceNorm is correct
per Zhu et al. 2017; that file is intentionally left unchanged.

## Testing

Unit tests in `tests/test_not_stability.py`:

1. **BatchNorm in UNet**: instantiate `UNetGenerator`, confirm no `InstanceNorm2d` module exists
   in the model (walk `model.modules()`), confirm `BatchNorm2d` is present.
2. **Gradient clip applied**: run a single NOT training step using a `Trainer` with a tiny
   synthetic dataset; after the T inner loop, inspect `opt_t`'s parameter gradients and confirm
   their global norm is ≤ `t_grad_clip * 1.01` (1% tolerance for floating point).
3. **Shape preservation**: `UNetGenerator(num_filters=48)(torch.randn(2, 3, 128, 128))` still
   produces shape `(2, 3, 128, 128)` and output in `[−1, 1]`.

All tests run via `uv run pytest`.
