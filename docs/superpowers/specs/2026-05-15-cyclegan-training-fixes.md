# CycleGAN Training Fixes Design

## Goal

Fix CycleGAN "colors-only" domain translation failure by replacing the discriminator norm, switching to LSGAN loss, and correcting the optimizer and loss-weight hyperparameters.

## Root Causes Identified

- `disc_steps=5` starved the generator: discriminator trained 5× per batch, became too confident too quickly, leaving the adversarial gradient numerically small relative to `lambda_cycle=10`. Generator fell back to minimizing only cycle+identity loss, for which color-shift is the trivial minimum.
- `lambda_cycle=10` with a dominant discriminator made color-mapping the path of least resistance — color shifts satisfy cycle consistency at near-zero L1 cost and are trivially invertible.
- RMSprop at `lr=1e-3` is non-standard for CycleGAN; the original paper uses Adam with `beta1=0.5`.
- Spectral norm in PatchDiscriminator is from SNGAN/WGAN; the original CycleGAN discriminator uses instance norm.
- BCEWithLogitsLoss is correct for non-saturating GAN but the original CycleGAN paper uses LSGAN (MSE), which maintains gradient signal across the full output range.

## Changes

### `kaoanime/models/patch_discriminator.py`

Replace all `nn.utils.spectral_norm(...)` wrappers with plain `nn.Conv2d`. Add `nn.InstanceNorm2d` after each intermediate conv:

- Conv 1 (first, raw pixels): no norm
- Conv 2–5 (intermediate): `InstanceNorm2d` after each
- Conv 6 (output): no norm, no activation

```python
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, num_filters: int = 64) -> None:
        super().__init__()
        nf = num_filters
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, nf * 8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, 1, 4, stride=1, padding=1),
        )
```

### `kaoanime/losses/cyclegan.py`

Replace `nn.BCEWithLogitsLoss()` with `nn.MSELoss()`. Target values remain `ones_like` / `zeros_like` — MSELoss penalises distance from 0 or 1 instead of log-likelihood. The 0.5 scaling on the discriminator loss is unchanged.

```python
self.gan = nn.MSELoss()
```

No other changes to `generator()` or `discriminator()` method bodies.

### `kaoanime/config_cyclegan.py`

| Field | Old | New |
|-------|-----|-----|
| `lr` | `1e-3` | `2e-4` |
| `alpha` | `0.993` | _(removed)_ |
| `beta1` | _(missing)_ | `0.5` |
| `disc_steps` | `5` | `1` |
| `lambda_cycle` | `10.0` | `2.0` |
| `lambda_identity` | `5.0` | `1.0` |

### `kaoanime/model_cyclegan.py`

Remove the dead `if self.cfg.model_type == 'not':` branch in `configure_optimizers` — it referenced `cfg.train.beta1` which does not exist in `CycleGANTrainConfig` and would crash. Replace with a single Adam path for both optimizers:

```python
def configure_optimizers(self):
    lr = self.cfg.train.lr
    betas = (self.cfg.train.beta1, 0.999)
    opt_g = torch.optim.Adam(
        list(self.g_ab.parameters()) + list(self.g_ba.parameters()),
        lr=lr, betas=betas,
    )
    opt_d = torch.optim.Adam(
        list(self.d_a.parameters()) + list(self.d_b.parameters()),
        lr=lr, betas=betas,
    )
    self._sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, self._lr_lambda)
    self._sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, self._lr_lambda)
    return [opt_g, opt_d]
```

## What Is Not Changed

- Generator architecture (UNet, num_filters=48) — unchanged
- Cycle-consistency and identity loss formulas — unchanged
- Image pool (replay buffer) — unchanged
- LR decay schedule — unchanged
- All NOT model code — unchanged
