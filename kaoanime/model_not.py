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
from kaoanime.utils import fid_should_accumulate, fid_should_compute


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
            torch.nn.utils.clip_grad_norm_(self.T.parameters(), self.cfg.not_.t_grad_clip)
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

        f_on_fake = self.f(T_X).mean()
        f_on_real = self.f(real_b).mean()
        # NOT trains as a single max_steps run (max_epochs=-1); on_epoch
        # aggregation never fires and only adds Lightning's _step/_epoch
        # name suffixes, so log on_step only.
        self.log_dict(
            {
                "train/t_loss": t_loss,
                "train/f_loss": f_loss,
                "train/f_on_fake": f_on_fake,
                "train/f_on_real": f_on_real,
                "train/mse": F.mse_loss(real_a, T_X.detach()),
                "train/lr": opt_t.param_groups[0]["lr"],
            },
            on_step=True,
            on_epoch=False,
        )

        # --- FID: accumulate in the window ENDING at each compute boundary
        # so the score reflects the current generator, not a stale one. ---
        n_fid = self.cfg.not_.fid_every_n_steps
        fid_limit = self.cfg.not_.fid_num_images
        bs = real_a.shape[0]
        num_batches = max(1, -(-fid_limit // bs))  # ceil(fid_limit / bs)

        self._train_step += 1
        step = self._train_step

        if (
            fid_should_accumulate(step, n_fid, num_batches)
            and self._fid_images_seen < fid_limit
        ):
            take = min(bs, fid_limit - self._fid_images_seen)
            with torch.no_grad():
                real_f = real_b[:take].float().add(1).div(2).clamp(0, 1)
                fake_f = T_X[:take].float().add(1).div(2).clamp(0, 1)
            self.fid.update(real_f, real=True)
            self.fid.update(fake_f, real=False)
            self._fid_images_seen += take

        if fid_should_compute(step, n_fid) and self._fid_images_seen > 0:
            score = self.fid.compute()
            if isinstance(self.logger, MLFlowLogger):
                self.logger.experiment.log_metric(
                    self.logger.run_id,
                    "val/fid",
                    score.item(),
                    step=self.trainer.global_step,
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
                run_id,
                _tensor_to_image(log_tb[0]),
                f"images/{self.trainer.global_step:06d}_output.png",
            )

        self._sch_t.step()
        self._sch_f.step()

    def configure_optimizers(self):
        betas = (self.cfg.not_.beta1, self.cfg.not_.beta2)
        opt_t = torch.optim.Adam(
            self.T.parameters(), lr=self.cfg.not_.t_lr, weight_decay=1e-10, betas=betas
        )
        opt_f = torch.optim.Adam(
            self.f.parameters(), lr=self.cfg.not_.f_lr, weight_decay=1e-10, betas=betas
        )
        self._sch_t = torch.optim.lr_scheduler.StepLR(
            opt_t, step_size=self.cfg.not_.lr_step_size, gamma=self.cfg.not_.lr_gamma
        )
        self._sch_f = torch.optim.lr_scheduler.StepLR(
            opt_f, step_size=self.cfg.not_.lr_step_size, gamma=self.cfg.not_.lr_gamma
        )
        return [opt_t, opt_f]
