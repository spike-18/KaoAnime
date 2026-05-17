from __future__ import annotations

import numpy as np
import lightning as pl
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from torchmetrics.image.fid import FrechetInceptionDistance

from kaoanime.config import Config
from kaoanime.losses import CycleGANLoss
from kaoanime.models import PatchDiscriminator, ResNetDiscriminator, ResNetGenerator, UNetGenerator
from kaoanime.utils import ImagePool, fid_should_accumulate, fid_should_compute


def _make_generator(cfg: Config) -> torch.nn.Module:
    if cfg.model.generator == "unet":
        return UNetGenerator(num_filters=cfg.model.num_filters)
    return ResNetGenerator(
        num_filters=cfg.model.num_filters,
        num_residual_blocks=cfg.model.num_residual_blocks,
    )


def _make_discriminator(cfg: Config) -> torch.nn.Module:
    if cfg.model.discriminator == "resnet":
        return ResNetDiscriminator(num_filters=cfg.model.num_filters)
    return PatchDiscriminator(num_filters=cfg.model.num_filters)


def _tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) tensor in [-1, 1] to a (H, W, 3) uint8 array."""
    img = t.float().clamp(-1.0, 1.0).add(1.0).div(2.0)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


class KaoAnimeModel(pl.LightningModule):
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
        self.g_ab = _make_generator(cfg)
        self.g_ba = _make_generator(cfg)
        self.d_a  = _make_discriminator(cfg)
        self.d_b  = _make_discriminator(cfg)
        self.criterion = CycleGANLoss(
            lambda_cycle=cfg.model.lambda_cycle,
            lambda_identity=cfg.model.lambda_identity,
        )
        self._train_step = 0
        pool_size = max(50, cfg.data.batch_size * 8)
        self.pool_a = ImagePool(pool_size)
        self.pool_b = ImagePool(pool_size)
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
        self._fid_images_seen = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g_ab(x)

    def training_step(self, batch: dict, batch_idx: int) -> None:
        if batch_idx == 0 and not hasattr(self, "_log_batch"):
            self._log_batch = {k: v[:1].detach().cpu() for k, v in batch.items()}
        real_a, real_b = batch["A"], batch["B"]
        opt_g, opt_d = self.optimizers()

        # --- Generator update(s) ---
        self.toggle_optimizer(opt_g)
        for _ in range(self.cfg.train.gen_steps):
            fake_b = self.g_ab(real_a)  # A → B
            fake_a = self.g_ba(real_b)  # B → A
            rec_a  = self.g_ba(fake_b)  # A → B → A
            rec_b  = self.g_ab(fake_a)  # B → A → B
            idt_b  = self.g_ab(real_b)  # G_AB(real_b) should ≈ real_b
            idt_a  = self.g_ba(real_a)  # G_BA(real_a) should ≈ real_a
            disc_fake_a = self.d_a(fake_a)
            disc_fake_b = self.d_b(fake_b)
            loss_g = self.criterion.generator(
                real_a, real_b, fake_a, fake_b, rec_a, rec_b,
                disc_fake_a, disc_fake_b, idt_a, idt_b,
            )
            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()
        self.untoggle_optimizer(opt_g)
        # fake_a, fake_b from last generator iteration are used by disc and FID below

        # --- Discriminator update(s) ---
        self.toggle_optimizer(opt_d)
        for _ in range(self.cfg.train.disc_steps):
            disc_real_a     = self.d_a(real_a)
            disc_fake_a_det = self.d_a(self.pool_a.query(fake_a))
            disc_real_b     = self.d_b(real_b)
            disc_fake_b_det = self.d_b(self.pool_b.query(fake_b))
            loss_d = self.criterion.discriminator(
                disc_real_a, disc_fake_a_det, disc_real_b, disc_fake_b_det,
            )
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.log_dict(
            {"train/loss_g": loss_g, "train/loss_d": loss_d},
            on_step=True,
            on_epoch=True,
        )

        # --- FID: accumulate in the window ENDING at each compute boundary
        # so the score reflects the current generator, not a stale one. ---
        n_fid = self.cfg.train.fid_every_n_steps
        fid_limit = self.cfg.train.fid_num_images
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
                fake_f = fake_b[:take].detach().float().add(1).div(2).clamp(0, 1)
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
        n = self.cfg.train.log_image_every_n_steps
        if isinstance(self.logger, MLFlowLogger) and hasattr(self, "_log_batch") and self._train_step % n == 0:
            with torch.no_grad():
                real_a = self._log_batch["A"].to(self.device)
                fake_b = self.g_ab(real_a)
            run_id = self.logger.run_id
            if self._train_step == n:
                self.logger.experiment.log_image(
                    run_id, _tensor_to_image(real_a[0]), "images/input.png"
                )
            self.logger.experiment.log_image(
                run_id,
                _tensor_to_image(fake_b[0]),
                f"images/{self.trainer.global_step:06d}_output.png",
            )


    def test_step(self, batch: dict, batch_idx: int) -> None:
        _ = self.g_ab(batch["A"])

    def on_train_epoch_end(self) -> None:
        self._sch_g.step()
        self._sch_d.step()

    def _lr_lambda(self, epoch: int) -> float:
        decay_start = self.cfg.train.lr_decay_start_epoch
        max_epochs = self.cfg.train.max_epochs
        if epoch < decay_start or max_epochs <= decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (max_epochs - decay_start))

    def configure_optimizers(self):
        lr = self.cfg.train.lr

        if self.cfg.model_type == 'not':
            betas = (self.cfg.train.beta1, 0.999)
            opt_g = torch.optim.Adam(
                list(self.g_ab.parameters()) + list(self.g_ba.parameters()),
                lr=lr,
                betas=betas,
            )
            opt_d = torch.optim.Adam(
                list(self.d_a.parameters()) + list(self.d_b.parameters()),
                lr=lr,
                betas=betas,
            )
        else:
            alpha = self.cfg.train.alpha
            opt_g = torch.optim.RMSprop(
                list(self.g_ab.parameters()) + list(self.g_ba.parameters()),
                lr=lr,
                alpha=alpha,
            )
            opt_d = torch.optim.RMSprop(
                list(self.d_a.parameters()) + list(self.d_b.parameters()),
                lr=lr,
                alpha=alpha,
            )


        self._sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, self._lr_lambda)
        self._sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, self._lr_lambda)
        return [opt_g, opt_d]
