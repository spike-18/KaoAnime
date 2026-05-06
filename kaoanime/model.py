from __future__ import annotations

import lightning as pl
import torch
from omegaconf import OmegaConf

from kaoanime.config import Config
from kaoanime.losses import CycleGANLoss
from kaoanime.models import PatchDiscriminator, ResNetGenerator


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
        self.g_ab = ResNetGenerator(
            num_filters=cfg.model.num_filters,
            num_residual_blocks=cfg.model.num_residual_blocks,
        )
        self.g_ba = ResNetGenerator(
            num_filters=cfg.model.num_filters,
            num_residual_blocks=cfg.model.num_residual_blocks,
        )
        self.d_a = PatchDiscriminator(num_filters=cfg.model.num_filters)
        self.d_b = PatchDiscriminator(num_filters=cfg.model.num_filters)
        self.criterion = CycleGANLoss(
            lambda_cycle=cfg.model.lambda_cycle,
            lambda_identity=cfg.model.lambda_identity,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g_ab(x)

    def training_step(self, batch: dict, batch_idx: int) -> None:
        real_a, real_b = batch["A"], batch["B"]
        opt_g, opt_d = self.optimizers()

        # --- Forward pass ---
        fake_b = self.g_ab(real_a)  # A → B
        fake_a = self.g_ba(real_b)  # B → A
        rec_a = self.g_ba(fake_b)  # A → B → A
        rec_b = self.g_ab(fake_a)  # B → A → B
        idt_a = self.g_ba(real_a)  # G_BA on A domain (identity)
        idt_b = self.g_ab(real_b)  # G_AB on B domain (identity)

        # --- Generator update ---
        self.toggle_optimizer(opt_g)
        disc_fake_a = self.d_a(fake_a)
        disc_fake_b = self.d_b(fake_b)
        loss_g = self.criterion.generator(
            real_a,
            real_b,
            fake_a,
            fake_b,
            rec_a,
            rec_b,
            idt_a,
            idt_b,
            disc_fake_a,
            disc_fake_b,
        )
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # --- Discriminator update ---
        self.toggle_optimizer(opt_d)
        disc_real_a = self.d_a(real_a)
        disc_fake_a_det = self.d_a(fake_a.detach())
        disc_real_b = self.d_b(real_b)
        disc_fake_b_det = self.d_b(fake_b.detach())
        loss_d = self.criterion.discriminator(
            disc_real_a,
            disc_fake_a_det,
            disc_real_b,
            disc_fake_b_det,
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

    def test_step(self, batch: dict, batch_idx: int) -> None:
        _ = self.g_ab(batch["A"])

    def configure_optimizers(self):
        lr = self.cfg.train.lr
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
        return [opt_g, opt_d]
