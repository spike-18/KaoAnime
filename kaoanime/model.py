from __future__ import annotations

import lightning as pl
import torch
from omegaconf import OmegaConf

from kaoanime.config import Config
from kaoanime.models import UNetGenerator


class KaoAnimeModel(pl.LightningModule):
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
        self.generator = UNetGenerator(
            num_filters=cfg.model.num_filters,
            num_down=cfg.model.num_down,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        fake_b = self.generator(batch["A"])
        loss = fake_b.abs().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> None:
        _ = self.generator(batch["A"])

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)
