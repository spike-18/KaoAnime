# kaoanime/model.py
from __future__ import annotations

import lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from kaoanime.config import Config


class DummyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        self.generator = DummyGenerator()

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
