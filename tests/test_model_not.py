import dataclasses
from unittest.mock import MagicMock

import torch
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from kaoanime.config import Config
from kaoanime.model_not import NOTModel


class _PairedDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 12

    def __getitem__(self, idx: int) -> dict:
        return {"A": torch.randn(3, 32, 32), "B": torch.randn(3, 32, 32)}


def _tiny_cfg() -> Config:
    base = Config()
    not_ = dataclasses.replace(
        base.not_,
        t_iters=2,
        t_filters=8,
        f_filters=8,
        fid_every_n_steps=3,
        fid_num_images=2,
        log_image_every_n_steps=10_000,
        log_every_n_steps=1,
    )
    return dataclasses.replace(base, not_=not_)


def test_fid_logged_on_global_step_axis(tmp_path):
    """FID must be logged at trainer.global_step (the loss axis), not the
    per-batch counter. With t_iters=2 -> 3 optimizer steps/batch, FID computed
    at batches 3 and 6 must land at global_step 9 and 18."""
    cfg = _tiny_cfg()
    model = NOTModel(cfg)

    # Mock the Inception-based FID so the test is offline and fast; we only
    # care about the scheduling + the step value passed to log_metric.
    model.fid.update = MagicMock()
    model.fid.reset = MagicMock()
    model.fid.compute = MagicMock(return_value=torch.tensor(0.5))

    logger = MLFlowLogger(
        experiment_name="not-test", tracking_uri=f"file:{tmp_path}/mlruns"
    )
    loader = DataLoader(_PairedDataset(), batch_size=2, num_workers=0)
    trainer = Trainer(
        max_steps=(cfg.not_.t_iters + 1) * 6,  # exactly 6 batches
        max_epochs=-1,
        accelerator="cpu",
        precision="32-true",
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=loader)

    client = MlflowClient(tracking_uri=f"file:{tmp_path}/mlruns")
    history = client.get_metric_history(logger.run_id, "val/fid")
    steps = sorted(m.step for m in history)

    assert steps == [9, 18], (
        f"FID should be logged on the global_step axis [9, 18]; got {steps}. "
        "If this is [3, 6] the per-batch counter is still being used."
    )
