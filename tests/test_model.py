import pytest
import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from kaoanime.config import Config
from kaoanime.model import KaoAnimeModel


def test_model_forward_pass():
    """Generator g_ab produces correct output shape."""
    model = KaoAnimeModel(Config())
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 3, 128, 128)


def _make_fake_loader() -> DataLoader:
    """Single-batch dataloader that yields {A, B} dicts."""

    class PairedDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int) -> dict:
            return {"A": torch.randn(3, 128, 128), "B": torch.randn(3, 128, 128)}

    return DataLoader(PairedDataset(), batch_size=2)


def test_model_training_step_runs():
    """Full training step (G + D update) completes without error."""
    model = KaoAnimeModel(Config())
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=_make_fake_loader())


def test_config_defaults():
    cfg = Config()
    assert cfg.train.precision == "16-mixed"
    assert cfg.train.max_epochs == 200
    assert cfg.train.lr == pytest.approx(2e-4)
    assert cfg.train.beta1 == pytest.approx(0.5)
    assert cfg.data.batch_size == 1
    assert cfg.data.image_size == 128
    assert cfg.eval.output_dir == "outputs/eval"
    assert cfg.model.num_filters == 64
    assert cfg.model.num_residual_blocks == 9
    assert cfg.model.lambda_cycle == pytest.approx(10.0)
    assert cfg.model.lambda_identity == pytest.approx(5.0)
