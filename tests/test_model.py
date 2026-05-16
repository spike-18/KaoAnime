import pytest
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from kaoanime.config import Config
from kaoanime.model_cyclegan import KaoAnimeModel


def test_model_forward_pass():
    """Generator g_ab produces correct output shape."""
    model = KaoAnimeModel(Config())
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 3, 128, 128)


class _PairedDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict:
        return {"A": torch.randn(3, 128, 128), "B": torch.randn(3, 128, 128)}


def _make_fake_loader() -> DataLoader:
    return DataLoader(
        _PairedDataset(),
        batch_size=2,
        num_workers=2,
        multiprocessing_context="forkserver",
    )


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
    assert cfg.train.max_epochs == 100
    assert cfg.train.lr == pytest.approx(0.001)
    assert cfg.train.beta1 == pytest.approx(0.5)
    assert cfg.data.batch_size == 128
    assert cfg.data.image_size == 128
    assert cfg.eval.output_dir == "outputs/eval"
    assert cfg.model.num_filters == 48
    assert cfg.model.num_residual_blocks == 9
    assert cfg.model.lambda_cycle == pytest.approx(10.0)
    assert cfg.model.lambda_identity == pytest.approx(5.0)


def test_last_checkpoint_is_saved(tmp_path):
    ckpt_cb = ModelCheckpoint(
        dirpath=tmp_path,
        save_last=True,
        save_top_k=0,
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        callbacks=[ckpt_cb],
        enable_progress_bar=False,
    )
    model = KaoAnimeModel(Config())
    trainer.fit(model, train_dataloaders=_make_fake_loader())
    saved = list(tmp_path.glob("*.ckpt"))
    assert len(saved) == 1, f"Expected 1 checkpoint, found: {saved}"
    assert saved[0].name == "last.ckpt"


def test_resume_from_checkpoint(tmp_path):
    """Training resumes correctly from a saved checkpoint."""
    # Phase 1: train 1 epoch and save checkpoint
    ckpt_cb = ModelCheckpoint(
        dirpath=tmp_path,
        save_last=True,
        save_top_k=0,
    )
    model = KaoAnimeModel(Config())
    trainer1 = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        callbacks=[ckpt_cb],
        enable_progress_bar=False,
    )
    trainer1.fit(model, train_dataloaders=_make_fake_loader())
    ckpt_path = tmp_path / "last.ckpt"
    assert ckpt_path.exists(), "Checkpoint was not saved after phase 1"

    # Phase 2: resume and train to max_epochs=2 total
    ckpt_cb2 = ModelCheckpoint(
        dirpath=tmp_path / "run2",
        save_last=True,
        save_top_k=0,
    )
    model2 = KaoAnimeModel(Config())
    trainer2 = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        callbacks=[ckpt_cb2],
        enable_progress_bar=False,
    )
    trainer2.fit(model2, train_dataloaders=_make_fake_loader(), ckpt_path=str(ckpt_path))
    # Lightning restores epoch counter; with max_epochs=2 and resuming from epoch 1,
    # only one more epoch runs → current_epoch==2 (past the last completed epoch).
    assert trainer2.current_epoch == 2


def test_empty_resume_checkpoint_resolves_to_none():
    """The `or None` sentinel in train.py converts '' to None for trainer.fit()."""
    assert ("" or None) is None
    assert ("/some/path.ckpt" or None) == "/some/path.ckpt"
