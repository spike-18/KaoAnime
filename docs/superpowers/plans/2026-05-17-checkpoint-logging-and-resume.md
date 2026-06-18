# Checkpoint Logging and Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log the model checkpoint to MLflow during training (not after `trainer.fit()` returns) so early-stopped runs have their checkpoint archived; and add a `resume_from_checkpoint` config field so training can be resumed from any saved `.ckpt` file.

**Architecture:** A new `MLflowCheckpointCallback` subclasses `ModelCheckpoint` and overrides `on_train_epoch_end` and `on_train_end` to call `log_artifact` immediately after each save — Lightning calls both hooks during graceful interrupts (Ctrl+C), so the checkpoint is always uploaded. The post-`fit()` artifact logging is removed from `train.py`. Resume is wired by reading `cfg.train.resume_from_checkpoint` / `cfg.not_.resume_from_checkpoint` and passing it as `ckpt_path` to `trainer.fit()`.

**Tech Stack:** PyTorch Lightning 2.6.1, MLflow, Python, pytest. Always invoke Python as `uv run python`, pytest as `uv run pytest`.

---

## File Map

| File | Change |
|------|--------|
| `kaoanime/callbacks.py` | Create: `MLflowCheckpointCallback` |
| `kaoanime/config_cyclegan.py` | Add `resume_from_checkpoint: str = ""` to `CycleGANTrainConfig` |
| `kaoanime/config_not.py` | Add `resume_from_checkpoint: str = ""` to `NOTConfig` |
| `train.py` | Use `MLflowCheckpointCallback`; remove post-fit logging; pass `ckpt_path` to `trainer.fit()` |
| `tests/test_callbacks.py` | Create: unit tests for `MLflowCheckpointCallback._log_last_to_mlflow` |
| `tests/test_model.py` | Add: `test_resume_from_checkpoint` |

---

## Task 1 — `MLflowCheckpointCallback` and its tests

**Files:**
- Create: `kaoanime/callbacks.py`
- Create: `tests/test_callbacks.py`

- [ ] **Step 1.1: Write failing tests**

Create `tests/test_callbacks.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from lightning.pytorch.loggers import MLFlowLogger

from kaoanime.callbacks import MLflowCheckpointCallback


def test_logs_artifact_when_last_path_set(tmp_path):
    cb = MLflowCheckpointCallback(dirpath=tmp_path, save_last=True, save_top_k=0)
    ckpt = tmp_path / "last.ckpt"
    ckpt.touch()
    cb.last_model_path = str(ckpt)

    mock_logger = MagicMock(spec=MLFlowLogger)
    mock_logger.run_id = "run-abc"
    mock_trainer = MagicMock()
    mock_trainer.logger = mock_logger

    cb._log_last_to_mlflow(mock_trainer)

    mock_logger.experiment.log_artifact.assert_called_once_with("run-abc", str(ckpt))


def test_does_not_log_when_no_checkpoint(tmp_path):
    cb = MLflowCheckpointCallback(dirpath=tmp_path, save_last=True, save_top_k=0)
    cb.last_model_path = ""

    mock_logger = MagicMock(spec=MLFlowLogger)
    mock_trainer = MagicMock()
    mock_trainer.logger = mock_logger

    cb._log_last_to_mlflow(mock_trainer)

    mock_logger.experiment.log_artifact.assert_not_called()


def test_does_not_log_when_logger_not_mlflow(tmp_path):
    cb = MLflowCheckpointCallback(dirpath=tmp_path, save_last=True, save_top_k=0)
    ckpt = tmp_path / "last.ckpt"
    ckpt.touch()
    cb.last_model_path = str(ckpt)

    mock_trainer = MagicMock()
    mock_trainer.logger = MagicMock()  # generic logger, not MLFlowLogger

    cb._log_last_to_mlflow(mock_trainer)  # must not raise or call log_artifact


def test_is_subclass_of_model_checkpoint():
    from lightning.pytorch.callbacks import ModelCheckpoint
    assert issubclass(MLflowCheckpointCallback, ModelCheckpoint)
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_callbacks.py -v
```

Expected: all 4 FAIL with `ModuleNotFoundError: No module named 'kaoanime.callbacks'`.

- [ ] **Step 1.3: Create `kaoanime/callbacks.py`**

```python
from __future__ import annotations

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger


class MLflowCheckpointCallback(ModelCheckpoint):
    """ModelCheckpoint that uploads last.ckpt to MLflow after each save.

    Overrides on_train_epoch_end and on_train_end so the checkpoint is
    archived during training. Lightning calls both hooks on graceful
    interrupts (Ctrl+C), ensuring the checkpoint is logged even when
    training is stopped early.
    """

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self._log_last_to_mlflow(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        super().on_train_end(trainer, pl_module)
        self._log_last_to_mlflow(trainer)

    def _log_last_to_mlflow(self, trainer) -> None:
        path = self.last_model_path
        if path and isinstance(trainer.logger, MLFlowLogger):
            trainer.logger.experiment.log_artifact(trainer.logger.run_id, path)
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_callbacks.py -v
```

Expected: all 4 PASS.

- [ ] **Step 1.5: Commit**

```bash
git add kaoanime/callbacks.py tests/test_callbacks.py
git commit -m "feat: MLflowCheckpointCallback logs checkpoint to MLflow during training"
```

---

## Task 2 — Config fields, train.py wiring, and resume test

**Files:**
- Modify: `kaoanime/config_cyclegan.py`
- Modify: `kaoanime/config_not.py`
- Modify: `train.py`
- Modify: `tests/test_model.py`

- [ ] **Step 2.1: Write failing resume test**

Append to `tests/test_model.py`:

```python
from pathlib import Path


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
```

- [ ] **Step 2.2: Run test to confirm it fails**

```bash
uv run pytest tests/test_model.py::test_resume_from_checkpoint -v
```

Expected: FAIL (the test itself is valid; we haven't changed the application yet, but the test should pass since `trainer.fit(..., ckpt_path=path)` is already a Lightning feature). Actually it may PASS already — if so, note it and continue. The config-field test below will fail until config is updated.

- [ ] **Step 2.3: Add `resume_from_checkpoint` to `kaoanime/config_cyclegan.py`**

Add one field to `CycleGANTrainConfig` (after the existing fields):

```python
    resume_from_checkpoint: str = ""
```

Full `CycleGANTrainConfig` after the change:

```python
@dataclass
class CycleGANTrainConfig:
    max_epochs: int = 100
    lr: float = 1e-3
    lr_decay_start_epoch: int = 30
    alpha: float = 0.993
    precision: str = "16-mixed"
    log_every_n_steps: int = 50
    mlflow_tracking_uri: str = "http://10.0.111.233:9999"
    gen_steps: int = 1
    disc_steps: int = 5
    log_image_every_n_steps: int = 5000
    fid_every_n_steps: int = 2000
    fid_num_images: int = 512
    resume_from_checkpoint: str = ""
```

- [ ] **Step 2.4: Add `resume_from_checkpoint` to `kaoanime/config_not.py`**

Add one field to `NOTConfig` (after the existing fields):

```python
    resume_from_checkpoint: str = ""
```

Full `NOTConfig` after the change:

```python
@dataclass
class NOTConfig:
    t_iters: int = 5
    t_lr: float = 3e-4
    f_lr: float = 3e-4
    t_filters: int = 64
    f_filters: int = 64
    max_steps: int = 200001
    precision: str = "16-mixed"
    log_every_n_steps: int = 50
    mlflow_tracking_uri: str = "http://10.0.111.233:9999"
    log_image_every_n_steps: int = 500
    fid_every_n_steps: int = 1000
    fid_num_images: int = 512
    resume_from_checkpoint: str = ""
```

- [ ] **Step 2.5: Update `train.py`**

Replace the entire `train.py` with:

```python
# train.py
import hydra
import lightning as pl
import torch
from lightning.pytorch.loggers import MLFlowLogger

from kaoanime.callbacks import MLflowCheckpointCallback
from kaoanime.config import Config, register_configs
from kaoanime.model_cyclegan import KaoAnimeModel
from kaoanime.model_not import NOTModel
from kaoanime.utils import UnpairedImageDataset, create_dataloader

torch.set_float32_matmul_precision("medium")
register_configs()


def _make_model(cfg: Config) -> pl.LightningModule:
    if cfg.model_type == "cyclegan":
        return KaoAnimeModel(cfg)
    if cfg.model_type == "not":
        return NOTModel(cfg)
    raise ValueError(f"Unknown model_type {cfg.model_type!r}. Choose 'cyclegan' or 'not'.")


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config) -> None:
    dataset = UnpairedImageDataset(
        cfg.data.root_a,
        cfg.data.root_b,
        cfg.data.image_size,
        extra_roots_a=list(cfg.data.extra_roots_a),
        extra_roots_b=list(cfg.data.extra_roots_b),
        align_a=cfg.data.align_a,
        anime_offset_x=cfg.data.anime_offset_x,
        anime_offset_y=cfg.data.anime_offset_y,
    )

    train_dl = create_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = _make_model(cfg)

    if cfg.model_type == "not":
        logger = MLFlowLogger(
            experiment_name="kaoanime-not",
            tracking_uri=cfg.not_.mlflow_tracking_uri,
        )
        _lightning_max_steps = cfg.not_.max_steps * (cfg.not_.t_iters + 1)
        trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            max_steps=_lightning_max_steps,
            max_epochs=-1,
            precision=cfg.not_.precision,
            log_every_n_steps=cfg.not_.log_every_n_steps,
            logger=logger,
            callbacks=[MLflowCheckpointCallback(filename="step{step:06d}", save_last=True, save_top_k=0)],
        )
        ckpt_path = cfg.not_.resume_from_checkpoint or None
    else:
        logger = MLFlowLogger(
            experiment_name="kaoanime",
            tracking_uri=cfg.train.mlflow_tracking_uri,
        )
        trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            max_epochs=cfg.train.max_epochs,
            precision=cfg.train.precision,
            log_every_n_steps=cfg.train.log_every_n_steps,
            logger=logger,
            callbacks=[MLflowCheckpointCallback(filename="epoch{epoch:03d}", save_last=True, save_top_k=0)],
        )
        ckpt_path = cfg.train.resume_from_checkpoint or None

    trainer.fit(model, train_dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
```

Key changes from the original:
1. `from kaoanime.callbacks import MLflowCheckpointCallback` added, `ModelCheckpoint` import removed
2. Both `ModelCheckpoint(...)` calls replaced with `MLflowCheckpointCallback(...)`
3. `ckpt_path` read from config (`cfg.not_.resume_from_checkpoint` or `cfg.train.resume_from_checkpoint`), empty string converted to `None`
4. `trainer.fit(model, train_dl, ckpt_path=ckpt_path)` replaces the old call
5. The 3 post-fit lines (`ckpt_path = trainer.checkpoint_callback...`, `if ckpt_path:`, `logger.experiment.log_artifact(...)`) are removed

- [ ] **Step 2.6: Verify config fields exist**

```bash
uv run python -c "
from kaoanime.config import Config
cfg = Config()
print('cyclegan resume_from_checkpoint:', repr(cfg.train.resume_from_checkpoint))
print('not resume_from_checkpoint:', repr(cfg.not_.resume_from_checkpoint))
assert cfg.train.resume_from_checkpoint == ''
assert cfg.not_.resume_from_checkpoint == ''
print('Config OK')
"
```

Expected:
```
cyclegan resume_from_checkpoint: ''
not resume_from_checkpoint: ''
Config OK
```

- [ ] **Step 2.7: Run the resume test**

```bash
uv run pytest tests/test_model.py::test_resume_from_checkpoint -v
```

Expected: PASS.

- [ ] **Step 2.8: Run full test suite**

```bash
uv run pytest tests/test_callbacks.py tests/test_model.py -v
```

Expected: all callback tests pass; all model tests pass (the pre-existing failures in `test_config_defaults`, `test_cyclegan_loss.py`, `test_cyclegan_model.py` are unrelated to this change and are expected until the CycleGAN training fixes are implemented).

- [ ] **Step 2.9: Commit**

```bash
git add kaoanime/config_cyclegan.py kaoanime/config_not.py train.py tests/test_model.py
git commit -m "feat: wire MLflowCheckpointCallback and resume_from_checkpoint config into train.py"
```

---

## Usage after this plan is complete

**Resume training (CycleGAN):**
```bash
uv run python -m kaoanime.train train.resume_from_checkpoint=/path/to/last.ckpt
```

**Resume training (NOT):**
```bash
uv run python -m kaoanime.train not_.resume_from_checkpoint=/path/to/last.ckpt
```

The checkpoint path is the local `.ckpt` file. Download it from MLflow artifacts first if needed.

---

## Self-review

**Spec coverage:**
- ✅ Checkpoint logged during training (not after): `MLflowCheckpointCallback.on_train_epoch_end` + `on_train_end` — Task 1
- ✅ Logged even on early stop: `on_train_end` is called by Lightning on graceful Ctrl+C — Task 1
- ✅ Post-fit artifact logging removed: Task 2 step 2.5
- ✅ Resume from checkpoint — CycleGAN: `cfg.train.resume_from_checkpoint` — Task 2
- ✅ Resume from checkpoint — NOT: `cfg.not_.resume_from_checkpoint` — Task 2

**Placeholder scan:** No TBDs. All code blocks are complete.

**Type consistency:** `MLflowCheckpointCallback` used as drop-in for `ModelCheckpoint` everywhere. `ckpt_path: str | None` passed directly to `trainer.fit()`. `resume_from_checkpoint: str = ""` converted to `None` with `or None` before passing to Lightning.
