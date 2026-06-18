# MLflow Model Logging + Inference CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log the trained checkpoint as an MLflow artifact after training completes, and add a standalone inference CLI that loads a checkpoint and translates a single image or directory.

**Architecture:** Two independent additions: (1) a `ModelCheckpoint` callback in `train.py` that saves `last.ckpt`, then logs it to MLflow via `log_artifact` after `trainer.fit()` returns; (2) a `kaoanime/inference/` module with a `run_inference()` function that accepts a loaded model, input path (file or dir), and output dir, plus a `infer.py` Hydra CLI entry point that mirrors the pattern of `eval.py`.

**Tech Stack:** PyTorch Lightning, MLflow, Hydra, torchvision transforms. Always invoke Python as `uv run python`.

---

## File Map

| File                             | Change                                                              |
| -------------------------------- | ------------------------------------------------------------------- |
| `train.py`                       | Add `ModelCheckpoint` callback; log `last.ckpt` to MLflow after fit |
| `kaoanime/config.py`             | Add `input: str` and `direction: str` to `EvalConfig`               |
| `kaoanime/inference/__init__.py` | New: `run_inference()` function                                     |
| `infer.py`                       | New: Hydra CLI entry point for inference                            |
| `tests/test_inference.py`        | New: tests for `run_inference`                                      |
| `tests/test_model.py`            | Add checkpoint-save smoke test                                      |

---

## Task 1: Save final checkpoint and log to MLflow

**Files:**

- Modify: `train.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1.1: Write the failing checkpoint test**

Add to `tests/test_model.py`:

```python
from lightning.pytorch.callbacks import ModelCheckpoint


def test_last_checkpoint_is_saved(tmp_path):
    ckpt_cb = ModelCheckpoint(
        dirpath=tmp_path,
        filename="last",
        save_last=False,
        save_top_k=1,
        monitor=None,
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
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
uv run pytest tests/test_model.py::test_last_checkpoint_is_saved -v
```

Expected: FAIL — `ModelCheckpoint` is not yet imported in `test_model.py` (or passes trivially if Lightning saves by default; either way this verifies the scaffolding).

- [ ] **Step 1.3: Update `train.py` to add checkpoint callback and artifact logging**

Replace the entire `train.py` with:

```python
# train.py
import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from kaoanime.config import Config, register_configs
from kaoanime.model import KaoAnimeModel
from kaoanime.utils import UnpairedImageDataset, create_dataloader

torch.set_float32_matmul_precision("medium")
register_configs()


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config) -> None:
    dataset = UnpairedImageDataset(
        cfg.data.root_a,
        cfg.data.root_b,
        cfg.data.image_size,
        extra_roots_a=list(cfg.data.extra_roots_a),
        extra_roots_b=list(cfg.data.extra_roots_b),
    )

    train_dl = create_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = KaoAnimeModel(cfg)

    logger = MLFlowLogger(
        experiment_name="kaoanime",
        tracking_uri=cfg.train.mlflow_tracking_uri,
    )

    ckpt_callback = ModelCheckpoint(
        filename="epoch{epoch:03d}",
        save_last=True,
        save_top_k=0,
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="auto",
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=logger,
        callbacks=[ckpt_callback],
    )
    trainer.fit(model, train_dl)

    if ckpt_callback.last_model_path:
        logger.experiment.log_artifact(logger.run_id, ckpt_callback.last_model_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
uv run pytest tests/test_model.py::test_last_checkpoint_is_saved -v
```

Expected: PASS

- [ ] **Step 1.5: Commit**

```bash
git add train.py tests/test_model.py
git commit -m "feat: save last checkpoint and log to MLflow artifact store after training"
```

---

## Task 2: Inference pipeline module

**Files:**

- Create: `kaoanime/inference/__init__.py`
- Modify: `kaoanime/config.py`
- Create: `tests/test_inference.py`

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_inference.py`:

```python
from pathlib import Path

import torch
from PIL import Image

from kaoanime.config import Config
from kaoanime.inference import run_inference
from kaoanime.model import KaoAnimeModel


def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (128, 128), color=(i * 30, 100, 200)).save(directory / f"{i}.jpg")


def _cpu_model() -> KaoAnimeModel:
    model = KaoAnimeModel(Config())
    model.eval()
    return model


def test_infer_single_image(tmp_path):
    src = tmp_path / "input.jpg"
    Image.new("RGB", (128, 128)).save(src)
    out_dir = tmp_path / "out"

    run_inference(_cpu_model(), src, out_dir, image_size=128, direction="a2b", device="cpu")

    assert (out_dir / "input.jpg").exists()


def test_infer_directory(tmp_path):
    src_dir = tmp_path / "inputs"
    _make_images(src_dir, 3)
    out_dir = tmp_path / "out"

    run_inference(_cpu_model(), src_dir, out_dir, image_size=128, direction="a2b", device="cpu")

    assert len(list(out_dir.glob("*.jpg"))) == 3


def test_infer_b2a_direction(tmp_path):
    src = tmp_path / "anime.jpg"
    Image.new("RGB", (128, 128)).save(src)
    out_dir = tmp_path / "out"

    run_inference(_cpu_model(), src, out_dir, image_size=128, direction="b2a", device="cpu")

    assert (out_dir / "anime.jpg").exists()


def test_infer_output_filenames_match_input(tmp_path):
    src_dir = tmp_path / "inputs"
    src_dir.mkdir()
    Image.new("RGB", (64, 64)).save(src_dir / "face_001.png")
    Image.new("RGB", (64, 64)).save(src_dir / "face_002.png")
    out_dir = tmp_path / "out"

    run_inference(_cpu_model(), src_dir, out_dir, image_size=128, direction="a2b", device="cpu")

    assert (out_dir / "face_001.png").exists()
    assert (out_dir / "face_002.png").exists()
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
uv run pytest tests/test_inference.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'kaoanime.inference'`

- [ ] **Step 2.3: Add `input` and `direction` to `EvalConfig` in `kaoanime/config.py`**

Replace the `EvalConfig` dataclass:

```python
@dataclass
class EvalConfig:
    checkpoint: str = ""
    output_dir: str = "outputs/eval"
    input: str = ""
    direction: str = "a2b"
```

- [ ] **Step 2.4: Create `kaoanime/inference/__init__.py`**

```python
from __future__ import annotations

from pathlib import Path

import torch

from kaoanime.utils import load_image, save_image, get_transforms

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_inference(
    model,
    input_path: str | Path,
    output_dir: str | Path,
    image_size: int = 128,
    direction: str = "a2b",
    device: str = "cuda",
) -> list[Path]:
    """Translate images from domain A→B (or B→A) and save to output_dir.

    Args:
        model: Loaded KaoAnimeModel instance (already on correct device).
        input_path: Path to a single image file or a directory of images.
        output_dir: Directory where translated images are written.
        image_size: Size used for center-crop resize preprocessing.
        direction: "a2b" uses g_ab (selfie→anime); "b2a" uses g_ba (anime→selfie).
        device: Torch device string, e.g. "cuda" or "cpu".

    Returns:
        List of output file paths that were written.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transforms("test", image_size=image_size)
    generator = model.g_ab if direction == "a2b" else model.g_ba

    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)

    written: list[Path] = []
    with torch.no_grad():
        for p in paths:
            img = load_image(p)
            tensor = transform(img).unsqueeze(0).to(device)
            out = generator(tensor.float())[0]
            dst = output_dir / p.name
            save_image(out, dst)
            written.append(dst)

    return written
```

- [ ] **Step 2.5: Run tests to verify they pass**

```bash
uv run pytest tests/test_inference.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 2.6: Commit**

```bash
git add kaoanime/config.py kaoanime/inference/__init__.py tests/test_inference.py
git commit -m "feat: add inference pipeline module and extend EvalConfig with input/direction fields"
```

---

## Task 3: Inference CLI entry point

**Files:**

- Create: `infer.py`

- [ ] **Step 3.1: Create `infer.py`**

```python
# infer.py
import torch
import hydra

from kaoanime.config import Config, register_configs
from kaoanime.inference import run_inference
from kaoanime.model import KaoAnimeModel

register_configs()


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config) -> None:
    if not cfg.eval.checkpoint:
        raise ValueError("eval.checkpoint must be specified, e.g.: eval.checkpoint=outputs/.../last.ckpt")
    if not cfg.eval.input:
        raise ValueError("eval.input must be specified, e.g.: eval.input=data/selfie2anime/testA")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KaoAnimeModel.load_from_checkpoint(cfg.eval.checkpoint, cfg=cfg, map_location=device)
    model.to(device).float()
    model.eval()

    written = run_inference(
        model,
        input_path=cfg.eval.input,
        output_dir=cfg.eval.output_dir,
        image_size=cfg.data.image_size,
        direction=cfg.eval.direction,
        device=device,
    )
    print(f"Saved {len(written)} image(s) to {cfg.eval.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Verify the CLI prints usage error when required args are missing**

```bash
uv run python infer.py 2>&1 | grep -E "ValueError|eval.checkpoint"
```

Expected: `ValueError: eval.checkpoint must be specified`

- [ ] **Step 3.3: Commit**

```bash
git add infer.py
git commit -m "feat: add inference CLI (infer.py) — load checkpoint, translate image or directory"
```

---

## Usage Examples

After completing all tasks:

**Translate a directory of selfies to anime:**

```bash
uv run python infer.py \
  eval.checkpoint=outputs/checkpoints/last.ckpt \
  eval.input=data/selfie2anime/testA \
  eval.output_dir=outputs/infer/testA_anime
```

**Translate a single image:**

```bash
uv run python infer.py \
  eval.checkpoint=outputs/checkpoints/last.ckpt \
  eval.input=my_photo.jpg \
  eval.output_dir=outputs/infer
```

**Reverse direction (anime → selfie):**

```bash
uv run python infer.py \
  eval.checkpoint=outputs/checkpoints/last.ckpt \
  eval.input=data/selfie2anime/testB \
  eval.output_dir=outputs/infer/testB_real \
  eval.direction=b2a
```

## Notes

- The checkpoint logged to MLflow is `last.ckpt` (final training epoch), not a "best" checkpoint — CycleGAN losses don't monotonically improve, so monitoring a single loss metric for "best" isn't meaningful.
- `run_inference` calls `.float()` on the input tensor because checkpoints trained with AMP (`16-mixed`) have fp16 generator weights in the computation graph; casting to fp32 ensures clean inference without an AMP context.
- `load_from_checkpoint(..., cfg=cfg)` passes the config explicitly because `KaoAnimeModel.__init__` requires a `Config` dataclass, but `save_hyperparameters` serialises it as a plain dict in the checkpoint — Lightning cannot auto-reconstruct the dataclass on load.
