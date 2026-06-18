# DVC Data & Model Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Version datasets and models with DVC on two Google Drive remotes, ship a small DVC-tracked demo dataset, and wire data fetching into `train`/`infer`.

**Architecture:** A `variant` switch (`demo`/`full`) selects repo-relative data paths via OmegaConf interpolation. `kaoanime/data/` provides `download_data()` (demo → `dvc pull`; full → Kaggle + gdown) and `ensure_data()` (called by `train.py`/`infer.py`). `scripts/prepare_dataset.py` builds the `data/<variant>/{trainA,trainB,testA,testB}` layout (and copies `list_attr_celeba.csv` so the existing women-only filter works). `scripts/export_models.py` pushes selected checkpoints to the `models` remote.

**Tech Stack:** DVC (`dvc[gdrive]`), `gdown`, `kaggle`, `fire`, Hydra/OmegaConf, PyTorch, pytest. Always invoke Python via `uv run python`.

---

## File Structure

| File                            | Responsibility                                             |
| ------------------------------- | ---------------------------------------------------------- |
| `pyproject.toml`                | Add `dvc[gdrive]`, `gdown`, `kaggle`, `fire` deps          |
| `.dvc/config`                   | Two gdrive remotes (`data` default, `models`)              |
| `kaoanime/config.py`            | `DataConfig.variant` + interpolated repo-relative paths    |
| `kaoanime/data/__init__.py`     | Package exports (`download_data`, `ensure_data`)           |
| `kaoanime/data/layout.py`       | `required_data_dirs(cfg)` — dirs that must exist           |
| `kaoanime/data/download.py`     | `download_data(variant, ...)` — demo/full backends         |
| `kaoanime/data/ensure.py`       | `ensure_data(cfg)` — download only when dirs missing       |
| `scripts/prepare_dataset.py`    | fire CLI: build `data/<variant>/...` + copy attr CSV       |
| `scripts/export_models.py`      | fire CLI: stage selected `.pt`, `dvc add`+`push -r models` |
| `train.py`, `infer.py`          | Call `ensure_data(cfg)` at start                           |
| `tests/test_data_layout.py`     | `required_data_dirs` + interpolated paths                  |
| `tests/test_prepare_dataset.py` | demo sampling determinism, women-only A, CSV copy          |
| `tests/test_download.py`        | `download_data` branches (mocked externals)                |
| `tests/test_ensure_data.py`     | `ensure_data` calls downloader only when missing           |
| `tests/test_export_models.py`   | checkpoint staging + dvc calls (mocked)                    |

ONNX export is **out of scope here** (handled by the separate model-packaging plan); `export_models.py` ships the `.pt` + DVC push now and the packaging plan extends it with ONNX.

---

## Task 1: Add dependencies

**Files:**

- Modify: `pyproject.toml`

- [ ] **Step 1: Add the runtime deps**

In `pyproject.toml`, under `[project] dependencies`, add these four entries (keep the list alphabetical-ish, matching existing style):

```toml
    "dvc[gdrive]>=3.0",
    "fire>=0.6",
    "gdown>=5.0",
    "kaggle>=1.6",
```

- [ ] **Step 2: Sync and verify imports**

Run:

```bash
uv sync
uv run python -c "import dvc, gdown, kaggle, fire; print('deps OK')"
```

Expected: `deps OK` (after install). If `kaggle` import warns about missing credentials, that is fine — it only matters at download time.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add dvc[gdrive], gdown, kaggle, fire for data management"
```

---

## Task 2: Configure DVC remotes

**Files:**

- Modify: `.dvc/config`

- [ ] **Step 1: Write the two-remote config**

Replace the contents of `.dvc/config` with:

```ini
[core]
    remote = data
['remote "data"']
    url = gdrive://158t5Cy5bHckNjxzbXI5D5G1EI7Nh27fy
['remote "models"']
    url = gdrive://16RC8Zc2cnDvc2Dh46CFoy9iEzPZohFGF
```

- [ ] **Step 2: Verify DVC sees both remotes**

Run:

```bash
uv run dvc remote list
```

Expected output:

```
data	gdrive://158t5Cy5bHckNjxzbXI5D5G1EI7Nh27fy
models	gdrive://16RC8Zc2cnDvc2Dh46CFoy9iEzPZohFGF
```

- [ ] **Step 3: Commit**

```bash
git add .dvc/config
git commit -m "feat(dvc): configure data and models gdrive remotes"
```

---

## Task 3: Add `variant` and interpolated data paths to config

**Files:**

- Modify: `kaoanime/config.py`
- Test: `tests/test_data_layout.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_layout.py`:

```python
from omegaconf import OmegaConf

from kaoanime.config import Config


def test_demo_variant_paths_resolve():
    cfg = OmegaConf.structured(Config)
    assert cfg.data.variant == "demo"
    assert cfg.data.root_a == "data/demo/trainA"
    assert cfg.data.root_b == "data/demo/trainB"
    assert cfg.data.test_a == "data/demo/testA"
    assert cfg.data.test_b == "data/demo/testB"


def test_full_variant_paths_resolve():
    cfg = OmegaConf.structured(Config)
    cfg.data.variant = "full"
    assert cfg.data.root_a == "data/full/trainA"
    assert cfg.data.test_b == "data/full/testB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_layout.py -v`
Expected: FAIL — current `root_a` default is the hardcoded `/beta/...` path.

- [ ] **Step 3: Implement the config change**

In `kaoanime/config.py`, replace the `DataConfig` path fields. The full `DataConfig` becomes:

```python
@dataclass
class DataConfig:
    variant: str = "demo"  # "demo" (DVC-tracked sample) or "full" (downloaded)
    root_a: str = "data/${data.variant}/trainA"
    root_b: str = "data/${data.variant}/trainB"
    test_a: str = "data/${data.variant}/testA"
    test_b: str = "data/${data.variant}/testB"
    extra_roots_a: list[str] = field(default_factory=list)
    extra_roots_b: list[str] = field(default_factory=list)
    batch_size: int = 64
    image_size: int = 128
    num_workers: int = 12
    pin_memory: bool = True
    align_a: bool = False
    # Fixed crop transform for domain B (anime). Pixel offsets from center (256, 256).
    anime_offset_x: int = 0
    anime_offset_y: int = -7
```

(The `${data.variant}` interpolation resolves against the root config, so changing `data.variant` updates all four paths.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data_layout.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add kaoanime/config.py tests/test_data_layout.py
git commit -m "feat(config): variant switch with repo-relative interpolated data paths"
```

---

## Task 4: `required_data_dirs` helper

**Files:**

- Create: `kaoanime/data/__init__.py`
- Create: `kaoanime/data/layout.py`
- Test: `tests/test_data_layout.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_layout.py`:

```python
from kaoanime.data import required_data_dirs


def test_required_data_dirs_lists_four_split_dirs():
    cfg = OmegaConf.structured(Config)
    dirs = required_data_dirs(cfg)
    assert [str(d) for d in dirs] == [
        "data/demo/trainA",
        "data/demo/trainB",
        "data/demo/testA",
        "data/demo/testB",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_layout.py::test_required_data_dirs_lists_four_split_dirs -v`
Expected: FAIL — `kaoanime.data` does not exist.

- [ ] **Step 3: Implement**

Create `kaoanime/data/layout.py`:

```python
from __future__ import annotations

from pathlib import Path


def required_data_dirs(cfg) -> list[Path]:
    """The four split directories that must exist before training/inference."""
    data = cfg.data
    return [Path(data.root_a), Path(data.root_b), Path(data.test_a), Path(data.test_b)]
```

Create `kaoanime/data/__init__.py`:

```python
from kaoanime.data.layout import required_data_dirs

__all__ = ["required_data_dirs"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data_layout.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add kaoanime/data/__init__.py kaoanime/data/layout.py tests/test_data_layout.py
git commit -m "feat(data): required_data_dirs helper"
```

---

## Task 5: `scripts/prepare_dataset.py` — demo sampler + full layout

**Files:**

- Create: `scripts/prepare_dataset.py`
- Test: `tests/test_prepare_dataset.py`

The sampler reads CelebA attribute labels to sample women-only domain-A images, copies the chosen images into the split tree, and copies `list_attr_celeba.csv` into the output root so the dataset's women-only filter (which searches upward from `root_a`) succeeds.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_dataset.py`:

```python
import csv
from pathlib import Path

import pytest

from scripts.prepare_dataset import build_layout


def _make_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    src_a = tmp_path / "celeba"
    src_b = tmp_path / "anime"
    src_a.mkdir()
    src_b.mkdir()
    # 6 CelebA images: even ids female (Male=-1), odd ids male (Male=1)
    rows = []
    for i in range(6):
        name = f"{i:06d}.jpg"
        (src_a / name).write_bytes(b"a")
        rows.append({"image_id": name, "Male": "-1" if i % 2 == 0 else "1"})
    for i in range(6):
        (src_b / f"anime_{i}.jpg").write_bytes(b"b")
    csv_path = tmp_path / "list_attr_celeba.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "Male"])
        writer.writeheader()
        writer.writerows(rows)
    return src_a, src_b, csv_path


def test_demo_layout_counts_and_women_only(tmp_path):
    src_a, src_b, csv_path = _make_source(tmp_path)
    out = tmp_path / "out"
    build_layout(
        src_a=str(src_a), src_b=str(src_b), attr_csv=str(csv_path), out=str(out),
        n_train_a=2, n_train_b=2, n_test_a=1, n_test_b=1, seed=0,
    )
    train_a = sorted(p.name for p in (out / "trainA").iterdir())
    test_a = sorted(p.name for p in (out / "testA").iterdir())
    # only even (female) ids selected, train and test disjoint
    assert len(train_a) == 2 and len(test_a) == 1
    selected_a = set(train_a) | set(test_a)
    assert all(int(n[:6]) % 2 == 0 for n in selected_a)
    assert len(selected_a) == 3  # no overlap between train and test
    assert len(list((out / "trainB").iterdir())) == 2
    assert len(list((out / "testB").iterdir())) == 1
    assert (out / "list_attr_celeba.csv").exists()


def test_demo_layout_is_deterministic(tmp_path):
    src_a, src_b, csv_path = _make_source(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    for out in (out1, out2):
        build_layout(
            src_a=str(src_a), src_b=str(src_b), attr_csv=str(csv_path), out=str(out),
            n_train_a=2, n_train_b=2, n_test_a=1, n_test_b=1, seed=0,
        )
    assert sorted(p.name for p in (out1 / "trainA").iterdir()) == sorted(
        p.name for p in (out2 / "trainA").iterdir()
    )


def test_missing_source_raises(tmp_path):
    _, src_b, csv_path = _make_source(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_layout(
            src_a=str(tmp_path / "nope"), src_b=str(src_b), attr_csv=str(csv_path),
            out=str(tmp_path / "out"), n_train_a=1, n_train_b=1, n_test_a=1, n_test_b=1, seed=0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prepare_dataset.py -v`
Expected: FAIL — `scripts.prepare_dataset` does not exist.

- [ ] **Step 3: Implement**

Create `scripts/prepare_dataset.py`:

```python
#!/usr/bin/env python
"""Build the data/<variant>/{trainA,trainB,testA,testB} layout.

Domain A (CelebA) is sampled women-only using list_attr_celeba.csv; the CSV is
copied into the output root so the dataset's women-only filter works.

Examples:
    uv run python scripts/prepare_dataset.py demo \\
        --src-a <celeba_img_dir> --src-b <anime_dir> \\
        --attr-csv <list_attr_celeba.csv> --out data/demo
    uv run python scripts/prepare_dataset.py full \\
        --src-a <celeba_img_dir> --src-b <anime_dir> \\
        --attr-csv <list_attr_celeba.csv> --out data/full
"""
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import fire

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _female_ids(attr_csv: Path) -> set[str]:
    with attr_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "Male" not in reader.fieldnames:
            raise ValueError(f"{attr_csv} has no 'Male' column")
        return {row["image_id"] for row in reader if row["Male"] == "-1"}


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Source directory not found: {directory}")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def _copy_into(paths: list[Path], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for src in paths:
        shutil.copy2(src, dst / src.name)


def build_layout(
    src_a: str,
    src_b: str,
    attr_csv: str,
    out: str,
    n_train_a: int,
    n_train_b: int,
    n_test_a: int,
    n_test_b: int,
    seed: int = 0,
) -> None:
    """Sample/copy images into out/{trainA,trainB,testA,testB} and copy the CSV."""
    out_dir = Path(out)
    rng = random.Random(seed)

    female = _female_ids(Path(attr_csv))
    a_images = [p for p in _list_images(Path(src_a)) if p.name in female]
    b_images = _list_images(Path(src_b))

    a_pick = rng.sample(a_images, min(n_train_a + n_test_a, len(a_images)))
    b_pick = rng.sample(b_images, min(n_train_b + n_test_b, len(b_images)))

    _copy_into(a_pick[:n_train_a], out_dir / "trainA")
    _copy_into(a_pick[n_train_a : n_train_a + n_test_a], out_dir / "testA")
    _copy_into(b_pick[:n_train_b], out_dir / "trainB")
    _copy_into(b_pick[n_train_b : n_train_b + n_test_b], out_dir / "testB")

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(attr_csv), out_dir / "list_attr_celeba.csv")


def demo(src_a, src_b, attr_csv, out="data/demo", n_train_a=3000, n_train_b=3000,
         n_test_a=500, n_test_b=500, seed=0):
    """Build a small demo dataset (DVC-tracked)."""
    build_layout(src_a, src_b, attr_csv, out, n_train_a, n_train_b, n_test_a, n_test_b, seed)


def full(src_a, src_b, attr_csv, out="data/full", n_test_a=1000, n_test_b=1000, seed=0):
    """Lay out the full dataset; train splits take all remaining images."""
    a_total = len(_list_images(Path(src_a)))
    b_total = len(_list_images(Path(src_b)))
    build_layout(src_a, src_b, attr_csv, out,
                 a_total - n_test_a, b_total - n_test_b, n_test_a, n_test_b, seed)


if __name__ == "__main__":
    fire.Fire({"demo": demo, "full": full})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prepare_dataset.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_dataset.py tests/test_prepare_dataset.py
git commit -m "feat(scripts): prepare_dataset CLI for demo/full layout"
```

---

## Task 6: `download_data()` with demo/full backends

**Files:**

- Create: `kaoanime/data/download.py`
- Modify: `kaoanime/data/__init__.py`
- Test: `tests/test_download.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_download.py`:

```python
import pytest

from kaoanime.data import download_data


def test_demo_calls_dvc_pull(monkeypatch):
    calls = {}

    def fake_pull(remote, targets):
        calls["remote"] = remote
        calls["targets"] = targets

    monkeypatch.setattr("kaoanime.data.download._dvc_pull", fake_pull)
    download_data("demo")
    assert calls["remote"] == "data"


def test_full_invokes_kaggle_and_gdown(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr("kaoanime.data.download._download_celeba", lambda dest: seen.append("celeba"))
    monkeypatch.setattr("kaoanime.data.download._download_anime", lambda dest: seen.append("anime"))
    monkeypatch.setattr("kaoanime.data.download._layout_full", lambda dest: seen.append("layout"))
    download_data("full", dest=str(tmp_path))
    assert seen == ["celeba", "anime", "layout"]


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        download_data("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL — `download_data` not importable.

- [ ] **Step 3: Implement**

Create `kaoanime/data/download.py`:

```python
from __future__ import annotations

from pathlib import Path


def _dvc_pull(remote: str, targets: list[str] | None = None) -> None:
    from dvc.repo import Repo

    with Repo() as repo:
        repo.pull(remote=remote, targets=targets)


def _download_anime(dest: Path) -> None:
    """Download reitanaka/alignedanimefaces via the Kaggle API into dest."""
    import kaggle

    dest.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()  # raises a clear error if ~/.kaggle/kaggle.json is missing
    kaggle.api.dataset_download_files(
        "reitanaka/alignedanimefaces", path=str(dest), unzip=True
    )


def _download_celeba(dest: Path) -> None:
    """Download CelebA aligned images + attribute CSV from the CUHK Google Drive."""
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    # CelebA aligned&cropped folder (img_align_celeba + annotations) on Google Drive.
    gdown.download_folder(
        id="0B7EVK8r0v71pWEZsZE9oNnFzTm8", output=str(dest), quiet=False, use_cookies=False
    )


def _layout_full(dest: Path) -> None:
    """Arrange downloaded CelebA/anime into dest/full/{trainA,...} via the CLI helper."""
    from scripts.prepare_dataset import full as _full

    _full(
        src_a=str(dest / "img_align_celeba"),
        src_b=str(dest / "safebooru_jpeg"),
        attr_csv=str(dest / "list_attr_celeba.csv"),
        out=str(dest / "full"),
    )


def download_data(variant: str, dest: str = "data") -> None:
    """Fetch the dataset for the given variant.

    demo → `dvc pull` the demo dataset from the `data` remote.
    full → download CelebA (gdown) + AlignedAnimeFaces (kaggle), then lay out.
    """
    if variant == "demo":
        _dvc_pull("data", targets=["data/demo"])
        return
    if variant == "full":
        dest_path = Path(dest)
        _download_celeba(dest_path)
        _download_anime(dest_path)
        _layout_full(dest_path)
        return
    raise ValueError(f"Unknown variant {variant!r}; expected 'demo' or 'full'")
```

Update `kaoanime/data/__init__.py`:

```python
from kaoanime.data.download import download_data
from kaoanime.data.layout import required_data_dirs

__all__ = ["download_data", "required_data_dirs"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_download.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add kaoanime/data/download.py kaoanime/data/__init__.py tests/test_download.py
git commit -m "feat(data): download_data with demo (dvc pull) and full (kaggle+gdown) backends"
```

---

## Task 7: `ensure_data()` — fetch only when missing

**Files:**

- Create: `kaoanime/data/ensure.py`
- Modify: `kaoanime/data/__init__.py`
- Test: `tests/test_ensure_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ensure_data.py`:

```python
from omegaconf import OmegaConf

from kaoanime.config import Config
from kaoanime.data import ensure_data


def _cfg(tmp_path, variant="demo"):
    cfg = OmegaConf.structured(Config)
    cfg.data.variant = variant
    base = tmp_path / variant
    cfg.data.root_a = str(base / "trainA")
    cfg.data.root_b = str(base / "trainB")
    cfg.data.test_a = str(base / "testA")
    cfg.data.test_b = str(base / "testB")
    return cfg


def test_downloads_when_dirs_missing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    called = {}
    monkeypatch.setattr(
        "kaoanime.data.ensure.download_data",
        lambda variant, **kw: called.setdefault("variant", variant),
    )
    ensure_data(cfg)
    assert called["variant"] == "demo"


def test_no_download_when_all_dirs_present(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    for key in ("root_a", "root_b", "test_a", "test_b"):
        from pathlib import Path

        Path(getattr(cfg.data, key)).mkdir(parents=True)
    called = {}
    monkeypatch.setattr(
        "kaoanime.data.ensure.download_data",
        lambda variant, **kw: called.setdefault("variant", variant),
    )
    ensure_data(cfg)
    assert called == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ensure_data.py -v`
Expected: FAIL — `ensure_data` not importable.

- [ ] **Step 3: Implement**

Create `kaoanime/data/ensure.py`:

```python
from __future__ import annotations

from kaoanime.data.download import download_data
from kaoanime.data.layout import required_data_dirs


def ensure_data(cfg) -> None:
    """Download the dataset if any required split directory is missing."""
    missing = [d for d in required_data_dirs(cfg) if not d.is_dir()]
    if missing:
        download_data(cfg.data.variant)
```

Update `kaoanime/data/__init__.py`:

```python
from kaoanime.data.download import download_data
from kaoanime.data.ensure import ensure_data
from kaoanime.data.layout import required_data_dirs

__all__ = ["download_data", "ensure_data", "required_data_dirs"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ensure_data.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add kaoanime/data/ensure.py kaoanime/data/__init__.py tests/test_ensure_data.py
git commit -m "feat(data): ensure_data fetches only when split dirs are missing"
```

---

## Task 8: Wire `ensure_data` into train.py and infer.py

**Files:**

- Modify: `train.py`
- Modify: `infer.py`

- [ ] **Step 1: Add the call in `train.py`**

In `train.py`, add the import near the other `kaoanime` imports:

```python
from kaoanime.data import ensure_data
```

Then, inside `main`, as the **first** statement of the function body (before building the dataset):

```python
    ensure_data(cfg)
```

- [ ] **Step 2: Add the call in `infer.py`**

In `infer.py`, add the import:

```python
from kaoanime.data import ensure_data
```

Then, inside `main`, as the first statement after the `eval.checkpoint`/`eval.input` validation (so a missing checkpoint still fails fast first):

```python
    ensure_data(cfg)
```

- [ ] **Step 3: Verify both modules import cleanly**

Run:

```bash
uv run python -c "import train, infer; print('entrypoints import OK')"
```

Expected: `entrypoints import OK`

- [ ] **Step 4: Commit**

```bash
git add train.py infer.py
git commit -m "feat: wire ensure_data into train and infer entrypoints"
```

---

## Task 9: `scripts/export_models.py` — push selected checkpoints to `models` remote

**Files:**

- Create: `scripts/export_models.py`
- Test: `tests/test_export_models.py`

The user picks the 1–2 best checkpoints by path. This task ships the `.pt`
export + DVC push to the `models` remote. ONNX is added by the separate
model-packaging plan.

- [ ] **Step 1: Write the failing test**

Create `tests/test_export_models.py`:

```python
from pathlib import Path

from scripts.export_models import stage_models


def test_stage_copies_checkpoints_and_returns_paths(tmp_path):
    ckpt1 = tmp_path / "a.ckpt"
    ckpt2 = tmp_path / "b.ckpt"
    ckpt1.write_bytes(b"x")
    ckpt2.write_bytes(b"y")
    out = tmp_path / "models" / "export"
    staged = stage_models([str(ckpt1), str(ckpt2)], out_dir=str(out))
    assert [p.name for p in staged] == ["a.pt", "b.pt"]
    assert all(p.exists() for p in staged)


def test_stage_missing_checkpoint_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        stage_models([str(tmp_path / "nope.ckpt")], out_dir=str(tmp_path / "out"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_export_models.py -v`
Expected: FAIL — `scripts.export_models` does not exist.

- [ ] **Step 3: Implement**

Create `scripts/export_models.py`:

```python
#!/usr/bin/env python
"""Stage the best checkpoints as .pt and push them to the DVC `models` remote.

Example:
    uv run python scripts/export_models.py \\
        --checkpoints checkpoints/not_ep10.ckpt checkpoints/not_ep9.ckpt
"""
from __future__ import annotations

import shutil
from pathlib import Path

import fire


def stage_models(checkpoints: list[str], out_dir: str = "models/export") -> list[Path]:
    """Copy each checkpoint to out_dir as <stem>.pt; return the staged paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for ckpt in checkpoints:
        src = Path(ckpt)
        if not src.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {src}")
        dst = out / f"{src.stem}.pt"
        shutil.copy2(src, dst)
        staged.append(dst)
    return staged


def _dvc_add_push(paths: list[Path]) -> None:
    from dvc.repo import Repo

    with Repo() as repo:
        repo.add([str(p) for p in paths])
        repo.push(remote="models", targets=[str(p) for p in paths])


def export(*checkpoints: str, out_dir: str = "models/export") -> None:
    """Stage checkpoints and push to the `models` remote."""
    if not checkpoints:
        raise ValueError("Pass at least one --checkpoints path")
    staged = stage_models(list(checkpoints), out_dir=out_dir)
    _dvc_add_push(staged)
    print(f"Staged and pushed {len(staged)} model(s) to the 'models' remote.")


if __name__ == "__main__":
    fire.Fire(export)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_export_models.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_models.py tests/test_export_models.py
git commit -m "feat(scripts): export_models stages .pt and pushes to models remote"
```

---

## Task 10: Retire stale `data.dvc` and update `.gitignore`

**Files:**

- Delete: `data.dvc`
- Modify: `.gitignore`

- [ ] **Step 1: Remove the stale tracking file**

The old `data.dvc` tracks a 411 MB `data/` directory that is no longer used.

```bash
git rm data.dvc
```

- [ ] **Step 2: Ensure `data/` stays ignored except DVC pointers**

Confirm `.gitignore` contains `/data` (it already does). No change needed unless
absent — in that case add a line `/data`.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore(dvc): retire stale data.dvc (superseded by variant layout)"
```

---

## Task 11: Run the full test + lint gate

**Files:** none (verification)

- [ ] **Step 1: Run the test suite**

Run: `uv run pytest -q -p no:cacheprovider`
Expected: all pass (existing 94 + new data tests), 2 skipped.

- [ ] **Step 2: Run the hooks**

Run: `uv run pre-commit run -a`
Expected: all hooks pass.

- [ ] **Step 3: If anything was reformatted, commit it**

```bash
git add -A
git commit -m "style: apply hooks after data-management changes"
```

---

## Task 12: Build & publish the demo dataset (runbook — needs Drive auth + real data)

**Files:** none (operational; run once by the maintainer)

This step needs the real datasets on disk and Google Drive auth, so it is a
manual runbook rather than an automated test. Paths below are passed as
arguments — none are hardcoded in the repo.

- [ ] **Step 1: Build the demo layout from the full datasets**

```bash
uv run python scripts/prepare_dataset.py demo \
    --src-a <path-to>/CelebA/img_align_celeba/img_align_celeba \
    --src-b <path-to>/alignedanimefaces/safebooru_jpeg \
    --attr-csv <path-to>/CelebA/list_attr_celeba.csv \
    --out data/demo
```

- [ ] **Step 2: Track and push to the `data` remote**

```bash
uv run dvc add data/demo
uv run dvc push -r data       # first push opens a Google Drive auth flow
git add data/demo.dvc data/.gitignore
git commit -m "data: add DVC-tracked demo dataset"
```

- [ ] **Step 3: Verify a clean pull works**

```bash
uv run dvc pull -r data data/demo
```

Expected: demo splits present under `data/demo/`.

---

## Task 13: Update README data section

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Replace the "Данные (DVC)" section**

Update the README "Данные (DVC)" section to document:

- `dvc pull` fetches the demo dataset (default `variant=demo`);
- full training: set `data.variant=full` (or override `data.root_a`/`root_b` to an existing copy); `download_data` needs `~/.kaggle/kaggle.json` and Google Drive auth;
- `scripts/prepare_dataset.py` and `scripts/export_models.py` usage;
- the two remotes (`data`, `models`).

Concrete replacement block:

```markdown
### Данные (DVC)

Данные и модели версионируются через DVC на двух Google Drive remote
(`data` — демо-датасет, `models` — обученные модели).

Демо-датасет (по умолчанию, `data.variant=demo`) скачивается автоматически —
`train.py`/`infer.py` вызывают `ensure_data()`, который при отсутствии данных
делает `dvc pull`. Вручную:

\`\`\`bash
uv run dvc pull -r data data/demo
\`\`\`

Полный датасет (`data.variant=full`) скачивается `download_data()`:
CelebA через `gdown`, AlignedAnimeFaces через Kaggle API
(нужен `~/.kaggle/kaggle.json`). Раскладка по директориям —
`scripts/prepare_dataset.py`. Чтобы не качать 100 ГБ заново, можно указать
существующие пути: `data.variant=full data.root_a=<...> data.root_b=<...>`.

Экспорт лучших моделей в remote `models`:

\`\`\`bash
uv run python scripts/export_models.py --checkpoints checkpoints/not_ep10.ckpt
\`\`\`
```

- [ ] **Step 2: Verify hooks on the README**

Run: `uv run pre-commit run prettier --files README.md`
Expected: Passed.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document demo/full data variants and model export"
```

---

## Self-Review Notes

- **Spec coverage:** two remotes (Task 2), variant switch + repo-relative paths (Task 3), demo dataset (Tasks 5, 12), `download_data` demo/full (Task 6), `ensure_data` in train/infer (Tasks 7–8), `prepare_dataset.py` (Task 5), `export_models.py` (Task 9), retire stale `data.dvc` (Task 10), deps (Task 1), README (Task 13). ONNX explicitly deferred to the packaging plan.
- **Women-only filter integration:** the dataset searches upward from `root_a` for `list_attr_celeba.csv` and raises if absent; `prepare_dataset.build_layout` copies the CSV into the output root so demo/full both satisfy the filter (Task 5).
- **Type consistency:** `build_layout(...)` signature is identical in Task 5 (def) and Task 6 (`_layout_full` calls `full(...)` which calls `build_layout`). `download_data(variant, dest)`, `ensure_data(cfg)`, `required_data_dirs(cfg)`, `stage_models(checkpoints, out_dir)` are used with matching signatures across tasks.
- **No placeholders:** every code step contains complete code; the only manual steps (Task 12) are operational by necessity (real data + Drive auth) and give exact commands.
