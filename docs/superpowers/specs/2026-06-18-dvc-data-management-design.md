# DVC Data & Model Management — Design

**Date:** 2026-06-18
**Status:** Approved (pending spec review)

## Goal

Bring the project's data management in line with the course requirements:

- Datasets and trained models are versioned with **DVC**, never committed to git.
- **Two different storages**: one for data, one for models.
- `dvc pull` of the data is wired into the `train` and `infer` entry points.
- A `download_data()` function fetches the datasets from open sources.

Today `.dvc/config` is empty (no remotes), the only `data.dvc` tracks a stale
411 MB `data/` directory that is **not** what training uses, and nothing pulls
data automatically. This design replaces that with a portable, reproducible
setup.

## Context & Constraints

- Real training data lives outside the repo and is large:
  - **CelebA** (aligned & cropped): ~1.7 GB, 201 599 images + attribute CSVs.
  - **AlignedAnimeFaces** (`safebooru_jpeg`): **~100 GB**, 499 088 images.
- Local free disk is only ~121 GB, so a full local DVC remote copy of the 100 GB
  set (cache + remote ≈ 200 GB) does **not** fit, and a 100 GB zip cannot be
  re-downloaded + extracted locally either.
- A **local** DVC remote does not transport to a grader who clones the repo on a
  different machine — `dvc pull` from a local dir would fetch nothing.
- Absolute machine paths (`/beta/students/madorskii/...`) must **not** be
  hardcoded — the repo must work when cloned elsewhere.
- Women-only CelebA filtering is already implemented inside
  `UnpairedImageDataset` (parses `list_attr_celeba.csv` at construction) and is
  **out of scope** here — the layout step does not re-filter.

## Decisions (from brainstorming)

1. **Variant C + demo.** Datasets are downloaded from the internet for full
   training; DVC versions a small **demo dataset** (< 2 GB) so graders can
   `dvc pull` it and verify training works without the 100 GB download.
2. **Two cloud (Google Drive) remotes** — portable, no hardcoded paths:
   - `data` (default) → demo dataset, folder id `158t5Cy5bHckNjxzbXI5D5G1EI7Nh27fy`.
   - `models` → best models, folder id `16RC8Zc2cnDvc2Dh46CFoy9iEzPZohFGF`.
     Two different Drive folders satisfy the "two different storages" requirement.
     The local-storage `download_data()` requirement is covered by the full-mode
     downloader below.
3. **`variant` switch** in config: `"demo"` (default) vs `"full"`.
4. **`extra_roots_*` are unused** and stay out of the data layout.
5. **Models:** before upload, select 1–2 best checkpoints and store **both**
   `.pt` and `.onnx` versions on the `models` remote.

## Architecture

### Data layout (both variants)

A single repo-relative layout, `data/` already in `.gitignore`:

```
data/<variant>/{trainA,trainB,testA,testB}
```

- `trainA` / `testA` — CelebA images (women-only filter applied later, inside
  the dataset; not duplicated in the layout).
- `trainB` / `testB` — anime images.

Demo composition (deterministic seed): `trainA=3000`, `trainB=3000`,
`testA=500`, `testB=500`. Anime ≈ 210 KB/img → ~735 MB; CelebA ≈ 9 KB/img →
~30 MB; total < 1 GB.

### Config

`DataConfig` gains:

- `variant: str = "demo"` — `"demo"` or `"full"`.
- Paths derived from `variant` (e.g. a helper returns
  `data/<variant>/trainA` etc.). Full-mode source directories are configurable
  via Hydra CLI so the user can point at an already-downloaded copy instead of
  re-downloading 100 GB. No absolute paths in the committed defaults.

### Components

1. **`scripts/prepare_dataset.py`** — CLI (fire/click, **not argparse**) that
   builds the `data/<variant>/...` directory tree.
   - `--variant demo` — deterministically samples N images per split from a
     source dataset into `data/demo/...`.
   - `--variant full` — lays out the fully downloaded data into
     `data/full/...` (or accepts existing source paths, to avoid re-copying
     100 GB).

2. **`kaoanime/data/download.py` → `download_data(variant, dest, ...)`**
   - `demo` → `dvc pull -r data` via the DVC Python API — fetches the demo
     dataset from the `data` Drive remote.
   - `full` → CelebA via `gdown` (CUHK Google Drive: `img_align_celeba` +
     annotation files), AlignedAnimeFaces via the `kaggle` API
     (`reitanaka/alignedanimefaces`); unzip; then invoke the `full` layout of
     `prepare_dataset.py`. Requires `~/.kaggle/kaggle.json`. Target directory is
     configurable (default `data/full`).

3. **`ensure_data(cfg)`** (in `kaoanime/data/`) — called at the start of
   `train.py` and `infer.py`. If the required `data/<variant>/...` directories
   are missing, it calls `download_data(cfg.data.variant, ...)`. This wires
   `dvc pull` (demo) / downloading (full) into the train and infer commands per
   the requirement.

4. **`scripts/export_models.py`** — CLI to select 1–2 best checkpoints, write
   `.pt` + `.onnx`, `dvc add` them, and `dvc push -r models`. (ONNX conversion
   overlaps with the separate "model production packaging" task; the exporter is
   defined here but the ONNX details are finalised there.)

### `.dvc/config`

```ini
[core]
    remote = data
['remote "data"']
    url = gdrive://158t5Cy5bHckNjxzbXI5D5G1EI7Nh27fy
['remote "models"']
    url = gdrive://16RC8Zc2cnDvc2Dh46CFoy9iEzPZohFGF
```

The stale `data.dvc` (old 411 MB `data/`) is removed; `data/demo` is tracked
instead (`dvc add data/demo`, `dvc push -r data`).

### Dependencies

Added (all used): `dvc[gdrive]`, `gdown`, `kaggle`.

## Data Flow

| Actor  | Command                          | Data path                                               |
| ------ | -------------------------------- | ------------------------------------------------------- |
| Grader | `uv run python train.py`         | `ensure_data` → `dvc pull -r data` → `data/demo/...`    |
| User   | `train.py data.variant=full ...` | `ensure_data` → `download_data(full)` → `data/full/...` |
| User   | `scripts/export_models.py`       | best `.pt`+`.onnx` → `dvc add` → `dvc push -r models`   |

## Error Handling

- **Missing `~/.kaggle/kaggle.json`** (full mode) → clear error naming the file
  and how to obtain a Kaggle token.
- **`dvc pull` fails / no Drive auth** (demo mode) → clear error pointing at the
  Drive auth setup.
- **Layout source missing** → `prepare_dataset.py` fails loudly naming the
  expected source directory.
- Insufficient disk for full mode is the user's responsibility (documented);
  graders use demo mode.

## Testing

- `prepare_dataset.py` demo sampling is deterministic (seeded) and produces the
  expected per-split counts — unit-tested with a tiny synthetic source tree.
- `ensure_data` calls the downloader only when directories are absent
  (mock/monkeypatch the downloader; assert no call when data present).
- `download_data` source selection branches on `variant` (mock `gdown`/`kaggle`/
  `dvc` calls; assert the right backend is invoked).
- No network or Drive/Kaggle access in unit tests — all external calls mocked.
- All Python invoked via `uv run python` / `uv run pytest`.

## Out of Scope

- ONNX/TensorRT conversion internals (separate "model production packaging").
- Women-only filtering (already implemented in `UnpairedImageDataset`).
- Inference server.
