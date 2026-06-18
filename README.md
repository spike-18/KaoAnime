# KaoAnime — selfie-to-anime style transfer

KaoAnime turns a real face photo into an anime-style portrait via neural
image-to-image translation between two unpaired domains. The main model is
**Neural Optimal Transport (NOT)** (Korotin et al., ICLR 2023); a **CycleGAN**
alternative that shares the same building blocks is also available.

## Project overview

### Task

There are two sets of face images with no paired correspondence: domain **A** —
real photos (CelebA), domain **B** — anime faces (safebooru). The goal is to learn
a mapping `T: A → B` that transfers a photo into anime style while preserving face
geometry and identity (unpaired image-to-image translation).

### Data

- **CelebA** — a public face dataset (~200k images). The aligned `img_align_celeba`
  version is used; domain A is additionally filtered to women only via
  `list_attr_celeba.csv`.
- **AlignedAnimeFaces (safebooru)** — ~500k pre-aligned anime faces.
- Both domains are normalized to a `128×128` square: real faces by landmark
  alignment (MediaPipe), anime faces by a center crop. Each domain has a held-out
  test split.

Data is versioned with **DVC** and never stored in git. A small **demo subset**
(~800 MB, 3000/3000 train + 500/500 test) is downloaded automatically for quick
checks; the full dataset (~100 GB) is fetched from open sources on demand
(see [Data](#data-1)).

### Method

NOT (strong OT) trains two networks: a transport map `T` (UNet) that moves A to B,
and a Kantorovich potential `f` (a ResNet without BatchNorm) that tells them apart.

```
T_loss = MSE(X, T(X)) − f(T(X))      # t_iters T-steps per one f-step
f_loss = f(T(X))      − f(Y)
```

The `MSE(X, T(X))` term penalizes large pixel changes, so the transport stays
"cheap" and no cycle-consistency is needed. Translation quality is tracked with the
**FID** metric during training.

### Libraries

[PyTorch](https://pytorch.org/) + [PyTorch Lightning](https://lightning.ai/) —
models and training loop; [Hydra](https://hydra.cc/) — configuration;
[MLflow](https://mlflow.org/) — experiment tracking;
[DVC](https://dvc.org/) — data and model versioning;
[torchmetrics](https://lightning.ai/docs/torchmetrics/) — FID;
[MediaPipe](https://developers.google.com/mediapipe) — face alignment.

## Technical details

### Setup

The project uses the [uv](https://docs.astral.sh/uv/) package manager; PyTorch is
installed from the CUDA 12.8 index (see `pyproject.toml`).

```bash
uv sync                    # environment + all dependencies (including PyTorch)
uv sync --group dev        # dev tools (pytest, jupyter, pre-commit)
uv run pre-commit install  # git hooks
```

Always invoke Python via `uv run python ...`.

### Data

The dataset is fetched automatically: `train.py`/`infer.py` call `ensure_data()`,
which downloads the required split if it is missing.

- **demo** (default, `data.variant=demo`) — a public `demo.zip` is downloaded from
  Google Drive via `gdown` with no authentication and unpacked into `data/demo/`.
- **full** (`data.variant=full`) — CelebA (`gdown`) + AlignedAnimeFaces (Kaggle API,
  requires `~/.kaggle/kaggle.json`), then laid out by `scripts/prepare_dataset.py`.
  To avoid re-downloading 100 GB, point at existing paths:

  ```bash
  uv run python train.py data.variant=full data.root_a=<celeba_dir> data.root_b=<anime_dir>
  ```

### Train

Single entry point — `train.py`; the model is selected by `model_type`.

```bash
uv run python train.py                      # NOT (default)
uv run python train.py model_type=cyclegan  # CycleGAN
```

Any hyperparameter can be overridden via the Hydra CLI:

```bash
uv run python train.py not_.t_iters=5 not_.t_lr=5e-5 data.batch_size=32
```

Metrics, hyperparameters, and training curves are logged to MLflow (address is set
in the config, default `http://127.0.0.1:8080`; local server:
`uv run mlflow server --host 127.0.0.1 --port 8080`).

### Infer

Entry point — `infer.py`: takes a checkpoint and an image (or a directory) and
writes the translations to an output folder.

```bash
uv run python infer.py \
    eval.checkpoint=<path/to/last.ckpt> \
    eval.input=data/demo/testA \
    eval.output_dir=outputs/eval
```

Input — `.jpg/.png/.webp/.bmp` images; `eval.align=true` enables face alignment
before translation.

### Overall — project structure

```
kaoanime-selfie2anime/
├── kaoanime/                # main Python package
│   ├── models/              # network building blocks (ResNet, UNet, PatchGAN, NOT potential)
│   ├── losses/              # CycleGAN losses
│   ├── data/                # dataset download (demo/full) and paths
│   ├── utils/               # datasets, transforms, alignment, FID
│   ├── inference/           # inference pipeline
│   ├── callbacks.py         # MLflow checkpoint callback
│   ├── config*.py           # Hydra configs (shared + NOT + CycleGAN)
│   ├── model_not.py         # NOT LightningModule
│   └── model_cyclegan.py    # CycleGAN LightningModule
├── scripts/                 # prepare_dataset, export_models, align_dataset
├── notebooks/               # exploratory notebooks
├── tests/                   # pytest tests
├── train.py · infer.py · eval.py   # CLI entry points
├── pyproject.toml · uv.lock        # dependencies
└── .pre-commit-config.yaml         # code-quality hooks
```

### Inference & delivery

A trained model is saved as a Lightning checkpoint (`.ckpt`) and logged to MLflow.
The best checkpoints are published to the DVC `models` remote via
`scripts/export_models.py`. ONNX/TensorRT export and an inference server are planned.
