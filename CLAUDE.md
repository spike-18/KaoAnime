# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KaoAnime converts real selfie photos to anime-style images via neural image-to-image translation. The dataset is unpaired (selfie domain A ↔ anime domain B), making CycleGAN the natural architecture. GPU acceleration uses AMD ROCm 7.2.

## Development Commands

```bash
uv sync --group rocm          # install PyTorch (ROCm) + all deps
uv sync --group dev           # install pre-commit tools
uv run pre-commit install     # set up git hooks after first sync
dvc pull                      # fetch dataset (~411 MB, 7 000 images)
uvx ruff check --fix .        # lint
uvx ruff format .             # format
uv run python -m kaoanime.train   # train (module entry point)
uv run python -m kaoanime.infer   # infer (module entry point)
```

## Data

Tracked by DVC (`data.dvc`). After `dvc pull`, data lives in `data/selfie2anime/`:
- `trainA/` / `trainB/` — 3 400 paired-domain training images
- `testA/` / `testB/` — 100 test images per domain

## Intended Project Structure

```
kaoanime/
  data/        # PyTorch Dataset classes and transforms
  models/      # Generator, Discriminator, CycleGAN wrapper
  training/    # training loop, loss functions (adversarial + cycle + identity)
  inference/   # inference pipeline and checkpoint loading
  utils/       # image helpers, metric utilities
train.py       # CLI entry point — delegates to kaoanime.training
infer.py       # CLI entry point — delegates to kaoanime.inference
```

## MLflow

Planned for experiment tracking but not yet wired up. Add `# TODO: mlflow` stubs rather than implementing until explicitly asked.
