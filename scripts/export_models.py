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
