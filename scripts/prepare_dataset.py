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


def demo(
    src_a,
    src_b,
    attr_csv,
    out="data/demo",
    n_train_a=3000,
    n_train_b=3000,
    n_test_a=500,
    n_test_b=500,
    seed=0,
):
    """Build a small demo dataset (DVC-tracked)."""
    build_layout(
        src_a, src_b, attr_csv, out, n_train_a, n_train_b, n_test_a, n_test_b, seed
    )


def full(src_a, src_b, attr_csv, out="data/full", n_test_a=1000, n_test_b=1000, seed=0):
    """Lay out the full dataset; train splits take all remaining images."""
    a_total = len(_list_images(Path(src_a)))
    b_total = len(_list_images(Path(src_b)))
    build_layout(
        src_a,
        src_b,
        attr_csv,
        out,
        a_total - n_test_a,
        b_total - n_test_b,
        n_test_a,
        n_test_b,
        seed,
    )


if __name__ == "__main__":
    fire.Fire({"demo": demo, "full": full})
