# kaoanime/utils/dataset.py
import csv
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .image import load_image
from .transforms import get_transforms

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_female_ids(csv_path: Path) -> set[str]:
    """Return CelebA image_ids whose 'Male' attribute is -1 (female)."""
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "Male" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} has no 'Male' column; columns={reader.fieldnames}"
            )
        id_field = reader.fieldnames[0]
        return {row[id_field] for row in reader if row["Male"] == "-1"}


# One AlignFaceProcessor per DataLoader worker process, created on first use.
# Stored at module level so it survives across __getitem__ calls within a worker.
_worker_processor = None


def _get_worker_processor():
    global _worker_processor
    if _worker_processor is None:
        from .align import AlignFaceProcessor
        _worker_processor = AlignFaceProcessor()
    return _worker_processor


def _align_pil(img: Image.Image, size: int) -> Image.Image:
    """Try landmark alignment; return original PIL image unchanged on failure."""
    arr = np.array(img)
    aligned = _get_worker_processor().align(arr, size=size)
    if aligned is not None:
        return Image.fromarray(aligned)
    return img


def _anime_crop(img: Image.Image, offset_x: int = 0, offset_y: int = -7) -> Image.Image:
    """Resize to 512×512, crop 256×256 centred at (256+offset_x, 256+offset_y).

    The returned image is 256×256 PIL; the caller's transform resizes it to
    the final model input size.
    """
    s512 = img.resize((512, 512), Image.LANCZOS)
    cx = 256 + offset_x
    cy = 256 + offset_y
    return s512.crop((cx - 128, cy - 128, cx + 128, cy + 128))


def _collect_files(roots: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for r in roots:
        files.extend(p for p in Path(r).iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    return sorted(files)


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        root_a: str | Path,
        root_b: str | Path,
        image_size: int,
        train: bool = True,
        extra_roots_a: list[str | Path] | None = None,
        extra_roots_b: list[str | Path] | None = None,
        align_a: bool = False,
        anime_offset_x: int = 0,
        anime_offset_y: int = -7,
    ) -> None:
        self._files_a = _collect_files([root_a] + list(extra_roots_a or []))
        self._files_b = _collect_files([root_b] + list(extra_roots_b or []))
        self._transform = self._create_transform(image_size, train)
        self._image_size = image_size
        self._align_a = align_a
        self._anime_offset_x = anime_offset_x
        self._anime_offset_y = anime_offset_y

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_a = load_image(self._files_a[index % len(self._files_a)])
        img_b = load_image(self._files_b[index % len(self._files_b)])
        if self._align_a:
            img_a = _align_pil(img_a, self._image_size)
        img_b = _anime_crop(img_b, self._anime_offset_x, self._anime_offset_y)
        return {"A": self._transform(img_a), "B": self._transform(img_b)}

    def __len__(self) -> int:
        return max(len(self._files_a), len(self._files_b))

    def _create_transform(self, image_size: int, train: bool) -> Callable:
        return get_transforms("train" if train else "test", image_size=image_size)
