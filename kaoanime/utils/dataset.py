# kaoanime/utils/dataset.py
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .image import load_image
from .transforms import get_transforms

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

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


def _anime_crop(img: Image.Image, scale: float, shift_x: float, shift_y: float) -> Image.Image:
    """Fixed-ratio crop for domain B (anime). Parameters are fractions of image size."""
    w, h = img.size
    crop = int(min(w, h) / scale)
    cx = w / 2 + shift_x * w
    cy = h / 2 + shift_y * h
    x0 = int(max(0, cx - crop / 2))
    y0 = int(max(0, cy - crop / 2))
    x1 = min(w, x0 + crop)
    y1 = min(h, y0 + crop)
    return img.crop((x0, y0, x1, y1))


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
        anime_scale  : float = 1.85,
        anime_shift_x: float = 0.00,
        anime_shift_y: float = -0.01,
    ) -> None:
        self._files_a = _collect_files([root_a] + list(extra_roots_a or []))
        self._files_b = _collect_files([root_b] + list(extra_roots_b or []))
        self._transform = self._create_transform(image_size, train)
        self._image_size = image_size
        self._align_a = align_a
        self._anime_scale   = anime_scale
        self._anime_shift_x = anime_shift_x
        self._anime_shift_y = anime_shift_y

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_a = load_image(self._files_a[index % len(self._files_a)])
        img_b = load_image(self._files_b[index % len(self._files_b)])
        if self._align_a:
            img_a = _align_pil(img_a, self._image_size)
        img_b = _anime_crop(img_b, self._anime_scale, self._anime_shift_x, self._anime_shift_y)
        return {"A": self._transform(img_a), "B": self._transform(img_b)}

    def __len__(self) -> int:
        return max(len(self._files_a), len(self._files_b))

    def _create_transform(self, image_size: int, train: bool) -> Callable:
        return get_transforms("train" if train else "test", image_size=image_size)
