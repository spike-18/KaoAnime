# kaoanime/utils/dataset.py
from pathlib import Path
from typing import Callable

from torch import Tensor
from torch.utils.data import Dataset

from .image import load_image

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UnpairedImageDataset(Dataset):
    def __init__(
        self, root_a: str | Path, root_b: str | Path, transform: Callable
    ) -> None:
        self._files_a = sorted(
            p for p in Path(root_a).iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        self._files_b = sorted(
            p for p in Path(root_b).iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        self._transform = transform

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_a = load_image(self._files_a[index % len(self._files_a)])
        img_b = load_image(self._files_b[index % len(self._files_b)])
        return {"A": self._transform(img_a), "B": self._transform(img_b)}

    def __len__(self) -> int:
        return max(len(self._files_a), len(self._files_b))
