# kaoanime/utils/dataset.py
from pathlib import Path
from typing import Callable

from torch import Tensor
from torch.utils.data import Dataset

from .image import load_image
from .transforms import get_transforms

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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
    ) -> None:
        self._files_a = _collect_files([root_a] + list(extra_roots_a or []))
        self._files_b = _collect_files([root_b] + list(extra_roots_b or []))
        self._transform = self._create_transform(image_size, train)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_a = load_image(self._files_a[index % len(self._files_a)])
        img_b = load_image(self._files_b[index % len(self._files_b)])
        return {"A": self._transform(img_a), "B": self._transform(img_b)}

    def __len__(self) -> int:
        return max(len(self._files_a), len(self._files_b))

    def _create_transform(self, image_size: int, train: bool) -> Callable:
        return get_transforms("train" if train else "test", image_size=image_size)
