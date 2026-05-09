# Utils Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `kaoanime/utils/` package with image I/O, transforms, dataset, and dataloader utilities.

**Architecture:** Five focused modules under `kaoanime/utils/` — `image.py` handles PIL I/O, `transforms.py` builds torchvision pipelines, `dataset.py` provides `UnpairedImageDataset`, `dataloader.py` wraps `DataLoader`. All public symbols re-exported from `__init__.py`.

**Tech Stack:** Python 3.13, PyTorch 2.11 (ROCm), torchvision 0.26, Pillow 12, pytest, uv.

---

### Task 1: Package skeleton + pytest

**Files:**
- Create: `kaoanime/__init__.py`
- Create: `kaoanime/utils/__init__.py`
- Create: `tests/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create package init files**

```python
# kaoanime/__init__.py
# (empty)
```

```python
# kaoanime/utils/__init__.py
# (empty — filled in Task 5)
```

```python
# tests/__init__.py
# (empty)
```

- [ ] **Step 2: Add pytest to dev dependencies**

```toml
# pyproject.toml — replace the dev group with:
[dependency-groups]
dev = [
    "pre-commit>=4.6.0",
    "pytest>=8.0.0",
]
```

- [ ] **Step 3: Install**

```bash
uv sync --group dev
```

Expected: resolves and installs pytest.

- [ ] **Step 4: Verify pytest runs**

```bash
uv run pytest tests/ -v
```

Expected: `no tests ran` (exit 0 or `no tests found`).

- [ ] **Step 5: Commit**

```bash
git add kaoanime/__init__.py kaoanime/utils/__init__.py tests/__init__.py pyproject.toml uv.lock
git commit -m "chore: scaffold kaoanime package and add pytest"
```

---

### Task 2: `image.py` — load_image and save_image

**Files:**
- Create: `kaoanime/utils/image.py`
- Create: `tests/test_image.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_image.py
from pathlib import Path
import torch
from PIL import Image
from kaoanime.utils.image import load_image, save_image


def test_load_image_returns_rgb_pil(tmp_path):
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path)
    img = load_image(img_path)
    assert img.mode == "RGB"
    assert img.size == (64, 64)


def test_load_image_accepts_string_path(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (32, 32)).save(img_path)
    img = load_image(str(img_path))
    assert isinstance(img, Image.Image)


def test_save_image_creates_parent_dirs(tmp_path):
    out_path = tmp_path / "subdir" / "nested" / "out.png"
    tensor = torch.zeros(3, 32, 32)
    save_image(tensor, out_path)
    assert out_path.exists()


def test_save_image_white_tensor_produces_white_pixels(tmp_path):
    out_path = tmp_path / "out.png"
    tensor = torch.ones(3, 16, 16)  # all 1.0 in [-1,1] = white
    save_image(tensor, out_path)
    img = Image.open(out_path).convert("RGB")
    pixels = list(img.getdata())
    assert all(v == 255 for px in pixels for v in px)


def test_save_image_black_tensor_produces_black_pixels(tmp_path):
    out_path = tmp_path / "out.png"
    tensor = torch.full((3, 16, 16), -1.0)  # all -1.0 in [-1,1] = black
    save_image(tensor, out_path)
    img = Image.open(out_path).convert("RGB")
    pixels = list(img.getdata())
    assert all(v == 0 for px in pixels for v in px)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_image.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `image.py`**

```python
# kaoanime/utils/image.py
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(tensor: Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor.detach().cpu().clamp(-1.0, 1.0)
    img = ((img + 1.0) / 2.0 * 255.0).byte()
    img = img.permute(1, 2, 0).numpy()
    Image.fromarray(img, mode="RGB").save(path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_image.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add kaoanime/utils/image.py tests/test_image.py
git commit -m "feat: add load_image and save_image utilities"
```

---

### Task 3: `transforms.py` — get_transforms

**Files:**
- Create: `kaoanime/utils/transforms.py`
- Create: `tests/test_transforms.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transforms.py
import torch
from PIL import Image
from kaoanime.utils.transforms import get_transforms


def _dummy_image(size: int = 200) -> Image.Image:
    return Image.new("RGB", (size, size), color=(128, 64, 32))


def test_train_transforms_output_shape():
    t = get_transforms("train", image_size=128)
    result = t(_dummy_image())
    assert result.shape == (3, 128, 128)


def test_test_transforms_output_shape():
    t = get_transforms("test", image_size=128)
    result = t(_dummy_image())
    assert result.shape == (3, 128, 128)


def test_default_image_size_is_128():
    t = get_transforms("test")
    result = t(_dummy_image())
    assert result.shape == (3, 128, 128)


def test_transforms_output_in_neg1_to_1_range():
    t = get_transforms("test", image_size=64)
    result = t(_dummy_image())
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_train_transforms_resize_before_crop():
    # Input smaller than image_size+30 should still produce correct output shape
    t = get_transforms("train", image_size=64)
    result = t(_dummy_image(size=50))
    assert result.shape == (3, 64, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_transforms.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `transforms.py`**

```python
# kaoanime/utils/transforms.py
from typing import Literal

from torchvision import transforms


def get_transforms(mode: Literal["train", "test"], image_size: int = 128) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if mode == "train":
        return transforms.Compose([
            transforms.Resize(image_size + 30),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_transforms.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add kaoanime/utils/transforms.py tests/test_transforms.py
git commit -m "feat: add get_transforms utility"
```

---

### Task 4: `dataset.py` — UnpairedImageDataset

**Files:**
- Create: `kaoanime/utils/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset.py
from pathlib import Path
import torch
from PIL import Image
from kaoanime.utils.dataset import UnpairedImageDataset
from kaoanime.utils.transforms import get_transforms


def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (64, 64), color=(i * 10, 0, 0)).save(directory / f"{i}.jpg")


def test_len_is_max_of_both_domains(tmp_path):
    _make_images(tmp_path / "A", 5)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(tmp_path / "A", tmp_path / "B", get_transforms("test", 64))
    assert len(ds) == 5


def test_getitem_returns_a_and_b_tensors(tmp_path):
    _make_images(tmp_path / "A", 3)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(tmp_path / "A", tmp_path / "B", get_transforms("test", 64))
    item = ds[0]
    assert set(item.keys()) == {"A", "B"}
    assert item["A"].shape == (3, 64, 64)
    assert item["B"].shape == (3, 64, 64)


def test_b_wraps_around_when_a_is_larger(tmp_path):
    _make_images(tmp_path / "A", 5)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(tmp_path / "A", tmp_path / "B", get_transforms("test", 64))
    # index 4: A[4], B[4 % 3 = 1] — both must be valid
    item = ds[4]
    assert item["B"].shape == (3, 64, 64)


def test_accepts_string_paths(tmp_path):
    _make_images(tmp_path / "A", 2)
    _make_images(tmp_path / "B", 2)
    ds = UnpairedImageDataset(str(tmp_path / "A"), str(tmp_path / "B"), get_transforms("test", 64))
    assert len(ds) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `dataset.py`**

```python
# kaoanime/utils/dataset.py
from pathlib import Path
from typing import Callable

from torch import Tensor
from torch.utils.data import Dataset

from .image import load_image

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UnpairedImageDataset(Dataset):
    def __init__(self, root_a: str | Path, root_b: str | Path, transform: Callable) -> None:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add kaoanime/utils/dataset.py tests/test_dataset.py
git commit -m "feat: add UnpairedImageDataset"
```

---

### Task 5: `dataloader.py` — create_dataloader

**Files:**
- Create: `kaoanime/utils/dataloader.py`
- Create: `tests/test_dataloader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataloader.py
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from kaoanime.utils.dataloader import create_dataloader
from kaoanime.utils.dataset import UnpairedImageDataset
from kaoanime.utils.transforms import get_transforms


def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (64, 64)).save(directory / f"{i}.jpg")


def test_create_dataloader_returns_dataloader():
    ds = TensorDataset(torch.zeros(10, 3, 64, 64))
    dl = create_dataloader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    assert isinstance(dl, DataLoader)


def test_create_dataloader_respects_batch_size(tmp_path):
    _make_images(tmp_path / "A", 4)
    _make_images(tmp_path / "B", 4)
    ds = UnpairedImageDataset(tmp_path / "A", tmp_path / "B", get_transforms("test", 64))
    dl = create_dataloader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(dl))
    assert batch["A"].shape == (2, 3, 64, 64)
    assert batch["B"].shape == (2, 3, 64, 64)


def test_create_dataloader_drop_last(tmp_path):
    # 3 samples, batch_size=2 -> drop_last means only 1 batch of 2
    _make_images(tmp_path / "A", 3)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(tmp_path / "A", tmp_path / "B", get_transforms("test", 64))
    dl = create_dataloader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    batches = list(dl)
    assert len(batches) == 1
    assert batches[0]["A"].shape[0] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_dataloader.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `dataloader.py`**

```python
# kaoanime/utils/dataloader.py
from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_dataloader.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add kaoanime/utils/dataloader.py tests/test_dataloader.py
git commit -m "feat: add create_dataloader utility"
```

---

### Task 6: `__init__.py` re-exports + full suite

**Files:**
- Modify: `kaoanime/utils/__init__.py`
- Create: `tests/test_utils_exports.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_utils_exports.py
from kaoanime import utils


def test_all_public_symbols_exported():
    assert hasattr(utils, "load_image")
    assert hasattr(utils, "save_image")
    assert hasattr(utils, "get_transforms")
    assert hasattr(utils, "UnpairedImageDataset")
    assert hasattr(utils, "create_dataloader")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_utils_exports.py -v
```

Expected: `AttributeError` or assertion error (symbols not yet re-exported).

- [ ] **Step 3: Implement `__init__.py`**

```python
# kaoanime/utils/__init__.py
from .dataloader import create_dataloader
from .dataset import UnpairedImageDataset
from .image import load_image, save_image
from .transforms import get_transforms

__all__ = [
    "load_image",
    "save_image",
    "get_transforms",
    "UnpairedImageDataset",
    "create_dataloader",
]
```

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all 18 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add kaoanime/utils/__init__.py tests/test_utils_exports.py
git commit -m "feat: wire up kaoanime.utils public API"
```
