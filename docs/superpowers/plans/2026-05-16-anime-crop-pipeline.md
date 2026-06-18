# Anime Crop Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old fraction-based `_anime_crop` (scale/shift params) with a fixed 3-stage pipeline: resize to 512×512, crop 256×256 at centre+(offset_x, offset_y), let the existing transform resize to 128×128.

**Architecture:** `_anime_crop` loses its scale/shift params and gains integer pixel offsets. `DataConfig` drops three float fields and gains two int fields. `UnpairedImageDataset` and `train.py` are updated to pass the new params. Notebooks 04 and 06 are updated to use the new API. The downstream `get_transforms` resize step is unchanged — it already resizes whatever PIL image `_anime_crop` returns to `image_size`.

**Tech Stack:** Python, Pillow, pytest, Jupyter/nbconvert. Always invoke Python as `uv run python`, pytest as `uv run pytest`.

---

## File Map

| File                                | Change                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------ |
| `kaoanime/utils/dataset.py`         | Replace `_anime_crop` body + signature; update `UnpairedImageDataset.__init__` and `__getitem__` |
| `kaoanime/config.py`                | Remove `anime_scale`, `anime_shift_x`, `anime_shift_y`; add `anime_offset_x`, `anime_offset_y`   |
| `train.py`                          | Update `UnpairedImageDataset(...)` call: remove 3 old kwargs, add 2 new ones                     |
| `tests/test_dataset.py`             | Add `test_anime_crop_*` tests                                                                    |
| `notebooks/04_data_pipeline.ipynb`  | Update Section 2 cells to use new `_anime_crop(img, 0, -7)` signature                            |
| `notebooks/06_anime_pipeline.ipynb` | Update `current_pipeline` helper and offset default to `-7`                                      |

---

## Task 1 — Replace `_anime_crop` and test it

**Files:**

- Modify: `kaoanime/utils/dataset.py:37-47`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1.1: Write failing tests**

Add to `tests/test_dataset.py`:

```python
from PIL import Image as PILImage
from kaoanime.utils.dataset import _anime_crop


def test_anime_crop_output_size():
    img = PILImage.new("RGB", (400, 300))
    result = _anime_crop(img, offset_x=0, offset_y=0)
    assert result.size == (256, 256), f"Expected (256, 256), got {result.size}"


def test_anime_crop_zero_offset_is_centred():
    # Fill a gradient so we can verify the crop window.
    # At 512×512, centre crop (offset=0) runs x:[128,384], y:[128,384].
    # Pixel (0,0) of the crop should equal pixel (128,128) of the 512 image.
    img = PILImage.new("RGB", (512, 512))
    pixels = img.load()
    for y in range(512):
        for x in range(512):
            pixels[x, y] = (x % 256, y % 256, 0)
    result = _anime_crop(img, offset_x=0, offset_y=0)
    r, g, _ = result.getpixel((0, 0))
    assert r == 128 and g == 128, f"Expected (128,128,_), got ({r},{g},_)"


def test_anime_crop_negative_offset_y_shifts_up():
    # offset_y=-7 → crop centre at y=249; top-left of crop at y=121
    img = PILImage.new("RGB", (512, 512))
    pixels = img.load()
    for y in range(512):
        for x in range(512):
            pixels[x, y] = (x % 256, y % 256, 0)
    result = _anime_crop(img, offset_x=0, offset_y=-7)
    r, g, _ = result.getpixel((0, 0))
    # top-left of crop: x=128, y=256-7-128=121
    assert r == 128 and g == 121, f"Expected (128,121,_), got ({r},{g},_)"


def test_anime_crop_works_on_non_square_input():
    img = PILImage.new("RGB", (800, 600))
    result = _anime_crop(img, offset_x=0, offset_y=-7)
    assert result.size == (256, 256)
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_dataset.py::test_anime_crop_output_size \
              tests/test_dataset.py::test_anime_crop_zero_offset_is_centred \
              tests/test_dataset.py::test_anime_crop_negative_offset_y_shifts_up \
              tests/test_dataset.py::test_anime_crop_works_on_non_square_input \
              -v
```

Expected: all 4 FAIL (wrong number of arguments or old behaviour).

- [ ] **Step 1.3: Replace `_anime_crop` in `kaoanime/utils/dataset.py`**

Replace lines 37–47 (the entire `_anime_crop` function):

```python
def _anime_crop(img: Image.Image, offset_x: int = 0, offset_y: int = -7) -> Image.Image:
    """Resize to 512×512, crop 256×256 centred at (256+offset_x, 256+offset_y).

    The returned image is 256×256 PIL; the caller's transform resizes it to
    the final model input size.
    """
    s512 = img.resize((512, 512), Image.LANCZOS)
    cx = 256 + offset_x
    cy = 256 + offset_y
    return s512.crop((cx - 128, cy - 128, cx + 128, cy + 128))
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_dataset.py::test_anime_crop_output_size \
              tests/test_dataset.py::test_anime_crop_zero_offset_is_centred \
              tests/test_dataset.py::test_anime_crop_negative_offset_y_shifts_up \
              tests/test_dataset.py::test_anime_crop_works_on_non_square_input \
              -v
```

Expected: all 4 PASS.

- [ ] **Step 1.5: Run full test suite to catch regressions**

```bash
uv run pytest tests/ -v
```

Expected: all existing tests still pass (they create 64×64 images so `_anime_crop` will work; `UnpairedImageDataset` tests don't pass anime params explicitly, so defaults are used).

- [ ] **Step 1.6: Commit**

```bash
git add kaoanime/utils/dataset.py tests/test_dataset.py
git commit -m "feat: replace _anime_crop with fixed 3-stage pipeline (resize-512 → crop-256 → caller-resizes)"
```

---

## Task 2 — Update config, dataset constructor, and train.py

**Files:**

- Modify: `kaoanime/config.py:25-27`
- Modify: `kaoanime/utils/dataset.py:60-85`
- Modify: `train.py:34-36`

- [ ] **Step 2.1: Update `DataConfig` in `kaoanime/config.py`**

Replace the three old anime fields:

```python
    anime_scale  : float = 1.20
    anime_shift_x: float = 0.00
    anime_shift_y: float = -0.02
```

With:

```python
    anime_offset_x: int = 0
    anime_offset_y: int = -7
```

- [ ] **Step 2.2: Update `UnpairedImageDataset.__init__` in `kaoanime/utils/dataset.py`**

Replace the constructor signature and body for the anime fields. Change lines 67–78 from:

```python
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
```

To:

```python
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
```

- [ ] **Step 2.3: Update `__getitem__` in `kaoanime/utils/dataset.py`**

Replace line 85:

```python
        img_b = _anime_crop(img_b, self._anime_scale, self._anime_shift_x, self._anime_shift_y)
```

With:

```python
        img_b = _anime_crop(img_b, self._anime_offset_x, self._anime_offset_y)
```

- [ ] **Step 2.4: Update `train.py`**

Replace lines 34–36:

```python
        anime_scale=cfg.data.anime_scale,
        anime_shift_x=cfg.data.anime_shift_x,
        anime_shift_y=cfg.data.anime_shift_y,
```

With:

```python
        anime_offset_x=cfg.data.anime_offset_x,
        anime_offset_y=cfg.data.anime_offset_y,
```

- [ ] **Step 2.5: Verify config and dataset wire up correctly**

```bash
uv run python -c "
from kaoanime.config import Config
cfg = Config()
print(f'anime_offset_x={cfg.data.anime_offset_x}')
print(f'anime_offset_y={cfg.data.anime_offset_y}')
assert cfg.data.anime_offset_x == 0
assert cfg.data.anime_offset_y == -7
print('Config OK')
"
```

Expected:

```
anime_offset_x=0
anime_offset_y=-7
Config OK
```

- [ ] **Step 2.6: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2.7: Commit**

```bash
git add kaoanime/config.py kaoanime/utils/dataset.py train.py
git commit -m "feat: wire new anime_offset_x/y params through config, dataset, and train"
```

---

## Task 3 — Update notebooks

**Files:**

- Modify: `notebooks/04_data_pipeline.ipynb`
- Modify: `notebooks/06_anime_pipeline.ipynb`

- [ ] **Step 3.1: Update `notebooks/04_data_pipeline.ipynb` Section 2**

Open the notebook. In the cell that imports and calls `_anime_crop` (currently references `cfg.anime_scale`, `cfg.anime_shift_x`, `cfg.anime_shift_y`), replace the domain B image construction block with:

```python
b_imgs = [
    np.array(
        _anime_crop(Image.open(p).convert('RGB'), 0, -7)
        .resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    )
    for p in b_paths
]
```

Also remove or update any import or reference to `cfg.anime_scale` / `cfg.anime_shift_x` / `cfg.anime_shift_y` in that cell.

- [ ] **Step 3.2: Update `notebooks/06_anime_pipeline.ipynb`**

In the `proposed_pipeline` function cell, change the default argument to match the approved value:

```python
def proposed_pipeline(path: Path, offset_x: int = 0, offset_y: int = -7) -> dict:
```

In the `current_pipeline` function cell, update it to show the old pipeline using its original params as a historical comparison (or remove the comparison row entirely since the old API no longer exists in the codebase):

```python
def current_pipeline_historical(path: Path) -> Image.Image:
    """Old pipeline for comparison: scale-based crop then resize."""
    from PIL import Image
    img = Image.open(path).convert('RGB')
    w, h = img.size
    scale = 1.85
    crop = int(min(w, h) / scale)
    cx, cy = w / 2, h / 2 + (-0.01 * h)
    x0, y0 = int(cx - crop / 2), int(cy - crop / 2)
    cropped = img.crop((x0, y0, x0 + crop, y0 + crop))
    return cropped.resize((128, 128), Image.LANCZOS)
```

Update the Section 3 suptitle to read:

```
'Top: proposed (offset_y=−7)    Bottom: old scale-based crop (historical)'
```

- [ ] **Step 3.3: Re-execute both notebooks**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_data_pipeline.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_anime_pipeline.ipynb
```

Expected: both complete without errors.

- [ ] **Step 3.4: Commit**

```bash
git add notebooks/04_data_pipeline.ipynb notebooks/06_anime_pipeline.ipynb
git commit -m "chore: update notebooks to use new anime_crop API (offset_x=0, offset_y=-7)"
```

---

## Self-review

**Spec coverage:**

- ✅ New pipeline (resize 512 → crop 256 at offset → caller resizes to 128): Task 1
- ✅ offset_x=0, offset_y=-7 defaults: Tasks 1 (function default) + 2 (config default)
- ✅ Old params removed from config: Task 2
- ✅ train.py updated: Task 2
- ✅ Notebooks updated: Task 3

**Placeholder scan:** No TBDs, no vague steps, all code blocks are concrete.

**Type consistency:** `_anime_crop(img, offset_x, offset_y)` — `int` offsets used consistently across all tasks.
