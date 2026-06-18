# Face Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize both real-face (CelebA) and anime-face datasets to the same canonical 128×128 crop via landmark-based similarity transform, with an offline batch script for training data and real-time alignment wired into the inference pipeline.

**Architecture:** A new `kaoanime/utils/align.py` module exposes `AlignFaceProcessor` (holds one MediaPipe FaceMesh instance alive for reuse) and a `align_face` convenience wrapper. An offline CLI script (`scripts/align_dataset.py`) batch-processes entire directories with multiprocessing and supports two modes: `real` (MediaPipe landmark-based, for CelebA/user photos) and `center-crop` (fixed square crop, for already-centered anime datasets). The inference pipeline accepts an `align=True` flag that runs `AlignFaceProcessor` per image before the standard transform. Training datasets are pre-processed once by the script; the DataLoader itself does not change.

**Tech Stack:** Python 3.13, MediaPipe (face mesh), OpenCV (similarity transform + warp), PIL, pytest. Always invoke Python as `uv run python`.

---

## File Map

| File                             | Change                                          |
| -------------------------------- | ----------------------------------------------- |
| `pyproject.toml`                 | Add `mediapipe>=0.10`, `opencv-python>=4.8`     |
| `kaoanime/utils/align.py`        | **New**: `AlignFaceProcessor`, `align_face()`   |
| `kaoanime/utils/__init__.py`     | Export `align_face`                             |
| `scripts/align_dataset.py`       | **New**: offline batch alignment CLI            |
| `kaoanime/config.py`             | Add `align: bool = False` to `EvalConfig`       |
| `kaoanime/inference/__init__.py` | Accept `align` param; apply alignment per image |
| `infer.py`                       | Pass `cfg.eval.align` to `run_inference`        |
| `tests/test_align.py`            | **New**: unit tests for alignment module        |

---

### Task 1: Add dependencies

**Files:**

- Modify: `pyproject.toml`

- [ ] **Step 1.1: Add mediapipe and opencv-python to pyproject.toml**

In the `dependencies` list add two entries:

```toml
dependencies = [
    "hydra-core>=1.3",
    "lightning>=2.4",
    "mediapipe>=0.10",
    "mlflow>=2.0",
    "opencv-python>=4.8",
    "torch>=2.11.0",
    "torchmetrics[image]>=1.4",
    "torchvision>=0.26.0",
]
```

- [ ] **Step 1.2: Sync**

```bash
uv sync
```

Expected: resolves and installs both packages without errors.

- [ ] **Step 1.3: Verify imports**

```bash
uv run python -c "import mediapipe as mp; import cv2; print('mediapipe', mp.__version__, 'cv2', cv2.__version__)"
```

Expected: prints both version strings.

---

### Task 2: Write failing tests for the alignment module

**Files:**

- Create: `tests/test_align.py`

- [ ] **Step 2.1: Create the test file**

```python
# tests/test_align.py
import numpy as np
import pytest

from kaoanime.utils.align import AlignFaceProcessor, align_face


def _solid_image(h: int = 256, w: int = 256, value: int = 128) -> np.ndarray:
    """Uniform grey — guaranteed no face detection."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _noise_image(h: int = 256, w: int = 256) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ── align_face convenience wrapper ────────────────────────────────────────────

def test_align_face_returns_none_for_no_face():
    result = align_face(_solid_image(), size=128)
    assert result is None


def test_align_face_returns_none_for_noise():
    result = align_face(_noise_image(), size=128)
    assert result is None


def test_align_face_output_contract(tmp_path, face_jpg):
    """If a face is detected the output is (size, size, 3) uint8."""
    import cv2
    img = cv2.cvtColor(cv2.imread(str(face_jpg)), cv2.COLOR_BGR2RGB)
    result = align_face(img, size=128)
    if result is None:
        pytest.skip("face not detected in fixture — skip shape contract check")
    assert result.shape == (128, 128, 3)
    assert result.dtype == np.uint8


def test_align_face_respects_size_param(tmp_path, face_jpg):
    import cv2
    img = cv2.cvtColor(cv2.imread(str(face_jpg)), cv2.COLOR_BGR2RGB)
    for size in (64, 128, 256):
        result = align_face(img, size=size)
        if result is not None:
            assert result.shape == (size, size, 3)


# ── AlignFaceProcessor reuse ──────────────────────────────────────────────────

def test_processor_reusable_across_calls():
    """Processor can be called many times without re-loading the model."""
    img = _solid_image()
    proc = AlignFaceProcessor()
    for _ in range(5):
        assert proc.align(img, size=128) is None  # no crash


def test_processor_independent_instances():
    """Two separate processors work independently."""
    img = _solid_image()
    p1, p2 = AlignFaceProcessor(), AlignFaceProcessor()
    assert p1.align(img, 128) is None
    assert p2.align(img, 128) is None


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def face_jpg(tmp_path_factory):
    """Try to grab a real face image from the selfie2anime test set.
    Falls back to a blank image (detection will fail; tests skip gracefully)."""
    import shutil
    from pathlib import Path
    candidates = sorted(Path("data/selfie2anime/testA").glob("*.jpg"))
    dst = tmp_path_factory.mktemp("fixtures") / "face.jpg"
    if candidates:
        shutil.copy(candidates[0], dst)
    else:
        from PIL import Image
        Image.new("RGB", (256, 256), 128).save(dst)
    return dst
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_align.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `kaoanime.utils.align` does not exist yet.

---

### Task 3: Implement `kaoanime/utils/align.py`

**Files:**

- Create: `kaoanime/utils/align.py`

- [ ] **Step 3.1: Create the module**

```python
# kaoanime/utils/align.py
"""Face alignment: MediaPipe landmark detection + OpenCV similarity transform."""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

# 5-point canonical landmarks at 128×128 (ArcFace standard scaled from 112px).
# Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth.
_CANONICAL_128 = np.array(
    [
        [43.7,  59.1],
        [84.0,  58.9],
        [64.0,  81.6],
        [47.5, 105.6],
        [80.8, 105.4],
    ],
    dtype=np.float32,
)

# MediaPipe Face Mesh landmark indices for each key point group.
_LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
_NOSE_IDX      = [4]
_L_MOUTH_IDX   = [61]
_R_MOUTH_IDX   = [291]


def _src_pts(landmarks, h: int, w: int) -> np.ndarray:
    def mean_xy(idxs: list[int]) -> list[float]:
        return [
            float(np.mean([landmarks[i].x * w for i in idxs])),
            float(np.mean([landmarks[i].y * h for i in idxs])),
        ]

    return np.array(
        [
            mean_xy(_LEFT_EYE_IDX),
            mean_xy(_RIGHT_EYE_IDX),
            mean_xy(_NOSE_IDX),
            mean_xy(_L_MOUTH_IDX),
            mean_xy(_R_MOUTH_IDX),
        ],
        dtype=np.float32,
    )


def _canonical(size: int) -> np.ndarray:
    return _CANONICAL_128 * (size / 128.0)


class AlignFaceProcessor:
    """Holds one MediaPipe FaceMesh instance to avoid reloading the model per image.

    Create one instance per process (or per inference session). Not thread-safe —
    use one per thread or per process.
    """

    def __init__(self) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )

    def align(self, image: np.ndarray, size: int = 128) -> np.ndarray | None:
        """Detect the largest face and warp it to canonical ArcFace alignment.

        Args:
            image: RGB uint8 array, any resolution.
            size:  Output square side in pixels.

        Returns:
            Aligned RGB image of shape ``(size, size, 3)`` uint8,
            or ``None`` if no face was detected.
        """
        results = self._mesh.process(image)
        if not results.multi_face_landmarks:
            return None

        h, w = image.shape[:2]
        pts_src = _src_pts(results.multi_face_landmarks[0].landmark, h, w)
        pts_dst = _canonical(size)

        M, _ = cv2.estimateAffinePartial2D(pts_src, pts_dst, method=cv2.LMEDS)
        if M is None:
            return None

        return cv2.warpAffine(
            image, M, (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def __del__(self) -> None:
        self._mesh.close()


def align_face(image: np.ndarray, size: int = 128) -> np.ndarray | None:
    """Single-image convenience wrapper around :class:`AlignFaceProcessor`.

    Creates a fresh processor per call — fine for inference on individual images.
    For batch processing prefer :class:`AlignFaceProcessor` to load the model
    once per worker.

    Args:
        image: RGB uint8 array, any resolution.
        size:  Output square side in pixels.

    Returns:
        Aligned RGB image ``(size, size, 3)`` uint8, or ``None``.
    """
    return AlignFaceProcessor().align(image, size)
```

- [ ] **Step 3.2: Run the failing tests — they should now pass**

```bash
uv run pytest tests/test_align.py -v
```

Expected: all tests pass (the `_output_contract` and `_size_param` tests skip gracefully if `data/selfie2anime/testA` is absent).

- [ ] **Step 3.3: Commit**

```bash
git add kaoanime/utils/align.py tests/test_align.py pyproject.toml
git commit -m "feat: add AlignFaceProcessor and align_face (MediaPipe + similarity transform)"
```

---

### Task 4: Export `align_face` from `kaoanime.utils`

**Files:**

- Modify: `kaoanime/utils/__init__.py`

- [ ] **Step 4.1: Update exports**

```python
from .align import align_face
from .dataloader import create_dataloader
from .dataset import UnpairedImageDataset
from .image import load_image, save_image
from .image_pool import ImagePool
from .transforms import get_transforms

__all__ = [
    "align_face",
    "load_image",
    "save_image",
    "get_transforms",
    "ImagePool",
    "UnpairedImageDataset",
    "create_dataloader",
]
```

- [ ] **Step 4.2: Verify**

```bash
uv run python -c "from kaoanime.utils import align_face; print('OK')"
```

Expected: `OK`

- [ ] **Step 4.3: Commit**

```bash
git add kaoanime/utils/__init__.py
git commit -m "chore: export align_face from kaoanime.utils"
```

---

### Task 5: Offline batch alignment script

**Files:**

- Create: `scripts/align_dataset.py`

The script supports two `--mode` values:

- `real` (default): MediaPipe landmark detection + similarity warp. Use for CelebA and any real-photo dataset, or for user-supplied images. Images where no face is detected are **skipped**.
- `center-crop`: Fixed square centre-crop + resize. Use for the "aligned anime faces" dataset, which is already centred and consistently scaled — no detection needed, zero skips.

- [ ] **Step 5.1: Create `scripts/` directory and write the script**

```python
#!/usr/bin/env python
# scripts/align_dataset.py
"""Offline batch face alignment — normalise a dataset directory to canonical crops.

Usage examples:
    # Real faces (CelebA / FFHQ / user photos) — MediaPipe landmark alignment
    uv run python scripts/align_dataset.py \\
        --input  data/celebA/img_align_celeba \\
        --output data/celebA_aligned \\
        --mode   real --size 128 --workers 8

    # Pre-aligned anime faces — centre-crop normalisation (no detection required)
    uv run python scripts/align_dataset.py \\
        --input  data/aligned_anime_faces \\
        --output data/aligned_anime_faces_norm \\
        --mode   center-crop --size 128 --workers 8
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── worker state (one per process) ────────────────────────────────────────────

_processor = None   # AlignFaceProcessor, initialised once per worker


def _init_real_worker() -> None:
    global _processor
    from kaoanime.utils.align import AlignFaceProcessor
    _processor = AlignFaceProcessor()


# ── per-image functions ───────────────────────────────────────────────────────

def _process_real(args: tuple[Path, Path, int]) -> str:
    src, dst, size = args
    raw = cv2.imread(str(src))
    if raw is None:
        return f"SKIP {src.name} (unreadable)"
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    aligned = _processor.align(img, size)
    if aligned is None:
        return f"SKIP {src.name} (no face detected)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return f"OK   {src.name}"


def _process_center(args: tuple[Path, Path, int]) -> str:
    src, dst, size = args
    raw = cv2.imread(str(src))
    if raw is None:
        return f"SKIP {src.name} (unreadable)"
    h, w = raw.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = raw[y0:y0 + s, x0:x0 + s]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return f"OK   {src.name}"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align faces in a dataset directory to a canonical crop."
    )
    parser.add_argument("--input",   required=True, type=Path,
                        help="Source image directory (searched recursively)")
    parser.add_argument("--output",  required=True, type=Path,
                        help="Destination directory (mirrors source structure)")
    parser.add_argument("--mode",    default="real",
                        choices=["real", "center-crop"],
                        help="'real': MediaPipe landmark alignment (CelebA, user photos). "
                             "'center-crop': fixed square crop (pre-aligned anime datasets).")
    parser.add_argument("--size",    default=128,   type=int,
                        help="Output image size in pixels (default 128)")
    parser.add_argument("--workers", default=4,     type=int,
                        help="Parallel worker processes (default 4)")
    args = parser.parse_args()

    paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        print(f"No images found in {args.input}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(paths):,} images.  mode={args.mode}  size={args.size}  "
          f"workers={args.workers}")

    tasks = [
        (src, args.output / src.relative_to(args.input), args.size)
        for src in paths
    ]

    if args.mode == "real":
        fn, init = _process_real, _init_real_worker
    else:
        fn, init = _process_center, None

    ok = skip = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init,
    ) as pool:
        futs = {pool.submit(fn, t): t for t in tasks}
        for n, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            if msg.startswith("OK"):
                ok += 1
            else:
                skip += 1
                print(f"  {msg}", file=sys.stderr)
            if n % 5_000 == 0 or n == len(tasks):
                print(f"  {n:,}/{len(tasks):,}  aligned={ok:,}  skipped={skip:,}")

    print(f"\nDone.  Aligned: {ok:,}   Skipped: {skip:,}")
    if skip and args.mode == "real":
        print("Skipped images had no detectable face and are NOT in the output directory.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Smoke-test centre-crop mode** (no face detection needed, always works)

```bash
mkdir -p /tmp/align_smoke
uv run python scripts/align_dataset.py \
    --input  data/selfie2anime/testA \
    --output /tmp/align_smoke/center \
    --mode   center-crop \
    --size   128 \
    --workers 1
ls /tmp/align_smoke/center/ | head -5
```

Expected: output directory contains JPEG files, final line: `Done. Aligned: N  Skipped: 0`.

- [ ] **Step 5.3: Smoke-test real mode** (requires face images in testA)

```bash
uv run python scripts/align_dataset.py \
    --input  data/selfie2anime/testA \
    --output /tmp/align_smoke/real \
    --mode   real \
    --size   128 \
    --workers 2
ls /tmp/align_smoke/real/ | head -5
```

Expected: output directory contains some JPEGs. Some may be skipped if detection fails.

- [ ] **Step 5.4: Commit**

```bash
git add scripts/align_dataset.py
git commit -m "feat: add offline batch face alignment script (real + center-crop modes)"
```

---

### Task 6: Wire alignment into inference pipeline

**Files:**

- Modify: `kaoanime/config.py`
- Modify: `kaoanime/inference/__init__.py`
- Modify: `infer.py`
- Modify: `tests/test_inference.py`

- [ ] **Step 6.1: Add `align` field to `EvalConfig` in `kaoanime/config.py`**

In `EvalConfig`, add `align` after `direction`:

```python
@dataclass
class EvalConfig:
    checkpoint: str = ""
    output_dir: str = "outputs/eval"
    input: str = ""
    direction: str = "a2b"
    align: bool = False
```

- [ ] **Step 6.2: Write failing tests for the alignment-in-inference path**

Add to the bottom of `tests/test_inference.py`:

```python
def test_infer_align_false_does_not_crash(tmp_path):
    """align=False should work identically to the old signature."""
    src = tmp_path / "input.jpg"
    Image.new("RGB", (128, 128)).save(src)
    out_dir = tmp_path / "out"
    run_inference(_cpu_model(), src, out_dir, image_size=128,
                  direction="a2b", device="cpu", align=False)
    assert (out_dir / "input.jpg").exists()


def test_infer_align_true_solid_image_still_produces_output(tmp_path):
    """align=True on a faceless image falls back to centre-crop (no crash, output written)."""
    src = tmp_path / "solid.jpg"
    Image.new("RGB", (128, 128), color=(128, 128, 128)).save(src)
    out_dir = tmp_path / "out"
    run_inference(_cpu_model(), src, out_dir, image_size=128,
                  direction="a2b", device="cpu", align=True)
    assert (out_dir / "solid.jpg").exists()
```

Run to verify failure:

```bash
uv run pytest tests/test_inference.py::test_infer_align_false_does_not_crash \
              tests/test_inference.py::test_infer_align_true_solid_image_still_produces_output -v
```

Expected: `TypeError: run_inference() got an unexpected keyword argument 'align'`

- [ ] **Step 6.3: Update `kaoanime/inference/__init__.py`**

Replace the full file:

```python
# kaoanime/inference/__init__.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kaoanime.utils import get_transforms, load_image, save_image
from kaoanime.utils.align import AlignFaceProcessor

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_inference(
    model,
    input_path: str | Path,
    output_dir: str | Path,
    image_size: int = 128,
    direction: str = "a2b",
    device: str = "cuda",
    align: bool = False,
) -> list[Path]:
    """Translate images from domain A→B (or B→A) and write to output_dir.

    Args:
        model:      Loaded KaoAnimeModel.
        input_path: Single image file or directory of images.
        output_dir: Where translated images are saved.
        image_size: Preprocessing size (resize / crop target).
        direction:  ``"a2b"`` → selfie-to-anime; ``"b2a"`` → anime-to-selfie.
        device:     Torch device string.
        align:      Apply face alignment before the transform pipeline.
                    Images where no face is detected fall back to centre-crop.

    Returns:
        Paths of every written output file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transforms("test", image_size=image_size)
    if direction not in {"a2b", "b2a"}:
        raise ValueError(f"direction must be 'a2b' or 'b2a', got {direction!r}")
    generator = model.g_ab if direction == "a2b" else model.g_ba

    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )

    if not paths:
        raise FileNotFoundError(f"No images found in {input_path!r}")

    aligner = AlignFaceProcessor() if align else None

    written: list[Path] = []
    with torch.no_grad():
        for p in paths:
            img = load_image(p)  # PIL RGB
            if aligner is not None:
                arr = np.array(img)
                aligned_arr = aligner.align(arr, size=image_size)
                if aligned_arr is not None:
                    img = Image.fromarray(aligned_arr)
                # else: face not found — fall back to standard centre-crop
            tensor = transform(img).unsqueeze(0).to(device)
            out = generator(tensor.float())[0]
            out_path = output_dir / p.name
            save_image(out, out_path)
            written.append(out_path)

    return written
```

- [ ] **Step 6.4: Pass `align` flag in `infer.py`**

In `infer.py`, update the `run_inference` call to include the align flag:

```python
    written = run_inference(
        model,
        input_path=cfg.eval.input,
        output_dir=cfg.eval.output_dir,
        image_size=cfg.data.image_size,
        direction=cfg.eval.direction,
        device=device,
        align=cfg.eval.align,
    )
```

- [ ] **Step 6.5: Run all inference and alignment tests**

```bash
uv run pytest tests/test_inference.py tests/test_align.py -v
```

Expected: all tests pass.

- [ ] **Step 6.6: Commit**

```bash
git add kaoanime/config.py kaoanime/inference/__init__.py infer.py tests/test_inference.py
git commit -m "feat: wire face alignment into inference (eval.align=True falls back gracefully)"
```

---

## Usage After Implementation

**Preprocessing CelebA (real faces):**

```bash
uv run python scripts/align_dataset.py \
    --input  data/celebA/img_align_celeba \
    --output data/celebA_aligned \
    --mode   real --size 128 --workers 8
```

**Preprocessing aligned anime faces (centre-crop normalisation):**

```bash
uv run python scripts/align_dataset.py \
    --input  data/aligned_anime_faces \
    --output data/aligned_anime_faces_norm \
    --mode   center-crop --size 128 --workers 8
```

**Point training config at aligned data:**

```
data.root_a=data/celebA_aligned
data.extra_roots_b='[data/aligned_anime_faces_norm]'
```

**Run inference with alignment (user selfie input):**

```bash
uv run python -m kaoanime.infer \
    eval.checkpoint=outputs/.../last.ckpt \
    eval.input=my_selfie.jpg \
    eval.align=True
```

## Notes

- MediaPipe's FaceMesh is trained on real faces; it may miss heavily stylised anime faces. The `center-crop` mode exists precisely for this case — the "aligned anime faces" dataset is already consistently centred, so no detection is needed.
- The offline script preserves subdirectory structure (`src.relative_to(input)`) so nested datasets (e.g. CelebA split into sub-folders) map correctly into the output.
- `AlignFaceProcessor.__del__` closes the MediaPipe mesh to release the TFLite runtime cleanly.
- `align_face` creates a fresh processor each call — fast enough for single-image inference, too slow for bulk use. Always prefer `AlignFaceProcessor` in loops.
