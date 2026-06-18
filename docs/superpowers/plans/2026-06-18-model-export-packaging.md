# Model Export & Production Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the NOT transport map to ONNX, provide a TensorRT conversion script, and ship a torch-free onnxruntime inference script — plus README production docs.

**Architecture:** Three `scripts/` entries — `export_onnx.py` (checkpoint → ONNX + parity check), `export_tensorrt.sh` (trtexec FP16 wrapper), `infer_onnx.py` (numpy/PIL preprocess + onnxruntime + postprocess, torch-free by default; alignment lazy-imported). No new package.

**Tech Stack:** PyTorch (export only), onnx, onnxruntime, numpy, Pillow, fire, pytest. `uv run python` always.

---

## File Structure

| File                         | Responsibility                                              |
| ---------------------------- | ----------------------------------------------------------- |
| `pyproject.toml`             | Add `onnx`, `onnxruntime` deps                              |
| `scripts/export_onnx.py`     | `export_module` + `export` (checkpoint → ONNX + parity)     |
| `scripts/infer_onnx.py`      | `preprocess_image`, `postprocess`, `main` (onnxruntime CLI) |
| `scripts/export_tensorrt.sh` | trtexec FP16 wrapper (no engine built here)                 |
| `README.md`                  | Production preparation + Infer (ONNX) docs                  |
| `tests/test_export_onnx.py`  | parity export on a tiny UNet                                |
| `tests/test_infer_onnx.py`   | preprocess/postprocess + torch-free import                  |

---

## Task 1: Add onnx + onnxruntime dependencies

**Files:** Modify `pyproject.toml`

- [ ] **Step 1: Add deps** — in `[project] dependencies`, insert (keep alphabetical-ish):

```toml
    "onnx>=1.16",
    "onnxruntime>=1.18",
```

- [ ] **Step 2: Sync and verify**

Run:

```bash
uv sync
uv run python -c "import onnx, onnxruntime; print('onnx', onnx.__version__, 'ort', onnxruntime.__version__)"
```

Expected: versions print, no error.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add onnx and onnxruntime for model export"
```

---

## Task 2: `scripts/export_onnx.py`

**Files:** Create `scripts/export_onnx.py`, `tests/test_export_onnx.py`

- [ ] **Step 1: Write the failing test** — `tests/test_export_onnx.py`:

```python
from pathlib import Path

import onnxruntime as ort
import torch

from kaoanime.models import UNetGenerator
from scripts.export_onnx import export_module


def test_export_module_writes_onnx_and_passes_parity(tmp_path):
    transport = UNetGenerator(num_filters=8)
    out = tmp_path / "m.onnx"
    # export_module runs an internal parity check and raises on mismatch
    export_module(transport, str(out), image_size=128)
    assert out.exists()


def test_exported_onnx_supports_dynamic_batch(tmp_path):
    transport = UNetGenerator(num_filters=8)
    out = tmp_path / "m.onnx"
    export_module(transport, str(out), image_size=128)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for batch in (1, 4):
        result = sess.run(None, {"input": torch.randn(batch, 3, 128, 128).numpy()})[0]
        assert result.shape == (batch, 3, 128, 128)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_export_onnx.py -v`
Expected: FAIL — `scripts.export_onnx` does not exist.

- [ ] **Step 3: Implement** — `scripts/export_onnx.py`:

```python
#!/usr/bin/env python
"""Export the NOT transport map (T) to ONNX, verifying parity with PyTorch.

Example:
    uv run python scripts/export_onnx.py --checkpoint checkpoints/not_ep10.ckpt \\
        --out models/export/model.onnx
"""
from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import torch

from kaoanime.config import Config
from kaoanime.model_not import NOTModel


def _verify_parity(transport: torch.nn.Module, onnx_path: str, image_size: int, tol: float = 1e-3) -> None:
    import onnxruntime as ort

    sample = torch.randn(2, 3, image_size, image_size)
    with torch.no_grad():
        torch_out = transport(sample).numpy()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": sample.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    if max_diff > tol:
        raise ValueError(f"ONNX parity check failed: max abs diff {max_diff:.2e} > {tol:.0e}")


def export_module(transport: torch.nn.Module, out: str, image_size: int = 128, opset: int = 17) -> str:
    """Export a transport module to ONNX (dynamic batch) and verify parity."""
    transport.eval()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        transport,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset,
    )
    _verify_parity(transport, str(out_path), image_size)
    return str(out_path)


def export(
    checkpoint: str,
    out: str = "models/export/model.onnx",
    t_filters: int = 48,
    image_size: int = 128,
    opset: int = 17,
) -> str:
    """Load a NOT checkpoint and export its transport map T to ONNX."""
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    cfg = Config()
    cfg.not_.t_filters = t_filters
    model = NOTModel.load_from_checkpoint(
        str(ckpt), cfg=cfg, map_location="cpu", strict=False
    )
    model.eval()
    path = export_module(model.T, out, image_size=image_size, opset=opset)
    print(f"Exported ONNX to {path}")
    return path


if __name__ == "__main__":
    fire.Fire(export)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_export_onnx.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format scripts/export_onnx.py tests/test_export_onnx.py
uv run ruff check --fix scripts/export_onnx.py tests/test_export_onnx.py
uv run pytest tests/test_export_onnx.py -q
git add scripts/export_onnx.py tests/test_export_onnx.py
git commit -m "feat(scripts): export_onnx — NOT transport to ONNX with parity check"
```

---

## Task 3: `scripts/infer_onnx.py`

**Files:** Create `scripts/infer_onnx.py`, `tests/test_infer_onnx.py`

- [ ] **Step 1: Write the failing test** — `tests/test_infer_onnx.py`:

```python
import subprocess
import sys

import numpy as np
from PIL import Image

from scripts.infer_onnx import postprocess, preprocess_image


def _gradient_image(path, w=200, h=170):
    # smooth content so PIL vs torchvision resize stay close
    xs = np.linspace(0, 255, w, dtype=np.float32)
    ys = np.linspace(0, 255, h, dtype=np.float32)
    grid = (xs[None, :] + ys[:, None]) / 2.0
    arr = np.stack([grid, grid[::-1], np.full_like(grid, 128)], axis=-1).astype(np.uint8)
    Image.fromarray(arr).save(path)


def test_preprocess_shape_dtype_range(tmp_path):
    p = tmp_path / "x.png"
    _gradient_image(p)
    arr = preprocess_image(p, image_size=128)
    assert arr.shape == (1, 3, 128, 128)
    assert arr.dtype == np.float32
    assert arr.min() >= -1.0 and arr.max() <= 1.0


def test_preprocess_approx_matches_test_transform(tmp_path):
    import torch
    from kaoanime.utils.transforms import get_transforms

    p = tmp_path / "x.png"
    _gradient_image(p)
    mine = preprocess_image(p, image_size=128)[0]
    ref = get_transforms("test", image_size=128)(Image.open(p).convert("RGB")).numpy()
    assert np.abs(mine - ref).mean() < 5e-2


def test_postprocess_roundtrip():
    out = postprocess(np.zeros((1, 3, 8, 8), dtype=np.float32))  # 0.0 -> ~127
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8
    assert 120 <= int(out.mean()) <= 135


def test_module_import_is_torch_free():
    code = "import scripts.infer_onnx, sys; print('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_infer_onnx.py -v`
Expected: FAIL — `scripts.infer_onnx` does not exist.

- [ ] **Step 3: Implement** — `scripts/infer_onnx.py`:

```python
#!/usr/bin/env python
"""Torch-free ONNX inference (selfie -> anime) for the exported NOT transport map.

Example:
    uv run python scripts/infer_onnx.py --onnx models/export/model.onnx \\
        --input data/demo/testA --output-dir outputs/onnx
"""
from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
from PIL import Image

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def preprocess_image(path, image_size: int = 128, align: bool = False) -> np.ndarray:
    """Load an image and produce a (1, 3, S, S) float32 array in [-1, 1].

    Mirrors transforms.get_transforms("test"): Resize((S, S)) -> [0,1] -> normalize.
    Alignment (MediaPipe) is optional and lazy-imported only when requested.
    """
    img = Image.open(path).convert("RGB")
    if align:
        from kaoanime.utils.align import AlignFaceProcessor

        processor = AlignFaceProcessor()
        aligned = processor.align(np.array(img), size=image_size)
        processor.close()
        if aligned is not None:
            img = Image.fromarray(aligned)
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(arr, dtype=np.float32)


def postprocess(array: np.ndarray) -> np.ndarray:
    """Convert a (1, 3, H, W) or (3, H, W) array in [-1, 1] to a uint8 HWC image."""
    arr = np.asarray(array)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, -1.0, 1.0)
    arr = (arr + 1.0) / 2.0
    return (arr.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)


def main(
    onnx: str,
    input: str,
    output_dir: str = "outputs/onnx",
    image_size: int = 128,
    align: bool = False,
) -> None:
    import onnxruntime as ort

    onnx_path = Path(onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    in_path = Path(input)
    if in_path.is_file():
        paths = [in_path]
    else:
        paths = sorted(p for p in in_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {in_path!r}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for image_path in paths:
        arr = preprocess_image(image_path, image_size=image_size, align=align)
        out = session.run(None, {"input": arr})[0]
        Image.fromarray(postprocess(out)).save(out_dir / image_path.name)
    print(f"Saved {len(paths)} image(s) to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_infer_onnx.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format scripts/infer_onnx.py tests/test_infer_onnx.py
uv run ruff check --fix scripts/infer_onnx.py tests/test_infer_onnx.py
uv run pytest tests/test_infer_onnx.py -q
git add scripts/infer_onnx.py tests/test_infer_onnx.py
git commit -m "feat(scripts): infer_onnx — torch-free onnxruntime inference"
```

---

## Task 4: `scripts/export_tensorrt.sh`

**Files:** Create `scripts/export_tensorrt.sh`

- [ ] **Step 1: Implement** — `scripts/export_tensorrt.sh`:

```bash
#!/usr/bin/env bash
# Convert an ONNX model to a TensorRT FP16 engine via trtexec.
# Requires TensorRT (trtexec) installed on the target machine.
#
# Usage: bash scripts/export_tensorrt.sh <model.onnx> <model.engine>
set -euo pipefail

ONNX="${1:?usage: export_tensorrt.sh <model.onnx> <model.engine>}"
ENGINE="${2:?usage: export_tensorrt.sh <model.onnx> <model.engine>}"

trtexec \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --fp16 \
    --minShapes=input:1x3x128x128 \
    --optShapes=input:8x3x128x128 \
    --maxShapes=input:16x3x128x128
```

- [ ] **Step 2: Make executable and check syntax**

```bash
chmod +x scripts/export_tensorrt.sh
bash -n scripts/export_tensorrt.sh && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/export_tensorrt.sh
git commit -m "feat(scripts): export_tensorrt.sh — trtexec FP16 wrapper"
```

---

## Task 5: README — Production preparation + Infer (ONNX)

**Files:** Modify `README.md`

- [ ] **Step 1: Add a "Production preparation" subsection** under "Технические детали" (after the "Infer" section, before "Overall"):

```markdown
### Production preparation

The trained NOT checkpoint is packaged for production by exporting the transport
map `T` to ONNX:

\`\`\`bash
uv run python scripts/export_onnx.py --checkpoint checkpoints/not_ep10.ckpt \
 --out models/export/model.onnx
\`\`\`

Optionally build a TensorRT FP16 engine on a machine with TensorRT installed:

\`\`\`bash
bash scripts/export_tensorrt.sh models/export/model.onnx models/export/model.engine
\`\`\`

**Delivery bundle:** `model.onnx` + `scripts/infer_onnx.py`. Alignment is optional
and adds `models/face_landmarker.task` + `kaoanime/utils/align.py`. Default runtime
deps: `onnxruntime, numpy, pillow` (alignment adds `opencv-python, mediapipe`).
\`\`\`
```

- [ ] **Step 2: Add an ONNX paragraph to the existing "Infer" section** (after the torch `infer.py` example):

```markdown
Lightweight, torch-free inference on the exported ONNX model:

\`\`\`bash
uv run python scripts/infer_onnx.py --onnx models/export/model.onnx \
 --input data/demo/testA --output-dir outputs/onnx
\`\`\`
```

(Use literal triple backticks in the README; the escaped ones above are only for this plan.)

- [ ] **Step 3: Format and commit**

```bash
uv run pre-commit run prettier --files README.md
git add README.md
git commit -m "docs: document ONNX export, TensorRT, and onnxruntime inference"
```

---

## Task 6: Full test + lint gate

**Files:** none (verification)

- [ ] **Step 1:** `uv run pytest -q -p no:cacheprovider` — all pass (existing + new), 2 skipped.
- [ ] **Step 2:** `uv run pre-commit run -a` — all hooks pass.
- [ ] **Step 3:** If anything was reformatted, `git add -A && git commit -m "style: apply hooks after export packaging"`.

---

## Self-Review Notes

- **Spec coverage:** ONNX export (Task 2), TensorRT script (Task 4), torch-free onnxruntime inference (Task 3), deps (Task 1), README Production preparation + Infer (Task 5). Matches the spec's components.
- **Torch-free guarantee:** `kaoanime/__init__.py` is empty and `scripts` is a namespace package; `infer_onnx.py` top-level imports avoid torch and `kaoanime.utils.*`; alignment import is inside the `align` branch. Verified by `test_module_import_is_torch_free`.
- **Type consistency:** `export_module(transport, out, image_size, opset)` and `export(checkpoint, out, t_filters, image_size, opset)` in Task 2; `preprocess_image(path, image_size, align)`, `postprocess(array)`, `main(onnx, input, output_dir, image_size, align)` in Task 3 — used consistently in tests.
- **Parity tolerances:** ONNX vs PyTorch < 1e-3 (same framework math); preprocess vs torchvision mean-abs < 5e-2 (interpolation differs) — intentionally loose.
- **No placeholders:** every code step is complete.
