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
