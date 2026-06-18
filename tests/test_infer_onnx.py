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
    arr = np.stack([grid, grid[::-1], np.full_like(grid, 128)], axis=-1).astype(
        np.uint8
    )
    Image.fromarray(arr).save(path)


def test_preprocess_shape_dtype_range(tmp_path):
    p = tmp_path / "x.png"
    _gradient_image(p)
    arr = preprocess_image(p, image_size=128)
    assert arr.shape == (1, 3, 128, 128)
    assert arr.dtype == np.float32
    assert arr.min() >= -1.0 and arr.max() <= 1.0


def test_preprocess_approx_matches_test_transform(tmp_path):
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
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
