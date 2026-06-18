#!/usr/bin/env python
"""Test client for the KaoAnime Triton ensemble (selfie -> anime).

Sends raw image bytes to the ``kaoanime`` ensemble (preprocess -> transport ->
postprocess) over HTTP and saves the returned RGB images. Depends only on
``tritonclient[http]``, ``numpy`` and ``pillow`` — no torch.

Example:
    uv run python triton/client.py --input data/demo/testA --output_dir outputs/triton
"""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import tritonclient.http as httpclient
from PIL import Image

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_MODEL = "kaoanime"


def _infer_one(client: httpclient.InferenceServerClient, data: bytes) -> np.ndarray:
    """Run one image (raw encoded bytes) through the ensemble; return HWC uint8."""
    inp = httpclient.InferInput("IMAGE", [1, 1], "BYTES")
    inp.set_data_from_numpy(np.array([[data]], dtype=object))
    out = httpclient.InferRequestedOutput("OUTPUT_IMAGE")
    response = client.infer(_MODEL, inputs=[inp], outputs=[out])
    return response.as_numpy("OUTPUT_IMAGE")[0]


def main(
    input: str,
    output_dir: str = "outputs/triton",
    url: str = "localhost:8000",
) -> None:
    in_path = Path(input)
    if in_path.is_file():
        paths = [in_path]
    else:
        paths = sorted(p for p in in_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {in_path!r}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = httpclient.InferenceServerClient(url=url)
    if not client.is_model_ready(_MODEL):
        raise RuntimeError(
            f"Model {_MODEL!r} is not ready on {url}. Is the server up "
            "(triton/run_server.sh) and the repository populated?"
        )

    for image_path in paths:
        result = _infer_one(client, image_path.read_bytes())
        Image.fromarray(result).save(out_dir / image_path.name)
    print(f"Saved {len(paths)} image(s) to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
